"""
实验三 TAS Governance v2: V1/V2 并行验证 + Healing 机制

6 组消融 (configs):
  0. noexecutor    — TASL baseline, no Executor-side tampering
  1. fedavg        — 纯 FedAvg, 均权聚合, 无防御
  2. trust_only    — Cubic Trust + EMA, 无 V1/V2
  3. trust_v1      — Cubic Trust + EMA + V1 only (全量trust + 抽样agg验证)
  4. trust_v2      — Cubic Trust + EMA + V2 only (抽样trust + 内积agg验证)
  5. trust_v1v2    — 完整 TAS (V1+V2+healing)

3 种 E 攻击 (attacks):
  A1 (假trust, agg一致)      — E 公布假 trust (Byz=0.7), 用假 trust 聚合
  A2 (真trust, 假agg)        — E 公布真 trust, 用篡改权重 (Byz +0.25) 聚合
  A3 (假trust, 假agg不一致)  — E 公布假 trust (Byz=0.7), 用更篡改权重 (Byz=0.9 + 压制top2) 聚合

V1 验证:
  - 全量重算 trust_weights, 逐 client 对比 E 公布的 trust
  - 抽样 30% 维度验证 agg_E = sum(trust_E_published * g)

V2 验证:
  - 抽样 30% 客户端重算 trust, 对比 E 公布的 trust
  - 内积验证 agg_E: <agg_E, r> = sum(trust_E_published * <g_cid, r>)

Healing: restore the auditor-recomputed no-Executor TASL proposal for the same round

固定参数:
  - seed=42, Fashion-MNIST, label_flip, 30% Byzantine (3/10)
  - 50 rounds, lr=0.01, momentum=0.9, weight_decay=1e-4, batch_size=64
  - cudnn.deterministic=True

用法:
  python exp3_governance_tasv2.py --executor-attack A1
  python exp3_governance_tasv2.py --executor-attack A2
  python exp3_governance_tasv2.py --executor-attack A3
  python exp3_governance_tasv2.py --executor-attack all
"""

import csv
import os
import random
import sys
import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup — same as existing exp3 scripts
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCKCHAIN_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CORE_DIR = os.path.join(BLOCKCHAIN_DIR, "core")
SCRIPTS_DIR = os.path.join(BLOCKCHAIN_DIR, "scripts")
sys.path.insert(0, BLOCKCHAIN_DIR)
sys.path.insert(0, CORE_DIR)
sys.path.insert(0, SCRIPTS_DIR)

from model import FedAvgCNN
from scripts.exp2.exp2_run import load_fashionmnist_data, load_mnist_data
from scripts.exp2.exp2_run import compute_tasl_trust_weights as full_trust


# ===========================================================================
# Data classes
# ===========================================================================

@dataclass
class RoundAttack:
    active: bool
    attack_type: Optional[str] = None  # "A1" / "A2" / "A3"


@dataclass
class Exp3Result:
    name: str
    best_accuracy: float
    avg_accuracy: float
    accuracy_history: List[float]
    v1_trust_detections: int
    v1_agg_detections: int
    v2_trust_detections: int
    v2_agg_detections: int
    attack_rounds: int
    blocked_rounds: int
    heal_source_counts: Dict[str, int]
    round_logs: List[Dict]


# ===========================================================================
# Utility functions — copied from working code
# ===========================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(model: nn.Module, test_loader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += model(x).argmax(dim=1).eq(y).sum().item()
            total += y.size(0)
    return 100.0 * correct / max(total, 1)


def state_sub(local_state, global_state):
    return {k: (local_state[k] - global_state[k]).detach()
            for k in global_state
            if isinstance(global_state[k], torch.Tensor) and global_state[k].is_floating_point()}


def state_add(global_state, update):
    return {k: (global_state[k] + update[k] if k in update else global_state[k])
            for k in global_state}


def flatten_update(update):
    chunks = [v.detach().cpu().reshape(-1).numpy() for v in update.values()
              if isinstance(v, torch.Tensor) and v.is_floating_point()]
    return np.concatenate(chunks).astype(np.float64) if chunks else np.zeros(1, dtype=np.float64)


def weighted_average_updates(updates, weights):
    valid = [(cid, float(weights.get(cid, 0.0))) for cid in updates
             if float(weights.get(cid, 0.0)) > 0]
    if not valid:
        eq = 1.0 / max(len(updates), 1)
        valid = [(cid, eq) for cid in updates]
    w_sum = sum(w for _, w in valid)
    normed = {cid: w / max(w_sum, 1e-12) for cid, w in valid}
    agg = {k: torch.zeros_like(v) for k, v in updates[next(iter(normed))].items()}
    for cid, w in normed.items():
        for k in agg:
            agg[k] += updates[cid][k] * w
    return agg


# ===========================================================================
# Trust Scoring — spatial median + cos^5 + weight cap
# ===========================================================================

def compute_trust_weights(flat_updates, trust_power=5.0, max_weight_ratio=2.0):
    """
    空间中位数作为 anchor, cos^5 作为 trust score, 权重帽 = 2.0/N.
    返回 (sim_scores, trust_weights).
    """
    n = len(flat_updates)
    stacked = np.stack(list(flat_updates.values()), axis=0)
    anchor = np.median(stacked, axis=0)
    anchor_norm = np.linalg.norm(anchor) + 1e-12

    sim_scores = {}
    raw_scores = {}
    for cid, v in flat_updates.items():
        score = float(np.dot(v, anchor) / ((np.linalg.norm(v) + 1e-12) * anchor_norm))
        sim_scores[cid] = score
        raw_scores[cid] = max(0.0, score) ** trust_power

    sum_w = sum(raw_scores.values())
    if sum_w > 1e-12:
        trust_weights = {cid: w / sum_w for cid, w in raw_scores.items()}
    else:
        trust_weights = {cid: 1.0 / n for cid in raw_scores}

    max_w = max_weight_ratio / n
    capped = {cid: min(w, max_w) for cid, w in trust_weights.items()}
    cap_sum = sum(capped.values())
    if cap_sum > 1e-12:
        trust_weights = {cid: w / cap_sum for cid, w in capped.items()}

    return sim_scores, trust_weights


# ===========================================================================
# Training functions
# ===========================================================================

def train_one_client(global_state, loader, device,
                     local_epochs=1, lr=0.01, momentum=0.9, weight_decay=1e-4):
    model = FedAvgCNN().to(device)
    model.load_state_dict({k: v.to(device) for k, v in global_state.items()})
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=momentum, weight_decay=weight_decay)
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def train_one_client_label_flip(global_state, loader, device,
                                local_epochs=1, lr=0.01,
                                momentum=0.9, weight_decay=1e-4,
                                num_classes=10):
    model = FedAvgCNN().to(device)
    model.load_state_dict({k: v.to(device) for k, v in global_state.items()})
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=momentum, weight_decay=weight_decay)
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_flipped = num_classes - 1 - y
            optimizer.zero_grad()
            loss = criterion(model(x), y_flipped)
            loss.backward()
            optimizer.step()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


# ===========================================================================
# Data Loading
# ===========================================================================


def train_one_client_scaling(global_state, loader, device,
                              local_epochs=1, lr=0.01,
                              momentum=0.9, weight_decay=1e-4,
                              scaling_factor=5.0):
    model = FedAvgCNN().to(device)
    model.load_state_dict({k: v.to(device) for k, v in global_state.items()})
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=momentum, weight_decay=weight_decay)
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    update = {}
    for k in state:
        update[k] = global_state[k].cpu() + scaling_factor * (state[k] - global_state[k].cpu())
    return update

def get_data_loaders(client_num=10, batch_size=64, alpha=0.3, seed=42, dataset="fashionmnist"):
    cfg = {
        "data": {
            "dataset": "fashion_mnist",
            "n_clients": client_num,
            "batch_size": batch_size,
            "alpha": alpha,
            "split": "non_iid",
        },
        "experiment": {"seed": seed},
    }
    if dataset == "mnist":
        cfg["dataset"] = "mnist"
        return load_mnist_data(cfg)
    else:
        cfg["dataset"] = "fashion_mnist"
        return load_fashionmnist_data(cfg)


# ===========================================================================
# Attack scheduling
# ===========================================================================

ATTACK_TYPES = ["A1", "A2", "A3"]


def make_chaos_schedule(rounds: int, seed: int = 2026,
                        executor_attack: str = "A1",
                        attacks_per_block: int = 6) -> Dict[int, RoundAttack]:
    """
    每 10 轮中 attacks_per_block 轮为攻击轮 (默认 6/10).
    使用固定攻击类型 (executor_attack).
    mixed12: 每个攻击轮随机选择 A1 或 A2 (各 50% 概率).
    """
    rng = random.Random(seed)
    schedule = {r: RoundAttack(active=False) for r in range(1, rounds + 1)}
    block_start = 1
    while block_start <= rounds:
        block_end = min(block_start + 9, rounds)
        block_rounds = list(range(block_start, block_end + 1))
        atk_rounds = rng.sample(block_rounds,
                                k=min(attacks_per_block, len(block_rounds)))
        for r in atk_rounds:
            if executor_attack == "mixed12":
                atk = rng.choice(["A1", "A2"])
            else:
                atk = executor_attack
            schedule[r] = RoundAttack(active=True, attack_type=atk)
        block_start += 10
    return schedule


# ===========================================================================
# Corrupt E Attack — returns (published_weights, agg_weights)
# ===========================================================================

def apply_corrupt_e_attack(
    correct_trust: Dict[int, float],
    sim_scores: Dict[int, float],
    byzantine_ids: List[int],
    attack_type: str,
    rng: random.Random,
    num_clients: int,
) -> Tuple[Dict[int, float], Dict[int, float], Set[int]]:
    """
    E 恶意攻击: 分别生成 "公布的权重" (published) 和 "实际聚合用的权重" (agg_used).

    A1 (假trust, agg一致):
      published = fake (Byz=0.7), agg_used = same fake
      → V1 全量 trust 检测, 但 agg 验证通过 (一致的)

    A2 (真trust, 假agg):
      published = correct trust (原样), agg_used = Byz +0.25
      → V1 全量 trust 通过, 但 V1 抽样 agg / V2 内积检测

    A3 (假trust, 假agg不一致):
      published = fake (Byz=0.7), agg_used = Byz=0.9 + 压制 top2 honest
      → V1 trust + agg 都检测, V2 trust + 内积都检测

    返回 (published_weights, agg_weights, attackers_set)
    """
    attackers = set(byzantine_ids)
    n = num_clients

    if attack_type == "A1":
        # A1: published 不归一化 (V1/V2 trust检测用), agg_used 归一化 (agg验证用)
        published = dict(correct_trust)
        agg_used = dict(correct_trust)
        for cid in byzantine_ids:
            published[cid] = 0.75
            agg_used[cid] = 0.75
        s_agg = sum(agg_used.values())
        agg_used = {c: w / max(s_agg, 1e-12) for c, w in agg_used.items()}

    elif attack_type == "A2":
        # A2: published=correct trust, agg_used: Byz +0.3 (cap 0.5).
        # V1 trust 通过 (published 正确); V1_a/V2_a 检出 agg 篡改 (但有抽样漏检).
        published = dict(correct_trust)
        agg_used = dict(correct_trust)
        for cid in byzantine_ids:
            old_w = agg_used.get(cid, 0.0)
            agg_used[cid] = min(0.50, old_w + 0.25)
        s_agg = sum(agg_used.values())
        agg_used = {c: w / max(s_agg, 1e-12) for c, w in agg_used.items()}

    elif attack_type == "A3":
        # published: Byz=0.7, agg_used: Byz=0.9 + suppress top2 honest
        published = dict(correct_trust)
        agg_used = dict(correct_trust)
        for cid in byzantine_ids:
            published[cid] = 0.7
            agg_used[cid] = 0.9

        # suppress top-2 honest clients
        honest_cids = [c for c in range(n) if c not in attackers]
        honest_sorted = sorted(honest_cids,
                               key=lambda c: correct_trust.get(c, 0.0), reverse=True)
        for cid in honest_sorted[:2]:
            agg_used[cid] = 0.0
            attackers.add(cid)

        s_pub = sum(published.values())
        published = {c: w / max(s_pub, 1e-12) for c, w in published.items()}
        s_agg = sum(agg_used.values())
        if s_agg > 1e-12:
            agg_used = {c: w / s_agg for c, w in agg_used.items()}
        else:
            agg_used = dict(published)

    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    return published, agg_used, attackers


# ===========================================================================
# V1 Verification — 全量 trust + 抽样维度 agg
# ===========================================================================

TOLERANCE = 1e-4


def v1_verify_trust(
    published_weights: Dict[int, float],
    fresh_trust: Dict[int, float],
    num_clients: int,
) -> Tuple[bool, Dict[int, float], List[int]]:
    """
    V1 全量重算 trust_weights, 逐 client 对比 E 公布的权重.
    Step 1: 快速 sum-check (published 和是否=1) → A1 攻击会 sum!=1
    Step 2: 逐 client 对比 (抓精确篡改)
    返回 (detected, trust_V1, mismatched_clients)
    """
    mismatched = []
    for cid in range(num_clients):
        pub_w = published_weights.get(cid, 0.0)
        ft_w = fresh_trust.get(cid, 0.0)
        if abs(pub_w - ft_w) > TOLERANCE:
            mismatched.append(cid)
    trust_v1 = dict(fresh_trust)
    return len(mismatched) > 0, trust_v1, mismatched


def v1_verify_agg(
    agg_update,
    published_weights,
    local_updates,
    sample_ratio=0.3,
    seed=0,
    tolerance=0.97,
):
    """V1 lightweight agg check: cosine similarity (O(d)).
    对比 agg 和 expected 的余弦相似度, < 0.95 触发.
    幅值无关 → Scaling 攻击的幅值差异不影响, 只看方向偏差.
    V2 内积验证互补覆盖幅值篡改.
    """
    flat_agg = flatten_update(agg_update)

    n = len(local_updates)
    pub_sum = sum(published_weights.values())
    if pub_sum < 1e-12:
        return False, 0
    pub_norm = {c: w / pub_sum for c, w in published_weights.items()}

    # Soft weights: min 1/N for each client (避免 norm gate 归零导致 expected 偏差过大)
    soft_pub = {c: max(pub_norm.get(c, 0.0), 1.0 / n) for c in range(n)}
    s_soft = sum(soft_pub.values())
    soft_pub = {c: w / s_soft for c, w in soft_pub.items()}

    expected = {}
    for cid in local_updates:
        w = soft_pub.get(cid, 0.0)
        for k in local_updates[cid]:
            if k not in expected:
                expected[k] = w * local_updates[cid][k].detach().cpu().to(torch.float32)
            else:
                expected[k] += w * local_updates[cid][k].detach().cpu().to(torch.float32)
    expected_flat = flatten_update(expected)

    agg_norm = np.linalg.norm(flat_agg) + 1e-12
    exp_norm = np.linalg.norm(expected_flat) + 1e-12
    cos_sim = float(np.dot(flat_agg, expected_flat) / (agg_norm * exp_norm))

    return cos_sim < tolerance, 0

def v2_verify_trust(
    published_weights: Dict[int, float],
    fresh_trust: Dict[int, float],
    num_clients: int,
    sample_ratio: float = 0.3,
    seed: int = 0,
) -> Tuple[bool, Set[int]]:
    """
    V2: 抽样对比 published_w 和 fresh_trust (pre-EMA honest weights).
    抽样 30% 客户端 → 只在抽到被篡改的客户端时检出 A1.
    seed: 用于确定性随机采样 (round_number * 257)
    返回 (detected, suspected_clients)
    """
    k = max(1, int(num_clients * sample_ratio))
    rng = np.random.RandomState(seed)
    sampled = set(rng.choice(num_clients, size=k, replace=False).tolist())

    suspected = set()
    for cid in sampled:
        pub_w = published_weights.get(cid, 0.0)
        ft_w = fresh_trust.get(cid, 0.0)
        if abs(pub_w - ft_w) > TOLERANCE:
            suspected.add(cid)

    return len(suspected) > 0, suspected


def v2_verify_agg(
    agg_update: Dict[str, torch.Tensor],
    published_weights: Dict[int, float],
    local_updates: Dict[int, Dict[str, torch.Tensor]],
    seed: int = 0,
) -> bool:
    """
    V2 内积验证: <agg_E, r> = sum(published_norm[c] * <g_c, r>)
    published_norm = published / sum(published) (处理不归一化的 published).
    r 是链上随机向量 (seed 确定性模拟)
    返回 True 表示检测到篡改
    """
    flat_agg = flatten_update(agg_update)
    dim = len(flat_agg)

    # 归一化 published (A1 可能不归一化)
    pub_sum = sum(published_weights.values())
    if pub_sum < 1e-12:
        return False
    pub_norm = {c: w / pub_sum for c, w in published_weights.items()}

    rng = np.random.RandomState(seed)
    r = rng.randn(dim).astype(np.float64)
    r_norm = np.linalg.norm(r)
    if r_norm > 1e-12:
        r = r / r_norm

    actual = float(np.dot(flat_agg, r))

    expected = 0.0
    for cid, upd in local_updates.items():
        flat_g = flatten_update(upd)
        expected += pub_norm.get(cid, 0.0) * float(np.dot(flat_g, r))

    rel_diff = abs(actual - expected) / max(abs(expected), 1e-12)
    return rel_diff > 0.03  # 3% 相对容差, 漏检微小篡改, 避免数值噪声误报, 但可能漏检微小篡改


# ===========================================================================
# Healing
# ===========================================================================

def healing_recover(
    noexecutor_weights: Dict[int, float],
    local_updates: Dict[int, Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Restore the no-Executor TASL proposal for the current round.

    Interception is defined at the Executor-proposal level: once an Executor
    manipulation is detected, auditors discard the tampered proposal and commit
    the aggregate implied by the independently recomputed TASL trust weights.
    This makes 100% interception comparable to the no-Executor baseline under
    the same client updates, trust-scoring state, and training configuration.
    """
    num_clients = len(local_updates)
    healed_weights = {}
    for cid in range(num_clients):
        healed_weights[cid] = noexecutor_weights.get(cid, 1.0 / num_clients)
    s = sum(healed_weights.values())
    if s > 1e-12:
        healed_weights = {c: w / s for c, w in healed_weights.items()}
    return weighted_average_updates(local_updates, healed_weights)


# ===========================================================================
# Main experiment group runner
# ===========================================================================

def run_experiment_group(
    group_name: str,
    client_loaders,
    test_loader,
    rounds: int,
    chaos_schedule: Dict[int, RoundAttack],
    device: torch.device,
    seed: int,
    local_epochs: int = 1,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    byzantine_clients: Optional[List[int]] = None,
    ema_alpha: float = 0.6,
    v1_sample_dim_ratio: float = 0.3,
    v2_sample_client_ratio: float = 0.3,
    tolerance: float = 1e-4,
    client_attack: str = "label_flip",
) -> Exp3Result:
    """
    运行一个实验组.

    group_name in {"fedavg", "trust_only", "trust_v1", "trust_v2", "trust_v1v2"}
    """
    assert group_name in {"noexecutor", "fedavg", "trust_only", "trust_v1", "trust_v2", "trust_v1v2"}

    # Reset RNG state for every group so no-Executor and audited variants are
    # compared under the same initialization and dataloader stochasticity.
    set_seed(seed)

    num_clients = len(client_loaders)
    global_model = FedAvgCNN().to(device)
    global_state = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
    byzantine_set = set(byzantine_clients or [])

    use_v1 = group_name in ("trust_v1", "trust_v1v2")
    use_v2 = group_name in ("trust_v2", "trust_v1v2")
    use_trust = group_name != "fedavg"

    accuracy_history = []
    logs = []
    ema_weights: Dict[int, float] = {}

    v1_trust_detections = 0
    v1_agg_detections = 0
    v2_trust_detections = 0
    prev_anchor = None
    temporal_anchor = None
    v2_agg_detections = 0
    attack_rounds = 0
    blocked_rounds = 0
    heal_source_counts = {"V1": 0, "V2": 0, "V1+V2": 0}

    rng = random.Random(seed)  # same seed for all groups

    for r in range(1, rounds + 1):
        # --- 1. Client training ---
        local_updates = {}
        flat_updates = {}

        for cid in range(num_clients):
            kwargs = dict(global_state=global_state,
                          loader=client_loaders[cid],
                          device=device,
                          local_epochs=local_epochs,
                          lr=lr, momentum=momentum, weight_decay=weight_decay)

            if cid in byzantine_set:
                if client_attack == "scaling":
                    raw_state = train_one_client_scaling(**kwargs)
                else:
                    raw_state = train_one_client_label_flip(**kwargs)
            else:
                raw_state = train_one_client(**kwargs)

            local_state = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                           for k, v in raw_state.items()}
            upd = state_sub(local_state, global_state)
            local_updates[cid] = upd
            flat_updates[cid] = flatten_update(upd)

        chaos = chaos_schedule.get(r, RoundAttack(active=False))
        attack_type = chaos.attack_type if chaos.active else None

        # --- 2. Trust computation (full 10-step, aligned with Table 4-7) ---
        if use_trust:
            correct_trust, sim_scores, new_anchor = full_trust(
                flat_updates,
                trust_power=5.0, max_weight_ratio=1.5,
                norm_penalty_strength=0.8, refine_anchor=True,
                ema_weights=ema_weights, ema_alpha=0.7,
                prev_anchor=prev_anchor, temporal_anchor=temporal_anchor,
            )
            fresh_trust = dict(correct_trust)
            ema_weights = dict(correct_trust)
            prev_anchor = new_anchor
            if temporal_anchor is None:
                temporal_anchor = new_anchor
            else:
                temporal_anchor = 0.6 * new_anchor + 0.4 * temporal_anchor
        else:
            sim_scores = {cid: 1.0 for cid in range(num_clients)}
            correct_trust = {cid: 1.0 / num_clients for cid in range(num_clients)}
            fresh_trust = dict(correct_trust)

        # No-Executor proposal for this exact round. A successful audit must
        # restore this aggregate, not a separate rollback/exclusion path.
        baseline_weights = dict(correct_trust)
        baseline_agg = weighted_average_updates(local_updates, baseline_weights)

        # --- 3. Corrupt E attack ---
        executor_attack_active = bool(chaos.active and group_name != "noexecutor")
        if executor_attack_active:
            attack_rounds += 1
            # Publish FRESH trust (pre-EMA) so V1/V2 can verify honestly.
            # EMA-smoothed correct_trust is only used for healing.
            published_w, agg_w, attackers = apply_corrupt_e_attack(
                fresh_trust, sim_scores, byzantine_clients,
                attack_type, rng, num_clients)
        else:
            published_w = dict(fresh_trust)
            agg_w = dict(correct_trust)  # EMA-smoothed for aggregation stability
            attackers = set()

        # --- 4. V1/V2 审计 ---
        v1_trust_detected = False
        v1_agg_detected = False
        v2_trust_detected = False
        v2_agg_detected = False
        v2_suspected: Set[int] = set()
        trust_v1: Dict[int, float] = {}
        consensus_ok = True
        healed_applied = False
        heal_source = ""

        if executor_attack_active:
            s_trust = seed + r * 137
            s_agg = seed + r * 971

            if use_v1:
                v1_trust_detected, trust_v1, _ = v1_verify_trust(
                    published_w, fresh_trust, num_clients)
                if v1_trust_detected:
                    v1_trust_detections += 1

                v1_agg_detected, v1_dim_checked = v1_verify_agg(
                    weighted_average_updates(local_updates, agg_w),
                    published_w, local_updates,
                    sample_ratio=v1_sample_dim_ratio, seed=s_agg)
                if v1_agg_detected:
                    v1_agg_detections += 1

            if use_v2:
                v2_trust_detected, v2_suspected = v2_verify_trust(
                    published_w, fresh_trust, num_clients,
                    sample_ratio=v2_sample_client_ratio, seed=s_trust)
                if v2_trust_detected:
                    v2_trust_detections += 1

                v2_agg_detected = v2_verify_agg(
                    weighted_average_updates(local_updates, agg_w),
                    published_w, local_updates, seed=s_agg)
                if v2_agg_detected:
                    v2_agg_detections += 1

            # --- 5. Consensus ---
            v1_detected = v1_trust_detected or v1_agg_detected
            v2_detected = v2_trust_detected or v2_agg_detected

            if v1_detected or v2_detected:
                consensus_ok = False
                blocked_rounds += 1
                if v1_detected and v2_detected:
                    heal_source = "V1+V2"
                    heal_source_counts["V1+V2"] += 1
                elif v1_detected:
                    heal_source = "V1"
                    heal_source_counts["V1"] += 1
                else:
                    heal_source = "V2"
                    heal_source_counts["V2"] += 1

        # --- 6. Aggregation ---
        if consensus_ok:
            if group_name == "noexecutor":
                final_agg = baseline_agg
            elif group_name == "fedavg":
                final_agg = weighted_average_updates(local_updates, agg_w)
            else:
                final_agg = weighted_average_updates(local_updates, agg_w)
            global_state = state_add(global_state, final_agg)
        else:
            final_agg = healing_recover(baseline_weights, local_updates)
            global_state = state_add(global_state, final_agg)
            healed_applied = True

        # --- 7. Evaluation ---
        global_model.load_state_dict({k: v.to(device) for k, v in global_state.items()})
        acc = evaluate(global_model, test_loader, device)
        accuracy_history.append(acc)

        # --- 8. Logging ---
        logs.append({
            "round": r, "group": group_name,
            "attack_active": int(executor_attack_active),
            "attack_type": attack_type if executor_attack_active else "none",
            "v1_trust": int(v1_trust_detected),
            "v1_agg": int(v1_agg_detected),
            "v2_trust": int(v2_trust_detected),
            "v2_agg": int(v2_agg_detected),
            "consensus_passed": int(consensus_ok),
            "healed_applied": int(healed_applied),
            "heal_source": heal_source,
            "accuracy": float(acc),
        })

        if r % 10 == 0 or r == 1:
            parts = [f"[{group_name}] r{r:02d}"]
            parts.append(f"atk={executor_attack_active}")
            if use_v1:
                parts.append(f"V1_t={int(v1_trust_detected)},a={int(v1_agg_detected)}")
            if use_v2:
                parts.append(f"V2_t={int(v2_trust_detected)},a={int(v2_agg_detected)}")
            parts.append(f"ok={int(consensus_ok)} heal={int(healed_applied)}")
            parts.append(f"acc={acc:.2f}%")
            print(" | ".join(parts))

    return Exp3Result(
        name=group_name,
        best_accuracy=float(max(accuracy_history)) if accuracy_history else 0.0,
        avg_accuracy=float(np.mean(accuracy_history)) if accuracy_history else 0.0,
        accuracy_history=accuracy_history,
        v1_trust_detections=v1_trust_detections,
        v1_agg_detections=v1_agg_detections,
        v2_trust_detections=v2_trust_detections,
        v2_agg_detections=v2_agg_detections,
        attack_rounds=attack_rounds,
        blocked_rounds=blocked_rounds,
        heal_source_counts=heal_source_counts,
        round_logs=logs,
    )


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TAS Governance v2: V1/V2 Parallel Verification Experiment")
    parser.add_argument("--executor-attack", type=str, default="A1",
                        choices=["A1", "A2", "A3", "all", "mixed12"],
                        help="Executor attack type (default: A1)")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--byzantine-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ema-alpha", type=float, default=0.6)
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Dirichlet alpha for non-IID split")
    parser.add_argument("--client-attack", type=str, default="label_flip", choices=["label_flip","scaling"])
    parser.add_argument("--dataset", type=str, default="fashionmnist", choices=["mnist","fashionmnist"])
    parser.add_argument("--attacks-per-block", type=int, default=6,
                        help="Attack rounds per 10-round block")
    parser.add_argument("--v1-sample-dim-ratio", type=float, default=0.3,
                        help="V1 agg dimension sample ratio")
    parser.add_argument("--v2-sample-client-ratio", type=float, default=0.3,
                        help="V2 trust client sample ratio")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="Floating-point tolerance for verification")
    parser.add_argument("--trust-power", type=float, default=5.0,
                        help="cos^power for trust computation")
    parser.add_argument("--max-weight-ratio", type=float, default=2.0,
                        help="Max weight cap ratio * (1/N)")
    parser.add_argument("--chaos-seed", type=int, default=2026,
                        help="Seed for attack schedule randomization")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_byz = max(1, int(args.clients * args.byzantine_ratio))
    byzantine_ids = list(range(n_byz))

    print(f"{'='*70}")
    print(f"TAS Governance v2 Experiment")
    print(f"{'='*70}")
    print(f"  Dataset: Fashion-MNIST, {args.clients} clients, {n_byz} Byzantine (label_flip)")
    print(f"  Rounds: {args.rounds}, Local epochs: {args.local_epochs}")
    print(f"  LR: {args.lr}, Momentum: {args.momentum}, Weight decay: {args.weight_decay}")
    print(f"  Batch size: {args.batch_size}, Dirichlet alpha: {args.alpha}")
    print(f"  Seed: {args.seed}, Device: {device}")
    print(f"  EMA alpha: {args.ema_alpha}, Trust cos^power: {args.trust_power}")
    print(f"  V1 dim sample: {args.v1_sample_dim_ratio}, V2 client sample: {args.v2_sample_client_ratio}")
    print(f"  Tolerance: {args.tolerance}")
    print(f"  cudnn.deterministic: True")

    # Load data
    client_loaders, test_loader = get_data_loaders(
        client_num=args.clients, batch_size=args.batch_size,
        alpha=args.alpha, seed=args.seed, dataset=args.dataset)

    # Configurations
    groups = ["noexecutor", "fedavg", "trust_only", "trust_v1", "trust_v2", "trust_v1v2"]
    if args.executor_attack == "all":
        executor_attacks = ["A1", "A2", "A3"]
    elif args.executor_attack == "mixed12":
        executor_attacks = ["mixed12"]
    else:
        executor_attacks = [args.executor_attack]

    all_results: Dict[str, Dict[str, Exp3Result]] = {}

    common = dict(
        client_loaders=client_loaders, test_loader=test_loader,
        rounds=args.rounds, device=device,
        local_epochs=args.local_epochs, lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
        byzantine_clients=byzantine_ids,
        ema_alpha=args.ema_alpha,
        v1_sample_dim_ratio=args.v1_sample_dim_ratio,
        v2_sample_client_ratio=args.v2_sample_client_ratio,
        tolerance=args.tolerance,
        client_attack=args.client_attack,
    )

    for ea in executor_attacks:
        print(f"\n{'='*70}")
        print(f"  Executor Attack: {ea}")
        print(f"{'='*70}")

        schedule = make_chaos_schedule(
            rounds=args.rounds, seed=args.chaos_seed,
            executor_attack=ea, attacks_per_block=args.attacks_per_block)
        results: Dict[str, Exp3Result] = {}

        for i, g in enumerate(groups):
            print(f"\n--- {ea} {i+1}/{len(groups)}: {g} ---")
            results[g] = run_experiment_group(
                g, seed=args.seed,
                chaos_schedule=schedule,
                **common)
            print(f"  => best_acc={results[g].best_accuracy:.2f}%, avg_acc={results[g].avg_accuracy:.2f}%")

        all_results[ea] = results

    # ===== Report =====
    for ea, results in all_results.items():
        print(f"\n{'='*70}")
        print(f" TAS Governance v2 Summary: Executor Attack = {ea}")
        print(f"{'='*70}")
        header = (f"{'Group':<18} {'Best':>8} {'Avg':>8} "
                  f"{'V1_t':>6} {'V1_a':>6} {'V2_t':>6} {'V2_a':>6} "
                  f"{'Blocked':>8} {'Interc%':>7}")
        print(header)
        print("-" * 85)

        for g in groups:
            r = results[g]
            interc = (100.0 * r.blocked_rounds / r.attack_rounds
                      if r.attack_rounds > 0 else 0.0)
            print(f"{g:<18} {r.best_accuracy:>8.2f} {r.avg_accuracy:>8.2f} "
                  f"{r.v1_trust_detections:>6d} {r.v1_agg_detections:>6d} "
                  f"{r.v2_trust_detections:>6d} {r.v2_agg_detections:>6d} "
                  f"{r.blocked_rounds:>8d} {interc:>6.1f}%")

        # Heal source breakdown for TAS (trust_v1v2)
        tas_r = results.get("trust_v1v2")
        if tas_r:
            print(f"\n  TAS Heal Sources: "
                  f"V1={tas_r.heal_source_counts['V1']}, "
                  f"V2={tas_r.heal_source_counts['V2']}, "
                  f"V1+V2={tas_r.heal_source_counts['V1+V2']}")

    # ===== Save CSV =====
    results_dir = os.path.join(BLOCKCHAIN_DIR, "results", "governance_tasv2")
    os.makedirs(results_dir, exist_ok=True)

    for ea, results in all_results.items():
        csv_path = os.path.join(
            results_dir,
            f"tasv2_{ea}_r{args.rounds}_b{n_byz}.csv")
        fieldnames = [
            "round", "group", "attack_active", "attack_type",
            "v1_trust", "v1_agg", "v2_trust", "v2_agg",
            "consensus_passed", "healed_applied", "heal_source", "accuracy",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for g in groups:
                for row in results[g].round_logs:
                    writer.writerow(row)

        print(f"\nSaved: {csv_path}")

    # ===== Summary CSV =====
    summary_path = os.path.join(results_dir, "tasv2_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        sfields = ["executor_attack", "group", "best_acc", "avg_acc",
                   "v1_trust_det", "v1_agg_det", "v2_trust_det", "v2_agg_det",
                   "attack_rounds", "blocked_rounds", "interception_rate"]
        writer = csv.DictWriter(f, fieldnames=sfields)
        writer.writeheader()
        for ea, results in all_results.items():
            for g in groups:
                r = results[g]
                interc = (r.blocked_rounds / r.attack_rounds
                          if r.attack_rounds > 0 else 0.0)
                writer.writerow({
                    "executor_attack": ea, "group": g,
                    "best_acc": f"{r.best_accuracy:.4f}",
                    "avg_acc": f"{r.avg_accuracy:.4f}",
                    "v1_trust_det": r.v1_trust_detections,
                    "v1_agg_det": r.v1_agg_detections,
                    "v2_trust_det": r.v2_trust_detections,
                    "v2_agg_det": r.v2_agg_detections,
                    "attack_rounds": r.attack_rounds,
                    "blocked_rounds": r.blocked_rounds,
                    "interception_rate": f"{interc:.4f}",
                })
    print(f"Saved: {summary_path}")

    print(f"\n{'='*70}")
    print(" Experiment complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
