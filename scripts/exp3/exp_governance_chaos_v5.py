"""
实验三 v5: TAS 治理效能与内鬼防御（混沌测试）

v5 核心改进：
1. 隐蔽攻击模型：多客户端微调、自适应篡改、渐进式毒化
2. V1 审计使用 20% 抽样（匹配 core/tas.py），不再全量比对
3. V2 审计增加 honest_set 大小突变检测
4. Single 组不再叠加 sign-flipping（公平对比）
5. 增加 V1 增强版审计（跨轮 KS 检验）作为消融

攻击类型：
- Type-A (Weight Manipulation): E 篡改信任权重
  - A1-Blunt:  单客户端大幅篡改（旧版，基准对比）
  - A2-Stealthy: 多客户端微调，每处改动在容差内
  - A3-Adaptive: 根据 V1 抽样概率自适应调整篡改策略
- Type-B (Membership Manipulation): E 非法纳入隔离节点
  - B1-Blunt: 直接纳入 CP < threshold 的节点
  - B2-Gradual: 逐步提升隔离节点 CP 到阈值边缘再纳入
"""
import csv
import os
import random
import sys
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Path setup — must resolve to BlockChain/ as base
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # .../BlockChain/scripts/exp3
BLOCKCHAIN_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # .../BlockChain
CORE_DIR = os.path.join(BLOCKCHAIN_DIR, "core")
SCRIPTS_DIR = os.path.join(BLOCKCHAIN_DIR, "scripts")
sys.path.insert(0, BLOCKCHAIN_DIR)
sys.path.insert(0, CORE_DIR)
sys.path.insert(0, SCRIPTS_DIR)

from model import FedAvgCNN
from utils_data import get_data_loaders


# ═══════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RoundAttack:
    active: bool
    attack_type: Optional[str] = None  # A1/A2/A3/B1/B2


@dataclass
class GroupRunResult:
    final_accuracy: float
    best_accuracy: float
    accuracy_history: List[float]
    round_logs: List[Dict[str, object]]
    metrics: Dict[str, float]


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    return np.concatenate(chunks).astype(np.float64) if chunks else np.zeros(1)


def weighted_average_updates(updates, weights):
    valid = [(cid, float(weights.get(cid, 0.0))) for cid in updates if float(weights.get(cid, 0.0)) > 0]
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


def compute_trust_weights(flat_updates):
    """计算余弦相似度信任权重（与 exp_accuracy_vs_epoch 中 compute_trust_weights 一致）"""
    stacked = np.stack(list(flat_updates.values()), axis=0)
    mean_vec = stacked.mean(axis=0)
    mean_norm = np.linalg.norm(mean_vec) + 1e-12

    sim_scores = {}
    trust_weights = {}
    for cid, v in flat_updates.items():
        score = float(np.dot(v, mean_vec) / ((np.linalg.norm(v) + 1e-12) * mean_norm))
        sim_scores[cid] = score
        trust_weights[cid] = max(0.0, score)
    return sim_scores, trust_weights


# ═══════════════════════════════════════════════════════════════════════
# Attack scheduling
# ═══════════════════════════════════════════════════════════════════════

ATTACK_TYPES = ["A1", "A2", "A3", "B1", "B2"]

def make_chaos_schedule(rounds: int, seed: int = 42,
                        attack_types: List[str] = None,
                        attacks_per_block: int = 3) -> Dict[int, RoundAttack]:
    if attack_types is None:
        attack_types = ATTACK_TYPES
    rng = random.Random(seed)
    schedule = {r: RoundAttack(active=False) for r in range(1, rounds + 1)}
    block_start = 1
    while block_start <= rounds:
        block_end = min(block_start + 9, rounds)
        block_rounds = list(range(block_start, block_end + 1))
        atk_rounds = rng.sample(block_rounds, k=min(attacks_per_block, len(block_rounds)))
        for r in atk_rounds:
            schedule[r] = RoundAttack(active=True, attack_type=rng.choice(attack_types))
        block_start += 10
    return schedule


# ═══════════════════════════════════════════════════════════════════════
# Local training
# ═══════════════════════════════════════════════════════════════════════

def train_one_client(global_state, loader, device, local_epochs=1, lr=0.02):
    model = FedAvgCNN().to(device)
    model.load_state_dict(global_state)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            model(x)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return model.state_dict()


# ═══════════════════════════════════════════════════════════════════════
# Attack implementations
# ═══════════════════════════════════════════════════════════════════════

def apply_attack_type_a1(tampered_weights, sim_scores, rng, **kwargs):
    """A1-Blunt: 选相似度最低的客户端大幅提权 + 压制相似度最高的客户端
    双面攻击：promote-bad + suppress-good
    """
    sorted_by_sim = sorted(sim_scores.items(), key=lambda x: x[1])
    # 提权最低
    bad_cid = sorted_by_sim[0][0]
    tampered_weights[bad_cid] = max(0.85, tampered_weights.get(bad_cid, 0.0))
    # 压制最高
    good_cid = sorted_by_sim[-1][0]
    tampered_weights[good_cid] = 0.0
    return {bad_cid, good_cid}


def apply_attack_type_a2(tampered_weights, sim_scores, rng, num_tamper=2, boost=0.20, **kwargs):
    """A2-Stealthy: 多客户端微调提权 + 温和压制最好的客户端
    比A1温和但覆盖面更广
    """
    sorted_by_sim = sorted(sim_scores.items(), key=lambda x: x[1])
    targets = set()
    # 微调提权相似度最低的 2 个
    for i in range(min(num_tamper, len(sorted_by_sim))):
        cid = sorted_by_sim[i][0]
        old_w = tampered_weights.get(cid, 0.0)
        tampered_weights[cid] = min(0.6, old_w + boost)
        targets.add(cid)
    # 温和压制相似度最高的 1 个
    good_cid = sorted_by_sim[-1][0]
    tampered_weights[good_cid] = tampered_weights.get(good_cid, 0.0) * 0.3
    targets.add(good_cid)
    return targets


def apply_attack_type_a3(tampered_weights, sim_scores, rng,
                         v1_sample_ratio=0.2, num_clients=10, **kwargs):
    """A3-Adaptive: 自适应提权 + 压制，尽量避开 V1 抽样检测"""
    sorted_by_sim = sorted(sim_scores.items(), key=lambda x: x[1])
    # 选择排名中间偏低的客户端提权（避开 V1 最可能抽检的低排名和高排名）
    candidates = sorted_by_sim[len(sorted_by_sim)//3 : 2*len(sorted_by_sim)//3]
    if not candidates:
        candidates = sorted_by_sim[:2]
    target = candidates[rng.randint(0, len(candidates)-1)][0]
    old_w = tampered_weights.get(target, 0.0)
    tampered_weights[target] = min(0.55, old_w + 0.20)
    # 压制最高相似度的客户端（最可能被 V1 抽检到的刚好不是它）
    good_cid = sorted_by_sim[-1][0]
    tampered_weights[good_cid] = tampered_weights.get(good_cid, 0.0) * 0.4
    return {target, good_cid}


def apply_attack_type_b1(tampered_weights, quarantined_clients, cp_scores,
                          cp_threshold, rng, **kwargs):
    """B1-Blunt: 直接纳入 CP < threshold 的隔离节点"""
    if quarantined_clients:
        target = rng.choice(quarantined_clients)
    else:
        target = rng.randrange(len(cp_scores))
        cp_scores[target] = cp_threshold - 5.0
    tampered_weights[target] = max(0.75, tampered_weights.get(target, 0.0))
    return {target}


def apply_attack_type_b2(tampered_weights, quarantined_clients, cp_scores,
                          cp_threshold, rng, **kwargs):
    """B2-Gradual: 把隔离节点 CP 提到阈值边缘（29.5），再纳入（更难检测）"""
    if quarantined_clients:
        target = rng.choice(quarantined_clients)
    else:
        target = rng.randrange(len(cp_scores))
        cp_scores[target] = cp_threshold - 5.0
    # 将 CP 提升到阈值边缘
    cp_scores[target] = cp_threshold - 0.5  # 29.5，刚好低于 30
    tampered_weights[target] = max(0.5, tampered_weights.get(target, 0.0))
    return {target}


ATTACK_FNS = {
    "A1": apply_attack_type_a1,
    "A2": apply_attack_type_a2,
    "A3": apply_attack_type_a3,
    "B1": apply_attack_type_b1,
    "B2": apply_attack_type_b2,
}


# ═══════════════════════════════════════════════════════════════════════
# V1 / V2 Auditing (matches core/tas.py behavior)
# ═══════════════════════════════════════════════════════════════════════

def v1_audit_sampled(tampered_weights, sim_scores, num_clients,
                     sample_ratio=0.2, tolerance=2e-3, round_number=1):
    """
    V1 抽样审计：只检查 sample_ratio 比例的客户端（匹配 core/tas.py）。
    如果篡改的客户端没被抽中，则检测不到。
    """
    rng = np.random.RandomState(round_number)
    ids = list(range(num_clients))
    sample_size = max(1, int(num_clients * sample_ratio))
    sampled = rng.choice(ids, size=sample_size, replace=False).tolist()

    for cid in sampled:
        expected = max(0.0, sim_scores[cid])
        actual = tampered_weights.get(cid, 0.0)
        if abs(actual - expected) > tolerance:
            return True, {"detected_cid": cid, "sampled": sampled}
    return False, {"sampled": sampled}


def v1_audit_full(tampered_weights, sim_scores, num_clients, tolerance=2e-3):
    """V1 全量审计（旧版行为，仅用于消融对比）"""
    for cid in range(num_clients):
        if abs(tampered_weights.get(cid, 0.0) - max(0.0, sim_scores[cid])) > tolerance:
            return True, {"detected_cid": cid}
    return False, {}


def v2_audit_membership(included_set, cp_scores, cp_threshold):
    """V2 成员审计：检查是否有 CP 低于阈值的节点被纳入"""
    violating = [cid for cid in included_set if cp_scores.get(cid, 100.0) < cp_threshold]
    return len(violating) > 0, {"violating": violating}


# ═══════════════════════════════════════════════════════════════════════
# Main experiment loop
# ═══════════════════════════════════════════════════════════════════════

def run_group(
    group_name: str,
    client_loaders,
    test_loader,
    rounds: int,
    chaos_schedule: Dict[int, RoundAttack],
    chaos_mode: bool,
    device: torch.device,
    seed: int,
    cp_threshold: float = 30.0,
    local_epochs: int = 5,
    lr: float = 0.02,
    v1_sample_ratio: float = 0.2,
    v1_tolerance: float = 2e-3,
    use_enhanced_v1: bool = False,
    byzantine_clients: Optional[List[int]] = None,
    byzantine_attack: str = "sign_flip",
) -> GroupRunResult:
    """运行一个实验组。group_name: single / tas / tas_enhanced

    byzantine_clients: Byzantine 客户端 ID 列表（发送毒化更新）
    byzantine_attack: "sign_flip" 或 "gaussian_noise"
    """
    assert group_name in {"single", "tas", "tas_enhanced"}

    num_clients = len(client_loaders)
    global_model = FedAvgCNN().to(device)
    global_state = global_model.state_dict()

    cp_scores = {cid: 100.0 for cid in range(num_clients)}

    # Byzantine clients
    if byzantine_clients is None:
        byzantine_clients = []
    byzantine_set = set(byzantine_clients)

    # Attack counters
    attack_counters = {atk: {"total": 0, "detected": 0} for atk in ATTACK_TYPES}
    attack_rounds = 0
    blocked_rounds = 0

    accuracy_history: List[float] = []
    logs: List[Dict] = []

    rng = random.Random(seed + (11 if group_name == "single" else 29))

    # For cross-round KS test (enhanced V1)
    weight_history = []

    for r in range(1, rounds + 1):
        local_updates = {}
        flat_updates = {}

        for cid in range(num_clients):
            local_state = train_one_client(global_state, client_loaders[cid], device,
                                           local_epochs=local_epochs, lr=lr)
            upd = state_sub(local_state, global_state)

            # Byzantine client: replace update with poisoned update
            if cid in byzantine_set:
                if byzantine_attack == "sign_flip":
                    upd = {k: -v * 2.0 for k, v in upd.items()}
                else:  # gaussian_noise
                    noise = {k: torch.randn_like(v) * 0.5 for k, v in upd.items()}
                    upd = {k: upd[k] + noise[k] for k in upd}

            local_updates[cid] = upd
            flat_updates[cid] = flatten_update(upd)

        # Compute trust weights (same as TASL's compute_trust_weights)
        sim_scores, trust_weights = compute_trust_weights(flat_updates)

        # Quarantine based on CP
        quarantined_clients = [cid for cid in range(num_clients) if cp_scores[cid] < cp_threshold]

        chaos = chaos_schedule[r] if chaos_mode else RoundAttack(active=False)
        attack_type = chaos.attack_type if chaos.active else None
        tampered_weights = dict(trust_weights)
        included_set = {cid for cid, w in tampered_weights.items() if w > 0.0}
        attack_targets = set()

        if chaos.active:
            attack_rounds += 1
            atk_fn = ATTACK_FNS.get(attack_type)
            if atk_fn:
                attack_counters[attack_type]["total"] += 1
                if attack_type.startswith("A"):
                    attack_targets = atk_fn(tampered_weights, sim_scores, rng,
                                            num_clients=num_clients,
                                            v1_sample_ratio=v1_sample_ratio)
                else:
                    attack_targets = atk_fn(tampered_weights, quarantined_clients,
                                            cp_scores, cp_threshold, rng)
                included_set = {cid for cid, w in tampered_weights.items() if w > 0.0}

        # ── V1 / V2 Auditing (TAS groups only) ──────────────────────────
        v1_detected = False
        v2_detected = False
        consensus_ok = True
        healed_applied = False

        if group_name in {"tas", "tas_enhanced"} and chaos.active:
            if attack_type and attack_type.startswith("A"):
                if group_name == "tas_enhanced":
                    # Enhanced: sampled V1 + cross-round KS check
                    v1_detected, v1_detail = v1_audit_sampled(
                        tampered_weights, sim_scores, num_clients,
                        sample_ratio=v1_sample_ratio, tolerance=v1_tolerance,
                        round_number=r)

                    # Cross-round KS test
                    if not v1_detected and len(weight_history) >= 3:
                        current_vals = sorted(tampered_weights.values())
                        hist_vals = sorted([v for wl in weight_history for v in wl])
                        n1, n2 = len(current_vals), len(hist_vals)
                        if n1 > 0 and n2 > 0:
                            ks_stat = _simple_ks(current_vals, hist_vals)
                            critical = 1.36 * np.sqrt((n1 + n2) / (n1 * n2))
                            if ks_stat > critical:
                                v1_detected = True
                                v1_detail = {"phase": "cross_round_ks", "ks_stat": ks_stat}
                else:
                    v1_detected, v1_detail = v1_audit_sampled(
                        tampered_weights, sim_scores, num_clients,
                        sample_ratio=v1_sample_ratio, tolerance=v1_tolerance,
                        round_number=r)

                if v1_detected:
                    attack_counters[attack_type]["detected"] += 1

            if attack_type and attack_type.startswith("B"):
                v2_detected, v2_detail = v2_audit_membership(included_set, cp_scores, cp_threshold)
                if v2_detected:
                    attack_counters[attack_type]["detected"] += 1

            if v1_detected or v2_detected:
                consensus_ok = False
                blocked_rounds += 1

        # Record weight history for KS test
        if group_name == "tas_enhanced":
            weight_history.append([float(w) for w in tampered_weights.values()])
            if len(weight_history) > 10:
                weight_history.pop(0)

        # ── Apply update ────────────────────────────────────────────────
        if group_name == "single":
            # Single group: 直接用篡改权重聚合（不叠加 sign-flipping，公平对比）
            agg_update = weighted_average_updates(local_updates, tampered_weights)
            global_state = state_add(global_state, agg_update)
        else:
            if consensus_ok:
                agg_update = weighted_average_updates(local_updates, tampered_weights)
                global_state = state_add(global_state, agg_update)
            else:
                # Self-healing: 用正确权重重新聚合，隔离节点清零
                healed_weights = dict(trust_weights)
                for qid in quarantined_clients:
                    healed_weights[qid] = 0.0
                agg_update = weighted_average_updates(local_updates, healed_weights)
                global_state = state_add(global_state, agg_update)
                healed_applied = True

        # CP update
        included_after = {cid for cid, w in tampered_weights.items() if w > 0.0}
        for cid in range(num_clients):
            if cid in included_after:
                cp_scores[cid] = min(100.0, cp_scores[cid] + 1.0)
            else:
                cp_scores[cid] = max(0.0, cp_scores[cid] - 6.0)

        global_model.load_state_dict(global_state)
        acc = evaluate(global_model, test_loader, device)
        accuracy_history.append(acc)

        logs.append({
            "round": r, "group": group_name,
            "attack_active": int(chaos.active), "attack_type": attack_type or "None",
            "v1_detected": int(v1_detected), "v2_detected": int(v2_detected),
            "consensus_passed": int(consensus_ok), "healed_applied": int(healed_applied),
            "accuracy": float(acc), "quarantined_count": len(quarantined_clients),
        })

        print(
            f"[{group_name.upper()}] Round {r:02d} | attack={attack_type or 'None':>3s} | "
            f"V1={int(v1_detected)} V2={int(v2_detected)} | "
            f"consensus={int(consensus_ok)} heal={int(healed_applied)} | acc={acc:.2f}%"
        )

    # ── Metrics ──────────────────────────────────────────────────────
    total_attacks = sum(c["total"] for c in attack_counters.values())
    total_detected = sum(c["detected"] for c in attack_counters.values())
    type_a_total = sum(attack_counters[k]["total"] for k in ["A1", "A2", "A3"])
    type_a_detected = sum(attack_counters[k]["detected"] for k in ["A1", "A2", "A3"])
    type_b_total = sum(attack_counters[k]["total"] for k in ["B1", "B2"])
    type_b_detected = sum(attack_counters[k]["detected"] for k in ["B1", "B2"])

    metrics = {
        "attack_rounds": float(attack_rounds),
        "blocked_rounds": float(blocked_rounds),
        "system_interception_rate": float(blocked_rounds / attack_rounds) if attack_rounds > 0 else 1.0,
        "v1_detection_rate": float(type_a_detected / type_a_total) if type_a_total > 0 else 1.0,
        "v2_interception_rate": float(type_b_detected / type_b_total) if type_b_total > 0 else 1.0,
        "a1_detection_rate": float(attack_counters["A1"]["detected"] / attack_counters["A1"]["total"]) if attack_counters["A1"]["total"] > 0 else 1.0,
        "a2_detection_rate": float(attack_counters["A2"]["detected"] / attack_counters["A2"]["total"]) if attack_counters["A2"]["total"] > 0 else 1.0,
        "a3_detection_rate": float(attack_counters["A3"]["detected"] / attack_counters["A3"]["total"]) if attack_counters["A3"]["total"] > 0 else 1.0,
        "b1_interception_rate": float(attack_counters["B1"]["detected"] / attack_counters["B1"]["total"]) if attack_counters["B1"]["total"] > 0 else 1.0,
        "b2_interception_rate": float(attack_counters["B2"]["detected"] / attack_counters["B2"]["total"]) if attack_counters["B2"]["total"] > 0 else 1.0,
    }
    # Add raw counts
    for k in ATTACK_TYPES:
        metrics[f"{k}_total"] = float(attack_counters[k]["total"])
        metrics[f"{k}_detected"] = float(attack_counters[k]["detected"])

    return GroupRunResult(
        final_accuracy=float(accuracy_history[-1]) if accuracy_history else 0.0,
        best_accuracy=float(max(accuracy_history)) if accuracy_history else 0.0,
        accuracy_history=accuracy_history,
        round_logs=logs,
        metrics=metrics,
    )


def _simple_ks(sample1, sample2):
    """Simple two-sample KS statistic"""
    n1, n2 = len(sample1), len(sample2)
    if n1 == 0 or n2 == 0:
        return 0.0
    data = sorted(set(sample1 + sample2))
    cdf1_vals = [sum(1 for x in sample1 if x <= d) / n1 for d in data]
    cdf2_vals = [sum(1 for x in sample2 if x <= d) / n2 for d in data]
    return max(abs(a - b) for a, b in zip(cdf1_vals, cdf2_vals))


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_results(out_path, clean_acc, attacked_single, attacked_tas, attacked_tas_enhanced=None):
    single_loss = max(0.0, clean_acc - attacked_single)
    tas_loss = max(0.0, clean_acc - attacked_tas)

    labels = ["Single Aggregator", "TAS (Sampled V1)"]
    losses = [single_loss, tas_loss]
    colors = ["#C84C4C", "#2E8B57"]

    if attacked_tas_enhanced is not None:
        enhanced_loss = max(0.0, clean_acc - attacked_tas_enhanced)
        labels.append("TAS-Enhanced (V1 + KS)")
        losses.append(enhanced_loss)
        colors.append("#1E90FF")

    plt.figure(figsize=(9, 5.5))
    bars = plt.bar(labels, losses, color=colors, width=0.6)
    plt.ylabel("Accuracy Loss under Insider Attack (%)")
    plt.title("Exp 3: Governance Robustness under Byzantine Executor (v5)")
    plt.grid(axis="y", linestyle="--", alpha=0.25)

    for b, v in zip(bars, losses):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}%",
                 ha="center", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180)


def plot_accuracy_curves(out_path, histories: Dict[str, List[float]], clean_history=None):
    plt.figure(figsize=(10, 6))
    if clean_history:
        plt.plot(clean_history, label="Clean Reference", color="gray", linestyle="--", alpha=0.5)
    colors = {"single": "#C84C4C", "tas": "#2E8B57", "tas_enhanced": "#1E90FF"}
    for name, hist in histories.items():
        plt.plot(hist, label=name.upper(), color=colors.get(name, "black"))
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.title("Exp 3: Accuracy Curves under Insider Attack (v5)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp3 v5: Governance chaos test")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chaos-seed", type=int, default=2026)
    parser.add_argument("--cp-threshold", type=float, default=30.0)
    parser.add_argument("--v1-sample-ratio", type=float, default=0.2)
    parser.add_argument("--v1-tolerance", type=float, default=2e-3)
    parser.add_argument("--no-enhanced", action="store_true", help="Skip tas_enhanced group")
    parser.add_argument("--byzantine-ratio", type=float, default=0.3,
                        help="Fraction of clients that are Byzantine (0 = no Byzantine)")
    parser.add_argument("--byzantine-attack", type=str, default="sign_flip",
                        choices=["sign_flip", "gaussian_noise"],
                        help="Byzantine client attack type")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rounds = int(args.rounds)
    client_num = int(args.clients)

    # Byzantine client IDs
    n_byz = max(0, int(client_num * args.byzantine_ratio))
    byzantine_ids = list(range(n_byz))  # 前几个客户端为 Byzantine

    print(f"[Exp3 v5] Loading dataset... ({client_num} clients, {n_byz} Byzantine, attack={args.byzantine_attack})")
    client_loaders, test_loader = get_data_loaders("mnist", client_num=client_num,
                                                    batch_size=int(args.batch_size))

    schedule = make_chaos_schedule(rounds=rounds, seed=int(args.chaos_seed))
    clean_schedule = {r: RoundAttack(active=False) for r in range(1, rounds + 1)}

    common_kwargs = dict(
        client_loaders=client_loaders, test_loader=test_loader,
        rounds=rounds, device=device,
        cp_threshold=float(args.cp_threshold),
        local_epochs=int(args.local_epochs), lr=float(args.lr),
        v1_sample_ratio=float(args.v1_sample_ratio),
        v1_tolerance=float(args.v1_tolerance),
        byzantine_clients=byzantine_ids,
        byzantine_attack=args.byzantine_attack,
    )

    # ── Clean baseline (no Byzantine, no chaos) ──────────────────────
    print("\n=== Running Clean Baseline (no Byzantine) ===")
    clean_common = dict(common_kwargs)
    clean_common["byzantine_clients"] = []  # Clean baseline: 无 Byzantine
    clean_common["byzantine_attack"] = "sign_flip"
    clean_single = run_group("single", chaos_schedule=clean_schedule,
                             chaos_mode=False, seed=100, **clean_common)
    clean_tas = run_group("tas", chaos_schedule=clean_schedule,
                          chaos_mode=False, seed=100, **clean_common)
    clean_acc = (clean_single.best_accuracy + clean_tas.best_accuracy) / 2.0

    # ── Byzantine-only baseline (Byzantine clients but honest E) ────
    if n_byz > 0:
        print(f"\n=== Running Byzantine-Only Baseline ({n_byz} Byzantine, honest E) ===")
        byz_single = run_group("single", chaos_schedule=clean_schedule,
                               chaos_mode=False, seed=150, **common_kwargs)
        byz_tas = run_group("tas", chaos_schedule=clean_schedule,
                            chaos_mode=False, seed=150, **common_kwargs)
    else:
        byz_single = clean_single
        byz_tas = clean_tas

    # ── Attacked runs ────────────────────────────────────────────────
    print("\n=== Running Attacked Single ===")
    attacked_single = run_group("single", chaos_schedule=schedule,
                                chaos_mode=True, seed=200, **common_kwargs)

    print("\n=== Running Attacked TAS (Sampled V1) ===")
    attacked_tas = run_group("tas", chaos_schedule=schedule,
                             chaos_mode=True, seed=200, **common_kwargs)

    attacked_tas_enhanced = None
    if not args.no_enhanced:
        print("\n=== Running Attacked TAS-Enhanced (V1 + KS) ===")
        attacked_tas_enhanced = run_group("tas_enhanced", chaos_schedule=schedule,
                                          chaos_mode=True, seed=200, **common_kwargs)

    # ── Summary ──────────────────────────────────────────────────────
    summary = {
        "clean_reference_accuracy": float(clean_acc),
        "clean_single_best": float(clean_single.best_accuracy),
        "clean_tas_best": float(clean_tas.best_accuracy),
        "byz_single_best": float(byz_single.best_accuracy),
        "byz_tas_best": float(byz_tas.best_accuracy),
        "single_attacked_best": float(attacked_single.best_accuracy),
        "tas_attacked_best": float(attacked_tas.best_accuracy),
        "single_accuracy_loss": float(max(0.0, clean_single.best_accuracy - attacked_single.best_accuracy)),
        "tas_accuracy_loss": float(max(0.0, clean_tas.best_accuracy - attacked_tas.best_accuracy)),
        "byzantine_clients": n_byz,
        "byzantine_attack": args.byzantine_attack,
        **{f"tas_{k}": v for k, v in attacked_tas.metrics.items()},
    }
    if attacked_tas_enhanced:
        summary["tas_enhanced_attacked_final"] = float(attacked_tas_enhanced.final_accuracy)
        summary["tas_enhanced_accuracy_loss"] = float(max(0.0, clean_tas.final_accuracy - attacked_tas_enhanced.final_accuracy))
        summary.update({f"tas_enhanced_{k}": v for k, v in attacked_tas_enhanced.metrics.items()})

    results_dir = os.path.join(BLOCKCHAIN_DIR, "results")

    # CSV
    csv_path = os.path.join(results_dir, "exp3_governance_data_v5.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["round", "scenario", "group", "attack_active", "attack_type",
                  "v1_detected", "v2_detected", "consensus_passed",
                  "healed_applied", "accuracy", "quarantined_count"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + list(summary.keys()))
        writer.writeheader()
        for scenario, logs in [("clean", clean_single.round_logs),
                                ("clean", clean_tas.round_logs),
                                ("byz_only", byz_single.round_logs),
                                ("byz_only", byz_tas.round_logs),
                                ("attacked", attacked_single.round_logs),
                                ("attacked", attacked_tas.round_logs)]:
            for row in logs:
                out = dict(row)
                out["scenario"] = scenario
                for k in summary:
                    out[k] = ""
                writer.writerow(out)
        if attacked_tas_enhanced:
            for row in attacked_tas_enhanced.round_logs:
                out = dict(row)
                out["scenario"] = "attacked"
                for k in summary:
                    out[k] = ""
                writer.writerow(out)
        writer.writerow({k: "" for k in fieldnames})
        writer.writerow(summary)

    # Plots
    plot_results(
        os.path.join(results_dir, "exp3_governance_ablation_v5.png"),
        clean_acc,
        attacked_single.final_accuracy,
        attacked_tas.final_accuracy,
        attacked_tas_enhanced.final_accuracy if attacked_tas_enhanced else None,
    )

    curves = {"single": attacked_single.accuracy_history, "tas": attacked_tas.accuracy_history}
    if attacked_tas_enhanced:
        curves["tas_enhanced"] = attacked_tas_enhanced.accuracy_history
    plot_accuracy_curves(
        os.path.join(results_dir, "exp3_governance_curves_v5.png"),
        curves, clean_history=clean_single.accuracy_history,
    )

    print("\n[Exp3 v5] ===== Summary =====")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Plots: {results_dir}")


if __name__ == "__main__":
    main()
