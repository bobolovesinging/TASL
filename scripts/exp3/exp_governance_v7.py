"""
实验三 v7: TAS 治理效能 — Byzantine(label_flip) + Corrupt E 权重覆盖

v6→v7 核心优化:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 移除 momentum anchor（prev_global_flat 反馈环导致污染传播）
2. 纯 spatial median anchor + trust_power=5（更锋利的过滤）
3. 单客户端权重硬帽: max_weight = 2.0 / n_clients（防止任意客户端独大）
4. EMA 信任平滑: trust_ema[cid] = α*current + (1-α)*history（平滑波动）
5. V2 成员审计: 连续3轮位于信任底部30% → 标记为可疑 → 触发healing
6. 攻击频率降低: 4/10轮（更真实的攻击节奏）
"""
import csv
import os
import random
import sys
import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCKCHAIN_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
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
    attack_type: Optional[str] = None  # A1 / A2 / A3


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


# ═══════════════════════════════════════════════════════════════════════
# Trust Scoring (v7: pure spatial median + power=5 + weight cap)
# ═══════════════════════════════════════════════════════════════════════

def compute_trust_weights(flat_updates, trust_power=5.0, max_weight_ratio=2.0):
    """Cubic Trust Scoring with Spatial-Median Anchor (v7 优化版)

    关键变更(vs v6):
    - 移除 momentum anchor（反馈环）
    - trust_power=5（更锋利: 0.5^5=0.031 vs 0.5^3=0.125）
    - 单客户端权重硬帽: max_weight = max_weight_ratio / n_clients
    """
    n = len(flat_updates)
    stacked = np.stack(list(flat_updates.values()), axis=0)

    # ── 纯 spatial median anchor ──────────────────────────────────────
    anchor = np.median(stacked, axis=0)
    anchor_norm = np.linalg.norm(anchor) + 1e-12

    # ── High-power trust scoring ──────────────────────────────────────
    sim_scores = {}
    raw_scores = {}
    for cid, v in flat_updates.items():
        score = float(np.dot(v, anchor) / ((np.linalg.norm(v) + 1e-12) * anchor_norm))
        sim_scores[cid] = score
        raw_scores[cid] = max(0.0, score) ** trust_power

    # ── 归一化 ────────────────────────────────────────────────────────
    sum_w = sum(raw_scores.values())
    if sum_w > 1e-12:
        trust_weights = {cid: w / sum_w for cid, w in raw_scores.items()}
    else:
        trust_weights = {cid: 1.0 / n for cid in raw_scores}

    # ── 单客户端权重硬帽 ─────────────────────────────────────────────
    max_w = max_weight_ratio / n
    capped = {cid: min(w, max_w) for cid, w in trust_weights.items()}
    cap_sum = sum(capped.values())
    if cap_sum > 1e-12:
        trust_weights = {cid: w / cap_sum for cid, w in capped.items()}

    return sim_scores, trust_weights


# ═══════════════════════════════════════════════════════════════════════
# Local training (honest + label_flip Byzantine)
# ═══════════════════════════════════════════════════════════════════════

def train_one_client(global_state, loader, device, local_epochs=1, lr=0.02):
    """诚实客户端正常训练"""
    model = FedAvgCNN().to(device)
    model.load_state_dict(global_state)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return model.state_dict()


def train_one_client_label_flip(global_state, loader, device,
                                  local_epochs=1, lr=0.02, num_classes=10):
    """Byzantine 客户端: label_flip 攻击 (y → num_classes-1-y)"""
    model = FedAvgCNN().to(device)
    model.load_state_dict(global_state)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_flipped = num_classes - 1 - y
            optimizer.zero_grad()
            loss = criterion(model(x), y_flipped)
            loss.backward()
            optimizer.step()
    return model.state_dict()


# ═══════════════════════════════════════════════════════════════════════
# Attack scheduling (v7: 4/10 rounds instead of 6/10)
# ═══════════════════════════════════════════════════════════════════════

ATTACK_TYPES = ["A1", "A2", "A3"]


def make_chaos_schedule(rounds: int, seed: int = 42,
                        attack_types: List[str] = None,
                        attacks_per_block: int = 4) -> Dict[int, RoundAttack]:
    """每 10 轮中 attacks_per_block 轮为攻击轮（v7 默认 4/10）"""
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
# Attack implementations — E overrides Byzantine clients' weights
# ═══════════════════════════════════════════════════════════════════════

def apply_attack_a1(tampered_weights, sim_scores, byzantine_ids, rng, **kwargs):
    """A1-Blunt: E 将所有 Byzantine 客户端权重设为 0.7（大幅提权）"""
    targets = set()
    for cid in byzantine_ids:
        tampered_weights[cid] = 0.7
        targets.add(cid)
    return targets


def apply_attack_a2(tampered_weights, sim_scores, byzantine_ids, rng,
                     boost=0.25, **kwargs):
    """A2-Stealthy: E 将 Byzantine 客户端权重增加 +0.25（微调提权）"""
    targets = set()
    for cid in byzantine_ids:
        old_w = tampered_weights.get(cid, 0.0)
        tampered_weights[cid] = min(0.50, old_w + boost)
        targets.add(cid)
    return targets


def apply_attack_a3(tampered_weights, sim_scores, byzantine_ids, rng, **kwargs):
    """A3-Blunt+Suppress: E 将 Byzantine 权重设为 0.7 + 压制前2个最优诚实客户端"""
    targets = set()
    for cid in byzantine_ids:
        tampered_weights[cid] = 0.7
        targets.add(cid)
    honest_sims = {cid: sim for cid, sim in sim_scores.items() if cid not in set(byzantine_ids)}
    if honest_sims:
        top2 = sorted(honest_sims, key=honest_sims.get, reverse=True)[:2]
        for cid in top2:
            tampered_weights[cid] = 0.0
            targets.add(cid)
    return targets


ATTACK_FNS = {
    "A1": apply_attack_a1,
    "A2": apply_attack_a2,
    "A3": apply_attack_a3,
}


# ═══════════════════════════════════════════════════════════════════════
# V1 / V2 Auditing
# ═══════════════════════════════════════════════════════════════════════

def v1_audit_sampled(tampered_weights, correct_weights, num_clients,
                     sample_ratio=0.3, tolerance=2e-3, round_number=1):
    """V1 抽样审计: 比对 E 报告的权重 vs 正确计算的信任权重"""
    rng = np.random.RandomState(round_number)
    ids = list(range(num_clients))
    sample_size = max(1, int(num_clients * sample_ratio))
    sampled = rng.choice(ids, size=sample_size, replace=False).tolist()

    for cid in sampled:
        expected = correct_weights.get(cid, 0.0)
        actual = tampered_weights.get(cid, 0.0)
        if abs(actual - expected) > tolerance:
            return True, {"detected_cid": cid, "sampled": sampled,
                          "expected": expected, "actual": actual}
    return False, {"sampled": sampled}


def v2_membership_audit(trust_rank_history, num_clients, byzantine_set,
                         low_rank_threshold=0.3, consecutive_rounds=3):
    """V2 成员审计: 如果一个客户端连续 consecutive_rounds 轮排名在底部
    low_rank_threshold 分位，则标记为可疑

    trust_rank_history: dict[cid] -> list of ranks (lower = worse trust)
    返回: (detected, suspected_ids)
    """
    suspected = set()
    for cid in range(num_clients):
        ranks = trust_rank_history.get(cid, [])
        if len(ranks) < consecutive_rounds:
            continue
        recent = ranks[-consecutive_rounds:]
        threshold_rank = int(num_clients * (1 - low_rank_threshold))
        if all(r >= threshold_rank for r in recent):
            suspected.add(cid)
    return len(suspected) > 0, suspected


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
    local_epochs: int = 2,
    lr: float = 0.02,
    v1_sample_ratio: float = 0.3,
    v1_tolerance: float = 2e-3,
    byzantine_clients: Optional[List[int]] = None,
    ema_alpha: float = 0.6,
) -> GroupRunResult:
    """运行一个实验组

    v7 变更:
    - 移除 prev_global_flat（反馈环）
    - trust_power=5 + 权重硬帽
    - EMA 信任平滑
    - V2 成员审计（连续低信任检测）
    """
    assert group_name in {"single", "tas"}

    num_clients = len(client_loaders)
    global_model = FedAvgCNN().to(device)
    global_state = global_model.state_dict()

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

    # EMA 信任权重（跨轮平滑）
    ema_weights: Dict[int, float] = {}

    # V2: 信任排名历史
    trust_rank_history: Dict[int, List[int]] = defaultdict(list)

    # V2 检测计数
    v2_detections = 0

    for r in range(1, rounds + 1):
        local_updates = {}
        flat_updates = {}

        for cid in range(num_clients):
            if cid in byzantine_set:
                local_state = train_one_client_label_flip(
                    global_state, client_loaders[cid], device,
                    local_epochs=local_epochs, lr=lr)
            else:
                local_state = train_one_client(
                    global_state, client_loaders[cid], device,
                    local_epochs=local_epochs, lr=lr)

            upd = state_sub(local_state, global_state)
            local_updates[cid] = upd
            flat_updates[cid] = flatten_update(upd)

        # Compute trust weights (pure spatial median + power=5 + weight cap)
        sim_scores, trust_weights = compute_trust_weights(
            flat_updates, trust_power=5.0, max_weight_ratio=2.0)

        # ── EMA 信任平滑 ─────────────────────────────────────────────
        if ema_weights:
            smoothed = {}
            for cid in range(num_clients):
                current = trust_weights.get(cid, 0.0)
                history = ema_weights.get(cid, 1.0 / num_clients)
                smoothed[cid] = ema_alpha * current + (1 - ema_alpha) * history
            # 归一化
            s_sum = sum(smoothed.values())
            if s_sum > 1e-12:
                trust_weights = {cid: w / s_sum for cid, w in smoothed.items()}
        ema_weights = dict(trust_weights)

        # ── V2: 更新信任排名 ──────────────────────────────────────────
        sorted_cids = sorted(trust_weights.keys(), key=lambda c: trust_weights[c])
        rank_map = {cid: rank for rank, cid in enumerate(sorted_cids)}
        for cid in range(num_clients):
            trust_rank_history[cid].append(rank_map[cid])

        # Log Byzantine clients' trust weights
        byz_weights = {cid: trust_weights.get(cid, 0.0) for cid in byzantine_set}

        chaos = chaos_schedule[r] if chaos_mode else RoundAttack(active=False)
        attack_type = chaos.attack_type if chaos.active else None
        tampered_weights = dict(trust_weights)
        attack_targets: Set[int] = set()

        if chaos.active:
            attack_rounds += 1
            atk_fn = ATTACK_FNS.get(attack_type)
            if atk_fn:
                attack_counters[attack_type]["total"] += 1
                attack_targets = atk_fn(tampered_weights, sim_scores,
                                         byzantine_ids=byzantine_clients, rng=rng)

        # ── V1 + V2 Auditing (TAS group only) ─────────────────────────
        v1_detected = False
        v2_detected = False
        v2_suspected: Set[int] = set()
        v1_detail: Dict = {}
        consensus_ok = True
        healed_applied = False
        heal_source = ""

        if group_name == "tas" and chaos.active:
            # V1: 抽样审计
            v1_detected, v1_detail = v1_audit_sampled(
                tampered_weights, trust_weights, num_clients,
                sample_ratio=v1_sample_ratio, tolerance=v1_tolerance,
                round_number=r)

            # V2: 成员审计（连续3轮低信任 → 可疑）
            if not v1_detected and r >= 3:
                v2_detected, v2_suspected = v2_membership_audit(
                    trust_rank_history, num_clients, byzantine_set,
                    low_rank_threshold=0.3, consecutive_rounds=3)

            if v1_detected:
                attack_counters[attack_type]["detected"] += 1
                consensus_ok = False
                heal_source = "V1"
                blocked_rounds += 1
            elif v2_detected:
                consensus_ok = False
                heal_source = "V2"
                blocked_rounds += 1
                v2_detections += 1

        # ── Apply update ────────────────────────────────────────────────
        if group_name == "single":
            agg_update = weighted_average_updates(local_updates, tampered_weights)
            global_state = state_add(global_state, agg_update)
        else:
            if consensus_ok:
                agg_update = weighted_average_updates(local_updates, tampered_weights)
                global_state = state_add(global_state, agg_update)
            else:
                # Healing: 用正确权重重新聚合
                healed_weights = dict(trust_weights)
                # 清零 Byzantine + V2 可疑客户端
                for cid in byzantine_set:
                    healed_weights[cid] = 0.0
                for cid in v2_suspected:
                    healed_weights[cid] = 0.0
                # 重新归一化
                h_sum = sum(healed_weights.values())
                if h_sum > 1e-12:
                    healed_weights = {cid: w / h_sum for cid, w in healed_weights.items()}
                agg_update = weighted_average_updates(local_updates, healed_weights)
                global_state = state_add(global_state, agg_update)
                healed_applied = True

        global_model.load_state_dict(global_state)
        acc = evaluate(global_model, test_loader, device)
        accuracy_history.append(acc)

        logs.append({
            "round": r, "group": group_name,
            "attack_active": int(chaos.active), "attack_type": attack_type or "None",
            "v1_detected": int(v1_detected), "v2_detected": int(v2_detected),
            "consensus_passed": int(consensus_ok), "healed_applied": int(healed_applied),
            "heal_source": heal_source,
            "accuracy": float(acc),
            "byz_weights_avg": float(np.mean(list(byz_weights.values()))) if byz_weights else 0.0,
        })

        print(
            f"[{group_name.upper()}] Round {r:02d} | attack={attack_type or 'None':>3s} | "
            f"V1={int(v1_detected)} V2={int(v2_detected)} | "
            f"consensus={int(consensus_ok)} heal={int(healed_applied)}{heal_source} | "
            f"acc={acc:.2f}% | byz_w={[f'{byz_weights.get(c,0):.3f}' for c in sorted(byzantine_set)]}"
        )

    # ── Metrics ──────────────────────────────────────────────────────
    total_attacks = sum(c["total"] for c in attack_counters.values())
    total_detected = sum(c["detected"] for c in attack_counters.values())

    metrics = {
        "attack_rounds": float(attack_rounds),
        "blocked_rounds": float(blocked_rounds),
        "v1_detection_rate": float(total_detected / attack_rounds) if attack_rounds > 0 else 1.0,
        "v2_detections": float(v2_detections),
        "system_interception_rate": float(blocked_rounds / attack_rounds) if attack_rounds > 0 else 1.0,
    }
    for k in ATTACK_TYPES:
        total_k = attack_counters[k]["total"]
        detected_k = attack_counters[k]["detected"]
        metrics[f"{k}_total"] = float(total_k)
        metrics[f"{k}_detected"] = float(detected_k)
        metrics[f"{k}_detection_rate"] = float(detected_k / total_k) if total_k > 0 else 1.0

    return GroupRunResult(
        final_accuracy=float(accuracy_history[-1]) if accuracy_history else 0.0,
        best_accuracy=float(max(accuracy_history)) if accuracy_history else 0.0,
        accuracy_history=accuracy_history,
        round_logs=logs,
        metrics=metrics,
    )


# ═══════════════════════════════════════════════════════════════════════
# Plotting (Publication Quality)
# ═══════════════════════════════════════════════════════════════════════

def plot_accuracy_curves(out_path, histories: Dict[str, List[float]],
                          clean_history=None, byz_honest_history=None,
                          attack_schedule=None):
    """出版级精度曲线: 含攻击标注"""
    fig, ax = plt.subplots(figsize=(12, 6.5))

    if clean_history:
        ax.plot(clean_history, label="Clean Baseline", color="#888888",
                linestyle="--", alpha=0.7, linewidth=1.5)
    if byz_honest_history:
        ax.plot(byz_honest_history, label="Byz + Honest E", color="#2196F3",
                linestyle="--", alpha=0.7, linewidth=1.5)

    colors = {"single": "#D32F2F", "tas": "#2E7D32"}
    labels_map = {"single": "Attacked - No Governance",
                  "tas": "Attacked - TAS (V1+V2 Audit)"}

    for name, hist in histories.items():
        ax.plot(hist, label=labels_map.get(name, name),
                color=colors.get(name, "black"), linewidth=2.0, alpha=0.9)

    # 攻击轮次底色标注
    if attack_schedule:
        for r, atk in attack_schedule.items():
            if atk.active:
                ax.axvspan(r - 0.5, r + 0.5, alpha=0.08, color='#FF5722')

    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("TAS Governance: Accuracy under Byzantine (label_flip) + Corrupt E",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-5, 105)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_bar_comparison(out_path, results: Dict[str, float]):
    """出版级柱状图: 各组 best accuracy 对比"""
    groups = ["Clean\nBaseline", "Byz +\nHonest E", "Attacked\n(No Governance)",
              "Attacked\n(TAS)"]
    accs = [
        results.get("clean_best", 0),
        results.get("byz_honest_best", 0),
        results.get("single_best", 0),
        results.get("tas_best", 0),
    ]
    colors = ["#78909C", "#42A5F5", "#EF5350", "#66BB6A"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(groups, accs, color=colors, width=0.6, edgecolor='white', linewidth=1.2)
    ax.set_ylabel("Best Accuracy (%)", fontsize=12)
    ax.set_title("TAS Governance Efficacy: Best Accuracy Comparison",
                 fontsize=13, fontweight='bold')
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.set_ylim(min(accs) - 8, max(accs) + 3)

    for b, v in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.2f}%",
                ha="center", fontsize=10, fontweight="bold")

    # 添加 loss reduction 注释
    if "single_loss" in results and "tas_loss" in results:
        single_loss = results["single_loss"]
        tas_loss = results["tas_loss"]
        if single_loss > 0:
            reduction = 100.0 * (1.0 - tas_loss / single_loss)
            ax.annotate(f"TAS reduces loss by {reduction:.0f}%",
                       xy=(3, accs[3]), xytext=(2.5, accs[3] + 5),
                       fontsize=11, fontweight="bold", color="#2E7D32",
                       arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp3 v7: Optimized TAS Governance")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chaos-seed", type=int, default=2026)
    parser.add_argument("--v1-sample-ratio", type=float, default=0.3)
    parser.add_argument("--v1-tolerance", type=float, default=2e-3)
    parser.add_argument("--byzantine-ratio", type=float, default=0.3)
    parser.add_argument("--ema-alpha", type=float, default=0.6,
                        help="EMA smoothing factor (0=no smoothing, 1=no history)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client_num = int(args.clients)
    n_byz = max(1, int(client_num * args.byzantine_ratio))
    byzantine_ids = list(range(n_byz))

    print(f"[Exp3 v7] {client_num} clients, {n_byz} Byzantine (label_flip), "
          f"device={device}")
    print(f"  trust_power=5, weight_cap=2.0/n, ema_alpha={args.ema_alpha}")

    # Load data
    client_loaders, test_loader = get_data_loaders(
        "mnist", client_num=client_num, batch_size=int(args.batch_size))

    # Attack schedule: 4/10 rounds
    schedule = make_chaos_schedule(rounds=args.rounds, seed=int(args.chaos_seed),
                                    attacks_per_block=4)
    clean_schedule = {r: RoundAttack(active=False) for r in range(1, args.rounds + 1)}

    common_kwargs = dict(
        client_loaders=client_loaders, test_loader=test_loader,
        rounds=int(args.rounds), device=device,
        local_epochs=int(args.local_epochs), lr=float(args.lr),
        v1_sample_ratio=float(args.v1_sample_ratio),
        v1_tolerance=float(args.v1_tolerance),
        byzantine_clients=byzantine_ids,
        ema_alpha=float(args.ema_alpha),
    )

    # ══════════════════════════════════════════════════════════════════
    # 1. Clean baseline
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("=== 1/4: Clean Baseline (no Byzantine) ===")
    print("=" * 60)
    clean_kwargs = dict(common_kwargs)
    clean_kwargs["byzantine_clients"] = []
    clean_result = run_group("single", chaos_schedule=clean_schedule,
                              chaos_mode=False, seed=100, **clean_kwargs)

    # ══════════════════════════════════════════════════════════════════
    # 2. Byz+HonestE
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"=== 2/4: Byz+HonestE ({n_byz} Byzantine, honest E) ===")
    print("=" * 60)
    byz_honest_result = run_group("single", chaos_schedule=clean_schedule,
                                   chaos_mode=False, seed=150, **common_kwargs)

    # ══════════════════════════════════════════════════════════════════
    # 3. Attacked Single (no governance)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("=== 3/4: Attacked Single (Corrupt E, no governance) ===")
    print("=" * 60)
    attacked_single = run_group("single", chaos_schedule=schedule,
                                 chaos_mode=True, seed=200, **common_kwargs)

    # ══════════════════════════════════════════════════════════════════
    # 4. Attacked TAS (V1+V2 audit + healing)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("=== 4/4: Attacked TAS (V1+V2 audit + healing) ===")
    print("=" * 60)
    attacked_tas = run_group("tas", chaos_schedule=schedule,
                              chaos_mode=True, seed=200, **common_kwargs)

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════
    results = {
        "clean_best": clean_result.best_accuracy,
        "byz_honest_best": byz_honest_result.best_accuracy,
        "single_best": attacked_single.best_accuracy,
        "tas_best": attacked_tas.best_accuracy,
        "single_loss": max(0.0, clean_result.best_accuracy - attacked_single.best_accuracy),
        "tas_loss": max(0.0, clean_result.best_accuracy - attacked_tas.best_accuracy),
        "byz_honest_loss": max(0.0, clean_result.best_accuracy - byz_honest_result.best_accuracy),
    }
    results.update({f"tas_{k}": v for k, v in attacked_tas.metrics.items()})

    results_dir = os.path.join(BLOCKCHAIN_DIR, "results")

    # ── CSV ─────────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, "exp3_governance_data_v7.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = ["round", "scenario", "group", "attack_active", "attack_type",
                  "v1_detected", "v2_detected", "consensus_passed",
                  "healed_applied", "heal_source", "accuracy", "byz_weights_avg"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scenario, res in [("clean", clean_result), ("byz_honest", byz_honest_result),
                               ("attacked_single", attacked_single),
                               ("attacked_tas", attacked_tas)]:
            for row in res.round_logs:
                out = dict(row)
                out["scenario"] = scenario
                writer.writerow(out)

    # ── Plots ───────────────────────────────────────────────────────
    curves = {"single": attacked_single.accuracy_history,
              "tas": attacked_tas.accuracy_history}

    plot_accuracy_curves(
        os.path.join(results_dir, "exp3_governance_curves_v7.png"),
        curves, clean_history=clean_result.accuracy_history,
        byz_honest_history=byz_honest_result.accuracy_history,
        attack_schedule=schedule if True else None,
    )

    plot_bar_comparison(
        os.path.join(results_dir, "exp3_governance_bar_v7.png"),
        results,
    )

    # ── Print summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Exp3 v7] ===== Summary =====")
    print("=" * 60)
    print(f"  Clean Baseline Best:     {results['clean_best']:.2f}%")
    print(f"  Byz+HonestE Best:        {results['byz_honest_best']:.2f}%  "
          f"(loss: {results['byz_honest_loss']:.2f}%)")
    print(f"  Attacked Single Best:    {results['single_best']:.2f}%  "
          f"(loss: {results['single_loss']:.2f}%)")
    print(f"  Attacked TAS Best:       {results['tas_best']:.2f}%  "
          f"(loss: {results['tas_loss']:.2f}%)")

    if results['single_loss'] > 0:
        reduction = 100.0 * (1.0 - results['tas_loss'] / results['single_loss'])
        print(f"\n  >>> TAS reduces accuracy loss by {reduction:.1f}% "
              f"({results['single_loss']:.2f}% -> {results['tas_loss']:.2f}%)")

    print(f"\n  V1 Detection Rate:   {results.get('tas_v1_detection_rate', 0):.2%}")
    print(f"  V2 Detections:       {int(results.get('tas_v2_detections', 0))}")
    print(f"  System Interception: {results.get('tas_system_interception_rate', 0):.2%}")
    for k in ATTACK_TYPES:
        total_k = results.get(f"tas_{k}_total", 0)
        detected_k = results.get(f"tas_{k}_detected", 0)
        rate_k = results.get(f"tas_{k}_detection_rate", 0)
        print(f"  {k}: {int(detected_k)}/{int(total_k)} detected ({rate_k:.1%})")

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Plots: {results_dir}")


if __name__ == "__main__":
    main()
