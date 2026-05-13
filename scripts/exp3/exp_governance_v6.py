"""
实验三 v6: TAS 治理效能 — Byzantine(label_flip) + Corrupt E 权重覆盖

核心设计变更（vs v5）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 引入 Byzantine 客户端发送 label_flip 毒化更新
   - label_flip: 训练时翻转标签 y → (num_classes-1-y)
   - 比 sign_flip / gaussian_noise 更隐蔽、更真实
   - compute_trust_weights 自然赋予 Byzantine 客户端 ~0 权重

2. Corrupt E 的攻击：覆盖 Byzantine 客户端的低权重为高权重
   - 旧版(v5): 无 Byzantine → E 篡改权重无法造成实质伤害
   - 新版(v6): 有 Byzantine → E 覆盖低权重 → 毒化更新进入聚合 → 模型受损

3. 简化攻击模型：聚焦"权重覆盖"这一核心攻击
   - A1-Blunt: E 将 Byzantine 权重设为 0.5
   - A2-Stealthy: E 将 Byzantine 权重增加 +0.15
   - A3-Blunt+Suppress: E 将 Byzantine 权重设为 0.5 + 压制最优诚实客户端

实验对比组:
- Clean: 无 Byzantine, 无攻击 (基线)
- Byz+HonestE: Byzantine 存在但 E 诚实 (信任权重自然过滤)
- Attacked Single: Byzantine + Corrupt E, 无治理
- Attacked TAS: Byzantine + Corrupt E, V1/V2 审计 + healing
- Attacked TAS-Enh: Byzantine + Corrupt E, 增强审计 (V1+KS)
"""
import csv
import os
import random
import sys
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
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


def compute_trust_weights(flat_updates, trust_power=3.0,
                           prev_global_update=None):
    """Cubic Trust Scoring with Spatial-Median Anchor（与 core/gradient_aggregator.py 一致）
    
    使用 coordinate-wise median (v_space) 作为空间锚点：
    - 比 mean 更鲁棒，不受 Byzantine 异常值拉偏
    - max(0, cos_sim_to_v_space)^3 门控
    - 归一化后总和为1
    
    如果提供 prev_global_update（时间动量锚点），则使用混合锚点：
    anchor = 0.5 * prev_global + 0.5 * v_space
    """
    stacked = np.stack(list(flat_updates.values()), axis=0)

    # ── Step 1: 空间中位数锚点 (robust to Byzantine outliers) ────────
    v_space = np.median(stacked, axis=0)

    # ── Step 2: 混合锚点（可选时间动量） ──────────────────────────────
    if prev_global_update is not None:
        anchor = 0.5 * prev_global_update + 0.5 * v_space
    else:
        anchor = v_space

    anchor_norm = np.linalg.norm(anchor) + 1e-12

    # ── Step 3 & 4: Cubic Trust Score ────────────────────────────────
    sim_scores = {}
    raw_scores = {}
    for cid, v in flat_updates.items():
        score = float(np.dot(v, anchor) / ((np.linalg.norm(v) + 1e-12) * anchor_norm))
        sim_scores[cid] = score
        raw_scores[cid] = max(0.0, score) ** trust_power

    # ── Step 5: 归一化 ───────────────────────────────────────────────
    sum_w = sum(raw_scores.values())
    if sum_w > 1e-12:
        trust_weights = {cid: w / sum_w for cid, w in raw_scores.items()}
    else:
        n = len(raw_scores)
        trust_weights = {cid: 1.0 / n for cid in raw_scores}

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
            y_flipped = num_classes - 1 - y  # 标签翻转
            optimizer.zero_grad()
            loss = criterion(model(x), y_flipped)
            loss.backward()
            optimizer.step()
    return model.state_dict()


# ═══════════════════════════════════════════════════════════════════════
# Attack scheduling
# ═══════════════════════════════════════════════════════════════════════

ATTACK_TYPES = ["A1", "A2", "A3"]


def make_chaos_schedule(rounds: int, seed: int = 42,
                        attack_types: List[str] = None,
                        attacks_per_block: int = 6) -> Dict[int, RoundAttack]:
    """每 10 轮中 attacks_per_block 轮为攻击轮"""
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
    """A2-Stealthy: E 将 Byzantine 客户端权重增加 +0.25（微调提权，更难检测）"""
    targets = set()
    for cid in byzantine_ids:
        old_w = tampered_weights.get(cid, 0.0)
        tampered_weights[cid] = min(0.50, old_w + boost)
        targets.add(cid)
    return targets


def apply_attack_a3(tampered_weights, sim_scores, byzantine_ids, rng, **kwargs):
    """A3-Blunt+Suppress: E 将 Byzantine 权重设为 0.7 + 压制前2个最优诚实客户端"""
    targets = set()
    # 提权 Byzantine
    for cid in byzantine_ids:
        tampered_weights[cid] = 0.7
        targets.add(cid)
    # 压制前2个最优诚实客户端
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
                     sample_ratio=0.2, tolerance=2e-3, round_number=1):
    """V1 抽样审计: 20% 抽样，比对 E 报告的权重 vs 正确计算的信任权重

    tampered_weights: E 报告的权重（可能被篡改）
    correct_weights: 正确计算的信任权重（cubic scoring + 归一化后的）
    """
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


def _simple_ks(sample1, sample2):
    """双样本 KS 统计量"""
    n1, n2 = len(sample1), len(sample2)
    if n1 == 0 or n2 == 0:
        return 0.0
    data = sorted(set(sample1 + sample2))
    cdf1_vals = [sum(1 for x in sample1 if x <= d) / n1 for d in data]
    cdf2_vals = [sum(1 for x in sample2 if x <= d) / n2 for d in data]
    return max(abs(a - b) for a, b in zip(cdf1_vals, cdf2_vals))


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
    v1_sample_ratio: float = 0.2,
    v1_tolerance: float = 2e-3,
    use_enhanced_v1: bool = False,
    byzantine_clients: Optional[List[int]] = None,
) -> GroupRunResult:
    """运行一个实验组

    group_name: single / tas / tas_enhanced
    byzantine_clients: Byzantine 客户端 ID 列表（发送 label_flip 更新）
    """
    assert group_name in {"single", "tas", "tas_enhanced"}

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

    # For cross-round KS test (enhanced V1)
    weight_history: List[List[float]] = []

    # 时间动量锚点（上一轮聚合更新的方向）
    prev_global_flat: Optional[np.ndarray] = None

    for r in range(1, rounds + 1):
        local_updates = {}
        flat_updates = {}

        for cid in range(num_clients):
            if cid in byzantine_set:
                # Byzantine: label_flip 训练
                local_state = train_one_client_label_flip(
                    global_state, client_loaders[cid], device,
                    local_epochs=local_epochs, lr=lr)
            else:
                # 诚实客户端: 正常训练
                local_state = train_one_client(
                    global_state, client_loaders[cid], device,
                    local_epochs=local_epochs, lr=lr)

            upd = state_sub(local_state, global_state)
            local_updates[cid] = upd
            flat_updates[cid] = flatten_update(upd)

        # Compute trust weights (with spatial-median anchor + temporal momentum)
        sim_scores, trust_weights = compute_trust_weights(
            flat_updates, trust_power=3.0, prev_global_update=prev_global_flat)

        # Log Byzantine clients' trust weights for debugging
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

        # ── V1 / V2 Auditing (TAS groups only) ──────────────────────────
        v1_detected = False
        v1_detail: Dict = {}
        consensus_ok = True
        healed_applied = False

        if group_name in {"tas", "tas_enhanced"} and chaos.active:
            # V1 audit: 比对 E 报告的权重 vs 正确计算的信任权重
            if group_name == "tas_enhanced":
                # Enhanced: sampled V1 + cross-round KS check
                v1_detected, v1_detail = v1_audit_sampled(
                    tampered_weights, trust_weights, num_clients,
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
                    tampered_weights, trust_weights, num_clients,
                    sample_ratio=v1_sample_ratio, tolerance=v1_tolerance,
                    round_number=r)

            if v1_detected:
                attack_counters[attack_type]["detected"] += 1

            if v1_detected:
                consensus_ok = False
                blocked_rounds += 1

        # Record weight history for KS test
        if group_name == "tas_enhanced":
            weight_history.append([float(w) for w in tampered_weights.values()])
            if len(weight_history) > 10:
                weight_history.pop(0)

        # ── Apply update ────────────────────────────────────────────────
        if group_name == "single":
            # Single: 直接用篡改权重聚合（无治理）
            agg_update = weighted_average_updates(local_updates, tampered_weights)
            global_state = state_add(global_state, agg_update)
            # 更新时间动量
            agg_flat_vec = flatten_update(agg_update)
        else:
            if consensus_ok:
                # 审计通过或无攻击：用当前权重（可能被篡改）
                agg_update = weighted_average_updates(local_updates, tampered_weights)
                global_state = state_add(global_state, agg_update)
            else:
                # Healing: 用正确权重重新聚合，Byzantine 客户端清零
                healed_weights = dict(trust_weights)
                for cid in byzantine_set:
                    healed_weights[cid] = 0.0
                agg_update = weighted_average_updates(local_updates, healed_weights)
                global_state = state_add(global_state, agg_update)
                healed_applied = True
            agg_flat_vec = flatten_update(agg_update)

        # 更新时间动量锚点
        prev_global_flat = agg_flat_vec

        global_model.load_state_dict(global_state)
        acc = evaluate(global_model, test_loader, device)
        accuracy_history.append(acc)

        logs.append({
            "round": r, "group": group_name,
            "attack_active": int(chaos.active), "attack_type": attack_type or "None",
            "v1_detected": int(v1_detected),
            "consensus_passed": int(consensus_ok), "healed_applied": int(healed_applied),
            "accuracy": float(acc),
            "byz_weights_avg": float(np.mean(list(byz_weights.values()))) if byz_weights else 0.0,
        })

        print(
            f"[{group_name.upper()}] Round {r:02d} | attack={attack_type or 'None':>3s} | "
            f"V1={int(v1_detected)} | consensus={int(consensus_ok)} heal={int(healed_applied)} | "
            f"acc={acc:.2f}% | byz_w={[f'{byz_weights.get(c,0):.3f}' for c in byzantine_set]}"
        )

    # ── Metrics ──────────────────────────────────────────────────────
    total_attacks = sum(c["total"] for c in attack_counters.values())
    total_detected = sum(c["detected"] for c in attack_counters.values())

    metrics = {
        "attack_rounds": float(attack_rounds),
        "blocked_rounds": float(blocked_rounds),
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
# Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_accuracy_curves(out_path, histories: Dict[str, List[float]],
                          clean_history=None, byz_honest_history=None):
    """绘制精度曲线: Clean / Byz+HonestE / Attacked-Single / Attacked-TAS"""
    plt.figure(figsize=(11, 6.5))
    if clean_history:
        plt.plot(clean_history, label="Clean Baseline", color="#888888",
                 linestyle="--", alpha=0.6, linewidth=1.5)
    if byz_honest_history:
        plt.plot(byz_honest_history, label="Byz+HonestE", color="#2196F3",
                 linestyle="--", alpha=0.6, linewidth=1.5)

    colors = {"single": "#C84C4C", "tas": "#2E8B57", "tas_enhanced": "#1E90FF"}
    labels_map = {"single": "Attacked-Single", "tas": "Attacked-TAS",
                  "tas_enhanced": "Attacked-TAS-Enh"}
    for name, hist in histories.items():
        plt.plot(hist, label=labels_map.get(name, name),
                 color=colors.get(name, "black"), linewidth=1.8)

    plt.xlabel("Round", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.title("Exp3 v6: Accuracy under Byzantine(label_flip) + Corrupt E", fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_bar_comparison(out_path, results: Dict[str, float]):
    """柱状图: 各组 best accuracy 对比"""
    groups = ["Clean", "Byz+HonestE", "Attacked\nSingle", "Attacked\nTAS"]
    accs = [
        results.get("clean_best", 0),
        results.get("byz_honest_best", 0),
        results.get("single_best", 0),
        results.get("tas_best", 0),
    ]
    colors = ["#888888", "#2196F3", "#C84C4C", "#2E8B57"]

    if "tas_enhanced_best" in results:
        groups.append("Attacked\nTAS-Enh")
        accs.append(results["tas_enhanced_best"])
        colors.append("#1E90FF")

    plt.figure(figsize=(9, 5.5))
    bars = plt.bar(groups, accs, color=colors, width=0.6)
    plt.ylabel("Best Accuracy (%)", fontsize=11)
    plt.title("Exp3 v6: TAS Governance Efficacy", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.ylim(min(accs) - 5, max(accs) + 2)

    for b, v in zip(bars, accs):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.2f}%",
                 ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp3 v6: Byzantine(label_flip) + Corrupt E")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chaos-seed", type=int, default=2026)
    parser.add_argument("--v1-sample-ratio", type=float, default=0.3)
    parser.add_argument("--v1-tolerance", type=float, default=2e-3)
    parser.add_argument("--byzantine-ratio", type=float, default=0.3,
                        help="Fraction of clients that are Byzantine")
    parser.add_argument("--no-enhanced", action="store_true",
                        help="Skip tas_enhanced group")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client_num = int(args.clients)
    n_byz = max(1, int(client_num * args.byzantine_ratio))
    byzantine_ids = list(range(n_byz))  # 前 n_byz 个客户端为 Byzantine

    print(f"[Exp3 v6] {client_num} clients, {n_byz} Byzantine (label_flip), "
          f"device={device}")

    # Load data
    client_loaders, test_loader = get_data_loaders(
        "mnist", client_num=client_num, batch_size=int(args.batch_size))

    # Attack schedule: 6 out of every 10 rounds
    schedule = make_chaos_schedule(rounds=args.rounds, seed=int(args.chaos_seed),
                                    attacks_per_block=6)
    clean_schedule = {r: RoundAttack(active=False) for r in range(1, args.rounds + 1)}

    common_kwargs = dict(
        client_loaders=client_loaders, test_loader=test_loader,
        rounds=int(args.rounds), device=device,
        local_epochs=int(args.local_epochs), lr=float(args.lr),
        v1_sample_ratio=float(args.v1_sample_ratio),
        v1_tolerance=float(args.v1_tolerance),
        byzantine_clients=byzantine_ids,
    )

    # ══════════════════════════════════════════════════════════════════
    # 1. Clean baseline (no Byzantine, no attack)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("=== 1/5: Clean Baseline (no Byzantine) ===")
    print("=" * 60)
    clean_kwargs = dict(common_kwargs)
    clean_kwargs["byzantine_clients"] = []
    clean_result = run_group("single", chaos_schedule=clean_schedule,
                              chaos_mode=False, seed=100, **clean_kwargs)

    # ══════════════════════════════════════════════════════════════════
    # 2. Byz+HonestE (Byzantine present, E honest — trust weights filter)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"=== 2/5: Byz+HonestE ({n_byz} Byzantine, honest E) ===")
    print("=" * 60)
    byz_honest_result = run_group("single", chaos_schedule=clean_schedule,
                                   chaos_mode=False, seed=150, **common_kwargs)

    # ══════════════════════════════════════════════════════════════════
    # 3. Attacked Single (Byzantine + Corrupt E, no governance)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("=== 3/5: Attacked Single (Corrupt E, no governance) ===")
    print("=" * 60)
    attacked_single = run_group("single", chaos_schedule=schedule,
                                 chaos_mode=True, seed=200, **common_kwargs)

    # ══════════════════════════════════════════════════════════════════
    # 4. Attacked TAS (Byzantine + Corrupt E, V1/V2 audit + healing)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("=== 4/5: Attacked TAS (Corrupt E, V1/V2 audit) ===")
    print("=" * 60)
    attacked_tas = run_group("tas", chaos_schedule=schedule,
                              chaos_mode=True, seed=200, **common_kwargs)

    # ══════════════════════════════════════════════════════════════════
    # 5. Attacked TAS-Enhanced (V1 + cross-round KS)
    # ══════════════════════════════════════════════════════════════════
    attacked_tas_enhanced = None
    if not args.no_enhanced:
        print("\n" + "=" * 60)
        print("=== 5/5: Attacked TAS-Enhanced (V1 + KS) ===")
        print("=" * 60)
        attacked_tas_enhanced = run_group("tas_enhanced", chaos_schedule=schedule,
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
    if attacked_tas_enhanced:
        results["tas_enhanced_best"] = attacked_tas_enhanced.best_accuracy
        results["tas_enhanced_loss"] = max(0.0, clean_result.best_accuracy - attacked_tas_enhanced.best_accuracy)

    # Add TAS metrics
    results.update({f"tas_{k}": v for k, v in attacked_tas.metrics.items()})
    if attacked_tas_enhanced:
        results.update({f"tas_enhanced_{k}": v for k, v in attacked_tas_enhanced.metrics.items()})

    results_dir = os.path.join(BLOCKCHAIN_DIR, "results")

    # ── CSV ─────────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, "exp3_governance_data_v6.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = ["round", "scenario", "group", "attack_active", "attack_type",
                  "v1_detected", "consensus_passed", "healed_applied",
                  "accuracy", "byz_weights_avg"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scenario, res in [("clean", clean_result), ("byz_honest", byz_honest_result),
                               ("attacked", attacked_single), ("attacked", attacked_tas)]:
            for row in res.round_logs:
                out = dict(row)
                out["scenario"] = scenario
                writer.writerow(out)
        if attacked_tas_enhanced:
            for row in attacked_tas_enhanced.round_logs:
                out = dict(row)
                out["scenario"] = "attacked"
                writer.writerow(out)

    # ── Plots ───────────────────────────────────────────────────────
    curves = {"single": attacked_single.accuracy_history,
              "tas": attacked_tas.accuracy_history}
    if attacked_tas_enhanced:
        curves["tas_enhanced"] = attacked_tas_enhanced.accuracy_history

    plot_accuracy_curves(
        os.path.join(results_dir, "exp3_governance_curves_v6.png"),
        curves, clean_history=clean_result.accuracy_history,
        byz_honest_history=byz_honest_result.accuracy_history,
    )

    plot_bar_comparison(
        os.path.join(results_dir, "exp3_governance_bar_v6.png"),
        results,
    )

    # ── Print summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Exp3 v6] ===== Summary =====")
    print("=" * 60)
    print(f"  Clean Baseline Best:     {results['clean_best']:.2f}%")
    print(f"  Byz+HonestE Best:        {results['byz_honest_best']:.2f}%  "
          f"(loss: {results['byz_honest_loss']:.2f}%)")
    print(f"  Attacked Single Best:    {results['single_best']:.2f}%  "
          f"(loss: {results['single_loss']:.2f}%)")
    print(f"  Attacked TAS Best:       {results['tas_best']:.2f}%  "
          f"(loss: {results['tas_loss']:.2f}%)")
    if attacked_tas_enhanced:
        print(f"  Attacked TAS-Enh Best:   {results['tas_enhanced_best']:.2f}%  "
              f"(loss: {results['tas_enhanced_loss']:.2f}%)")

    print(f"\n  TAS V1 Detection Rate:   {results.get('tas_system_interception_rate', 0):.2%}")
    for k in ATTACK_TYPES:
        total_k = results.get(f"tas_{k}_total", 0)
        detected_k = results.get(f"tas_{k}_detected", 0)
        rate_k = results.get(f"tas_{k}_detection_rate", 0)
        print(f"  {k}: {int(detected_k)}/{int(total_k)} detected ({rate_k:.1%})")

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Plots: {results_dir}")


if __name__ == "__main__":
    main()
