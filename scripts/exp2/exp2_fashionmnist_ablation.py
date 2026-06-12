"""
Fashion-MNIST 消融实验
======================
消融模式对照：

| 模式     | 数据层 | 梯度层 | 信任层融合 |
|---------|:-----:|:-----:|:----------:|
| full     | ✅    | ✅    | 乘法 p×ω  |
| nodata   | ❌ p≡1 | ✅    | 乘法 1×ω  |
| nograd   | ✅    | ❌ ω≡1/n | 乘法 p×(1/n) |
| additive | ✅    | ✅    | 加法 (p+ω)/2 |

攻击：Label Flip (y→9-y), Sign Flip
配置：Non-IID α=0.3, 10个客户端, 4个Byzantine, 50轮
"""

import csv
import os
import sys
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

# ── 复用 exp2_fashionmnist 中的函数 ────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# exp2_fashionmnist script uses BLOCKCHAIN_DIR as root
# __file__ resolves to scripts/exp2/exp2_fashionmnist_ablation.py
# So BLOCKCHAIN_DIR = ~/TLF/TASL
BLOCKCHAIN_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CORE_DIR = os.path.join(BLOCKCHAIN_DIR, "core")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, BLOCKCHAIN_DIR)
sys.path.insert(0, CORE_DIR)

os.chdir(BLOCKCHAIN_DIR)

from model import FedAvgCNN
from unified_data_layer import TrainStatsCollector, compute_unified_data_trust

# 从原脚本导入关键函数
from exp2_fashionmnist import (
    set_seed, prepare_data_loaders, evaluate_full,
    flatten_update, state_sub, state_add, fedavg_aggregate,
    compute_tasl_trust_weights, ATTACK_TRAIN_FNS, train_honest
)


ABLATION_MODES = ["full", "nodata", "nograd", "additive"]
ATTACKS = ["label_flip", "sign_flip"]
SAVE_DIR = os.path.join(BLOCKCHAIN_DIR, "results", "exp2_fashionmnist_ablation")


def run_ablation_single(
    attack: str,
    ablation: str,
    client_loaders,
    test_loader,
    n_clients: int,
    n_byz: int,
    rounds: int,
    device: torch.device,
    local_epochs: int = 2,
    lr: float = 0.02,
    seed: int = 42,
) -> dict:
    """Run TASL with specific ablation mode."""
    set_seed(seed)
    byzantine_set = set(range(n_byz))
    global_model = FedAvgCNN().to(device)
    global_state = global_model.state_dict()

    acc_history = []
    asr_history = []
    ema_weights = None
    cos_history = []
    prev_anchor = None
    temporal_anchor = None
    best_acc = 0.0
    best_asr = 0.0

    for r in range(1, rounds + 1):
        stats_collector = TrainStatsCollector()
        local_updates = {}
        flat_updates = {}

        for cid in range(n_clients):
            if cid in byzantine_set and attack != "none":
                train_fn = ATTACK_TRAIN_FNS.get(attack, train_honest)
                local_state = train_fn(global_state, client_loaders[cid], device,
                                       local_epochs=local_epochs, lr=lr,
                                       stats_collector=stats_collector, cid=cid)
            else:
                local_state = train_honest(global_state, client_loaders[cid], device,
                                           local_epochs=local_epochs, lr=lr,
                                           stats_collector=stats_collector, cid=cid)

            upd = state_sub(local_state, global_state)
            local_updates[cid] = upd
            flat_updates[cid] = flatten_update(upd)

        # ── Data-layer trust ───────────────────────────────────────────
        round_stats = stats_collector.get_stats()
        data_trust, diag = compute_unified_data_trust(round_stats, n_clients)

        # ── Ablation: nodata → 跳过数据层 ────────────────────────────
        if ablation == "nodata":
            data_trust = {c: 1.0 for c in range(n_clients)}

        # ── TASL aggregation ─────────────────────────────────────────
        if ablation == "nograd":
            # 只使用数据层信任，梯度层等权
            ds = {c: data_trust.get(c, 1.0) for c in range(n_clients)}
            s = sum(ds.values()) or 1e-12
            trust_weights = {c: v / s for c, v in ds.items()}
        else:
            # 正常计算梯度信任（保持 numpy 类型一致）
            trust_weights, cos_sims, new_anchor = compute_tasl_trust_weights(
                flat_updates,
                trust_power=5.0,
                max_weight_ratio=1.5,
                min_cos_threshold=0.2,
                norm_penalty_strength=0.8,
                refine_anchor=True,
                ema_weights=ema_weights,
                ema_alpha=0.7,
                cos_history=cos_history,
                prev_anchor=prev_anchor,
                temporal_anchor=temporal_anchor,
                data_trust=data_trust,
            )
            ema_weights = dict(trust_weights)
            cos_history.append(dict(cos_sims))
            prev_anchor = new_anchor

        # ── Ablation: additive → 加法融合梯度信任 + 数据信任 ─────────
        if ablation == "additive":
            ds = {c: data_trust.get(c, 1.0) for c in range(n_clients)}
            fused = {c: (trust_weights.get(c, 0.0) + ds.get(c, 0.0)) / 2.0 for c in range(n_clients)}
            s = sum(fused.values()) or 1e-12
            trust_weights = {c: v / s for c, v in fused.items()}

        agg_update = fedavg_aggregate(local_updates, trust_weights)

        # 更新 temporal anchor
        agg_flat = flatten_update(agg_update)
        if np.linalg.norm(agg_flat) > 1e-12:
            temporal_anchor = agg_flat

        global_state = state_add(global_state, agg_update)
        global_model.load_state_dict(global_state)

        # ── Evaluate ──────────────────────────────────────────────────
        acc, asr = evaluate_full(global_model, test_loader, device, attack=attack)
        acc_history.append(acc)
        asr_history.append(asr)
        best_acc = max(best_acc, acc)
        if r == rounds:
            best_asr = asr

    last_acc = acc_history[-1] if acc_history else 0.0
    avg_acc = sum(acc_history) / len(acc_history) if acc_history else 0.0
    final_asr = asr_history[-1] if asr_history else 0.0

    return {
        "best_acc": best_acc,
        "avg_acc": avg_acc,
        "last_acc": last_acc,
        "best_asr": final_asr,
        "acc_history": acc_history,
        "asr_history": asr_history,
    }


def plot_ablation_curves(save_path, all_results, attack, rounds):
    """Plot ablation comparison curves for one attack type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"full": "#2ecc71", "nodata": "#3498db", "nograd": "#e74c3c", "additive": "#f39c12"}
    markers = {"full": "o", "nodata": "s", "nograd": "^", "additive": "D"}

    for ablation in ABLATION_MODES:
        key = f"tasl_{ablation}"
        if key in all_results:
            res = all_results[key]
            axes[0].plot(range(1, rounds + 1), res["acc_history"],
                        label=ablation, color=colors.get(ablation, "gray"),
                        marker=markers.get(ablation, "."), markevery=max(1, rounds // 10))
            axes[1].plot(range(1, rounds + 1), res["asr_history"],
                        label=ablation, color=colors.get(ablation, "gray"),
                        marker=markers.get(ablation, "."), markevery=max(1, rounds // 10))

    axes[0].set_title(f"{attack} - Accuracy per Round")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_title(f"{attack} - ASR per Round")
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("ASR (%)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Fashion-MNIST Ablation v2")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--byzantine-ratio", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--attack", type=str, default=None,
                        choices=["label_flip", "sign_flip"],
                        help="Only run specified attack (default: run all)")
    # Quick mode
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 5 clients, 10 rounds")
    args = parser.parse_args()

    if args.quick:
        args.clients = 5
        args.rounds = 10
        args.local_epochs = 1
        args.byzantine_ratio = 0.4

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: rounds={args.rounds}, clients={args.clients}, "
          f"byz_ratio={args.byzantine_ratio}, alpha={args.alpha}")

    n_clients = int(args.clients)
    n_byz = max(1, int(n_clients * args.byzantine_ratio))

    # ── Data ────────────────────────────────────────────────────────
    print("Loading Fashion-MNIST data...")
    train_loaders, test_loader = prepare_data_loaders(
        n_clients=n_clients, batch_size=args.batch_size,
        alpha=args.alpha, split="non_iid", seed=args.seed,
    )
    print(f"  Train clients: {len(train_loaders)}, Test: {len(test_loader.dataset)}")

    # ── Attacks to run ──────────────────────────────────────────────
    attacks = [args.attack] if args.attack else ATTACKS

    # ── SAVE ────────────────────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)

    all_results = {}
    rows = []

    for atk in attacks:
        print(f"\n{'='*60}")
        print(f"Attack: {atk}")
        print(f"{'='*60}")

        for ablation in ABLATION_MODES:
            key = f"tasl_{ablation}"
            print(f"\n  ▸ Ablation mode: {ablation}")

            res = run_ablation_single(
                attack=atk,
                ablation=ablation,
                client_loaders=train_loaders,
                test_loader=test_loader,
                n_clients=n_clients,
                n_byz=n_byz,
                rounds=args.rounds,
                device=device,
                local_epochs=args.local_epochs,
                lr=args.lr,
                seed=args.seed,
            )

            all_results[key] = res

            print(f"    Best Acc = {res['best_acc']:.2f}%")
            print(f"    Avg  Acc = {res['avg_acc']:.2f}%")
            print(f"    Best ASR = {res['best_asr']:.2f}%")

            rows.append({
                "attack": atk,
                "ablation": ablation,
                "best_acc": f"{res['best_acc']:.2f}",
                "avg_acc": f"{res['avg_acc']:.2f}",
                "last_acc": f"{res['last_acc']:.2f}",
                "asr": f"{res['best_asr']:.2f}",
            })

        # ── Plot curves for this attack ──────────────────────────────
        plot_path = os.path.join(SAVE_DIR, f"fmnist_ablation_{atk}_curves.png")
        plot_ablation_curves(plot_path, all_results, atk, args.rounds)

    # ── Save CSV ────────────────────────────────────────────────────
    csv_path = os.path.join(SAVE_DIR, "exp2_fashionmnist_ablation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["attack", "ablation", "best_acc", "avg_acc", "last_acc", "asr"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved: {csv_path}")

    # ── Print summary table ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary Table")
    print(f"{'='*60}")
    print(f"{'Attack':<15} {'Ablation':<10} {'Best Acc':<10} {'Avg Acc':<10} {'ASR':<8}")
    print("-" * 55)
    for row in rows:
        print(f"{row['attack']:<15} {row['ablation']:<10} {row['best_acc']:<10} {row['avg_acc']:<10} {row['asr']:<8}")
    print(f"{'='*60}")
    print(f"Results saved to: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
