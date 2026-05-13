"""
Exp2 v2 补充实验：跑完之前标 ⚠️ 的两组实验
1. 不同恶意比例 (10%/20%/30%/40%/50%) × label_flip × Non-IID
2. IID + 3种攻击 (ratio=0.4)

输出 CSV: results/exp2_v2_supplement.csv
"""
import csv
import os
import sys
import time
import random
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCKCHAIN_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CORE_DIR = os.path.join(BLOCKCHAIN_DIR, "core")
sys.path.insert(0, BLOCKCHAIN_DIR)
sys.path.insert(0, CORE_DIR)

from model import FedAvgCNN
from exp2_robustness_matrix import (
    prepare_data_loaders, evaluate, state_sub, state_add,
    flatten_update, ATTACK_TRAIN_FNS,
    fedavg_aggregate, multi_krum_aggregate, trimmed_mean_aggregate,
    fltrust_aggregate, compute_tasl_trust_weights, set_seed,
)


# ═══════════════════════════════════════════════════════════════════════
# Single run (same as exp2_robustness_matrix.py run_single)
# ═══════════════════════════════════════════════════════════════════════

def run_single(algo, attack, client_loaders, test_loader, n_clients, n_byz,
               rounds, device, local_epochs=2, lr=0.02, seed=42,
               ema_alpha=0.7, noise_std=0.1):
    set_seed(seed)
    byzantine_set = set(range(n_byz))
    global_model = FedAvgCNN().to(device)
    global_state = global_model.state_dict()

    acc_history = []
    ema_weights = None
    cos_history = []
    prev_anchor = None

    for rnd in range(rounds):
        # Collect updates
        updates = {}
        flat_updates = {}
        for cid in range(n_clients):
            if cid in byzantine_set and attack != 'none':
                fn = ATTACK_TRAIN_FNS.get(attack)
                if fn:
                    local_state = fn(global_state, client_loaders[cid], device,
                                     local_epochs=local_epochs, lr=lr)
                else:
                    local_state = {k: v.clone() for k, v in global_state.items()}
            else:
                model = FedAvgCNN().to(device)
                model.load_state_dict(global_state)
                model.train()
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()
                for _ in range(local_epochs):
                    for x, y in client_loaders[cid]:
                        x, y = x.to(device), y.to(device)
                        optimizer.zero_grad()
                        criterion(model(x), y).backward()
                        optimizer.step()
                local_state = model.state_dict()

            update = state_sub(local_state, global_state)
            updates[cid] = update
            flat_updates[cid] = np.concatenate(
                [v.detach().cpu().reshape(-1).numpy() for v in update.values()
                 if isinstance(v, torch.Tensor) and v.is_floating_point()]
            ).astype(np.float64)

        # Aggregate
        if algo == 'fedavg':
            agg_update = fedavg_aggregate(updates)

        elif algo == 'multi_krum':
            agg_update = multi_krum_aggregate(flat_updates, updates, n_byz)

        elif algo == 'trimmed_mean':
            agg_update = trimmed_mean_aggregate(updates)

        elif algo == 'fltrust':
            agg_update = fltrust_aggregate(flat_updates, updates, device)

        elif algo == 'tasl':
            trust_weights, cos_sims, anchor = compute_tasl_trust_weights(
                flat_updates,
                trust_power=5.0,
                max_weight_ratio=1.5,
                min_cos_threshold=0.2,
                norm_penalty_strength=0.8,
                refine_anchor=True,
                ema_weights=ema_weights,
                ema_alpha=ema_alpha,
                cos_history=cos_history if len(cos_history) > 0 else None,
                prev_anchor=prev_anchor,
            )
            prev_anchor = anchor
            ema_weights = trust_weights
            cos_history.append(cos_sims)
            if len(cos_history) > 5:
                cos_history = cos_history[-5:]

            agg_update = fedavg_aggregate(updates, weights=trust_weights)

        else:
            raise ValueError(f"Unknown algo: {algo}")

        global_state = state_add(global_state, agg_update)
        global_model.load_state_dict(global_state)
        acc = evaluate(global_model, test_loader, device)
        acc_history.append(acc)

        if (rnd + 1) % 10 == 0:
            print(f"    R{rnd+1:3d}: acc={acc:.2f}%")

    best_acc = max(acc_history)
    avg_acc = sum(acc_history) / len(acc_history)
    last_acc = acc_history[-1]

    return {
        'best_acc': best_acc,
        'avg_acc': avg_acc,
        'last_acc': last_acc,
        'acc_history': acc_history,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    algos = ['fedavg', 'multi_krum', 'trimmed_mean', 'fltrust', 'tasl']

    csv_rows = []

    # ── Part 1: Different Byzantine ratios × label_flip × Non-IID ────
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    attack = 'label_flip'

    print("\n" + "=" * 70)
    print("Part 1: Different Byzantine Ratios × Label Flip × Non-IID (α=0.3)")
    print("=" * 70)

    for ratio in ratios:
        n_clients = 10
        n_byz = max(1, int(n_clients * ratio))
        print(f"\n--- Byzantine Ratio = {ratio:.0%} ({n_byz}/{n_clients}) ---")

        client_loaders, test_loader = prepare_data_loaders(
            n_clients=n_clients, batch_size=128, alpha=0.3,
            split='non_iid', seed=42
        )

        for algo in algos:
            t0 = time.time()
            print(f"  [{algo}] ", end="", flush=True)
            result = run_single(
                algo, attack, client_loaders, test_loader,
                n_clients=n_clients, n_byz=n_byz,
                rounds=50, device=device,
                local_epochs=2, lr=0.02, seed=42,
                ema_alpha=0.7, noise_std=0.1,
            )
            elapsed = time.time() - t0
            print(f"best={result['best_acc']:.2f}%, avg={result['avg_acc']:.2f}%, "
                  f"last={result['last_acc']:.2f}% ({elapsed:.0f}s)")
            csv_rows.append({
                'part': 'ratio_sweep',
                'algo': algo,
                'attack': attack,
                'split': 'non_iid',
                'alpha': 0.3,
                'byzantine_ratio': ratio,
                'n_byz': n_byz,
                'best_acc': round(result['best_acc'], 2),
                'avg_acc': round(result['avg_acc'], 2),
                'last_acc': round(result['last_acc'], 2),
            })

    # ── Part 2: IID + 3 attacks × ratio=0.4 ─────────────────────────
    attacks = ['label_flip', 'sign_flip', 'gaussian_noise']
    ratio = 0.4
    n_clients = 10
    n_byz = int(n_clients * ratio)

    print("\n" + "=" * 70)
    print("Part 2: IID + 3 Attacks (ratio=0.4)")
    print("=" * 70)

    client_loaders, test_loader = prepare_data_loaders(
        n_clients=n_clients, batch_size=128, alpha=0.3,
        split='iid', seed=42
    )

    for attack in attacks:
        for algo in algos:
            t0 = time.time()
            print(f"  [{algo} × {attack}] ", end="", flush=True)
            result = run_single(
                algo, attack, client_loaders, test_loader,
                n_clients=n_clients, n_byz=n_byz,
                rounds=50, device=device,
                local_epochs=2, lr=0.02, seed=42,
                ema_alpha=0.7, noise_std=0.1,
            )
            elapsed = time.time() - t0
            print(f"best={result['best_acc']:.2f}%, avg={result['avg_acc']:.2f}%, "
                  f"last={result['last_acc']:.2f}% ({elapsed:.0f}s)")
            csv_rows.append({
                'part': 'iid_attacks',
                'algo': algo,
                'attack': attack,
                'split': 'iid',
                'alpha': 0.0,
                'byzantine_ratio': ratio,
                'n_byz': n_byz,
                'best_acc': round(result['best_acc'], 2),
                'avg_acc': round(result['avg_acc'], 2),
                'last_acc': round(result['last_acc'], 2),
            })

    # ── Save CSV ─────────────────────────────────────────────────────
    results_dir = os.path.join(BLOCKCHAIN_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "exp2_v2_supplement.csv")

    fieldnames = ['part', 'algo', 'attack', 'split', 'alpha',
                  'byzantine_ratio', 'n_byz', 'best_acc', 'avg_acc', 'last_acc']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # ── Print summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n--- Part 1: Ratio Sweep (Label Flip, Non-IID) ---")
    print(f"{'Ratio':<8s} {'FedAvg':>8s} {'M-Krum':>8s} {'T-Mean':>8s} {'FLTrust':>8s} {'TASL':>8s}")
    for ratio in ratios:
        row = f"{ratio:<8.0%}"
        for algo in algos:
            val = [r for r in csv_rows
                   if r['part'] == 'ratio_sweep' and r['algo'] == algo and r['byzantine_ratio'] == ratio]
            if val:
                row += f" {val[0]['best_acc']:>7.2f}"
            else:
                row += f" {'N/A':>7s}"
        print(row)

    print("\n--- Part 2: IID + 3 Attacks ---")
    print(f"{'Attack':<16s} {'FedAvg':>8s} {'M-Krum':>8s} {'T-Mean':>8s} {'FLTrust':>8s} {'TASL':>8s}")
    for attack in attacks:
        row = f"{attack:<16s}"
        for algo in algos:
            val = [r for r in csv_rows
                   if r['part'] == 'iid_attacks' and r['algo'] == algo and r['attack'] == attack]
            if val:
                row += f" {val[0]['best_acc']:>7.2f}"
            else:
                row += f" {'N/A':>7s}"
        print(row)

    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
