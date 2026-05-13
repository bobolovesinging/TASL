"""Run only TASL experiments (13-15/15) for exp2 v2"""
import csv
import os
import sys
import random
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCKCHAIN_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CORE_DIR = os.path.join(BLOCKCHAIN_DIR, "core")
sys.path.insert(0, BLOCKCHAIN_DIR)
sys.path.insert(0, CORE_DIR)

from exp2_robustness_matrix import *

set_seed(42)
device = torch.device("cpu")

n_clients = 10
n_byz = 4
rounds = 50
attacks = ['label_flip', 'sign_flip', 'gaussian_noise']

client_loaders, test_loader = prepare_data_loaders(
    n_clients=n_clients, batch_size=128, alpha=0.3, split='non_iid', seed=42)

print("=" * 70)
print("[Exp2 v2] Running TASL-only experiments (13-15/15)")
print(f"  TASL: trust_power=5, weight_cap=1.5/n, ema_alpha=0.7")
print(f"  TASL: norm_penalty=0.8, min_cos=0.2(adaptive), dual_anchor+norm_gate")
print("=" * 70)

# Pre-fill known results from v2 log (1-12/15)
known_results = {
    ('fedavg', 'label_flip'): {'best_acc': 93.74, 'avg_acc': 71.60, 'last_acc': 93.74},
    ('fedavg', 'sign_flip'): {'best_acc': 44.97, 'avg_acc': 27.07, 'last_acc': 24.01},
    ('fedavg', 'gaussian_noise'): {'best_acc': 97.98, 'avg_acc': 93.98, 'last_acc': 97.95},
    ('multi_krum', 'label_flip'): {'best_acc': 97.72, 'avg_acc': 90.52, 'last_acc': 97.61},
    ('multi_krum', 'sign_flip'): {'best_acc': 90.32, 'avg_acc': 66.79, 'last_acc': 90.32},
    ('multi_krum', 'gaussian_noise'): {'best_acc': 97.63, 'avg_acc': 92.21, 'last_acc': 97.63},
    ('trimmed_mean', 'label_flip'): {'best_acc': 90.68, 'avg_acc': 84.90, 'last_acc': 88.22},
    ('trimmed_mean', 'sign_flip'): {'best_acc': 73.96, 'avg_acc': 59.39, 'last_acc': 56.50},
    ('trimmed_mean', 'gaussian_noise'): {'best_acc': 97.95, 'avg_acc': 92.80, 'last_acc': 97.95},
    ('fltrust', 'label_flip'): {'best_acc': 95.43, 'avg_acc': 52.18, 'last_acc': 2.15},
    ('fltrust', 'sign_flip'): {'best_acc': 81.15, 'avg_acc': 63.14, 'last_acc': 11.35},
    # fltrust/gaussian_noise was interrupted, need to re-run
}

# Run FLTrust+gaussian_noise (12/15 was interrupted)
print("\n>>> [12/15] fltrust x gaussian_noise (re-run) <<<")
fltrust_gn = run_single('fltrust', 'gaussian_noise', client_loaders, test_loader,
    n_clients=n_clients, n_byz=n_byz, rounds=rounds, device=device,
    local_epochs=2, lr=0.02, seed=42 + hash('fltrustgaussian_noise') % 1000,
    ema_alpha=0.7, noise_std=0.1)
known_results[('fltrust', 'gaussian_noise')] = {
    'best_acc': fltrust_gn['best_acc'], 'avg_acc': fltrust_gn['avg_acc'], 'last_acc': fltrust_gn['last_acc']}
print(f"  => best={fltrust_gn['best_acc']:.2f}%, avg={fltrust_gn['avg_acc']:.2f}%, last={fltrust_gn['last_acc']:.2f}%")

# Run TASL experiments (13-15/15)
for idx, attack in enumerate(attacks):
    run_idx = 13 + idx
    print(f"\n>>> [{run_idx}/15] tasl x {attack} <<<")
    result = run_single('tasl', attack, client_loaders, test_loader,
        n_clients=n_clients, n_byz=n_byz, rounds=rounds, device=device,
        local_epochs=2, lr=0.02, seed=42 + hash('tasl' + attack) % 1000,
        ema_alpha=0.7, noise_std=0.1)
    known_results[('tasl', attack)] = {
        'best_acc': result['best_acc'], 'avg_acc': result['avg_acc'], 'last_acc': result['last_acc']}
    print(f"  => best={result['best_acc']:.2f}%, avg={result['avg_acc']:.2f}%, last={result['last_acc']:.2f}%")

# Print summary
algos = ['fedavg', 'multi_krum', 'trimmed_mean', 'fltrust', 'tasl']
algo_labels = ['FedAvg', 'M-Krum', 'T-Mean', 'FLTrust', 'TASL']

print("\n" + "=" * 70)
print("[Exp2 v2] Summary -- non_iid (a=0.3), 40% Byzantine")
print("=" * 70)
print(f"{'Attack':<18s} {'FedAvg':>8s} {'M-Krum':>8s} {'T-Mean':>8s} {'FLTrust':>8s} {'TASL':>8s}")
print("-" * 70)
for attack in attacks:
    row_str = f"{attack:<18s}"
    vals = []
    for algo in algos:
        key = (algo, attack)
        if key in known_results:
            val = known_results[key]['best_acc']
            vals.append(val)
        else:
            vals.append(0)
    best_val = max(vals)
    for i, algo in enumerate(algos):
        key = (algo, attack)
        if key in known_results:
            val = known_results[key]['best_acc']
            marker = " *" if val == best_val else ""
            row_str += f" {val:>6.2f}{marker:<2s}"
        else:
            row_str += f" {'N/A':>8s}"
    print(row_str)

# Print avg accuracy too
print(f"\n{'Attack':<18s} {'FedAvg':>8s} {'M-Krum':>8s} {'T-Mean':>8s} {'FLTrust':>8s} {'TASL':>8s}")
print("-" * 70)
for attack in attacks:
    row_str = f"{attack:<18s}"
    for algo in algos:
        key = (algo, attack)
        if key in known_results:
            row_str += f" {known_results[key]['avg_acc']:>7.2f}"
        else:
            row_str += f" {'N/A':>8s}"
    print(row_str)

print(f"\nClean Baseline: best=98.00%, avg=93.68%")

# Save all results to CSV
csv_rows = []
for algo in algos:
    for attack in attacks:
        key = (algo, attack)
        if key in known_results:
            r = known_results[key]
            csv_rows.append({
                'algo': algo, 'attack': attack, 'split': 'non_iid',
                'byzantine_ratio': 0.4,
                'best_acc': r['best_acc'], 'avg_acc': r['avg_acc'], 'last_acc': r['last_acc'],
            })
csv_rows.append({
    'algo': 'clean_baseline', 'attack': 'none', 'split': 'non_iid',
    'byzantine_ratio': 0.0, 'best_acc': 98.00, 'avg_acc': 93.68, 'last_acc': 98.00,
})

results_dir = os.path.join(BLOCKCHAIN_DIR, "results")
os.makedirs(results_dir, exist_ok=True)
csv_path = os.path.join(results_dir, "exp2_robustness_data.csv")
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['algo', 'attack', 'split', 'byzantine_ratio',
                                            'best_acc', 'avg_acc', 'last_acc'])
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"\nSaved CSV: {csv_path}")
