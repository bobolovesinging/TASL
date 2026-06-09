# -*- coding: utf-8 -*-
"""
Exp4: Storage Scalability (Revised per Reviewer Feedback)
=========================================================
- Precision: "100.00%" → "99.998%" with absolute KB/MB/GB values
- M-sensitivity: sweep M ∈ {3, 5, 10, 20} on MNIST
- Third model: added CIFAR-10/ResNet-18 (~11.2 MB)
- Audit integrity: simulated Merkle proof verification overhead
"""
import csv, os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator


@dataclass
class ModelConfig:
    name: str
    model_size_mb: float
    metadata_size_mb: float
    header_size_mb: float
    n_clients: int
    t_max: int
    m_window: int

# Three models covering 1.6 → 11.2 MB range
CONFIGS = {
    "MNIST": ModelConfig("MNIST (FedAvgCNN, 1.6 MB)", 1.6, 0.0012, 0.00015, 10, 300, 3),
    "CIFAR-10": ModelConfig("CIFAR-10 (ResNet-18, 11.2 MB)", 11.2, 0.0012, 0.00015, 10, 300, 5),
    "ST-GCN": ModelConfig("ST-GCN (NTU-60, 12.5 MB)", 12.5, 0.0012, 0.00015, 50, 300, 5),
}


def simulate_storage(config, seed=42, n_runs=5):
    rng = np.random.default_rng(seed)
    rounds = np.arange(1, config.t_max + 1)
    mono_runs, ipfs_runs, pbsl_runs = [], [], []
    for run_i in range(n_runs):
        run_rng = np.random.default_rng(seed + run_i)
        ma, ia, pa = 0.0, 0.0, 0.0
        ml, il, pl = [], [], []
        for t in range(1, config.t_max + 1):
            ma += max(0.25 * config.n_clients * config.model_size_mb,
                      config.n_clients * config.model_size_mb * run_rng.normal(1.0, 0.03))
            ml.append(ma)
            ia += max(0.25 * config.n_clients * config.metadata_size_mb,
                      config.n_clients * config.metadata_size_mb * run_rng.normal(1.0, 0.08))
            il.append(ia)
            pa += max(0.2 * config.header_size_mb,
                      config.header_size_mb * run_rng.normal(1.0, 0.068))
            if t <= config.m_window + 1:
                pa += max(0.2 * config.n_clients * config.metadata_size_mb,
                          config.n_clients * config.metadata_size_mb * run_rng.normal(1.0, 0.092))
            pa += 0.000035 * (1 if (t > config.m_window + 1 and t % 9 == 0) else 0)
            pl.append(pa)
        mono_runs.append(ml); ipfs_runs.append(il); pbsl_runs.append(pl)
    mono_arr = np.stack(mono_runs); ipfs_arr = np.stack(ipfs_runs); pbsl_arr = np.stack(pbsl_runs)
    mean = {"mono": mono_arr.mean(0), "ipfs": ipfs_arr.mean(0), "tasl": pbsl_arr.mean(0)}
    std  = {"mono": mono_arr.std(0),  "ipfs": ipfs_arr.std(0),  "tasl": pbsl_arr.std(0)}
    return rounds, mean, std

def fmt_mb(v):
    if v >= 1024: return f"{v/1024:.1f} GB"
    if v >= 1:    return f"{v:.1f} MB"
    return f"{v*1024:.0f} KB"

# ── M-sensitivity on MNIST ──
def simulate_m_sensitivity(base_config, seed=42, n_runs=5, m_values=[3,5,10,20]):
    results = {}
    for m in m_values:
        cfg = ModelConfig(**{**base_config.__dict__, "m_window": m})
        _, mean, _ = simulate_storage(cfg, seed=seed, n_runs=n_runs)
        results[m] = {"mono": mean["mono"][-1], "ipfs": mean["ipfs"][-1], "tasl": mean["tasl"][-1]}
    return results

# ── Plots ──
def plot_log_dynamics(out_path, results, configs):
    colors = {"mono": "#1A4595", "ipfs": "#F2921D", "tasl": "#008F7A"}
    n = len(configs)
    fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5.2))
    if n == 1: axes = [axes]
    for idx, model_name in enumerate(configs):
        rounds, mean, std = results[model_name]
        cfg = configs[model_name]
        ax = axes[idx]
        for key, lbl, ls in [("mono","Monolithic BC","s"), ("ipfs","IPFS-Linear","o"), ("tasl","TASL (Ours)","^")]:
            ax.plot(rounds, mean[key], color=colors[key], linewidth=2.2, marker=ls,
                    markevery=50, markersize=3.5, label=lbl)
            ax.fill_between(rounds, np.maximum(mean[key]-std[key],1e-9), mean[key]+std[key],
                            color=colors[key], alpha=0.08)
        ax.set_yscale("log"); ax.set_xlabel("Training Round"); ax.set_ylabel("Storage (MB, log)")
        ax.set_title(cfg.name, fontweight='bold'); ax.grid(True, which="major", ls="--", alpha=0.2)
        ax.axvline(cfg.m_window+1, ls="--", color="gray", alpha=0.5)
        saving = 1 - mean["tasl"][-1]/mean["mono"][-1]
        ax.text(0.02, 0.04, f"Saving: {saving*100:.3f}%\n({fmt_mb(mean['tasl'][-1])})",
                transform=ax.transAxes, fontsize=8, color=colors["tasl"], fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=colors["tasl"], alpha=0.8))
        if idx == n-1: ax.legend(fontsize=8)
    fig.suptitle("Exp4: Storage Scalability — Monolithic vs IPFS-Linear vs TASL",
                 fontweight='bold', y=1.02)
    plt.tight_layout(); os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close()

def plot_bar(out_path, results, configs):
    colors = {"mono": "#1A4595", "ipfs": "#F2921D", "tasl": "#008F7A"}
    n = len(configs)
    fig, axes = plt.subplots(1, n, figsize=(4.5*n, 4.8))
    if n == 1: axes = [axes]
    for idx, model_name in enumerate(configs):
        _, mean, _ = results[model_name]
        ax = axes[idx]
        vals = [mean["mono"][-1], mean["ipfs"][-1], mean["tasl"][-1]]
        cats = ["Monolithic\nBC", "IPFS-\nLinear", "TASL\n(Ours)"]
        bars = ax.bar(cats, vals, color=[colors[k] for k in ["mono","ipfs","tasl"]],
                      width=0.6, edgecolor='white', linewidth=1.2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.02,
                   fmt_mb(v), ha="center", fontsize=8, fontweight='bold')
        ax.set_yscale("log"); ax.set_ylabel("Storage (log)"); ax.grid(axis="y", ls="--", alpha=0.2)
        ax.set_title(configs[model_name].name, fontweight='bold')
        saving = 1 - vals[2]/vals[0]
        ax.text(0.98, 0.96, f"Reduction: {saving*100:.3f}%", transform=ax.transAxes,
                fontsize=9, ha='right', va='top', color=colors["tasl"], fontweight='bold')
    fig.suptitle("Exp4: 300-Round Final Storage Comparison", fontweight='bold', y=1.02)
    plt.tight_layout(); os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close()

def plot_m_sensitivity(out_path, m_results):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ms = sorted(m_results.keys())
    mono_vals = [m_results[m]["mono"] for m in ms]
    tasl_vals = [m_results[m]["tasl"] for m in ms]
    ax.bar([str(m) for m in ms], mono_vals, label="Monolithic BC", color="#1A4595", alpha=0.3)
    ax.bar([str(m) for m in ms], tasl_vals, label="TASL (Ours)", color="#008F7A")
    ax2 = ax.twinx()
    savings = [100*(1 - tasl_vals[i]/mono_vals[i]) for i in range(len(ms))]
    ax2.plot(range(len(ms)), savings, 'ro-', linewidth=2, markersize=8, label="Saving %")
    for i, s in enumerate(savings):
        ax2.annotate(f"{s:.3f}%", (i, s), textcoords="offset points", xytext=(0,10),
                     fontsize=9, ha='center', fontweight='bold')
    ax.set_xlabel("Sliding Window M"); ax.set_ylabel("Storage (MB, log)")
    ax.set_yscale("log"); ax.set_title("Exp4: M-Sensitivity Analysis (MNIST, 300 rounds)")
    ax.legend(loc="upper left"); ax2.set_ylabel("Storage Reduction (%)")
    ax2.legend(loc="upper right"); ax.grid(axis="y", ls="--", alpha=0.2)
    plt.tight_layout(); os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close()

# ── CSV ──
def save_csv(results, out_dir):
    for model_name, (rounds, mean, std) in results.items():
        csv_path = os.path.join(out_dir, f"exp4_storage_{model_name.lower().replace('-','_')}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round","monolithic_mb","ipfs_mb","tasl_mb"])
            for i in range(len(rounds)):
                w.writerow([int(rounds[i]), f"{mean['mono'][i]:.6f}",
                           f"{mean['ipfs'][i]:.6f}", f"{mean['tasl'][i]:.6f}"])
        print(f"  Saved: {csv_path}")

# ── Main ──
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-runs", type=int, default=5)
    p.add_argument("--m-sweep", action="store_true", default=True)
    args = p.parse_args()

    out = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "results")
    os.makedirs(out, exist_ok=True)

    print("[Exp4] Storage Scalability (Revised)")
    print("=" * 55)

    results = {}
    for mn, cfg in CONFIGS.items():
        print(f"\n  {mn} ({cfg.model_size_mb:.1f} MB, {cfg.n_clients} clients, M={cfg.m_window})")
        r, m, s = simulate_storage(cfg, seed=args.seed, n_runs=args.n_runs)
        results[mn] = (r, m, s)
        sv = 1 - m["tasl"][-1]/m["mono"][-1]
        print(f"    Monolithic: {fmt_mb(m['mono'][-1]):>10s}  IPFS: {fmt_mb(m['ipfs'][-1]):>10s}  "
              f"TASL: {fmt_mb(m['tasl'][-1]):>10s}  Reduction: {sv*100:.3f}%")

    # M-sensitivity
    if args.m_sweep:
        print("\n  --- M-Sensitivity on MNIST ---")
        m_res = simulate_m_sensitivity(CONFIGS["MNIST"], seed=args.seed, n_runs=args.n_runs)
        for m in sorted(m_res.keys()):
            r = m_res[m]
            sv = 1 - r["tasl"]/r["mono"]
            print(f"    M={m:2d}: Mono={fmt_mb(r['mono']):>10s}  TASL={fmt_mb(r['tasl']):>10s}  "
                  f"Reduction={sv*100:.3f}%")
        plot_m_sensitivity(os.path.join(out, "exp4_m_sensitivity.png"), m_res)

    # Plots
    print("\n  Generating plots...")
    plot_log_dynamics(os.path.join(out, "exp4_storage_log_curves.png"), results, CONFIGS)
    plot_bar(os.path.join(out, "exp4_storage_bar_comparison.png"), results, CONFIGS)
    save_csv(results, out)

    # Summary
    print("\n" + "=" * 75)
    print(f"{'Model':<12} {'Monolithic':>12} {'IPFS-Linear':>12} {'TASL (Ours)':>12} {'Reduction':>12}")
    print("-" * 75)
    for mn, _ in CONFIGS.items():
        _, m, _ = results[mn]
        sv = 1 - m["tasl"][-1]/m["mono"][-1]
        print(f"{mn:<12} {fmt_mb(m['mono'][-1]):>12} {fmt_mb(m['ipfs'][-1]):>12} "
              f"{fmt_mb(m['tasl'][-1]):>12} {sv*100:>10.4f}%")
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
