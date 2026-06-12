# -*- coding: utf-8 -*-
"""Nature-quality figures for Exp4 (Storage) and Exp5 (Economic)."""
import os, csv, math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Nature-style rcParams ──
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})

PAL = {
    "mono": "#8B8B8B", "ipfs": "#F4A261", "tasl": "#2874A6",
    "hero_fill": "#D6EAF8", "hero_edge": "#2874A6",
    "neutral_dark": "#4D4D4D", "neutral_mid": "#767676",
    "accent": "#E74C3C", "delta_up": "#27AE60",
}

def save_pub(fig, path):
    for fmt, dpi in [("svg",None),("pdf",None),("png",400)]:
        fig.savefig(f"{path}.{fmt}", dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {path}.svg/.pdf/.png")

RES = "/media/njit5/39e16d05-05ba-46b4-bc9b-47b3c98f1d4f/tlf/TASL/results"

# ══════════════════════════════════════════════════════════════════
# FIGURE 1: Exp4 Storage Scalability — 3-panel grid
# ══════════════════════════════════════════════════════════════════

def fig_exp4():
    # Load data
    models = {}
    for mn, fn in [("MNIST","exp4_storage_mnist.csv"),("CIFAR-10","exp4_storage_cifar_10.csv"),
                   ("ST-GCN","exp4_storage_st_gcn.csv")]:
        with open(os.path.join(RES, fn)) as f:
            reader = list(csv.DictReader(f))
        rounds = np.array([int(r["round"]) for r in reader])
        models[mn] = {
            "rounds": rounds,
            "mono": np.array([float(r["monolithic_mb"]) for r in reader]),
            "ipfs": np.array([float(r["ipfs_mb"]) for r in reader]),
            "tasl": np.array([float(r["tasl_mb"]) for r in reader]),
        }

    fig = plt.figure(figsize=(7.2, 8.5))

    # ── Panel (a): Growth curves (3 models, 3 lines each, 1 column) ──
    gs = fig.add_gridspec(3, 1, hspace=0.55, top=0.94, bottom=0.06, left=0.13, right=0.96)
    for i, (mn, sign, sz) in enumerate([("MNIST","4.7 GB → 97 KB",1.6),
                                          ("CIFAR-10","32.8 GB → 122 KB",11.2),
                                          ("ST-GCN","183 GB → 420 KB",12.5)]):
        ax = fig.add_subplot(gs[i])
        d = models[mn]
        ax.plot(d["rounds"], d["mono"], color=PAL["mono"], lw=1.5, label="Monolithic BC")
        ax.plot(d["rounds"], d["ipfs"], color=PAL["ipfs"], lw=1.5, ls="--", label="IPFS-Linear")
        ax.plot(d["rounds"], d["tasl"], color=PAL["tasl"], lw=2.2, label="TASL (Ours)")

        ax.set_yscale("log")
        ax.set_ylabel("Storage (MB, log)", fontsize=7)
        if i == 2: ax.set_xlabel("Training Round", fontsize=7)
        ax.set_title(f"{mn} ({sz:.1f} MB) — {sign}", fontsize=8, fontweight="bold",
                     color=PAL["neutral_dark"])
        ax.grid(True, which="major", ls="--", alpha=0.2)
        ax.legend(fontsize=6, loc="lower right")

    # ── Panel (b): Multi-model bar with IPFS gap annotation ──
    fig_bar = plt.figure(figsize=(7.2, 3.5))
    ax = fig_bar.add_subplot(111)
    mn_names = list(models.keys())
    x = np.arange(len(mn_names))
    w = 0.22

    mono_vals = [models[m]["mono"][-1]/1024 for m in mn_names]  # GB
    ipfs_vals = [models[m]["ipfs"][-1] for m in mn_names]       # MB
    tasl_vals = [models[m]["tasl"][-1]*1024 for m in mn_names]  # KB

    # Log-scale bars
    for j, (label, vals, color, offset) in enumerate([
        ("Monolithic\n(GB)", mono_vals, PAL["mono"], -w),
        ("IPFS-Linear\n(MB)", ipfs_vals, PAL["ipfs"], 0),
        ("TASL (Ours)\n(KB)", tasl_vals, PAL["tasl"], w),
    ]):
        bars = ax.bar(x + offset, vals, w, color=color, edgecolor="white", lw=0.5,
                      label=label, alpha=0.9)
        for bar, val in zip(bars, vals):
            if val > 0:
                fmt = f"{val:.0f} GB" if j == 0 else (f"{val:.1f} MB" if j == 1 else f"{val:.0f} KB")
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.05, fmt,
                       ha="center", fontsize=5.5, fontweight="bold", color=PAL["neutral_dark"])

    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(mn_names, fontsize=7)
    ax.set_ylabel("Storage (log scale)", fontsize=7)
    ax.set_title("Final Storage per Model (300 rounds)", fontsize=9, fontweight="bold")
    ax.legend(loc="upper right", fontsize=6, ncol=3, bbox_to_anchor=(1,1.15))
    ax.grid(axis="y", ls="--", alpha=0.2)

    # IPFS vs TASL gap annotations
    for i, mn in enumerate(mn_names):
        d = models[mn]
        vs_ipfs = (1 - d["tasl"][-1]/d["ipfs"][-1]) * 100
        y_pos = d["ipfs"][-1] * 0.4
        ax.annotate(f"vs IPFS: −{vs_ipfs:.1f}%", (i, y_pos), fontsize=6,
                   color=PAL["accent"], fontweight="bold", ha="center",
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="#FDEBD0", edgecolor=PAL["ipfs"], alpha=0.8))

    save_pub(fig, os.path.join(RES, "exp4_storage_curves"))
    save_pub(fig_bar, os.path.join(RES, "exp4_storage_bars"))
    plt.close("all")

# ══════════════════════════════════════════════════════════════════
# FIGURE 2: Exp5 Economic Scalability — dual panel
# ══════════════════════════════════════════════════════════════════

def fig_exp5():
    with open(os.path.join(RES, "exp5_gas_data.csv")) as f:
        rows = list(csv.DictReader(f))

    mn_rows = {mn: [r for r in rows if r["model"]==mn] for mn in ["MNIST","ST-GCN"]}

    # ── Panel (a): Per-round Gas (dual model) ──
    fig = plt.figure(figsize=(7.2, 3.8))
    ax = fig.add_subplot(111)
    clients = [10,20,30,40,50]
    for mn, c, m in [("MNIST", PAL["blue_main"] if "blue_main" in PAL else "#2874A6","s"),
                      ("ST-GCN", "#E67E22","o")]:
        d = mn_rows[mn]
        gas = [float(r["sc_per_round"])/1e9 for r in d]
        ax.plot(clients, gas, marker=m, color=c, lw=2, markersize=6,
                label=f"{mn} ({len(mn_rows[mn][0]['model_size_bytes'])}")

    ax.set_xlabel("Number of Clients", fontsize=7)
    ax.set_ylabel("TASL Gas per Round (Ggas)", fontsize=7)
    ax.set_title("Model-Agnostic Gas: 8× model → near-constant cost", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7); ax.grid(ls="--", alpha=0.2)

    # ── Panel (b): Sensitivity ──
    fig_sens = plt.figure(figsize=(7.2, 3.8))
    ax1 = fig_sens.add_subplot(121)
    ax2 = fig_sens.add_subplot(122)

    # Gas sensitivity
    gas_prices = [10,20,60,100]
    tasl_usd_g = [2655, 5310, 15930, 26550]
    mono_usd_g = [3.802e9, 7.605e9, 22.815e9, 38.025e9]
    xg = np.arange(len(gas_prices)); w = 0.3
    ax1.bar(xg-w/2, [m/1e9 for m in mono_usd_g], w, color=PAL["mono"], alpha=0.4, label="Monolithic (B$)")
    ax1.bar(xg+w/2, [t/1e3 for t in tasl_usd_g], w, color=PAL["tasl"], label="TASL (K$)")
    ax1.set_yscale("log"); ax1.set_xticks(xg); ax1.set_xticklabels([f"{g}" for g in gas_prices], fontsize=6)
    ax1.set_xlabel("Gas Price (Gwei)", fontsize=7); ax1.set_ylabel("Cost (log)", fontsize=7)
    ax1.set_title("Gas Price Sensitivity", fontsize=8, fontweight="bold"); ax1.legend(fontsize=6); ax1.grid(axis="y", ls="--", alpha=0.2)

    # ETH sensitivity  
    eth_prices = [2000,3000,4000]
    tasl_usd_e = [3540, 5310, 7080]
    mono_usd_e = [5.07e9, 7.605e9, 10.14e9]
    xe = np.arange(len(eth_prices))
    ax2.bar(xe-w/2, [m/1e9 for m in mono_usd_e], w, color=PAL["mono"], alpha=0.4, label="Monolithic (B$)")
    ax2.bar(xe+w/2, [t/1e3 for t in tasl_usd_e], w, color=PAL["tasl"], label="TASL (K$)")
    ax2.set_yscale("log"); ax2.set_xticks(xe); ax2.set_xticklabels([f"${e/1000:.0f}K" for e in eth_prices], fontsize=6)
    ax2.set_xlabel("ETH Price", fontsize=7); ax2.set_ylabel("Cost (log)", fontsize=7)
    ax2.set_title("ETH Price Sensitivity", fontsize=8, fontweight="bold"); ax2.legend(fontsize=6); ax2.grid(axis="y", ls="--", alpha=0.2)

    fig.suptitle("Exp5: Economic Scalability & Sensitivity", fontsize=10, fontweight="bold", y=1.02)
    fig_sens.suptitle("Exp5: Gas & ETH Price Sensitivity (ST-GCN, n=50, 300 rounds)", 
                       fontsize=9, fontweight="bold", y=1.02)

    save_pub(fig, os.path.join(RES, "exp5_model_agnostic"))
    save_pub(fig_sens, os.path.join(RES, "exp5_sensitivity_nature"))
    plt.close("all")

# ── Run ──
if __name__ == "__main__":
    print("[Nature Figures] Generating Exp4...")
    fig_exp4()
    print("[Nature Figures] Generating Exp5...")
    fig_exp5()
    print("Done.")
