# -*- coding: utf-8 -*-
"""
Exp4 总图：多M值 × 三数据集 × 实际轮次
- MNIST: T=50, M∈{3,5,10,16,25,50}
- CIFAR-10: T=100, M∈{5,10,20,33,50,100}
- ST-GCN (NTU-60): T=65, M∈{6,10,13,21,32,65}
Layout: 3行 × 2列 (左=存储vs M+IPFS线, 右=节省率%)
"""
import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Nature-style ──
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.labelpad": 3,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
})

PAL = {
    "teal":    "#42949E",
    "red":     "#B64342",
    "blue":    "#0F4D92",
    "grey":    "#767676",
    "dark":    "#4D4D4D",
    "black":   "#272727",
}

# ── Analytical model ──
CONFIGS = {
    "MNIST":   {"s": 1.6, "m": 0.0012, "h": 0.00015, "n": 10, "T": 50},
    "CIFAR-10":{"s": 11.2,"m": 0.0012, "h": 0.00015, "n": 10, "T": 100},
    "ST-GCN":  {"s": 12.5,"m": 0.0012, "h": 0.00015, "n": 50, "T": 65},
}

M_SETS = {
    "MNIST":    [3, 5, 10, 16, 25, 50],
    "CIFAR-10": [5, 10, 20, 33, 50, 100],
    "ST-GCN":   [6, 10, 13, 21, 32, 65],
}

def simulate(cfg, M, seed=42, n_runs=5):
    T = cfg["T"]
    rng = np.random.default_rng(seed)
    ma_r, ia_r, pa_r = [], [], []
    for ri in range(n_runs):
        rr = np.random.default_rng(seed + ri)
        ma = ia = pa = 0.0
        for t in range(1, T + 1):
            ma += max(0.25*cfg["n"]*cfg["s"], cfg["n"]*cfg["s"]*rr.normal(1,0.03))
            ia += max(0.25*cfg["n"]*cfg["m"], cfg["n"]*cfg["m"]*rr.normal(1,0.08))
            pa += max(0.2*cfg["h"], cfg["h"]*rr.normal(1,0.068))
            if t <= M + 1:
                pa += max(0.2*cfg["n"]*cfg["m"], cfg["n"]*cfg["m"]*rr.normal(1,0.092))
            pa += 0.000035*(1 if (t > M+1 and t%9==0) else 0)
        ma_r.append(ma); ia_r.append(ia); pa_r.append(pa)
    return {
        "mono": np.mean(ma_r), "ipfs": np.mean(ia_r), "tasl": np.mean(pa_r),
        "m_std": np.std(ma_r), "i_std": np.std(ia_r), "t_std": np.std(pa_r),
    }

def save_pub(fig, name):
    for ext, dpi in [("svg", None), ("pdf", None), ("png", 600)]:
        p = os.path.join(OUT_DIR, f"{name}.{ext}")
        kw = {"bbox_inches": "tight"}
        if dpi: kw["dpi"] = dpi
        fig.savefig(p, **kw)
    print(f"  Saved: {name}.{{svg,pdf,png}}")

def fmt(v):
    if v >= 1024: return f"{v/1024:.1f} GB"
    if v >= 1: return f"{v:.1f} MB"
    return f"{v*1024:.0f} KB"

# ═══════════════════════════════════════════════════════════════════
# FIGURE: 3行 × 2列
# ═══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.5), sharex="col")

datasets = ["MNIST", "CIFAR-10", "ST-GCN"]
row_labels = {
    "MNIST":    "MNIST (T=50, n=10, 1.6MB)",
    "CIFAR-10": "CIFAR-10 (T=100, n=10, 11.2MB)",
    "ST-GCN":   "ST-GCN / NTU-60 (T=65, n=50, 12.5MB)",
}

for row, name in enumerate(datasets):
    cfg = CONFIGS[name]
    Ms = M_SETS[name]
    T = cfg["T"]

    # Precompute
    data = [simulate(cfg, M) for M in Ms]
    tasl_vals = [d["tasl"] for d in data]
    ipfs_vals = [d["ipfs"] for d in data]
    mono_vals = [d["mono"] for d in data]
    savings = [(1 - tv/iv)*100 for tv, iv in zip(tasl_vals, ipfs_vals)]

    # ── LEFT: Storage vs M (log scale) ──
    axL = axes[row, 0]
    ipfs_ref = ipfs_vals[0]  # IPFS is nearly constant

    axL.plot(Ms, tasl_vals, "o-", color=PAL["teal"], lw=1.6, ms=5, zorder=5,
             label="TASL (ours)")
    # IPFS reference band
    axL.axhline(y=ipfs_ref, color=PAL["red"], ls="--", lw=1.0, alpha=0.6,
                label=f"IPFS ({fmt(ipfs_ref)})")

    # Cross point annotation
    cross = None
    for i in range(len(Ms)):
        if tasl_vals[i] > ipfs_ref and i > 0:
            frac = (ipfs_ref - tasl_vals[i-1]) / max(tasl_vals[i] - tasl_vals[i-1], 1e-12)
            cross = Ms[i-1] + frac * (Ms[i] - Ms[i-1])
            break
    if cross:
        axL.axvline(x=cross, color=PAL["grey"], ls=":", lw=0.7, alpha=0.5)
        axL.annotate(f"M={cross:.0f}", xy=(cross, ipfs_ref),
                     xytext=(cross + (T*0.05), ipfs_ref * 2.5),
                     fontsize=7, color=PAL["grey"],
                     arrowprops=dict(arrowstyle="->", color=PAL["grey"], lw=0.6))

    axL.set_yscale("log")
    axL.set_ylabel("Cumulative storage (MB, log)", fontsize=7)
    axL.set_title(row_labels[name], fontweight="bold", fontsize=8, loc="left",
                  color=PAL["black"])
    axL.grid(which="major", ls="--", alpha=0.12, lw=0.3)
    axL.tick_params(labelsize=7)
    axL.legend(fontsize=7, loc="upper left", handlelength=1.2)

    # ── RIGHT: Savings vs M ──
    axR = axes[row, 1]
    colors_bar = [PAL["teal"] if s > 0 else PAL["red"] for s in savings]
    bars = axR.bar(range(len(Ms)), savings, color=colors_bar, width=0.55,
                   edgecolor="white", lw=0.3, zorder=3)
    axR.axhline(y=0, color=PAL["dark"], lw=0.5, zorder=1)

    # Annotate values on bars
    for i, s in enumerate(savings):
        yoff = 3 if s > 0 else -6
        c = PAL["teal"] if s > 0 else PAL["red"]
        axR.annotate(f"{s:.0f}%", (i, s), textcoords="offset points",
                     xytext=(0, yoff), fontsize=6.5, ha="center", color=c,
                     fontweight="bold")

    axR.set_xticks(range(len(Ms)))
    axR.set_xticklabels([str(m) for m in Ms], fontsize=6.5)
    axR.set_xlabel("Sliding window M", fontsize=7)
    axR.set_ylabel("TASL vs IPFS savings (%)", fontsize=7)
    axR.set_title(f"M-sensitivity (T={T})", fontweight="bold", fontsize=8, loc="left",
                  color=PAL["black"])
    axR.grid(axis="y", ls="--", alpha=0.12, lw=0.3)
    axR.tick_params(labelsize=7)

    # Mark T (total rounds) on x-axis
    for i, m in enumerate(Ms):
        if m >= T:
            axR.annotate("=T", (i, savings[i]),
                         textcoords="offset points", xytext=(0, 15),
                         fontsize=6, color=PAL["grey"], ha="center")
            break

# ── Figure title ──
fig.suptitle("M-Window Sensitivity — All Datasets at Actual Training Rounds",
             fontsize=9, fontweight="bold", y=1.005)

plt.tight_layout(pad=1.5, h_pad=2.0, w_pad=2.5)
plt.subplots_adjust(top=0.95)

save_pub(fig, "figure_exp4_m_comprehensive")
plt.close()
print("[Exp4] Comprehensive M-sensitivity figure done.")
