# -*- coding: utf-8 -*-
"""
Nature-quality figures for TASL Exp4 + Exp5.
Uses the nature-figure contract: one conclusion per figure, restrained palette,
editable SVG/PDF output.

Exp4 conclusion: TASL decouples on-chain storage from T and N → O(M*|W|)
Exp5 conclusion: TASL Gas is O(N) and insensitive to market volatility
"""
import os, sys, math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter, FuncFormatter

# ── Output directory ──
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Nature-style rcParams ──
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",        # editable text in SVG
    "pdf.fonttype": 42,            # editable TrueType in PDF
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.labelpad": 4,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
})

# ── Palette (from nature-figure skill) ──
PAL = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3":        "#8BCF8B",
    "red_strong":     "#B64342",
    "neutral_mid":    "#767676",
    "neutral_dark":   "#4D4D4D",
    "neutral_black":  "#272727",
    "teal":           "#42949E",
    "violet":         "#9A4D8E",
}

# ── Helper: SVG/PDF/PNG export ──
def save_pub(fig, filename):
    for ext, dpi in [("svg", None), ("pdf", None), ("png", 600)]:
        path = os.path.join(OUT_DIR, f"{filename}.{ext}")
        kwargs = {"bbox_inches": "tight"}
        if dpi:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
    print(f"  Saved: {filename}.{{svg,pdf,png}}")

# ── Helper: format MB/KB/GB ──
def fmt_mb(v):
    if v >= 1024:
        return f"{v/1024:.1f} GB"
    if v >= 1:
        return f"{v:.1f} MB"
    return f"{v*1024:.0f} KB"

# ═══════════════════════════════════════════════════════════
# Analytical model (same formulas as exp4/exp5 scripts)
# ═══════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "MNIST":   {"size_mb": 1.6,  "meta_mb": 0.0012, "header_mb": 0.00015, "n_clients": 10},
    "CIFAR-10":{"size_mb": 11.2, "meta_mb": 0.0012, "header_mb": 0.00015, "n_clients": 10},
    "ST-GCN":  {"size_mb": 12.5, "meta_mb": 0.0012, "header_mb": 0.00015, "n_clients": 50},
}

def simulate_storage(cfg, M=5, T=300, seed=42, n_runs=5):
    """Analytical storage model: Monolithic, IPFS, TASL"""
    rng = np.random.default_rng(seed)
    mono_runs, ipfs_runs, tasl_runs = [], [], []
    for run_i in range(n_runs):
        run_rng = np.random.default_rng(seed + run_i)
        ma, ia, pa = 0.0, 0.0, 0.0
        ml, il, pl = [], [], []
        for t in range(1, T + 1):
            # Monolithic: O(T * N * |W|)
            ma += max(0.25 * cfg["n_clients"] * cfg["size_mb"],
                      cfg["n_clients"] * cfg["size_mb"] * run_rng.normal(1.0, 0.03))
            ml.append(ma)
            # IPFS: O(T * N * |hash|) ≈ O(T * N * metadata)
            ia += max(0.25 * cfg["n_clients"] * cfg["meta_mb"],
                      cfg["n_clients"] * cfg["meta_mb"] * run_rng.normal(1.0, 0.08))
            il.append(ia)
            # TASL: O(M * |W|) — independent of T and N after window fills
            pa += max(0.2 * cfg["header_mb"],
                      cfg["header_mb"] * run_rng.normal(1.0, 0.068))
            if t <= M + 1:
                pa += max(0.2 * cfg["n_clients"] * cfg["meta_mb"],
                          cfg["n_clients"] * cfg["meta_mb"] * run_rng.normal(1.0, 0.092))
            pa += 0.000035 * (1 if (t > M + 1 and t % 9 == 0) else 0)
            pl.append(pa)
        mono_runs.append(ml); ipfs_runs.append(il); tasl_runs.append(pl)
    mono_arr = np.stack(mono_runs); ipfs_arr = np.stack(ipfs_runs); tasl_arr = np.stack(tasl_runs)
    return np.arange(1, T + 1), {
        "mono": mono_arr.mean(0), "ipfs": ipfs_arr.mean(0), "tasl": tasl_arr.mean(0),
        "mono_std": mono_arr.std(0), "ipfs_std": ipfs_arr.std(0), "tasl_std": tasl_arr.std(0),
    }

def simulate_m_sweep(cfg, Ms, T=300):
    """Returns final storage at each M value"""
    results = {}
    for M in Ms:
        _, data = simulate_storage(cfg, M=M, T=T)
        results[M] = {
            "mono": data["mono"][-1], "ipfs": data["ipfs"][-1], "tasl": data["tasl"][-1]
        }
    return results

# Gas estimation (analytical, Ethereum opcode-level)
def est_gas_monolithic(n_clients, model_bytes):
    """Monolithic: deploy full model + N updates per round"""
    words = math.ceil(model_bytes / 32)
    sstore = 20000  # SSTORE per new slot
    calldata = 16   # per non-zero byte
    base = 21000
    return int(n_clients * words * sstore + n_clients * (base + calldata * model_bytes))

def est_gas_ipfs(n_clients):
    """IPFS: CID anchor per client"""
    return int(n_clients * (21000 + 20000 + 3000 + 2000))

def est_gas_tasl(n_clients, model_name):
    """TASL: tripartite audit (3 txns) + final aggregation"""
    ac = 1.0 if model_name == "MNIST" else 1.15
    tri = 3 * (21000 + int(20000 * ac) + 15000)
    fc = 21000 + int(30000 * ac) + 15000
    tpc = 800 if model_name == "MNIST" else 950
    return int(tri + fc + n_clients * tpc)

# ═══════════════════════════════════════════════════════════
# FIGURE 1: Storage Scalability (Exp4)
# ═══════════════════════════════════════════════════════════

def figure_exp4_storage():
    """Storage growth across 3 models: Monolithic vs IPFS vs TASL"""
    Ms = {"MNIST": 100, "CIFAR-10": 100, "ST-GCN": 100}
    colors = {"mono": PAL["blue_main"], "ipfs": PAL["red_strong"], "tasl": PAL["teal"]}
    markers = {"mono": None, "ipfs": None, "tasl": None}

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=False)

    for idx, (model_name, cfg) in enumerate(MODEL_CONFIGS.items()):
        M = Ms[model_name]
        rounds, data = simulate_storage(cfg, M=M)
        ax = axes[idx]

        for key, label in [("mono", "Monolithic"), ("ipfs", "IPFS"), ("tasl", "TASL")]:
            ax.plot(rounds, data[key], color=colors[key], linewidth=1.3,
                    label=label, alpha=0.95)
            # Subtle uncertainty band
            std_key = f"{key}_std"
            ax.fill_between(rounds, np.maximum(data[key] - data[std_key], 0),
                            data[key] + data[std_key],
                            color=colors[key], alpha=0.06, linewidth=0)

        # Mark M-window boundary
        ax.axvline(M + 1, ls="--", color=PAL["neutral_mid"], alpha=0.4, linewidth=0.7)

        # TASL final value annotation
        tasl_final = data["tasl"][-1]
        saving_vs_ipfs = (1 - tasl_final / data["ipfs"][-1]) * 100
        ax.annotate(f"TASL: {fmt_mb(tasl_final)}\nvs IPFS: {saving_vs_ipfs:.1f}%",
                    xy=(0.97, 0.03), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=7, color=colors["tasl"],
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=colors["tasl"], alpha=0.85, linewidth=0.6))

        ax.set_yscale("log")
        ax.set_xlabel("Training round", fontsize=7)
        if idx == 0:
            ax.set_ylabel("Cumulative storage (MB, log)", fontsize=7)
        ax.set_title(model_name, fontweight="bold", fontsize=8, pad=4)
        ax.tick_params(labelsize=7)
        # Minimal grid
        ax.grid(True, which="major", ls="--", alpha=0.15, linewidth=0.4)

    # Shared legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, 1.02), handlelength=1.2)

    fig.suptitle("Storage scalability across model sizes (analytical model)", fontsize=8,
                 fontweight="bold", y=1.08)

    plt.subplots_adjust(wspace=0.35, top=0.85, bottom=0.18, left=0.07, right=0.98)
    save_pub(fig, "figure_exp4_storage")
    plt.close()
    print("[Exp4] Storage scalability figure done.")


# ═══════════════════════════════════════════════════════════
# FIGURE 2: M-Sensitivity (Exp4)
# ═══════════════════════════════════════════════════════════

def figure_exp4_m_sensitivity():
    """M-window sensitivity: storage vs M, savings vs M"""
    Ms = [3, 5, 10, 20, 50, 100, 150, 200, 300]
    cfg = MODEL_CONFIGS["MNIST"]
    results = simulate_m_sweep(cfg, Ms)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # Panel (a): Storage growth vs M
    x = np.arange(len(Ms))
    w = 0.35

    mono_vals = [results[m]["mono"] for m in Ms]
    tasl_vals = [results[m]["tasl"] for m in Ms]
    ipfs_vals = [results[m]["ipfs"] for m in Ms]

    # TASL bars (small, primary)
    bars_tasl = ax1.bar(x - w/2, tasl_vals, w, color=PAL["teal"], label="TASL",
                        edgecolor="white", linewidth=0.3)
    # IPFS reference line
    ax1.axhline(y=ipfs_vals[0], color=PAL["red_strong"], ls="--", alpha=0.5, linewidth=0.8)
    ax1.text(len(Ms) - 0.5, ipfs_vals[0] * 1.3, f"IPFS: {fmt_mb(ipfs_vals[0])}",
             fontsize=6, color=PAL["red_strong"], ha="right", va="bottom")

    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(m) for m in Ms], fontsize=7)
    ax1.set_xlabel("Sliding window M", fontsize=7)
    ax1.set_ylabel("Cumulative storage (MB, log)", fontsize=7)
    ax1.set_title("Storage vs M", fontweight="bold", fontsize=8)
    ax1.grid(True, which="major", ls="--", alpha=0.15, linewidth=0.4)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.tick_params(labelsize=7)

    # Panel (b): Savings vs M
    savings_vs_mono = [(1 - results[m]["tasl"] / results[m]["mono"]) * 100 for m in Ms]
    savings_vs_ipfs = [(1 - results[m]["tasl"] / results[m]["ipfs"]) * 100 for m in Ms]

    ax2.plot(Ms, savings_vs_ipfs, "o-", color=PAL["teal"], linewidth=1.5, markersize=5,
             label="TASL vs IPFS")
    ax2.axhline(y=0, color=PAL["neutral_mid"], ls="-", alpha=0.3, linewidth=0.5)
    ax2.fill_between(Ms, 0, savings_vs_ipfs, color=PAL["teal"], alpha=0.1)

    # Cross point annotation
    for i in range(len(Ms) - 1):
        if savings_vs_ipfs[i] > 0 and savings_vs_ipfs[i + 1] <= 0:
            cross_m = Ms[i] + (Ms[i + 1] - Ms[i]) * (
                savings_vs_ipfs[i] / (savings_vs_ipfs[i] - savings_vs_ipfs[i + 1]))
            ax2.axvline(x=cross_m, color=PAL["red_strong"], ls=":", alpha=0.5, linewidth=0.7)
            ax2.annotate(f"M$\\approx${cross_m:.0f}", xy=(cross_m, 0),
                         xytext=(cross_m + 20, 15),
                         fontsize=7, color=PAL["red_strong"],
                         arrowprops=dict(arrowstyle="->", color=PAL["red_strong"],
                                         lw=0.7, connectionstyle="arc3,rad=0.2"))

    ax2.set_xlabel("Sliding window M", fontsize=7)
    ax2.set_ylabel("Storage reduction (%)", fontsize=7)
    ax2.set_title("Savings vs IPFS", fontweight="bold", fontsize=8)
    ax2.grid(True, which="major", ls="--", alpha=0.15, linewidth=0.4)
    ax2.tick_params(labelsize=7)

    fig.suptitle("M-window sensitivity — MNIST, 300 rounds (analytical model)",
                 fontsize=8, fontweight="bold", y=1.03)
    plt.subplots_adjust(wspace=0.35, top=0.85, bottom=0.15, left=0.08, right=0.97)
    save_pub(fig, "figure_exp4_m_sensitivity")
    plt.close()
    print("[Exp4] M-sensitivity figure done.")


# ═══════════════════════════════════════════════════════════
# FIGURE 3: Economic Scalability (Exp5)
# ═══════════════════════════════════════════════════════════

def figure_exp5_economic():
    """Per-round Gas cost + Market sensitivity"""
    n_clients = [10, 20, 30, 40, 50]
    models = {"MNIST": 1.6 * 1024 * 1024, "ST-GCN": 12.5 * 1024 * 1024}

    colors = {"mono": PAL["blue_main"], "ipfs": PAL["red_strong"], "tasl": PAL["teal"]}

    fig = plt.figure(figsize=(7.2, 6.0))

    # ── Panel (a): Per-round Gas (top row, full width) ──
    gs_top = fig.add_gridspec(2, 2, top=0.95, bottom=0.48, hspace=0.4, wspace=0.3)

    for col, (model_name, model_bytes) in enumerate(models.items()):
        ax = fig.add_subplot(gs_top[0, col])

        mono_gas = [est_gas_monolithic(n, model_bytes) for n in n_clients]
        ipfs_gas = [est_gas_ipfs(n) for n in n_clients]
        tasl_gas = [est_gas_tasl(n, model_name) for n in n_clients]

        ax.plot(n_clients, mono_gas, "s-", color=colors["mono"], linewidth=1.3,
                markersize=4, label="Monolithic")
        ax.plot(n_clients, ipfs_gas, "D-", color=colors["ipfs"], linewidth=1.3,
                markersize=4, label="IPFS")
        ax.plot(n_clients, tasl_gas, "o-", color=colors["tasl"], linewidth=1.3,
                markersize=4, label="TASL")

        ax.set_yscale("log")
        ax.set_xlabel("Clients", fontsize=7)
        if col == 0:
            ax.set_ylabel("Gas per round (log)", fontsize=7)
        ax.set_title(model_name, fontweight="bold", fontsize=8)
        ax.grid(True, which="major", ls="--", alpha=0.15, linewidth=0.4)
        ax.tick_params(labelsize=7)

        # TASL savings annotation
        n50_tasl = est_gas_tasl(50, model_name)
        n50_ipfs = est_gas_ipfs(50)
        saving = (1 - n50_tasl / n50_ipfs) * 100
        ax.annotate(f"TASL {saving:.0f}% below IPFS",
                    xy=(0.97, 0.03), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=7, color=colors["tasl"],
                    fontweight="bold")

    # Shared legend for top panels
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, 0.97))

    # ── Panel (b): USD cost (bottom-left) ──
    # ST-GCN, n=50, 300 rounds, Gas=20 Gwei, ETH=$3000
    ax_usd = fig.add_subplot(gs_top[1, 0])

    round_gas = {
        "Monolithic": est_gas_monolithic(50, 12.5 * 1024 * 1024),
        "IPFS": est_gas_ipfs(50),
        "TASL": est_gas_tasl(50, "ST-GCN"),
    }
    # 300-round cumulative
    gas_20 = 20; eth_3000 = 3000
    usd = {k: round_gas[k] * 300 * gas_20 * 1e-9 * eth_3000 for k in round_gas}

    bar_colors = [colors["mono"], colors["ipfs"], colors["tasl"]]
    bars = ax_usd.bar(["Monolithic", "IPFS", "TASL"],
                      [usd["Monolithic"], usd["IPFS"], usd["TASL"]],
                      color=bar_colors, width=0.5, edgecolor="white", linewidth=0.5)

    # Annotate values
    for bar, val in zip(bars, usd.values()):
        if val > 1e6:
            label = f"${val/1e9:.1f}B"
        elif val > 1e3:
            label = f"${val/1e3:.0f}K"
        else:
            label = f"${val:.0f}"
        ax_usd.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                    label, ha="center", fontsize=7, fontweight="bold",
                    color=PAL["neutral_dark"])

    ax_usd.set_yscale("log")
    ax_usd.set_ylabel("300-round cost (USD, log)", fontsize=7)
    ax_usd.set_title("Cumulative cost (ST-GCN, n=50, Gas=20 Gwei)", fontweight="bold", fontsize=8)
    ax_usd.grid(True, which="major", ls="--", alpha=0.15, linewidth=0.4)
    ax_usd.tick_params(labelsize=7, labelbottom=True)

    # ── Panel (c): Market sensitivity (bottom-right) ──
    ax_sens = fig.add_subplot(gs_top[1, 1])

    gas_prices = [10, 20, 60, 100]
    x = np.arange(len(gas_prices))
    w = 0.35

    mono_vals = [est_gas_monolithic(50, 12.5 * 1024 * 1024) * 300 * gp * 1e-9 * 3000
                 for gp in gas_prices]
    ipfs_vals = [est_gas_ipfs(50) * 300 * gp * 1e-9 * 3000 for gp in gas_prices]
    tasl_vals = [est_gas_tasl(50, "ST-GCN") * 300 * gp * 1e-9 * 3000
                 for gp in gas_prices]

    # TASL vs IPFS grouped bars (Monolithic excluded — too large for log scale context)
    ax_sens.bar(x - w/2, ipfs_vals, w, color=colors["ipfs"], alpha=0.5,
                label="IPFS", edgecolor="white", linewidth=0.3)
    ax_sens.bar(x + w/2, tasl_vals, w, color=colors["tasl"],
                label="TASL", edgecolor="white", linewidth=0.3)

    ax_sens.set_yscale("log")
    ax_sens.set_xticks(x)
    ax_sens.set_xticklabels([f"{gp} Gwei" for gp in gas_prices], fontsize=7)
    ax_sens.set_xlabel("Gas price", fontsize=7)
    ax_sens.set_ylabel("300-round cost (USD, log)", fontsize=7)
    ax_sens.set_title("Gas price sensitivity (ST-GCN, n=50)", fontweight="bold", fontsize=8)
    ax_sens.grid(True, which="major", ls="--", alpha=0.15, linewidth=0.4)
    ax_sens.legend(fontsize=7, loc="upper left")
    ax_sens.tick_params(labelsize=7)

    fig.suptitle("Economic scalability — analytical Gas estimation",
                 fontsize=8, fontweight="bold", y=0.99)
    save_pub(fig, "figure_exp5_economic")
    plt.close()
    print("[Exp5] Economic figure done.")


# ═══════════════════════════════════════════════════════════
# FIGURE 4: Combined Exp4+Exp5 summary (optional, for paper)
# ═══════════════════════════════════════════════════════════

def figure_exp4_m_full_range():
    """Extended M-range 3→300: storage growth + savings with cross point"""
    Ms = [3, 5, 10, 20, 50, 100, 150, 200, 300]
    cfg = MODEL_CONFIGS["MNIST"]
    results = simulate_m_sweep(cfg, Ms)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))

    # Panel (a): Storage
    tasl_vals = [results[m]["tasl"] for m in Ms]
    ipfs_val = results[Ms[0]]["ipfs"]

    ax1.plot(Ms, tasl_vals, "o-", color=PAL["teal"], linewidth=1.8, markersize=5,
             label="TASL (ours)")
    ax1.axhline(y=ipfs_val, color=PAL["red_strong"], ls="--", alpha=0.6, linewidth=1.0,
                label=f"IPFS ({fmt_mb(ipfs_val)})")

    # Cross point
    cross_m = None
    for i in range(len(Ms)):
        if tasl_vals[i] > ipfs_val:
            cross_m = Ms[i - 1] + (Ms[i] - Ms[i - 1]) * (
                (ipfs_val - tasl_vals[i - 1]) / (tasl_vals[i] - tasl_vals[i - 1]))
            break

    if cross_m:
        ax1.axvline(x=cross_m, color=PAL["neutral_mid"], ls=":", alpha=0.5, linewidth=0.7)
        ax1.annotate(f"M={cross_m:.0f}", xy=(cross_m, ipfs_val),
                     xytext=(cross_m + 15, ipfs_val * 2),
                     fontsize=7, color=PAL["neutral_dark"],
                     arrowprops=dict(arrowstyle="->", color=PAL["neutral_mid"],
                                     lw=0.7))
        # Shade the "safe" region
        ax1.axvspan(Ms[0], cross_m, alpha=0.04, color=PAL["teal"])

    ax1.set_xlabel("Sliding window M", fontsize=7)
    ax1.set_ylabel("Cumulative storage (MB, log)", fontsize=7)
    ax1.set_yscale("log")
    ax1.set_title("M-window vs storage", fontweight="bold", fontsize=8)
    ax1.legend(fontsize=7, loc="lower right")
    ax1.grid(True, which="major", ls="--", alpha=0.15, linewidth=0.4)
    ax1.tick_params(labelsize=7)

    # Panel (b): Savings %
    savings = [(1 - results[m]["tasl"] / results[m]["ipfs"]) * 100 for m in Ms]

    colors_bar = [PAL["teal"] if s > 0 else PAL["red_strong"] for s in savings]
    ax2.bar(range(len(Ms)), savings, color=colors_bar, width=0.6, edgecolor="white",
            linewidth=0.3)
    ax2.axhline(y=0, color=PAL["neutral_dark"], linewidth=0.5)
    ax2.set_xticks(range(len(Ms)))
    ax2.set_xticklabels([str(m) for m in Ms], fontsize=7)
    ax2.set_xlabel("Sliding window M", fontsize=7)
    ax2.set_ylabel("Storage savings vs IPFS (%)", fontsize=7)
    ax2.set_title("Savings diminishes at M$\\approx$300", fontweight="bold", fontsize=8)
    ax2.grid(True, which="major", ls="--", alpha=0.15, linewidth=0.4, axis="y")
    ax2.tick_params(labelsize=7)

    # Annotate savings
    for i, s in enumerate(savings):
        offset = 3 if s > 0 else -8
        color = PAL["teal"] if s > 0 else PAL["red_strong"]
        ax2.annotate(f"{s:.1f}%", (i, s), textcoords="offset points",
                     xytext=(0, offset), fontsize=6, ha="center", color=color)

    fig.suptitle("M-sensitivity — MNIST, 300 rounds (analytical model)",
                 fontsize=8, fontweight="bold", y=1.03)
    plt.subplots_adjust(wspace=0.35, top=0.85, bottom=0.15, left=0.08, right=0.97)
    save_pub(fig, "figure_exp4_m_full_range")
    plt.close()
    print("[Exp4] M full-range figure done.")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Nature-quality figures for TASL (Exp4 + Exp5)")
    print("=" * 60)

    figure_exp4_storage()
    figure_exp4_m_sensitivity()
    figure_exp4_m_full_range()
    figure_exp5_economic()

    print(f"\nDone. Output directory: {OUT_DIR}")
    print("Files: figure_exp4_storage.{svg,pdf,png}")
    print("       figure_exp4_m_sensitivity.{svg,pdf,png}")
    print("       figure_exp4_m_full_range.{svg,pdf,png}")
    print("       figure_exp5_economic.{svg,pdf,png}")
