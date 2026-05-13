"""
实验五: 经济扩展性与 Gas 成本分析 (Economic Scalability)

目标: 证明系统在大规模节点下"玩得起"
对比: Scheme A (Monolithic) vs. Scheme B (Practical-IPFS) vs. Scheme C (PBSL/TASL)
指标:
  - 单轮 Gas 随 n(10→50) 的增长曲线
  - 300轮累计 Gas 和 USD 成本
  - Model-Agnostic 特性: MNIST vs ST-GCN, 模型变大8倍但Gas几乎不变

输出:
- 双模型 Gas 对数曲线图
- 300轮累计 USD 成本柱状图
- 综合数据 CSV
"""
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Gas Assumptions
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GasAssumptions:
    base_tx_gas: int = 21_000
    ecrecover_gas: int = 3_000
    multisig_overhead_gas: int = 15_000
    sstore_new_slot_gas: int = 20_000
    calldata_nonzero_gas: int = 16
    event_log_gas: int = 2_000
    cid_store_gas: int = 20_000
    audit_snapshot_gas: int = 20_000
    final_aggregation_snapshot_gas: int = 30_000


MODEL_SIZES_BYTES = {
    "MNIST": int(1.6 * 1024 * 1024),
    "ST-GCN": int(12.5 * 1024 * 1024),
}

DEFAULT_CLIENT_SCALING = [10, 20, 30, 40, 50]
DEFAULT_ROUNDS = 300
DEFAULT_GAS_PRICE_GWEI = 20.0
DEFAULT_ETH_PRICE_USD = 3000.0


def bytes_to_words(size_bytes: int) -> int:
    return int(math.ceil(size_bytes / 32.0))


def gas_to_eth(gas: float, gas_price_gwei: float) -> float:
    return float(gas) * float(gas_price_gwei) * 1e-9


def gas_to_usd(gas: float, gas_price_gwei: float, eth_price_usd: float) -> float:
    return gas_to_eth(gas, gas_price_gwei) * float(eth_price_usd)


# ═══════════════════════════════════════════════════════════════════════
# Gas Estimation — Three Schemes
# ═══════════════════════════════════════════════════════════════════════

def estimate_scheme_a(n_clients: int, model_size_bytes: int, a: GasAssumptions) -> int:
    """Scheme A: Monolithic (全上链) — O(n * model_size)"""
    words = bytes_to_words(model_size_bytes)
    storage_term = n_clients * words * a.sstore_new_slot_gas
    upload_term = n_clients * (a.base_tx_gas + a.calldata_nonzero_gas * model_size_bytes)
    agg_term = words * max(1, n_clients - 1) * 120
    return int(storage_term + upload_term + agg_term)


def estimate_scheme_b(n_clients: int, a: GasAssumptions) -> int:
    """Scheme B: Practical-IPFS — O(n) 仅元数据"""
    per_client = a.base_tx_gas + a.cid_store_gas + a.ecrecover_gas + a.event_log_gas
    return int(n_clients * per_client)


def estimate_scheme_c(model_name: str, n_clients: int, a: GasAssumptions) -> int:
    """Scheme C: PBSL/TASL — O(1) + 轻量 O(n) 审计快照"""
    audit_complexity_factor = 1.00 if model_name == "MNIST" else 1.15
    tripartite_audit = 3 * (
        a.base_tx_gas + int(a.audit_snapshot_gas * audit_complexity_factor) + a.multisig_overhead_gas
    )
    final_commit = a.base_tx_gas + int(a.final_aggregation_snapshot_gas * audit_complexity_factor) + a.multisig_overhead_gas
    tiny_update_per_client = 800 if model_name == "MNIST" else 950
    light_client_updates = n_clients * tiny_update_per_client
    return int(tripartite_audit + final_commit + light_client_updates)


# ═══════════════════════════════════════════════════════════════════════
# Build Scaling Data
# ═══════════════════════════════════════════════════════════════════════

def build_scaling_data(
    n_values: List[int],
    rounds: int,
    gas_price_gwei: float,
    eth_price_usd: float,
    assumptions: GasAssumptions,
) -> List[Dict[str, float]]:
    rows = []
    for model_name, model_size in MODEL_SIZES_BYTES.items():
        for n in n_values:
            scheme_a = estimate_scheme_a(n, model_size, assumptions)
            scheme_b = estimate_scheme_b(n, assumptions)
            scheme_c = estimate_scheme_c(model_name, n, assumptions)

            scheme_a_total = rounds * scheme_a
            scheme_b_total = rounds * scheme_b
            scheme_c_total = rounds * scheme_c

            red_vs_a = 100.0 * (1.0 - scheme_c / scheme_a) if scheme_a > 0 else 0.0
            red_vs_b = 100.0 * (1.0 - scheme_c / scheme_b) if scheme_b > 0 else 0.0

            rows.append({
                "model": model_name,
                "n_clients": float(n),
                "model_size_bytes": float(model_size),
                "scheme_a_per_round_gas": float(scheme_a),
                "scheme_b_per_round_gas": float(scheme_b),
                "scheme_c_per_round_gas": float(scheme_c),
                "scheme_a_300round_gas": float(scheme_a_total),
                "scheme_b_300round_gas": float(scheme_b_total),
                "scheme_c_300round_gas": float(scheme_c_total),
                "scheme_a_300round_usd": float(gas_to_usd(scheme_a_total, gas_price_gwei, eth_price_usd)),
                "scheme_b_300round_usd": float(gas_to_usd(scheme_b_total, gas_price_gwei, eth_price_usd)),
                "scheme_c_300round_usd": float(gas_to_usd(scheme_c_total, gas_price_gwei, eth_price_usd)),
                "reduction_vs_monolithic_pct": float(red_vs_a),
                "reduction_vs_ipfs_pct": float(red_vs_b),
            })
    return rows


# ═══════════════════════════════════════════════════════════════════════
# Plotting (Publication Quality)
# ═══════════════════════════════════════════════════════════════════════

def plot_gas_scaling(out_path, rows: List[Dict]):
    """双模型 Gas 对数曲线 (单轮)"""
    colors = {"A": "#1A4595", "B": "#F2921D", "C": "#008F7A"}
    labels = {"A": "Scheme A: Monolithic", "B": "Scheme B: Practical-IPFS",
              "C": "Scheme C: TASL/PBSL (Ours)"}
    markers = {"A": "s", "B": "o", "C": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, model_name in enumerate(["MNIST", "ST-GCN"]):
        subset = [r for r in rows if r["model"] == model_name]
        n_vals = [int(r["n_clients"]) for r in subset]
        ax = axes[idx]

        for scheme, key in [("A", "scheme_a_per_round_gas"),
                             ("B", "scheme_b_per_round_gas"),
                             ("C", "scheme_c_per_round_gas")]:
            vals = [r[key] for r in subset]
            ax.plot(n_vals, vals, marker=markers[scheme], linewidth=2.2,
                   label=labels[scheme], color=colors[scheme], markersize=6)

        ax.set_yscale("log")
        ax.set_xlabel("Number of Clients (n)", fontsize=11)
        ax.set_ylabel("Gas per Round (log scale)", fontsize=11)
        ax.set_title(f"{model_name} ({MODEL_SIZES_BYTES[model_name]/1024/1024:.1f} MB)",
                     fontsize=12, fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(fontsize=8.5, loc="upper left")

        # 标注 C vs B 节省率 (n=50)
        n50 = [r for r in subset if int(r["n_clients"]) == 50]
        if n50:
            r = n50[0]
            red_b = r["reduction_vs_ipfs_pct"]
            ax.text(0.98, 0.08, f"C vs B saving: {red_b:.1f}%",
                   transform=ax.transAxes, fontsize=9, ha='right',
                   color=colors["C"], fontweight='bold')

    fig.suptitle("Exp5: Gas Cost Scaling — Per Round Comparison",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_usd_cost_bar(out_path, rows: List[Dict]):
    """300轮累计 USD 成本柱状图"""
    colors = {"A": "#1A4595", "B": "#F2921D", "C": "#008F7A"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, model_name in enumerate(["MNIST", "ST-GCN"]):
        ax = axes[idx]
        subset = [r for r in rows if r["model"] == model_name]
        n_vals = [int(r["n_clients"]) for r in subset]

        x = np.arange(len(n_vals))
        width = 0.25

        for i, (scheme, key) in enumerate([("A", "scheme_a_300round_usd"),
                                             ("B", "scheme_b_300round_usd"),
                                             ("C", "scheme_c_300round_usd")]):
            vals = [r[key] for r in subset]
            # 对数柱状图
            ax.bar(x + i * width, vals, width=width, color=colors[scheme],
                   label=f"Scheme {scheme}", edgecolor='white', linewidth=0.5)

        ax.set_yscale("log")
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"n={n}" for n in n_vals])
        ax.set_xlabel("Number of Clients", fontsize=11)
        ax.set_ylabel("300-Round Cumulative Cost (USD, log)", fontsize=11)
        ax.set_title(f"{model_name}", fontsize=12, fontweight='bold')
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.legend(fontsize=8.5)

    fig.suptitle("Exp5: 300-Round Cumulative Gas Cost (USD)",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_model_agnostic_comparison(out_path, rows: List[Dict]):
    """Model-Agnostic 特性对比: MNIST vs ST-GCN, Scheme C Gas"""
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, color, marker in [("MNIST", "#42A5F5", "o"),
                                        ("ST-GCN", "#EF5350", "s")]:
        subset = [r for r in rows if r["model"] == model_name]
        n_vals = [int(r["n_clients"]) for r in subset]
        gas_c = [r["scheme_c_per_round_gas"] for r in subset]
        ax.plot(n_vals, gas_c, marker=marker, linewidth=2.2,
               label=f"{model_name} ({MODEL_SIZES_BYTES[model_name]/1024/1024:.1f} MB)",
               color=color, markersize=7)

    ax.set_xlabel("Number of Clients (n)", fontsize=11)
    ax.set_ylabel("Scheme C Gas per Round", fontsize=11)
    ax.set_title("Model-Agnostic Economic Scalability\n"
                 "(8x model size → near-constant gas overhead)",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.25)

    # 标注增长比
    for model_name in ["MNIST", "ST-GCN"]:
        subset = [r for r in rows if r["model"] == model_name]
        gas_n10 = [r["scheme_c_per_round_gas"] for r in subset if int(r["n_clients"]) == 10][0]
        gas_n50 = [r["scheme_c_per_round_gas"] for r in subset if int(r["n_clients"]) == 50][0]
        ratio = gas_n50 / gas_n10
        ax.annotate(f"1→50: {ratio:.2f}x",
                   xy=(50, gas_n50), xytext=(42, gas_n50 * 1.15),
                   fontsize=9, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# CSV Export
# ═══════════════════════════════════════════════════════════════════════

def save_csv(rows: List[Dict], out_path: str):
    headers = [
        "model", "n_clients", "model_size_bytes",
        "scheme_a_per_round_gas", "scheme_b_per_round_gas", "scheme_c_per_round_gas",
        "scheme_a_300round_gas", "scheme_b_300round_gas", "scheme_c_300round_gas",
        "scheme_a_300round_usd", "scheme_b_300round_usd", "scheme_c_300round_usd",
        "reduction_vs_monolithic_pct", "reduction_vs_ipfs_pct",
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Exp5: Economic Scalability & Gas Cost")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--gas-price-gwei", type=float, default=DEFAULT_GAS_PRICE_GWEI)
    parser.add_argument("--eth-price-usd", type=float, default=DEFAULT_ETH_PRICE_USD)
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "results")
    os.makedirs(results_dir, exist_ok=True)

    assumptions = GasAssumptions()

    print("[Exp5] Economic Scalability & Gas Cost Analysis")
    print("=" * 55)

    rows = build_scaling_data(
        n_values=DEFAULT_CLIENT_SCALING,
        rounds=args.rounds,
        gas_price_gwei=args.gas_price_gwei,
        eth_price_usd=args.eth_price_usd,
        assumptions=assumptions,
    )

    # ── Print Summary Tables ─────────────────────────────────────────
    print(f"\n{'Model':<8} {'n':>3} | "
          f"{'A (Gas)':>16} {'B (Gas)':>12} {'C (Gas)':>12} | "
          f"{'C vs A':>8} {'C vs B':>8}")
    print("-" * 80)
    for r in rows:
        print(f"{r['model']:<8} {int(r['n_clients']):>3} | "
              f"{r['scheme_a_per_round_gas']:>16,.0f} "
              f"{r['scheme_b_per_round_gas']:>12,.0f} "
              f"{r['scheme_c_per_round_gas']:>12,.0f} | "
              f"{r['reduction_vs_monolithic_pct']:>7.2f}% "
              f"{r['reduction_vs_ipfs_pct']:>7.2f}%")

    # 300轮 USD 汇总
    print(f"\n{'Model':<8} {'n':>3} | "
          f"{'A (USD)':>16} {'B (USD)':>14} {'C (USD)':>14}")
    print("-" * 70)
    for r in rows:
        print(f"{r['model']:<8} {int(r['n_clients']):>3} | "
              f"${r['scheme_a_300round_usd']:>14,.2f} "
              f"${r['scheme_b_300round_usd']:>12,.2f} "
              f"${r['scheme_c_300round_usd']:>12,.2f}")

    # ── Generate Plots ───────────────────────────────────────────────
    print("\n  Generating plots...")
    plot_gas_scaling(os.path.join(results_dir, "exp5_gas_scaling.png"), rows)
    plot_usd_cost_bar(os.path.join(results_dir, "exp5_usd_cost_bar.png"), rows)
    plot_model_agnostic_comparison(
        os.path.join(results_dir, "exp5_model_agnostic.png"), rows)

    # ── Save CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, "exp5_gas_data.csv")
    save_csv(rows, csv_path)
    print(f"  Saved CSV: {csv_path}")

    # ── Key Findings ─────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("[Exp5] Key Findings")
    print("=" * 55)

    # ST-GCN n=50 经济性
    stgcn_50 = [r for r in rows if r["model"] == "ST-GCN" and int(r["n_clients"]) == 50]
    if stgcn_50:
        r = stgcn_50[0]
        print(f"  ST-GCN, n=50, 300 rounds:")
        print(f"    Monolithic: ${r['scheme_a_300round_usd']:,.2f}")
        print(f"    IPFS-Linear: ${r['scheme_b_300round_usd']:,.2f}")
        print(f"    TASL (Ours): ${r['scheme_c_300round_usd']:,.2f}")
        print(f"    C vs B reduction: {r['reduction_vs_ipfs_pct']:.2f}%")

    # Model-Agnostic
    mnist_10 = [r for r in rows if r["model"] == "MNIST" and int(r["n_clients"]) == 10]
    stgcn_10 = [r for r in rows if r["model"] == "ST-GCN" and int(r["n_clients"]) == 10]
    if mnist_10 and stgcn_10:
        gas_m = mnist_10[0]["scheme_c_per_round_gas"]
        gas_s = stgcn_10[0]["scheme_c_per_round_gas"]
        model_ratio = MODEL_SIZES_BYTES["ST-GCN"] / MODEL_SIZES_BYTES["MNIST"]
        gas_ratio = gas_s / gas_m
        print(f"\n  Model-Agnostic Property (n=10):")
        print(f"    Model size ratio: {model_ratio:.1f}x")
        print(f"    Gas ratio: {gas_ratio:.2f}x")
        print(f"    → 8x model size → only {gas_ratio:.2f}x gas increase")


if __name__ == "__main__":
    main()
