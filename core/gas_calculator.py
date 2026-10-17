import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt


@dataclass
class GasAssumptions:
    # Common
    base_tx_gas: int = 21_000
    ecrecover_gas: int = 3_000
    multisig_overhead_gas: int = 15_000

    # Storage / calldata
    sstore_new_slot_gas: int = 20_000      # per 32-byte slot
    calldata_nonzero_gas: int = 16         # per byte
    event_log_gas: int = 2_000

    # Practical-IPFS per client
    cid_store_gas: int = 20_000

    # PBSL-TAS constants (O(1) on-chain interactions)
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


# ------------------------
# Scheme A: Monolithic (upper bound)
# ------------------------
def estimate_scheme_a_monolithic_round_gas(
    n_clients: int,
    model_size_bytes: int,
    a: GasAssumptions,
) -> int:
    words = bytes_to_words(model_size_bytes)

    # fatal storage term: n * words * 20,000
    storage_term = n_clients * words * a.sstore_new_slot_gas
    upload_term = n_clients * (a.base_tx_gas + a.calldata_nonzero_gas * model_size_bytes)
    # naive on-chain aggregation loop
    agg_term = words * max(1, n_clients - 1) * 120

    return int(storage_term + upload_term + agg_term)


# ------------------------
# Scheme B: Practical-IPFS baseline
# n * (Base_Tx + CID_Storage + Signature_Verify + Event_Log)
# ------------------------
def estimate_scheme_b_practical_ipfs_round_gas(n_clients: int, a: GasAssumptions) -> int:
    per_client = a.base_tx_gas + a.cid_store_gas + a.ecrecover_gas + a.event_log_gas
    return int(n_clients * per_client)


# ------------------------
# Scheme C: PBSL (TAS optimized O(1))
# 3 * (Base_Tx + Audit_Snapshot + Multi_Sig) + 1 * final aggregation snapshot
# A small model-dependent commitment factor is included to reflect proof complexity.
# ------------------------
def estimate_scheme_c_pbsl_round_gas(model_name: str, n_clients: int, a: GasAssumptions) -> int:
    audit_complexity_factor = 1.00 if model_name == "MNIST" else 1.15

    tripartite_audit = 3 * (
        a.base_tx_gas + int(a.audit_snapshot_gas * audit_complexity_factor) + a.multisig_overhead_gas
    )
    final_commit = a.base_tx_gas + int(a.final_aggregation_snapshot_gas * audit_complexity_factor) + a.multisig_overhead_gas

    # small O(n) term: lightweight per-client stake/reputation touched on-chain
    tiny_update_per_client = 800 if model_name == "MNIST" else 950
    light_client_updates = n_clients * tiny_update_per_client

    return int(tripartite_audit + final_commit + light_client_updates)


def build_scaling_rows(
    n_values: List[int],
    rounds: int,
    gas_price_gwei: float,
    eth_price_usd: float,
    assumptions: GasAssumptions,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    for model_name, model_size in MODEL_SIZES_BYTES.items():
        for n in n_values:
            scheme_c_round = estimate_scheme_c_pbsl_round_gas(model_name, n, assumptions)
            scheme_a_round = estimate_scheme_a_monolithic_round_gas(n, model_size, assumptions)
            scheme_b_round = estimate_scheme_b_practical_ipfs_round_gas(n, assumptions)
            scheme_c = scheme_c_round

            scheme_a_total = rounds * scheme_a_round
            scheme_b_total = rounds * scheme_b_round
            scheme_c_total = rounds * scheme_c

            red_vs_a = 100.0 * (1.0 - scheme_c / scheme_a_round) if scheme_a_round > 0 else 0.0
            red_vs_b = 100.0 * (1.0 - scheme_c / scheme_b_round) if scheme_b_round > 0 else 0.0

            rows.append({
                "model": model_name,
                "n_clients": float(n),
                "model_size_bytes": float(model_size),
                "scheme_a_monolithic_round_gas": float(scheme_a_round),
                "scheme_b_practical_round_gas": float(scheme_b_round),
                "scheme_c_pbsl_round_gas": float(scheme_c),
                "scheme_a_300round_gas": float(scheme_a_total),
                "scheme_b_300round_gas": float(scheme_b_total),
                "scheme_c_300round_gas": float(scheme_c_total),
                "scheme_a_300round_usd": float(gas_to_usd(scheme_a_total, gas_price_gwei, eth_price_usd)),
                "scheme_b_300round_usd": float(gas_to_usd(scheme_b_total, gas_price_gwei, eth_price_usd)),
                "scheme_c_300round_usd": float(gas_to_usd(scheme_c_total, gas_price_gwei, eth_price_usd)),
                "reduction_vs_monolithic_percent": float(red_vs_a),
                "Reduction_vs_Practical": float(red_vs_b),
            })

    return rows


def save_scaling_csv(rows: List[Dict[str, float]], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    headers = [
        "model", "n_clients", "model_size_bytes",
        "scheme_a_monolithic_round_gas", "scheme_b_practical_round_gas", "scheme_c_pbsl_round_gas",
        "scheme_a_300round_gas", "scheme_b_300round_gas", "scheme_c_300round_gas",
        "scheme_a_300round_usd", "scheme_b_300round_usd", "scheme_c_300round_usd",
        "reduction_vs_monolithic_percent", "Reduction_vs_Practical",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _plot_three_lines_for_model(ax, rows: List[Dict[str, float]], model_name: str) -> None:
    subset = [r for r in rows if r["model"] == model_name]
    n_vals = [int(r["n_clients"]) for r in subset]
    scheme_a = [r["scheme_a_monolithic_round_gas"] for r in subset]
    scheme_b = [r["scheme_b_practical_round_gas"] for r in subset]
    scheme_c = [r["scheme_c_pbsl_round_gas"] for r in subset]

    ax.plot(n_vals, scheme_a, marker="o", linewidth=2.2, label="Scheme A: Monolithic (Upper Bound)")
    ax.plot(n_vals, scheme_b, marker="^", linewidth=2.2, label="Scheme B: Practical-IPFS")
    ax.plot(n_vals, scheme_c, marker="s", linewidth=2.2, label="Scheme C: PBSL (Ours)")
    ax.set_yscale("log")
    ax.set_xlabel("Number of clients (n)")
    ax.set_ylabel("Gas per round (log scale)")
    ax.set_title(f"{model_name}")
    ax.grid(True, linestyle="--", alpha=0.35)


def plot_comprehensive_figure(rows: List[Dict[str, float]], out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    _plot_three_lines_for_model(axes[0], rows, "MNIST")
    _plot_three_lines_for_model(axes[1], rows, "ST-GCN")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Exp5 Comprehensive Gas Comparison (Three-Scheme, Dual-Model)", y=1.04)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=240)
    plt.close()


def summarize_gas(n_clients: int = 50) -> Dict[str, Dict[str, float]]:
    # backward-compatible utility for existing scripts
    a = GasAssumptions()
    result: Dict[str, Dict[str, float]] = {}
    for model, size in MODEL_SIZES_BYTES.items():
        scheme_c = estimate_scheme_c_pbsl_round_gas(model, n_clients, a)
        scheme_a = estimate_scheme_a_monolithic_round_gas(n_clients, size, a)
        reduction = 100.0 * (1.0 - scheme_c / scheme_a) if scheme_a > 0 else 0.0
        result[model] = {
            "model_size_bytes": float(size),
            "baseline_full_onchain_gas": float(scheme_a),
            "pbsl_tas_gas": float(scheme_c),
            "saving_percent": float(reduction),
        }
    return result


def plot_gas_comparison(summary: Dict[str, Dict[str, float]], out_path: str) -> None:
    models = list(summary.keys())
    baseline_vals = [summary[m]["baseline_full_onchain_gas"] for m in models]
    pbsl_vals = [summary[m]["pbsl_tas_gas"] for m in models]

    x = range(len(models))
    width = 0.35
    plt.figure(figsize=(9, 5.5))
    plt.bar([i - width / 2 for i in x], baseline_vals, width=width, label="Scheme A")
    plt.bar([i + width / 2 for i in x], pbsl_vals, width=width, label="Scheme C")
    plt.yscale("log")
    plt.xticks(list(x), models)
    plt.ylabel("Gas (log scale)")
    plt.title("Exp5 Gas Comparison")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def print_terminal_summary(rows: List[Dict[str, float]], rounds: int) -> None:
    print("\n[Exp5 Three-Scheme Summary | Per-round]")
    for model in MODEL_SIZES_BYTES.keys():
        model_rows = [r for r in rows if r["model"] == model]
        for r in model_rows:
            print(
                f"{model:7s} n={int(r['n_clients']):2d}: "
                f"A={r['scheme_a_monolithic_round_gas']:.0f}, "
                f"B={r['scheme_b_practical_round_gas']:.0f}, "
                f"C={r['scheme_c_pbsl_round_gas']:.0f}, "
                f"C vs B={r['Reduction_vs_Practical']:.2f}%"
            )

    target = [r for r in rows if r["model"] == "ST-GCN" and int(r["n_clients"]) == 50]
    if target:
        r = target[0]
        print("\n[Exp5 Economic Projection | ST-GCN, n=50]")
        print(f"Scheme A Monolithic ({rounds} rounds): ${r['scheme_a_300round_usd'] / 1e4:.2f} x10^4 USD")
        print(f"Scheme B Practical-IPFS ({rounds} rounds): ${r['scheme_b_300round_usd']:.2f} USD")
        print(f"Scheme C PBSL ({rounds} rounds): ${r['scheme_c_300round_usd']:.2f} USD")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Exp5 enhanced three-scheme gas analysis")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--n_values", type=int, nargs="*", default=DEFAULT_CLIENT_SCALING)
    parser.add_argument("--gas_price_gwei", type=float, default=DEFAULT_GAS_PRICE_GWEI)
    parser.add_argument("--eth_price_usd", type=float, default=DEFAULT_ETH_PRICE_USD)

    default_results = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    parser.add_argument("--csv_out", type=str, default=os.path.join(default_results, "exp5_gas_scaling.csv"))
    parser.add_argument("--fig_out", type=str, default=os.path.join(default_results, "exp5_comprehensive_gas.png"))
    args = parser.parse_args()

    assumptions = GasAssumptions()
    rows = build_scaling_rows(
        n_values=list(args.n_values),
        rounds=int(args.rounds),
        gas_price_gwei=float(args.gas_price_gwei),
        eth_price_usd=float(args.eth_price_usd),
        assumptions=assumptions,
    )

    save_scaling_csv(rows, args.csv_out)
    plot_comprehensive_figure(rows, args.fig_out)
    print(f"Saved CSV to {args.csv_out}")
    print(f"Saved comprehensive figure to {args.fig_out}")

    print_terminal_summary(rows, rounds=int(args.rounds))


if __name__ == "__main__":
    main()
