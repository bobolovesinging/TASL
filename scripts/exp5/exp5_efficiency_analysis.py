import argparse
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from scripts.exp_accuracy_vs_epoch import (  # noqa: E402
    set_seed,
    get_client_loaders_iid,
    get_test_loader,
    train_fl_epoch,
)
from core.gas_calculator import summarize_gas, plot_gas_comparison  # noqa: E402


def plot_latency_breakdown(details: dict, out_path: str) -> None:
    t_train = np.array(details.get("latency_t_train", []), dtype=np.float64)
    t_enc = np.array(details.get("latency_t_enc", []), dtype=np.float64)
    t_sim = np.array(details.get("latency_t_sim", []), dtype=np.float64)
    t_audit = np.array(details.get("latency_t_audit", []), dtype=np.float64)

    if len(t_train) == 0:
        raise ValueError("No latency details found. Please run with method='our' and return_details=True.")

    rounds = np.arange(1, len(t_train) + 1)

    plt.figure(figsize=(10, 6))
    plt.bar(rounds, t_train, label="T_train")
    plt.bar(rounds, t_enc, bottom=t_train, label="T_enc")
    plt.bar(rounds, t_sim, bottom=t_train + t_enc, label="T_sim")
    plt.bar(rounds, t_audit, bottom=t_train + t_enc + t_sim, label="T_audit")

    plt.xlabel("FL Round")
    plt.ylabel("Time (seconds)")
    plt.title("Exp5 Latency Composition per Round (PBSL/TAS)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def export_latency_csv(details: dict, out_csv: str) -> None:
    t_train = details.get("latency_t_train", [])
    t_enc = details.get("latency_t_enc", [])
    t_sim = details.get("latency_t_sim", [])
    t_audit = details.get("latency_t_audit", [])

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "T_train", "T_enc", "T_sim", "T_audit", "T_total"])
        for i in range(len(t_train)):
            total = float(t_train[i]) + float(t_enc[i]) + float(t_sim[i]) + float(t_audit[i])
            writer.writerow([i + 1, t_train[i], t_enc[i], t_sim[i], t_audit[i], total])


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 5: Latency + Gas efficiency analysis")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--client_num", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--n_clients_gas", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_loaders = get_client_loaders_iid(client_num=args.client_num, batch_size=args.batch_size)
    test_loader = get_test_loader(batch_size=args.batch_size)

    accs, details = train_fl_epoch(
        client_loaders=client_loaders,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        client_num=args.client_num,
        malicious_clients=[],
        method="our",
        attack_config=None,
        return_details=True,
    )

    results_dir = os.path.join(PROJECT_ROOT, "results")
    latency_plot_path = os.path.join(results_dir, "exp5_efficiency_latency.png")
    latency_csv_path = os.path.join(results_dir, "exp5_efficiency_latency.csv")
    gas_plot_path = os.path.join(results_dir, "exp5_gas_comparison.png")

    plot_latency_breakdown(details, latency_plot_path)
    export_latency_csv(details, latency_csv_path)

    gas_summary = summarize_gas(n_clients=args.n_clients_gas)
    plot_gas_comparison(gas_summary, gas_plot_path)

    print("\n[Exp5] Latency summary (average over rounds):")
    avg_train = float(np.mean(details.get("latency_t_train", [0.0])))
    avg_enc = float(np.mean(details.get("latency_t_enc", [0.0])))
    avg_sim = float(np.mean(details.get("latency_t_sim", [0.0])))
    avg_audit = float(np.mean(details.get("latency_t_audit", [0.0])))
    avg_total = avg_train + avg_enc + avg_sim + avg_audit
    print(
        f"T_train={avg_train:.4f}s, T_enc={avg_enc:.4f}s, "
        f"T_sim={avg_sim:.4f}s, T_audit={avg_audit:.4f}s, T_total={avg_total:.4f}s"
    )

    print("\n[Exp5] Gas summary:")
    for model_name, stats in gas_summary.items():
        print(
            f"{model_name}: baseline={stats['baseline_full_onchain_gas']:.0f}, "
            f"pbsl={stats['pbsl_tas_gas']:.0f}, saving={stats['saving_percent']:.4f}%"
        )

    print(f"\nFinal FL accuracy: {accs[-1]:.2f}%")
    print(f"Saved: {latency_plot_path}")
    print(f"Saved: {latency_csv_path}")
    print(f"Saved: {gas_plot_path}")


if __name__ == "__main__":
    main()
