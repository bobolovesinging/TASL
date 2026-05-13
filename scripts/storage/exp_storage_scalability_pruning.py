import os
import sys
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
blockchain_dir = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)
sys.path.insert(0, blockchain_dir)

from exp_accuracy_vs_epoch import (
    set_seed,
    get_client_loaders_iid,
    get_test_loader,
    train_fl_epoch,
)


def run_storage_experiment(
    epochs: int = 50,
    client_num: int = 10,
    M: int = 3,
    seed: int = 100,
    batch_size: int = 128,
):
    """
    7.2.4 Storage Scalability and Pruning Effect

    Compare:
    1) Traditional single-layer chain (no pruning): linear growth with round * clients.
    2) Dual-layer chain with C_T sliding window (M+1) + C_D governance chain.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client_loaders = get_client_loaders_iid(client_num=client_num, batch_size=batch_size)
    test_loader = get_test_loader(batch_size=batch_size)

    # No attack needed for storage trend demonstration.
    details = train_fl_epoch(
        client_loaders=client_loaders,
        test_loader=test_loader,
        device=device,
        epochs=epochs,
        client_num=client_num,
        malicious_clients=[],
        method='our',
        attack_config=None,
        M=M,
        return_details=True,
        enable_storage_pruning=True,
        print_similarity_scores=False,
    )

    ct_sizes = details.get("ct_sizes", [])
    cd_sizes = details.get("cd_sizes", [])

    rounds = np.arange(1, epochs + 1)

    # Traditional single-layer chain: all client records are kept forever.
    # Storage unit here is number of on-chain metadata records.
    single_layer_storage = rounds * client_num

    # Proposed dual-layer + pruning storage.
    dual_layer_storage = np.array(ct_sizes) + np.array(cd_sizes)

    return {
        "rounds": rounds,
        "single_layer_storage": single_layer_storage,
        "dual_layer_storage": dual_layer_storage,
        "ct_sizes": np.array(ct_sizes),
        "cd_sizes": np.array(cd_sizes),
    }


def save_csv(result_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "single_layer_storage_records",
            "dual_layer_storage_records",
            "ct_records",
            "cd_records",
        ])
        for i in range(len(result_dict["rounds"])):
            writer.writerow([
                int(result_dict["rounds"][i]),
                int(result_dict["single_layer_storage"][i]),
                int(result_dict["dual_layer_storage"][i]),
                int(result_dict["ct_sizes"][i]),
                int(result_dict["cd_sizes"][i]),
            ])


def plot_storage(result_dict, save_path, M: int):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    rounds = result_dict["rounds"]
    single_layer = result_dict["single_layer_storage"]
    dual_layer = result_dict["dual_layer_storage"]

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, single_layer, label="Traditional Single-Layer Chain", linewidth=2)
    plt.plot(rounds, dual_layer, label=f"Dual-Layer + C_T Pruning (M+1={M+1})", linewidth=2)

    plt.xlabel("Training Round")
    plt.ylabel("Storage Occupancy (metadata records)")
    plt.title("7.2.4 Storage Scalability and Pruning Effect")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--M", type=int, default=3, help="Sliding window uses M+1 records for C_T")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    result = run_storage_experiment(
        epochs=args.epochs,
        client_num=args.clients,
        M=args.M,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    results_dir = os.path.join(blockchain_dir, "results")
    csv_path = os.path.join(results_dir, "storage_scalability_pruning.csv")
    fig_path = os.path.join(results_dir, "exp_storage_scalability_pruning.png")

    save_csv(result, csv_path)
    print(f"Saved CSV to {csv_path}")
    plot_storage(result, fig_path, M=args.M)
