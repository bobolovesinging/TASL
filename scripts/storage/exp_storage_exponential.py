import csv
import os
from typing import List

import matplotlib.pyplot as plt


def compute_storage_series(
    t_max: int,
    s_model: float,
    s_meta: float,
    s_audit: float,
    window_size: int,
    num_clients: int,
) -> List[dict]:
    rows = []
    for t in range(1, t_max + 1):
        scheme_a = s_model * (1.1 ** t)
        scheme_b = t * s_meta * num_clients
        scheme_c = (t * s_audit) + (min(t, window_size + 1) * num_clients * s_meta)
        rows.append({
            "round": t,
            "scheme_a": scheme_a,
            "scheme_b": scheme_b,
            "scheme_c": scheme_c,
        })
    return rows


def main():
    t_max = 100
    window_size = 5
    num_clients = 10
    s_meta = 1024.0
    s_audit = 102.4

    s_model = 1.5 * 1024 * 1024

    rows = compute_storage_series(
        t_max,
        s_model,
        s_meta,
        s_audit,
        window_size,
        num_clients,
    )

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "storage_exponential_v1.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "scheme_a", "scheme_b", "scheme_c"])
        writer.writeheader()
        writer.writerows(rows)

    rounds = [row["round"] for row in rows]
    scheme_a = [row["scheme_a"] for row in rows]
    scheme_b = [row["scheme_b"] for row in rows]
    scheme_c = [row["scheme_c"] for row in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, scheme_a, label="Scheme A: Monolithic-BC", linewidth=2.0)
    plt.plot(rounds, scheme_b, label="Scheme B: IPFS-Linear", linewidth=2.0)
    plt.plot(rounds, scheme_c, label="Scheme C: PBFL (Pruned)", linewidth=2.0)
    plt.yscale("log")
    plt.xlabel("Round")
    plt.ylabel("Storage (bytes, log scale)")
    plt.title("Exponential Storage Stress Test")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()

    plt.annotate(
        "Pruning Triggered",
        xy=(window_size + 1, scheme_c[window_size]),
        xytext=(window_size + 6, scheme_c[window_size] * 5),
        arrowprops=dict(arrowstyle="->"),
    )
    plt.annotate(
        "Storage Gap",
        xy=(t_max, scheme_a[-1]),
        xytext=(t_max - 20, scheme_a[-1] / 50),
        arrowprops=dict(arrowstyle="->"),
    )

    plot_path = os.path.join(output_dir, "storage_ablation_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)

    saving_ratio = 1.0 - (scheme_c[-1] / scheme_a[-1])
    print(f"S_model(bytes)={s_model:.2f}, S_meta(bytes)={s_meta:.2f}")
    print(f"Saving ratio at round {t_max}: {saving_ratio * 100:.6f}%")
    print(f"Saved CSV to {csv_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
