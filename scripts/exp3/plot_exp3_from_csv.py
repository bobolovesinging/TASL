import argparse
import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _read_exp3_csv(
    csv_path: str,
) -> Tuple[Dict[str, Dict[str, List[int]]], Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, List[int]]]]:
    scenarios = ["clean", "attacked"]
    groups = ["single", "tas"]
    rounds: Dict[str, Dict[str, List[int]]] = {s: {g: [] for g in groups} for s in scenarios}
    accs: Dict[str, Dict[str, List[float]]] = {s: {g: [] for g in groups} for s in scenarios}
    attack_active: Dict[str, Dict[str, List[int]]] = {s: {g: [] for g in groups} for s in scenarios}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r_raw = (row.get("round") or "").strip()
            g = (row.get("group") or "").strip().lower()
            s = (row.get("scenario") or "").strip().lower()
            if not r_raw.isdigit():
                continue
            if s not in rounds or g not in rounds[s]:
                continue

            rounds[s][g].append(int(r_raw))
            accs[s][g].append(float(row["accuracy"]))
            attack_active[s][g].append(int(float(row.get("attack_active", "0") or "0")))

    return rounds, accs, attack_active


def plot_accuracy_curves(
    csv_path: str,
    out_path: str,
    title: str,
    *,
    show_single_clean: bool,
):
    rounds, accs, attack_active = _read_exp3_csv(csv_path)

    plt.figure(figsize=(9, 5.2))
    # Clean curves (dashed)
    if show_single_clean and rounds.get("clean", {}).get("single"):
        plt.plot(
            rounds["clean"]["single"],
            accs["clean"]["single"],
            label="Single Aggregator (clean)",
            color="#C84C4C",
            linewidth=1.8,
            linestyle="--",
            alpha=0.75,
        )
    if rounds.get("clean", {}).get("tas"):
        plt.plot(
            rounds["clean"]["tas"],
            accs["clean"]["tas"],
            label="Full TAS (clean)",
            color="#2E8B57",
            linewidth=1.8,
            linestyle="--",
            alpha=0.75,
        )

    # Attacked curves (solid)
    if rounds.get("attacked", {}).get("single"):
        plt.plot(
            rounds["attacked"]["single"],
            accs["attacked"]["single"],
            label="Single Aggregator (attacked)",
            color="#C84C4C",
            linewidth=2.2,
        )
    if rounds.get("attacked", {}).get("tas"):
        plt.plot(
            rounds["attacked"]["tas"],
            accs["attacked"]["tas"],
            label="Full TAS (attacked)",
            color="#2E8B57",
            linewidth=2.2,
        )

    # Mark attack rounds (shared schedule) using light vertical shading.
    # Use union to be robust if one group is missing some rows.
    attack_rounds = sorted(
        {r for r, a in zip(rounds.get("attacked", {}).get("single", []), attack_active.get("attacked", {}).get("single", [])) if a == 1}
        | {r for r, a in zip(rounds.get("attacked", {}).get("tas", []), attack_active.get("attacked", {}).get("tas", [])) if a == 1}
    )
    for r in attack_rounds:
        plt.axvspan(r - 0.5, r + 0.5, color="#999999", alpha=0.08, linewidth=0)

    plt.xlabel("Round")
    plt.ylabel("Test Accuracy (%)")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180)


def main():
    parser = argparse.ArgumentParser(description="Plot Exp3 accuracy curves from CSV")
    parser.add_argument("--csv", type=str, required=True, help="Path to exp3_governance_data_v*.csv")
    parser.add_argument("--out", type=str, default="", help="Output png path (default: results/exp3_governance_curve_*.png)")
    parser.add_argument("--title", type=str, default="Experiment 3: Accuracy Curves under Byzantine Executor")
    parser.add_argument(
        "--show-single-clean",
        action="store_true",
        help="可选：显示 Single Aggregator (clean) 虚线（默认不显示）",
    )
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    if args.out:
        out_path = os.path.abspath(args.out)
    else:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        # exp3_governance_data_v3 -> exp3_governance_curve_v3
        out_name = base.replace("exp3_governance_data", "exp3_governance_curve") + ".png"
        out_path = os.path.join(os.path.dirname(csv_path), out_name)

    plot_accuracy_curves(csv_path, out_path, args.title, show_single_clean=bool(args.show_single_clean))
    print(f"Saved curve figure: {out_path}")


if __name__ == "__main__":
    main()

