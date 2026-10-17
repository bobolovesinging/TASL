import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from scripts.accuracy.exp_accuracy_vs_epoch import (  # noqa: E402
    AttackConfig,
    HAS_ATTACK,
    get_client_loaders_dirichlet,
    get_client_loaders_iid,
    get_test_loader,
    set_seed,
    train_fl_epoch,
)


@dataclass
class BenchmarkConfig:
    client_num: int = 10
    epochs: int = 50
    batch_size: int = 128
    seed: int = 100
    local_epochs: int = 1
    dirichlet_alpha: float = 0.3
    malicious_ratio: float = 0.4


METHODS = {
    "FedAvg": "avg",
    "Multi-Krum": "multi_krum",
    "FLTrust": "fltrust",
    "Trimmed Mean": "trimmed_mean",
}

ATTACKS = {
    "label_flip": "label_flip",
    "sign_flip": "sign_flip",
    "gaussian_noise": "gaussian_noise",
}


def make_attack_config(attack_name: str, malicious_clients: List[int]):
    if not HAS_ATTACK:
        return None
    cfg = AttackConfig()
    cfg.enable_attack = True
    cfg.malicious_clients = list(malicious_clients)
    cfg.attack_type = attack_name
    if attack_name == "label_flip":
        cfg.label_flip_map = {i: 9 - i for i in range(10)}
    elif attack_name == "sign_flip":
        cfg.sign_flip_factor = -2.0
    elif attack_name == "gaussian_noise":
        cfg.gaussian_noise_std = 1.0
    return cfg


def get_loaders(client_num: int, batch_size: int, non_iid: bool, alpha: float, seed: int):
    if non_iid:
        return get_client_loaders_dirichlet(client_num=client_num, batch_size=batch_size, alpha=alpha, seed=seed)
    return get_client_loaders_iid(client_num=client_num, batch_size=batch_size)


def cache_key(parts: Dict[str, object]) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=True)
    return str(abs(hash(payload)))


def load_cache(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def run_one_setting(
    client_loaders,
    test_loader,
    cfg: BenchmarkConfig,
    attack_name: str,
    split_name: str,
    method_name: str,
    rule: str,
    malicious_clients: List[int],
):
    local_attack_config = make_attack_config(attack_name, malicious_clients)
    # FLTrust 需要 root_loader 来计算服务器参考梯度
    from scripts.accuracy.exp_accuracy_vs_epoch import get_root_loader
    root_loader = get_root_loader(batch_size=128, root_size=100, seed=cfg.seed) if rule == 'fltrust' else None
    accs, details = train_fl_epoch(
        client_loaders=client_loaders,
        test_loader=test_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        epochs=cfg.epochs,
        client_num=cfg.client_num,
        malicious_clients=malicious_clients,
        method=rule,
        attack_config=local_attack_config,
        local_epochs=cfg.local_epochs,
        dirichlet_alpha=cfg.dirichlet_alpha,
        malicious_executor_round_rate=0.0,
        rng=np.random.RandomState(cfg.seed),
        root_loader=root_loader,
        return_details=True,
    )
    return accs, details


def plot_convergence(results: Dict[str, List[float]], out_path: str, title: str):
    plt.figure(figsize=(11, 6))
    for method, accs in results.items():
        plt.plot(range(1, len(accs) + 1), accs, label=method)
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_malicious_pressure(rows: List[Dict[str, float]], out_path: str, title: str):
    METHOD_STYLES = {
        "Our TASL":     {"color": "red",    "marker": "s", "linewidth": 2.5, "zorder": 5},
        "FLTrust":      {"color": "blue",   "marker": "^", "linewidth": 1.8},
        "Multi-Krum":   {"color": "green",  "marker": "D", "linewidth": 1.8},
        "FedAvg":       {"color": "orange", "marker": "o", "linewidth": 1.8},
        "Trimmed Mean": {"color": "purple", "marker": "v", "linewidth": 1.8},
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = sorted(set(r["method"] for r in rows))
    ratios = sorted(set(r["ratio"] for r in rows))
    for method in methods:
        xs, ys = [], []
        for ratio in ratios:
            match = [r for r in rows if r["method"] == method and r["ratio"] == ratio]
            if match:
                xs.append(ratio)
                ys.append(match[0]["best_acc"])
        style = METHOD_STYLES.get(method, {"color": "gray", "marker": "o", "linewidth": 1.5})
        ax.plot(xs, ys, label=method, **style)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=7, ha="center",
                        color=style.get("color", "black"))
    ax.axhline(y=90, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Malicious Ratio", fontsize=13)
    ax.set_ylabel("Best Accuracy (%)", fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(ratios)
    ax.set_ylim(0, 108)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MNIST robustness benchmark v4")
    parser.add_argument("--use_cache", action="store_true", default=True, help="Reuse cached results when available")
    parser.add_argument("--no_cache", action="store_true", help="Force recomputation and overwrite cache")
    parser.add_argument("--rerun_only", type=str, default="", help="Methods to rerun fresh, comma-separated (empty = all)")
    parser.add_argument("--client_num", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3)
    parser.add_argument("--malicious_ratio", type=float, default=0.4)
    args = parser.parse_args()

    cfg = BenchmarkConfig(
        client_num=args.client_num,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        local_epochs=args.local_epochs,
        dirichlet_alpha=args.dirichlet_alpha,
        malicious_ratio=args.malicious_ratio,
    )
    use_cache = args.use_cache and not args.no_cache
    rerun_only = {m.strip() for m in args.rerun_only.split(",") if m.strip()}
    results_dir = os.path.join(PROJECT_ROOT, "results")
    accuracy_dir = os.path.join(results_dir, "accuracy")
    cache_dir = os.path.join(results_dir, "cache", "mnist_benchmark_v4")
    csv_out = os.path.join(accuracy_dir, "mnist_benchmark_v4.csv")
    conv_out = os.path.join(accuracy_dir, "mnist_convergence_40pct_attack.png")
    pressure_out = os.path.join(accuracy_dir, "mnist_malicious_pressure.png")

    split_cases = [
        ("IID", False, 1.0),
        ("Non-IID(alpha=0.3)", True, 0.3),
    ]
    attack_cases = ["label_flip", "sign_flip", "gaussian_noise"]
    pressure_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

    all_rows = []
    convergence_plot_data = {}

    for split_name, non_iid, alpha in split_cases:
        cfg.dirichlet_alpha = alpha
        client_loaders = get_loaders(cfg.client_num, cfg.batch_size, non_iid, alpha, cfg.seed)
        test_loader = get_test_loader(batch_size=cfg.batch_size)
        malicious_clients_base = sorted(np.random.RandomState(cfg.seed).choice(range(cfg.client_num), size=max(1, int(cfg.client_num * cfg.malicious_ratio)), replace=False).tolist())

        for attack_name in attack_cases:
            for method_name, rule in METHODS.items():
                key = cache_key({
                    "split": split_name,
                    "attack": attack_name,
                    "method": method_name,
                    "ratio": cfg.malicious_ratio,
                    "epochs": cfg.epochs,
                    "seed": cfg.seed,
                    "alpha": cfg.dirichlet_alpha,
                })
                cache_path = os.path.join(cache_dir, f"{key}.json")
                cached = load_cache(cache_path)
                should_cache_hit = use_cache and (method_name not in rerun_only) and cached is not None
                if should_cache_hit:
                    accs = cached["accs"]
                    details = cached.get("details", {})
                else:
                    accs, details = run_one_setting(
                        client_loaders=client_loaders,
                        test_loader=test_loader,
                        cfg=cfg,
                        attack_name=attack_name,
                        split_name=split_name,
                        method_name=method_name,
                        rule=rule,
                        malicious_clients=malicious_clients_base,
                    )
                    if use_cache:
                        save_cache(cache_path, {"accs": accs, "details": details})
                convergence_plot_data[f"{split_name} | {attack_name} | {method_name}"] = accs
                all_rows.append({
                    "scenario": split_name,
                    "attack": attack_name,
                    "method": method_name,
                    "ratio": cfg.malicious_ratio,
                    "final_acc": float(accs[-1]),
                    "best_acc": float(max(accs)),
                    "epochs_to_90": next((i + 1 for i, v in enumerate(accs) if v >= 90.0), -1),
                    "num_quarantined": int(details.get("num_quarantined", [0])[-1]) if details else 0,
                })

        if non_iid:
            for ratio in pressure_ratios:
                malicious_clients = sorted(np.random.RandomState(cfg.seed).choice(range(cfg.client_num), size=max(1, int(cfg.client_num * ratio)), replace=False).tolist())
                for method_name, rule in METHODS.items():
                    key = cache_key({
                        "split": split_name,
                        "attack": "pressure_label_flip",
                        "method": method_name,
                        "ratio": ratio,
                        "epochs": cfg.epochs,
                        "seed": cfg.seed,
                        "alpha": cfg.dirichlet_alpha,
                    })
                    cache_path = os.path.join(cache_dir, f"pressure_{key}.json")
                    cached = load_cache(cache_path)
                    should_cache_hit = use_cache and (method_name not in rerun_only) and cached is not None
                    if should_cache_hit:
                        accs = cached["accs"]
                        details = cached.get("details", {})
                    else:
                        accs, details = run_one_setting(
                            client_loaders=client_loaders,
                            test_loader=test_loader,
                            cfg=cfg,
                            attack_name="label_flip",
                            split_name=split_name,
                            method_name=method_name,
                            rule=rule,
                            malicious_clients=malicious_clients,
                        )
                        if use_cache:
                            save_cache(cache_path, {"accs": accs, "details": details})
                    all_rows.append({
                        "scenario": split_name,
                        "attack": "label_flip",
                        "method": method_name,
                        "ratio": float(ratio),
                        "final_acc": float(accs[-1]),
                        "best_acc": float(max(accs)),
                        "epochs_to_90": next((i + 1 for i, v in enumerate(accs) if v >= 90.0), -1),
                        "num_quarantined": int(details.get("num_quarantined", [0])[-1]) if details else 0,
                    })

    os.makedirs(accuracy_dir, exist_ok=True)

    # ── 保留旧 CSV 中的 Our TASL 数据行 ────────────────────────────────────
    existing_tasl_rows = []
    if os.path.exists(csv_out):
        with open(csv_out, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("method", "") == "Our TASL":
                    existing_tasl_rows.append(row)

    # 合并：对照组新数据 + 旧 TASL 数据
    merged_rows = all_rows + existing_tasl_rows

    with open(csv_out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "attack", "method", "ratio", "final_acc", "best_acc", "epochs_to_90", "num_quarantined"])
        writer.writeheader()
        writer.writerows(merged_rows)

    # plot only 40% attack convergence for the first split
    plot_convergence(convergence_plot_data, conv_out, "MNIST Convergence Curves (40% Attack Settings)")
    # plot pressure curve: malicious ratio vs final acc using the pressure test rows
    pressure_rows = [r for r in all_rows if r["scenario"].startswith("Non-IID") and r["ratio"] in pressure_ratios]
    if pressure_rows:
        plot_malicious_pressure(pressure_rows, pressure_out, "MNIST Malicious-Ratio Pressure Test")

    print(f"Saved CSV to {csv_out}")
    print(f"Saved convergence plot to {conv_out}")
    print(f"Saved pressure plot to {pressure_out}")


if __name__ == "__main__":
    main()
