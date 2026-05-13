import csv
import os
import random
import sys
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCKCHAIN_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BLOCKCHAIN_DIR)
sys.path.insert(0, PROJECT_ROOT)
CORE_DIR = os.path.join(BLOCKCHAIN_DIR, "core")
sys.path.insert(0, CORE_DIR)

from model import FedAvgCNN
from utils_data import get_data_loaders


@dataclass
class RoundAttack:
    active: bool
    attack_type: Optional[str] = None  # Type-A / Type-B


@dataclass
class GroupRunResult:
    final_accuracy: float
    accuracy_history: List[float]
    round_logs: List[Dict[str, object]]
    metrics: Dict[str, float]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, test_loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    return 100.0 * correct / max(total, 1)


def state_sub(local_state: Dict[str, torch.Tensor], global_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    grad = {}
    for k in global_state:
        if isinstance(global_state[k], torch.Tensor) and global_state[k].dtype in [torch.int64, torch.int32, torch.long]:
            continue
        grad[k] = (local_state[k] - global_state[k]).detach()
    return grad


def state_add(global_state: Dict[str, torch.Tensor], update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state = {}
    for k in global_state:
        if k in update:
            new_state[k] = global_state[k] + update[k]
        else:
            new_state[k] = global_state[k]
    return new_state


def flatten_update(update: Dict[str, torch.Tensor]) -> np.ndarray:
    chunks = []
    for _, tensor in update.items():
        chunks.append(tensor.detach().cpu().reshape(-1).numpy())
    if not chunks:
        return np.zeros((1,), dtype=np.float64)
    return np.concatenate(chunks).astype(np.float64)


def weighted_average_updates(
    updates: Dict[int, Dict[str, torch.Tensor]],
    weights: Dict[int, float],
) -> Dict[str, torch.Tensor]:
    valid = [(cid, float(weights.get(cid, 0.0))) for cid in updates.keys() if float(weights.get(cid, 0.0)) > 0.0]
    if not valid:
        n = len(updates)
        equal_w = 1.0 / max(n, 1)
        valid = [(cid, equal_w) for cid in updates.keys()]

    w_sum = sum(w for _, w in valid)
    normed = [(cid, w / max(w_sum, 1e-12)) for cid, w in valid]

    first_cid = normed[0][0]
    agg = {k: torch.zeros_like(v) for k, v in updates[first_cid].items()}

    for cid, w in normed:
        for k in agg:
            agg[k] += updates[cid][k] * w
    return agg


def apply_sign_flipping(update: Dict[str, torch.Tensor], scale: float = -2.0) -> Dict[str, torch.Tensor]:
    return {k: (v * float(scale)) for k, v in update.items()}


def make_chaos_schedule(rounds: int, seed: int = 42) -> Dict[int, RoundAttack]:
    rng = random.Random(seed)
    schedule = {r: RoundAttack(active=False, attack_type=None) for r in range(1, rounds + 1)}

    block_start = 1
    while block_start <= rounds:
        block_end = min(block_start + 9, rounds)
        block_rounds = list(range(block_start, block_end + 1))
        attack_rounds = rng.sample(block_rounds, k=min(3, len(block_rounds)))
        for r in attack_rounds:
            schedule[r] = RoundAttack(active=True, attack_type=rng.choice(["Type-A", "Type-B"]))
        block_start += 10

    return schedule


def train_one_client(
    global_state: Dict[str, torch.Tensor],
    loader,
    device: torch.device,
    local_batches: int = 1,
    local_epochs: int = 1,
    lr: float = 0.02,
) -> Tuple[Dict[str, torch.Tensor], int]:
    model = FedAvgCNN().to(device)
    model.load_state_dict(global_state)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=float(lr))

    seen = 0
    batches_seen = 0
    for _ in range(int(local_epochs)):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            seen += y.size(0)
            batches_seen += 1
            if (batches_seen) >= int(local_batches):
                break
        if (batches_seen) >= int(local_batches):
            break

    return model.state_dict(), seen


def run_group(
    group_name: str,
    client_loaders,
    test_loader,
    rounds: int,
    chaos_schedule: Dict[int, RoundAttack],
    chaos_mode: bool,
    device: torch.device,
    seed: int,
    cp_threshold: float = 30.0,
    local_epochs: int = 5,
    max_local_batches: int = 0,
    lr: float = 0.02,
) -> GroupRunResult:
    assert group_name in {"single", "tas"}

    num_clients = len(client_loaders)
    global_model = FedAvgCNN().to(device)
    global_state = global_model.state_dict()

    cp_scores = {cid: 100.0 for cid in range(num_clients)}

    total_type_a = 0
    total_type_b = 0
    detected_type_a = 0
    intercepted_type_b = 0
    attack_rounds = 0
    blocked_rounds = 0

    accuracy_history: List[float] = []
    logs: List[Dict[str, object]] = []

    rng = random.Random(seed + (11 if group_name == "single" else 29))

    for r in range(1, rounds + 1):
        local_updates: Dict[int, Dict[str, torch.Tensor]] = {}
        flat_updates: Dict[int, np.ndarray] = {}

        # Match exp_accuracy_vs_epoch-style training strength:
        # local_epochs controls compute; max_local_batches can cap runtime (0 => no cap per epoch loop)
        local_batches = int(max_local_batches) if int(max_local_batches) > 0 else 10**9
        for cid in range(num_clients):
            local_state, _ = train_one_client(
                global_state,
                client_loaders[cid],
                device,
                local_batches=local_batches,
                local_epochs=int(local_epochs),
                lr=float(lr),
            )
            upd = state_sub(local_state, global_state)
            local_updates[cid] = upd
            flat_updates[cid] = flatten_update(upd)

        # Honest similarity score against mean direction
        stacked = np.stack([flat_updates[cid] for cid in range(num_clients)], axis=0)
        mean_vec = stacked.mean(axis=0)
        mean_norm = np.linalg.norm(mean_vec) + 1e-12

        sim_scores: Dict[int, float] = {}
        trust_weights: Dict[int, float] = {}
        for cid in range(num_clients):
            v = flat_updates[cid]
            score = float(np.dot(v, mean_vec) / ((np.linalg.norm(v) + 1e-12) * mean_norm))
            sim_scores[cid] = score
            trust_weights[cid] = max(0.0, score)

        # Quarantine based on CP tier C (< threshold)
        quarantined_clients = [cid for cid in range(num_clients) if cp_scores[cid] < cp_threshold]

        chaos = chaos_schedule[r] if chaos_mode else RoundAttack(active=False, attack_type=None)
        attack_type = chaos.attack_type if chaos.active else None
        attack_client = None

        tampered_weights = dict(trust_weights)
        included_set = {cid for cid, w in tampered_weights.items() if w > 0.0}

        if chaos.active:
            attack_rounds += 1

            if attack_type == "Type-A":
                total_type_a += 1
                sorted_by_sim = sorted(sim_scores.items(), key=lambda x: x[1])
                attack_client = sorted_by_sim[0][0]
                tampered_weights[attack_client] = max(0.85, tampered_weights.get(attack_client, 0.0))
                included_set.add(attack_client)

            elif attack_type == "Type-B":
                total_type_b += 1
                if quarantined_clients:
                    attack_client = rng.choice(quarantined_clients)
                else:
                    # Force one client into quarantine when none exists yet
                    attack_client = rng.randrange(num_clients)
                    cp_scores[attack_client] = cp_threshold - 5.0
                    quarantined_clients = [attack_client]

                tampered_weights[attack_client] = max(0.75, tampered_weights.get(attack_client, 0.0))
                included_set.add(attack_client)

        # V1 / V2 auditing (enabled only in TAS group)
        v1_detected = False
        v2_detected = False
        consensus_ok = True
        healed_applied = False

        if group_name == "tas" and chaos.active:
            if attack_type == "Type-A":
                # V1 recomputes score for all clients, checks tampering
                for cid in range(num_clients):
                    if abs(tampered_weights.get(cid, 0.0) - max(0.0, sim_scores[cid])) > 1e-6:
                        v1_detected = True
                        detected_type_a += 1
                        break

            if attack_type == "Type-B":
                # V2 checks whether included clients violate CP policy
                violating = [cid for cid in included_set if cp_scores.get(cid, 100.0) < cp_threshold]
                if violating:
                    v2_detected = True
                    intercepted_type_b += 1

            if v1_detected or v2_detected:
                consensus_ok = False
                blocked_rounds += 1

        # Apply update depending on governance outcome
        # - single: executor's tampered aggregation is applied; if malicious, inject sign-flipping payload
        # - tas: if audit passes, apply as usual; if intercepted, immediately self-heal by re-aggregating correctly
        if group_name == "single":
            agg_update = weighted_average_updates(local_updates, tampered_weights)
            if chaos.active:
                agg_update = apply_sign_flipping(agg_update, scale=-2.0)
            global_state = state_add(global_state, agg_update)

        else:
            if consensus_ok:
                agg_update = weighted_average_updates(local_updates, tampered_weights)
                global_state = state_add(global_state, agg_update)
            else:
                # Self-healing aggregation: honest auditor re-aggregates with correct weights and CP policy.
                # Enforce: quarantined clients cannot be included this round.
                healed_weights = {cid: trust_weights[cid] for cid in range(num_clients)}
                for qid in quarantined_clients:
                    healed_weights[qid] = 0.0
                agg_update = weighted_average_updates(local_updates, healed_weights)
                global_state = state_add(global_state, agg_update)
                healed_applied = True

        # CP update (honest included nodes up, excluded down)
        included_after_round = {cid for cid, w in tampered_weights.items() if w > 0.0}
        for cid in range(num_clients):
            if cid in included_after_round:
                cp_scores[cid] = min(100.0, cp_scores[cid] + 1.0)
            else:
                cp_scores[cid] = max(0.0, cp_scores[cid] - 6.0)

        global_model.load_state_dict(global_state)
        acc = evaluate(global_model, test_loader, device)
        accuracy_history.append(acc)

        logs.append(
            {
                "round": r,
                "group": group_name,
                "attack_active": int(chaos.active),
                "attack_type": attack_type if attack_type else "None",
                "attack_client": -1 if attack_client is None else int(attack_client),
                "quarantined_count": len(quarantined_clients),
                "v1_detected": int(v1_detected),
                "v2_detected": int(v2_detected),
                "consensus_passed": int(consensus_ok),
                "healed_applied": int(healed_applied),
                "accuracy": float(acc),
            }
        )

        print(
            f"[{group_name.upper()}] Round {r:02d} | attack={attack_type or 'None'} | "
            f"V1={int(v1_detected)} V2={int(v2_detected)} | consensus={int(consensus_ok)} "
            f"heal={int(healed_applied)} | acc={acc:.2f}%"
        )

    metrics = {
        "total_type_a": float(total_type_a),
        "total_type_b": float(total_type_b),
        "detected_type_a": float(detected_type_a),
        "intercepted_type_b": float(intercepted_type_b),
        "attack_rounds": float(attack_rounds),
        "blocked_rounds": float(blocked_rounds),
        "v1_detection_rate": float(detected_type_a / total_type_a) if total_type_a > 0 else 1.0,
        "v2_interception_rate": float(intercepted_type_b / total_type_b) if total_type_b > 0 else 1.0,
        "system_interception_rate": float(blocked_rounds / attack_rounds) if attack_rounds > 0 else 1.0,
    }

    return GroupRunResult(
        final_accuracy=float(accuracy_history[-1]) if accuracy_history else 0.0,
        accuracy_history=accuracy_history,
        round_logs=logs,
        metrics=metrics,
    )


def save_csv(
    csv_path: str,
    logs_clean_single: List[Dict[str, object]],
    logs_clean_tas: List[Dict[str, object]],
    logs_attacked_single: List[Dict[str, object]],
    logs_attacked_tas: List[Dict[str, object]],
    summary: Dict[str, float],
):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = [
        "round",
        "scenario",
        "group",
        "attack_active",
        "attack_type",
        "attack_client",
        "quarantined_count",
        "v1_detected",
        "v2_detected",
        "consensus_passed",
        "healed_applied",
        "accuracy",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + list(summary.keys()))
        writer.writeheader()

        def _write_rows(rows: List[Dict[str, object]], scenario: str):
            for row in rows:
                out = dict(row)
                out["scenario"] = scenario
                for k in summary:
                    out[k] = ""
                writer.writerow(out)

        _write_rows(logs_clean_single, "clean")
        _write_rows(logs_clean_tas, "clean")
        _write_rows(logs_attacked_single, "attacked")
        _write_rows(logs_attacked_tas, "attacked")

        writer.writerow({k: "" for k in fieldnames})
        writer.writerow({**{k: "" for k in fieldnames}, **summary})


def plot_ablation_from_summary(
    out_path: str,
    clean_single: float,
    clean_tas: float,
    attacked_single: float,
    attacked_tas: float,
):
    single_loss = max(0.0, clean_single - attacked_single)
    tas_loss = max(0.0, clean_tas - attacked_tas)

    labels = ["Single Aggregator", "Full TAS"]
    losses = [single_loss, tas_loss]
    colors = ["#C84C4C", "#2E8B57"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, losses, color=colors, width=0.62)
    plt.ylabel("Accuracy Loss under Insider Attack (%)")
    plt.title("Experiment 3: Governance Robustness under Byzantine Executor")
    plt.grid(axis="y", linestyle="--", alpha=0.25)

    for b, v in zip(bars, losses):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.2f}%", ha="center", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180)


def plot_ablation(
    out_path: str,
    clean_acc: float,
    attacked_single: float,
    attacked_tas: float,
):
    single_loss = max(0.0, clean_acc - attacked_single)
    tas_loss = max(0.0, clean_acc - attacked_tas)

    labels = ["Single Aggregator", "Full TAS"]
    losses = [single_loss, tas_loss]
    colors = ["#C84C4C", "#2E8B57"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, losses, color=colors, width=0.62)
    plt.ylabel("Accuracy Loss under Insider Attack (%)")
    plt.title("Experiment 3: Governance Robustness under Byzantine Executor")
    plt.grid(axis="y", linestyle="--", alpha=0.25)

    for b, v in zip(bars, losses):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.2f}%", ha="center", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180)


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Governance chaos test (Byzantine Executor)")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--local-epochs", type=int, default=5, help="每个全局轮次的本地训练轮数")
    parser.add_argument("--max-local-batches", type=int, default=20, help="每轮本地训练最多跑多少个 batch（20 默认更快且更稳定）")
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chaos-seed", type=int, default=2026)
    parser.add_argument("--cp-threshold", type=float, default=30.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rounds = int(args.rounds)
    client_num = int(args.clients)

    print("[Exp3] Loading dataset...")
    client_loaders, test_loader = get_data_loaders("mnist", client_num=client_num, batch_size=int(args.batch_size))

    schedule = make_chaos_schedule(rounds=rounds, seed=int(args.chaos_seed))

    # Clean references (no chaos): used to measure loss
    clean_schedule = {r: RoundAttack(active=False, attack_type=None) for r in range(1, rounds + 1)}
    clean_single = run_group(
        "single",
        client_loaders,
        test_loader,
        rounds,
        clean_schedule,
        chaos_mode=False,
        device=device,
        seed=100,
        cp_threshold=float(args.cp_threshold),
        local_epochs=int(args.local_epochs),
        max_local_batches=int(args.max_local_batches),
        lr=float(args.lr),
    )
    clean_tas = run_group(
        "tas",
        client_loaders,
        test_loader,
        rounds,
        clean_schedule,
        chaos_mode=False,
        device=device,
        seed=100,
        cp_threshold=float(args.cp_threshold),
        local_epochs=int(args.local_epochs),
        max_local_batches=int(args.max_local_batches),
        lr=float(args.lr),
    )

    # Insider attack runs
    attacked_single = run_group(
        "single",
        client_loaders,
        test_loader,
        rounds,
        schedule,
        chaos_mode=True,
        device=device,
        seed=200,
        cp_threshold=float(args.cp_threshold),
        local_epochs=int(args.local_epochs),
        max_local_batches=int(args.max_local_batches),
        lr=float(args.lr),
    )
    attacked_tas = run_group(
        "tas",
        client_loaders,
        test_loader,
        rounds,
        schedule,
        chaos_mode=True,
        device=device,
        seed=200,
        cp_threshold=float(args.cp_threshold),
        local_epochs=int(args.local_epochs),
        max_local_batches=int(args.max_local_batches),
        lr=float(args.lr),
    )

    clean_acc = (clean_single.final_accuracy + clean_tas.final_accuracy) / 2.0

    summary = {
        "clean_reference_accuracy": float(clean_acc),
        "clean_single_final_accuracy": float(clean_single.final_accuracy),
        "clean_tas_final_accuracy": float(clean_tas.final_accuracy),
        "single_attacked_final_accuracy": float(attacked_single.final_accuracy),
        "tas_attacked_final_accuracy": float(attacked_tas.final_accuracy),
        # Loss is computed against each group's clean baseline (more stable + paper-friendly)
        "single_accuracy_loss": float(max(0.0, clean_single.final_accuracy - attacked_single.final_accuracy)),
        "tas_accuracy_loss": float(max(0.0, clean_tas.final_accuracy - attacked_tas.final_accuracy)),
        "v1_detection_rate": attacked_tas.metrics["v1_detection_rate"],
        "v2_interception_rate": attacked_tas.metrics["v2_interception_rate"],
        "system_interception_rate": attacked_tas.metrics["system_interception_rate"],
        "total_type_a": attacked_tas.metrics["total_type_a"],
        "total_type_b": attacked_tas.metrics["total_type_b"],
    }

    results_dir = os.path.join(BLOCKCHAIN_DIR, "results")
    csv_path = os.path.join(results_dir, "exp3_governance_data_v4.csv")
    fig_path = os.path.join(results_dir, "exp3_governance_ablation_v4.png")

    save_csv(
        csv_path,
        clean_single.round_logs,
        clean_tas.round_logs,
        attacked_single.round_logs,
        attacked_tas.round_logs,
        summary,
    )
    plot_ablation_from_summary(
        fig_path,
        clean_single.final_accuracy,
        clean_tas.final_accuracy,
        attacked_single.final_accuracy,
        attacked_tas.final_accuracy,
    )

    print("\n[Exp3] ===== Summary =====")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Figure: {fig_path}")


if __name__ == "__main__":
    main()
