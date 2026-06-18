"""
Governance Ablation: 拆分 TAS 审计组件的贡献

5 组消融:
  1. fedavg:        纯 FedAvg（无防御，Corrupt E 攻击生效）
  2. trust_only:    Cubic Trust + EMA（无 V1/V2 审计，E 攻击无阻碍）
  3. trust+v1:      Cubic Trust + EMA + V1 only（无 V2）
  4. trust+v2:      Cubic Trust + EMA + V2 only（无 V1）
  5. trust+v1+v2:   完整 TAS（V1+V2+healing）

每个实验报告:
  - Best / Avg accuracy
  - V1 detection rate, V2 detections
  - System interception rate
"""
import csv, os, random, sys, argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCKCHAIN_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CORE_DIR = os.path.join(BLOCKCHAIN_DIR, "core")
SCRIPTS_DIR = os.path.join(BLOCKCHAIN_DIR, "scripts")
sys.path.insert(0, BLOCKCHAIN_DIR)
sys.path.insert(0, CORE_DIR)
sys.path.insert(0, SCRIPTS_DIR)

from model import FedAvgCNN
from scripts.exp2.exp2_run import load_mnist_data, load_fashionmnist_data, load_cifar10_data, FederatedDataset

# 复用 v8 的函数
from scripts.exp3.exp_governance_v8 import (
    set_seed, evaluate, state_sub, state_add, flatten_update,
    weighted_average_updates, compute_trust_weights,
    train_one_client, train_one_client_label_flip, train_one_client_sign_flip,
    make_chaos_schedule, apply_attack_a1, apply_attack_a2, apply_attack_a3,
    v1_audit_sampled, v2_membership_audit,
    RoundAttack, GroupRunResult, ATTACK_TYPES, ATTACK_FNS,
)

# ── 新增: scaling attack (高斯噪声缩放) ──
def train_one_client_scaling(global_state, loader, device,
                             local_epochs=1, lr=0.02, scale_factor=5.0):
    """Train normally, then scale the update by scale_factor (amplification attack)."""
    model = FedAvgCNN().to(device)
    model.load_state_dict(global_state)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    local_state = model.state_dict()
    # scale update = scale_factor * (local - global)
    scaled = {}
    for k in global_state:
        if isinstance(global_state[k], torch.Tensor) and global_state[k].is_floating_point():
            diff = local_state[k].to(global_state[k].device) - global_state[k]
            scaled[k] = (global_state[k] + scale_factor * diff).detach().cpu()
        else:
            scaled[k] = local_state[k].detach().cpu() if isinstance(local_state[k], torch.Tensor) else local_state[k]
    return scaled

def get_data_loaders(dataset, client_num=10, batch_size=128, alpha=0.3, seed=42):
    cfg = {
        "data": {"dataset": dataset, "n_clients": client_num,
                   "batch_size": batch_size, "alpha": alpha, "split": "non_iid"},
        "experiment": {"seed": seed},
    }
    if dataset == "mnist":
        return load_mnist_data(cfg)
    elif dataset in ("fashionmnist", "fashion_mnist"):
        return load_fashionmnist_data(cfg)
    elif dataset == "cifar10":
        return load_cifar10_data(cfg)
    raise ValueError(f"Unknown: {dataset}")


@dataclass
class AblationResult:
    name: str
    best_accuracy: float
    avg_accuracy: float
    accuracy_history: List[float]
    v1_detections: int
    v2_detections: int
    attack_rounds: int
    blocked_rounds: int
    round_logs: List[Dict]


def run_ablation_group(
    group_name: str,
    client_loaders,
    test_loader,
    rounds: int,
    chaos_schedule: Dict[int, RoundAttack],
    device: torch.device,
    seed: int,
    local_epochs: int = 2,
    lr: float = 0.02,
    v1_sample_ratio: float = 0.3,
    v1_tolerance: float = 2e-3,
    byzantine_clients: Optional[List[int]] = None,
    ema_alpha: float = 0.6,
    attack: str = "label_flip",
    model_class=None,
) -> AblationResult:
    """
    group_name in {"fedavg", "trust_only", "trust_v1", "trust_v2", "trust_v1v2"}
    """
    assert group_name in {"fedavg", "trust_only", "trust_v1", "trust_v2", "trust_v1v2"}

    if model_class is None:
        model_class = FedAvgCNN

    num_clients = len(client_loaders)
    global_model = model_class().to(device)
    # global_state 全在 CPU 上，避免 device mismatch
    global_state = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
    byzantine_set = set(byzantine_clients or [])

    accuracy_history = []
    logs = []
    ema_weights = {}
    trust_rank_history = defaultdict(list)
    v1_detections = 0
    v2_detections = 0
    attack_rounds = 0
    blocked_rounds = 0

    rng = random.Random(seed + hash(group_name) % 1000)

    for r in range(1, rounds + 1):
        local_updates = {}
        flat_updates = {}

        for cid in range(num_clients):
            if cid in byzantine_set:
                if attack == "sign_flip":
                    raw_state = train_one_client_sign_flip(
                        global_state, client_loaders[cid], device,
                        local_epochs=local_epochs, lr=lr)
                elif attack == "scaling":
                    raw_state = train_one_client_scaling(
                        global_state, client_loaders[cid], device,
                        local_epochs=local_epochs, lr=lr, scale_factor=5.0)
                else:
                    raw_state = train_one_client_label_flip(
                        global_state, client_loaders[cid], device,
                        local_epochs=local_epochs, lr=lr)
                # 确保 local_state 全部在 CPU 上
                local_state = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                               for k, v in raw_state.items()}
            else:
                raw_state = train_one_client(
                    global_state, client_loaders[cid], device,
                    local_epochs=local_epochs, lr=lr)
                local_state = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                               for k, v in raw_state.items()}
            upd = state_sub(local_state, global_state)
            local_updates[cid] = upd
            flat_updates[cid] = flatten_update(upd)

        chaos = chaos_schedule.get(r, RoundAttack(active=False))
        v1_detected = False
        v2_detected = False
        v2_suspected = set()
        consensus_ok = True
        healed_applied = False
        heal_source = ""

        if group_name == "fedavg":
            # 纯 FedAvg + Corrupt E 攻击
            equal_w = {cid: 1.0 / num_clients for cid in range(num_clients)}
            tampered_weights = dict(equal_w)
            if chaos.active:
                attack_rounds += 1
                atk_fn = ATTACK_FNS.get(chaos.attack_type)
                if atk_fn:
                    sim_scores = {cid: 1.0 for cid in range(num_clients)}
                    atk_fn(tampered_weights, sim_scores,
                           byzantine_ids=byzantine_clients, rng=rng)
            agg_update = weighted_average_updates(local_updates, tampered_weights)
            global_state = state_add(global_state, agg_update)

        else:
            # trust_only / trust_v1 / trust_v2 / trust_v1v2
            sim_scores, trust_weights = compute_trust_weights(
                flat_updates, trust_power=5.0, max_weight_ratio=2.0)

            # EMA
            if ema_weights:
                smoothed = {}
                for cid in range(num_clients):
                    current = trust_weights.get(cid, 0.0)
                    history = ema_weights.get(cid, 1.0 / num_clients)
                    smoothed[cid] = ema_alpha * current + (1 - ema_alpha) * history
                s_sum = sum(smoothed.values())
                if s_sum > 1e-12:
                    trust_weights = {cid: w / s_sum for cid, w in smoothed.items()}
            ema_weights = dict(trust_weights)

            # V2 排名
            sorted_cids = sorted(trust_weights.keys(), key=lambda c: trust_weights[c])
            rank_map = {cid: rank for rank, cid in enumerate(sorted_cids)}
            for cid in range(num_clients):
                trust_rank_history[cid].append(rank_map[cid])

            # Corrupt E 攻击
            tampered_weights = dict(trust_weights)
            if chaos.active:
                attack_rounds += 1
                atk_fn = ATTACK_FNS.get(chaos.attack_type)
                if atk_fn:
                    atk_fn(tampered_weights, sim_scores,
                           byzantine_ids=byzantine_clients, rng=rng)

            # V1/V2 审计
            use_v1 = group_name in ("trust_v1", "trust_v1v2")
            use_v2 = group_name in ("trust_v2", "trust_v1v2")

            if chaos.active:
                if use_v1:
                    v1_detected, _ = v1_audit_sampled(
                        tampered_weights, trust_weights, num_clients,
                        sample_ratio=v1_sample_ratio, tolerance=v1_tolerance,
                        round_number=r)
                    if v1_detected:
                        v1_detections += 1

                if not v1_detected and use_v2 and r >= 3:
                    v2_detected, v2_suspected = v2_membership_audit(
                        trust_rank_history, num_clients, byzantine_set,
                        low_rank_threshold=0.3, consecutive_rounds=3)
                    if v2_detected:
                        v2_detections += 1

                if v1_detected or v2_detected:
                    consensus_ok = False
                    blocked_rounds += 1
                    heal_source = "V1" if v1_detected else "V2"

            if consensus_ok:
                agg_update = weighted_average_updates(local_updates, tampered_weights)
                global_state = state_add(global_state, agg_update)
            else:
                # Healing
                healed_weights = dict(trust_weights)
                for cid in byzantine_set:
                    healed_weights[cid] = 0.0
                for cid in v2_suspected:
                    healed_weights[cid] = 0.0
                h_sum = sum(healed_weights.values())
                if h_sum > 1e-12:
                    healed_weights = {cid: w / h_sum for cid, w in healed_weights.items()}
                agg_update = weighted_average_updates(local_updates, healed_weights)
                global_state = state_add(global_state, agg_update)
                healed_applied = True

        global_model.load_state_dict(global_state)
        acc = evaluate(global_model, test_loader, device)
        accuracy_history.append(acc)

        logs.append({
            "round": r, "group": group_name,
            "attack_active": int(chaos.active),
            "v1_detected": int(v1_detected), "v2_detected": int(v2_detected),
            "consensus_passed": int(consensus_ok),
            "healed_applied": int(healed_applied),
            "accuracy": float(acc),
        })

        if r % 10 == 0 or r == 1:
            print(f"  [{group_name}] r{r:02d} | atk={int(chaos.active)} | "
                  f"V1={int(v1_detected)} V2={int(v2_detected)} | "
                  f"heal={int(healed_applied)}{heal_source} | acc={acc:.2f}%")

    return AblationResult(
        name=group_name,
        best_accuracy=float(max(accuracy_history)) if accuracy_history else 0.0,
        avg_accuracy=float(np.mean(accuracy_history)) if accuracy_history else 0.0,
        accuracy_history=accuracy_history,
        v1_detections=v1_detections,
        v2_detections=v2_detections,
        attack_rounds=attack_rounds,
        blocked_rounds=blocked_rounds,
        round_logs=logs,
    )


def main():
    parser = argparse.ArgumentParser(description="Governance Ablation: V1/V2 component analysis")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashionmnist", "cifar10"])
    parser.add_argument("--attack", type=str, default="label_flip",
                        choices=["label_flip", "sign_flip", "scaling"])
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--byzantine-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.02)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_byz = max(1, int(args.clients * args.byzantine_ratio))
    byzantine_ids = list(range(n_byz))

    print(f"\n{'='*70}")
    print(f"Governance Ablation: {args.dataset} / {args.attack} / {args.rounds}r / {n_byz}B")
    print(f"{'='*70}")

    client_loaders, test_loader = get_data_loaders(
        args.dataset, client_num=args.clients, batch_size=128,
        alpha=0.3, seed=args.seed)

    schedule = make_chaos_schedule(rounds=args.rounds, seed=2026, attacks_per_block=6)

    common = dict(
        client_loaders=client_loaders, test_loader=test_loader,
        rounds=args.rounds, chaos_schedule=schedule,
        device=device, local_epochs=args.local_epochs, lr=args.lr,
        byzantine_clients=byzantine_ids, ema_alpha=0.6,
        attack=args.attack,
    )

    groups = ["fedavg", "trust_only", "trust_v1", "trust_v2", "trust_v1v2"]
    results = {}

    for i, g in enumerate(groups):
        print(f"\n--- {i+1}/5: {g} ---")
        results[g] = run_ablation_group(g, seed=args.seed + i * 100, **common)

    # Summary
    print(f"\n{'='*70}")
    print(f"Governance Ablation Summary: {args.dataset} / {args.attack}")
    print(f"{'='*70}")
    print(f"{'Group':<18} {'Best':>8} {'Avg':>8} {'V1_det':>7} {'V2_det':>7} {'Blocked':>8} {'Interc%':>8}")
    print("-" * 70)
    for g in groups:
        r = results[g]
        interc = 100.0 * r.blocked_rounds / r.attack_rounds if r.attack_rounds > 0 else 0.0
        print(f"{g:<18} {r.best_accuracy:>8.2f} {r.avg_accuracy:>8.2f} "
              f"{r.v1_detections:>7d} {r.v2_detections:>7d} {r.blocked_rounds:>8d} {interc:>7.1f}%")

    # Save CSV
    results_dir = os.path.join(BLOCKCHAIN_DIR, "results", "governance_ablation")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir,
        f"ablation_{args.dataset}_{args.attack}_r{args.rounds}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "round", "group", "attack_active", "v1_detected", "v2_detected",
            "consensus_passed", "healed_applied", "accuracy"])
        writer.writeheader()
        for g in groups:
            for row in results[g].round_logs:
                writer.writerow(row)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
