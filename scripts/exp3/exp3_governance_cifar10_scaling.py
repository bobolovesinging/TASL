"""Unified CIFAR-10 Scaling governance experiment.

The CIFAR model, data pipeline, client training, Scaling attack, TASL trust
scoring, scheduler, and evaluation all remain in exp2_cifar10_v3.  This
module owns only Executor-side proposal attacks and governance checks.
"""

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch
import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP2_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "exp2")
sys.path.insert(0, EXP2_DIR)

import exp2_cifar10_v3 as sweep


CANONICAL_GROUPS = ("no_executor", "trust_only", "v1", "v2", "tas")
DEFAULT_CONFIG = {
    "clients": 10,
    "byzantine_ratio": 0.4,
    "alpha": 0.3,
    "batch_size": 128,
    "rounds": 200,
    "local_epochs": 5,
    "lr": 0.1,
    "seed": 42,
    "block_size": 10,
    "attacks_per_block": 6,
    "executor_attack": "mixed12",
    "groups": list(CANONICAL_GROUPS),
    "device": "auto",
    "output": os.path.join(
        os.path.dirname(os.path.dirname(SCRIPT_DIR)),
        "results",
        "exp3_cifar10_scaling_governance",
    ),
}


@dataclass(frozen=True)
class ExecutorProposal:
    attack_type: str
    published_weights: Dict[int, float]
    aggregation_weights: Dict[int, float]
    aggregate: Dict[str, torch.Tensor]


def normalize_weights(weights: Dict[int, float]) -> Dict[int, float]:
    total = float(sum(weights.values()))
    if total <= 1e-12:
        raise ValueError("Executor weights must have a positive sum")
    return {cid: float(weight) / total for cid, weight in weights.items()}


def apply_executor_attack(
    context: sweep.ExecutorRoundContext,
    attack_type: str,
) -> ExecutorProposal:
    """Build an A1 forged-trust or A2 tampered-aggregate proposal."""
    published = dict(context.trust_weights)
    aggregation = dict(context.trust_weights)

    if attack_type == "A1":
        for cid in context.byzantine_ids:
            published[cid] = 0.75
            aggregation[cid] = 0.75
    elif attack_type == "A2":
        for cid in context.byzantine_ids:
            aggregation[cid] = min(
                0.50,
                aggregation.get(cid, 0.0) + 0.25,
            )
    else:
        raise ValueError(f"unsupported Executor attack: {attack_type}")

    normalized_aggregation = normalize_weights(aggregation)
    aggregate = sweep.fedavg_aggregate(
        context.local_updates,
        normalized_aggregation,
    )
    return ExecutorProposal(
        attack_type=attack_type,
        published_weights=published,
        aggregation_weights=normalized_aggregation,
        aggregate=aggregate,
    )


def verify_v1_trust(
    proposal: ExecutorProposal,
    context: sweep.ExecutorRoundContext,
    tolerance: float = 1e-4,
) -> bool:
    """Detect forged published trust weights."""
    return any(
        abs(
            proposal.published_weights.get(cid, 0.0)
            - context.trust_weights.get(cid, 0.0)
        )
        > tolerance
        for cid in context.trust_weights
    )


def verify_v1_aggregate(
    proposal: ExecutorProposal,
    context: sweep.ExecutorRoundContext,
    cosine_threshold: float = 0.97,
) -> bool:
    """Weakly detect aggregate direction drift against published weights."""
    published = normalize_weights(proposal.published_weights)
    client_count = len(context.local_updates)
    minimum_weight = 1.0 / client_count
    soft_published = {
        cid: max(published.get(cid, 0.0), minimum_weight)
        for cid in context.local_updates
    }
    soft_published = normalize_weights(soft_published)
    expected = sweep.fedavg_aggregate(
        context.local_updates,
        soft_published,
    )
    actual_flat = sweep.flatten_update(proposal.aggregate)
    expected_flat = sweep.flatten_update(expected)
    actual_norm = np.linalg.norm(actual_flat) + 1e-12
    expected_norm = np.linalg.norm(expected_flat) + 1e-12
    cosine_similarity = float(
        np.dot(actual_flat, expected_flat)
        / (actual_norm * expected_norm)
    )
    return cosine_similarity < cosine_threshold


def verify_v2_trust(
    proposal: ExecutorProposal,
    context: sweep.ExecutorRoundContext,
    sample_ratio: float = 0.3,
    sample_seed: int = 0,
    tolerance: float = 1e-4,
) -> bool:
    """Weakly detect forged trust by checking a deterministic client sample."""
    client_ids = sorted(context.local_updates)
    sample_count = max(1, int(len(client_ids) * sample_ratio))
    rng = np.random.RandomState(sample_seed)
    sampled_indices = rng.choice(
        len(client_ids),
        size=sample_count,
        replace=False,
    )
    sampled_clients = [client_ids[index] for index in sampled_indices]
    return any(
        abs(
            proposal.published_weights.get(cid, 0.0)
            - context.trust_weights.get(cid, 0.0)
        )
        > tolerance
        for cid in sampled_clients
    )


def verify_v2_aggregate(
    proposal: ExecutorProposal,
    context: sweep.ExecutorRoundContext,
    projection_seed: int,
    rtol: float = 0.03,
    atol: float = 1e-8,
) -> bool:
    """Detect an aggregate inconsistent with the published weights."""
    published = normalize_weights(proposal.published_weights)
    expected = sweep.fedavg_aggregate(context.local_updates, published)
    actual_flat = sweep.flatten_update(proposal.aggregate)
    expected_flat = sweep.flatten_update(expected)

    rng = np.random.RandomState(projection_seed)
    projection = rng.randn(len(actual_flat)).astype(np.float64)
    projection_norm = np.linalg.norm(projection)
    if projection_norm > 1e-12:
        projection /= projection_norm

    actual_value = float(np.dot(actual_flat, projection))
    expected_value = float(np.dot(expected_flat, projection))
    difference = abs(actual_value - expected_value)
    scale = max(abs(actual_value), abs(expected_value), 1e-12)
    return difference > atol + rtol * scale


# Compatibility aliases for callers of the first unified-runner revision.
verify_v1 = verify_v1_trust
verify_v2 = verify_v2_aggregate


class GovernanceExecutor:
    """Apply the Executor attack and the defenses enabled for one group."""

    SUPPORTED_GROUPS = {"trust_only", "v1", "v2", "tas"}

    def __init__(
        self,
        group: str,
        schedule: Mapping[int, str],
        seed: int,
    ):
        if group not in self.SUPPORTED_GROUPS:
            raise ValueError(f"unsupported governance group: {group}")
        self.group = group
        self.schedule = dict(schedule)
        self.seed = int(seed)

    def __call__(
        self,
        context: sweep.ExecutorRoundContext,
    ) -> sweep.ExecutorRoundDecision:
        attack_type = self.schedule.get(context.round_number)
        if attack_type is None:
            return sweep.ExecutorRoundDecision(
                aggregate=context.baseline_aggregate,
            )

        proposal = apply_executor_attack(context, attack_type)
        v1_trust_detected = (
            verify_v1_trust(proposal, context)
            if self.group in {"v1", "tas"}
            else False
        )
        v1_aggregate_detected = (
            verify_v1_aggregate(proposal, context)
            if self.group in {"v1", "tas"}
            else False
        )
        v2_trust_detected = (
            verify_v2_trust(
                proposal,
                context,
                sample_ratio=0.3,
                sample_seed=self.seed + context.round_number * 137,
            )
            if self.group in {"v2", "tas"}
            else False
        )
        v2_aggregate_detected = (
            verify_v2_aggregate(
                proposal,
                context,
                projection_seed=self.seed + context.round_number * 971,
            )
            if self.group in {"v2", "tas"}
            else False
        )
        v1_detected = v1_trust_detected or v1_aggregate_detected
        v2_detected = v2_trust_detected or v2_aggregate_detected
        blocked = v1_detected or v2_detected
        aggregate = (
            context.baseline_aggregate
            if blocked
            else proposal.aggregate
        )
        return sweep.ExecutorRoundDecision(
            aggregate=aggregate,
            metadata={
                "round": context.round_number,
                "attack": attack_type,
                "v1_trust_detected": v1_trust_detected,
                "v1_aggregate_detected": v1_aggregate_detected,
                "v2_trust_detected": v2_trust_detected,
                "v2_aggregate_detected": v2_aggregate_detected,
                "v1_detected": v1_detected,
                "v2_detected": v2_detected,
                "blocked": blocked,
                "healed": blocked,
            },
        )


def make_attack_schedule(
    rounds: int,
    seed: int,
    attack_type: str = "mixed12",
    block_size: int = 10,
    attacks_per_block: int = 6,
) -> Dict[int, str]:
    """Create one deterministic schedule shared by every governance group."""
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if not 0 <= attacks_per_block <= block_size:
        raise ValueError("attacks_per_block must be between 0 and block_size")
    if attack_type not in {"A1", "A2", "mixed12"}:
        raise ValueError(f"unsupported Executor attack: {attack_type}")

    rng = random.Random(seed)
    schedule = {}
    for block_start in range(1, rounds + 1, block_size):
        block_rounds = list(
            range(block_start, min(block_start + block_size, rounds + 1))
        )
        attack_count = min(attacks_per_block, len(block_rounds))
        for round_number in sorted(rng.sample(block_rounds, attack_count)):
            schedule[round_number] = (
                rng.choice(["A1", "A2"])
                if attack_type == "mixed12"
                else attack_type
            )
    return schedule


def build_executor_hook(
    group: str,
    schedule: Mapping[int, str],
    seed: int,
) -> Optional[GovernanceExecutor]:
    """Map a canonical governance group to its shared-runner hook."""
    if group == "no_executor":
        return None
    if group not in GovernanceExecutor.SUPPORTED_GROUPS:
        raise ValueError(f"unsupported governance group: {group}")
    return GovernanceExecutor(group=group, schedule=schedule, seed=seed)


def parse_args(argv=None):
    """Parse CLI options, optionally taking defaults from a flat YAML file."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config")
    pre_args, _ = pre_parser.parse_known_args(argv)

    defaults = dict(DEFAULT_CONFIG)
    if pre_args.config:
        with open(pre_args.config, encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("governance config must contain a YAML mapping")
        defaults.update(loaded)

    parser = argparse.ArgumentParser(
        description="Unified CIFAR-10 Scaling Executor-governance experiment"
    )
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--clients", type=int, default=defaults["clients"])
    parser.add_argument(
        "--byzantine-ratio",
        type=float,
        default=defaults["byzantine_ratio"],
    )
    parser.add_argument("--alpha", type=float, default=defaults["alpha"])
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults["batch_size"],
    )
    parser.add_argument("--rounds", type=int, default=defaults["rounds"])
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=defaults["local_epochs"],
    )
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument(
        "--block-size",
        type=int,
        default=defaults["block_size"],
    )
    parser.add_argument(
        "--attacks-per-block",
        type=int,
        default=defaults["attacks_per_block"],
    )
    parser.add_argument(
        "--executor-attack",
        choices=("A1", "A2", "mixed12"),
        default=defaults["executor_attack"],
    )
    parser.add_argument(
        "--groups",
        default=defaults["groups"],
        help="Comma-separated: no_executor,trust_only,v1,v2,tas",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=defaults["device"],
    )
    parser.add_argument("--output", default=defaults["output"])
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 5 clients, 2 rounds, and 1 local epoch",
    )
    args = parser.parse_args(argv)

    if isinstance(args.groups, str):
        args.groups = [
            group.strip()
            for group in args.groups.split(",")
            if group.strip()
        ]
    else:
        args.groups = list(args.groups)
    unknown_groups = sorted(set(args.groups) - set(CANONICAL_GROUPS))
    if unknown_groups:
        parser.error(
            "unsupported governance group(s): "
            + ", ".join(unknown_groups)
        )
    if args.quick:
        args.clients = 5
        args.rounds = 2
        args.local_epochs = 1
    return args


def save_results(output_dir, config, results):
    """Write detailed JSON plus one compact CSV row per group."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    json_path = output_path / f"cifar_scaling_governance_seed{seed}.json"
    csv_path = output_path / f"cifar_scaling_governance_seed{seed}.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"config": config, "results": results},
            handle,
            ensure_ascii=False,
            indent=2,
        )

    fieldnames = [
        "group",
        "best_acc",
        "avg_acc",
        "last_acc",
        "attack_rounds",
        "blocked_rounds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group, result in results.items():
            history = result.get("executor_history", [])
            writer.writerow(
                {
                    "group": group,
                    "best_acc": result["best_acc"],
                    "avg_acc": result["avg_acc"],
                    "last_acc": result["last_acc"],
                    "attack_rounds": len(history),
                    "blocked_rounds": sum(
                        bool(event.get("blocked"))
                        for event in history
                    ),
                }
            )
    return str(json_path), str(csv_path)


def run_governance_groups(args):
    """Run every group through the shared CIFAR Sweep implementation."""
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    n_byzantine = max(1, int(args.clients * args.byzantine_ratio))
    schedule = make_attack_schedule(
        rounds=args.rounds,
        seed=args.seed,
        attack_type=args.executor_attack,
        block_size=args.block_size,
        attacks_per_block=args.attacks_per_block,
    )

    print("[Data] Loading the Sweep CIFAR-10 partition...")
    client_loaders, test_loader = sweep.prepare_cifar10_loaders(
        n_clients=args.clients,
        batch_size=args.batch_size,
        alpha=args.alpha,
        seed=args.seed,
    )

    results = {}
    for group in args.groups:
        print(f"\n>>> Governance group: {group} <<<")
        executor_hook = build_executor_hook(
            group=group,
            schedule=schedule,
            seed=args.seed,
        )
        result = sweep.run_single(
            algo="tasl",
            attack="scaling",
            client_loaders=client_loaders,
            test_loader=test_loader,
            n_clients=args.clients,
            n_byz=n_byzantine,
            rounds=args.rounds,
            device=device,
            local_epochs=args.local_epochs,
            lr=args.lr,
            seed=args.seed,
            noise_scale=5.0,
            executor_hook=executor_hook,
        )
        results[group] = result
        blocked = sum(
            bool(event.get("blocked"))
            for event in result["executor_history"]
        )
        print(
            f"  => best={result['best_acc']:.2f}% "
            f"avg={result['avg_acc']:.2f}% "
            f"last={result['last_acc']:.2f}% "
            f"blocked={blocked}/{len(result['executor_history'])}"
        )
    return results


def main(argv=None):
    args = parse_args(argv)
    config = vars(args).copy()

    print("=" * 72)
    print("[CIFAR-10 Scaling Governance] Unified Sweep TASL Runner")
    print(f"  groups={args.groups}")
    print(
        f"  clients={args.clients}, Byzantine={args.byzantine_ratio:.0%}, "
        f"alpha={args.alpha}"
    )
    print(
        f"  rounds={args.rounds}, local_epochs={args.local_epochs}, "
        f"lr={args.lr}"
    )
    print(
        f"  seed={args.seed}, Executor attack={args.executor_attack}, "
        f"schedule={args.attacks_per_block}/{args.block_size}"
    )
    print("  Scaling rule: global + 5 * (trained_local - global)")
    print("=" * 72)

    results = run_governance_groups(args)
    json_path, csv_path = save_results(args.output, config, results)
    print(f"\n[Saved] JSON: {json_path}")
    print(f"[Saved] CSV:  {csv_path}")
    return results


if __name__ == "__main__":
    main()
