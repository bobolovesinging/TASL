"""Unified CIFAR-10 Scaling governance experiment.

The CIFAR model, data pipeline, client training, Scaling attack, TASL trust
scoring, scheduler, and evaluation all remain in exp2_cifar10_v3.  This
module owns only Executor-side proposal attacks and governance checks.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP2_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "exp2")
sys.path.insert(0, EXP2_DIR)

import exp2_cifar10_v3 as sweep


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


def verify_v1(
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


def verify_v2(
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
        v1_detected = (
            verify_v1(proposal, context)
            if self.group in {"v1", "tas"}
            else False
        )
        v2_detected = (
            verify_v2(
                proposal,
                context,
                projection_seed=self.seed + context.round_number * 971,
            )
            if self.group in {"v2", "tas"}
            else False
        )
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
                "v1_detected": v1_detected,
                "v2_detected": v2_detected,
                "blocked": blocked,
                "healed": blocked,
            },
        )
