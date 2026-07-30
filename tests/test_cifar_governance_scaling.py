import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP2_DIR = PROJECT_ROOT / "scripts" / "exp2"
EXP3_DIR = PROJECT_ROOT / "scripts" / "exp3"
sys.path.insert(0, str(EXP2_DIR))
sys.path.insert(0, str(EXP3_DIR))

import exp2_cifar10_v3 as sweep
import exp3_governance_cifar10_scaling as governance


def make_context(baseline_aggregate=None, round_number=1):
    local_updates = {
        0: {"weight": torch.tensor([1.0, 0.0])},
        1: {"weight": torch.tensor([0.0, 2.0])},
    }
    if baseline_aggregate is None:
        baseline_aggregate = {"weight": torch.tensor([0.6, 0.8])}
    return sweep.ExecutorRoundContext(
        round_number=round_number,
        seed=42,
        local_updates=local_updates,
        trust_weights={0: 0.6, 1: 0.4},
        similarity_scores={0: 0.9, 1: 0.2},
        baseline_aggregate=baseline_aggregate,
        byzantine_ids=(1,),
    )


class ScalingTests(unittest.TestCase):
    def test_scale_model_update_multiplies_only_floating_delta(self):
        """Catches missing x5 amplification or accidental integer scaling."""
        global_state = {
            "weight": torch.tensor([1.0, -2.0]),
            "counter": torch.tensor(3, dtype=torch.int64),
        }
        trained_state = {
            "weight": torch.tensor([2.0, 0.0]),
            "counter": torch.tensor(4, dtype=torch.int64),
        }

        result = sweep.scale_model_update(
            global_state,
            trained_state,
            scale_factor=5.0,
        )

        torch.testing.assert_close(
            result["weight"],
            torch.tensor([6.0, 8.0]),
            rtol=0,
            atol=0,
        )
        self.assertEqual(result["counter"].item(), 4)

    def test_scaling_attack_is_registered(self):
        """Catches the CLI accepting Scaling while silently training honestly."""
        self.assertIs(
            sweep.ATTACK_TRAIN_FNS["scaling"],
            sweep.train_scaling,
        )

    def test_explicit_run_seed_replaces_process_hash_seed(self):
        """Catches reintroduction of process-randomized experiment seeds."""
        self.assertEqual(sweep.resolve_run_seed(base_seed=42, run_seed=None), 42)
        self.assertEqual(sweep.resolve_run_seed(base_seed=42, run_seed=314), 314)


class ExecutorHookTests(unittest.TestCase):
    def test_select_executor_aggregate_uses_baseline_without_hook(self):
        """Catches a no-Executor run entering a separate aggregation path."""
        baseline = {"weight": torch.tensor([1.0])}
        context = make_context(baseline)

        decision = sweep.select_executor_aggregate(context, executor_hook=None)

        self.assertIs(decision.aggregate, baseline)
        self.assertEqual(decision.metadata, {})

    def test_select_executor_aggregate_uses_hook_decision(self):
        """Catches the shared runner ignoring an Executor proposal."""
        baseline = {"weight": torch.tensor([1.0])}
        attacked = {"weight": torch.tensor([9.0])}
        context = make_context(baseline)

        decision = sweep.select_executor_aggregate(
            context,
            executor_hook=lambda _: sweep.ExecutorRoundDecision(
                aggregate=attacked,
                metadata={"attack": "A1"},
            ),
        )

        self.assertIs(decision.aggregate, attacked)
        self.assertEqual(decision.metadata["attack"], "A1")


class ExecutorAttackTests(unittest.TestCase):
    def test_a1_is_detected_only_by_v1(self):
        """Catches V1 missing forged trust or V2 claiming A1 coverage."""
        context = make_context()

        proposal = governance.apply_executor_attack(context, "A1")

        self.assertTrue(governance.verify_v1(proposal, context))
        self.assertFalse(
            governance.verify_v2(
                proposal,
                context,
                projection_seed=7,
            )
        )

    def test_a2_is_detected_only_by_v2(self):
        """Catches V2 missing aggregate tampering or V1 claiming A2 coverage."""
        context = make_context()

        proposal = governance.apply_executor_attack(context, "A2")

        self.assertFalse(governance.verify_v1(proposal, context))
        self.assertTrue(
            governance.verify_v2(
                proposal,
                context,
                projection_seed=7,
            )
        )


def policy(group, attack_type):
    return governance.GovernanceExecutor(
        group=group,
        schedule={1: attack_type},
        seed=42,
    )


class GovernancePolicyTests(unittest.TestCase):
    def test_trust_only_accepts_a1(self):
        """Catches Trust Only accidentally applying an Executor defense."""
        context = make_context()

        decision = policy("trust_only", "A1")(context)

        self.assertFalse(decision.metadata["blocked"])
        self.assertFalse(decision.metadata["healed"])
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(
                decision.aggregate["weight"],
                context.baseline_aggregate["weight"],
                rtol=0,
                atol=0,
            )

    def test_v1_heals_a1_but_not_a2(self):
        """Catches V1 defending the wrong Executor attack."""
        a1 = policy("v1", "A1")(make_context())
        a2 = policy("v1", "A2")(make_context())

        self.assertTrue(a1.metadata["healed"])
        self.assertFalse(a2.metadata["healed"])

    def test_v2_heals_a2_but_not_a1(self):
        """Catches V2 defending the wrong Executor attack."""
        a1 = policy("v2", "A1")(make_context())
        a2 = policy("v2", "A2")(make_context())

        self.assertFalse(a1.metadata["healed"])
        self.assertTrue(a2.metadata["healed"])

    def test_tas_heals_both_attacks_to_exact_baseline(self):
        """Catches TAS recovery using rollback or recomputed client weights."""
        for attack_type in ("A1", "A2"):
            with self.subTest(attack_type=attack_type):
                context = make_context()

                decision = policy("tas", attack_type)(context)

                self.assertTrue(decision.metadata["healed"])
                self.assertIs(
                    decision.aggregate,
                    context.baseline_aggregate,
                )
                torch.testing.assert_close(
                    decision.aggregate["weight"],
                    context.baseline_aggregate["weight"],
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
