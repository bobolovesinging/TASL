import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP2_DIR = PROJECT_ROOT / "scripts" / "exp2"
sys.path.insert(0, str(EXP2_DIR))

import exp2_cifar10_v3 as sweep


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


if __name__ == "__main__":
    unittest.main()
