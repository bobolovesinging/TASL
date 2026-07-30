import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP2_DIR = PROJECT_ROOT / "scripts" / "exp2"
sys.path.insert(0, str(EXP2_DIR))

import exp2_cifar10_v3 as sweep


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


if __name__ == "__main__":
    unittest.main()
