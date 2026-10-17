"""Negative controls for the new local disk bridge, not TAS algorithm tests."""
import tempfile
import unittest
from pathlib import Path
from storage_bridge import validate_checkpoint, commit_gate

class GuardTests(unittest.TestCase):
    def test_partial_trust_checkpoint_is_not_recoverable(self):
        checkpoint = {'schema': 'tasl-full-state-v1', 'round': 3,
                      'model': {'parameter': 1}, 'trust': ({}, [], None, None, set())}
        with self.assertRaisesRegex(ValueError, 'six'):
            validate_checkpoint(checkpoint, 3)

    def test_wrong_round_checkpoint_is_not_recoverable(self):
        checkpoint = {'schema': 'tasl-full-state-v1', 'round': 2,
                      'model': {'parameter': 1},
                      'trust': ({}, [], None, None, set(), {})}
        with self.assertRaisesRegex(ValueError, 'round'):
            validate_checkpoint(checkpoint, 3)

    def test_both_votes_and_valid_binding_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'commit.json'
            for vs, vp, bound in ((True,False,True), (False,True,True), (True,True,False)):
                self.assertFalse(commit_gate(path, round_number=3, parent='p',
                                            proposal='q', vs=vs, vp=vp, bound=bound))
                self.assertFalse(path.exists())

    def test_same_proposal_cannot_be_committed_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'commit.json'
            fields = dict(round_number=3, parent='p', proposal='q', vs=True, vp=True, bound=True)
            self.assertTrue(commit_gate(path, **fields))
            self.assertFalse(commit_gate(path, **fields))

if __name__ == '__main__':
    unittest.main()
