import tempfile
import unittest
from pathlib import Path
from checkpoint_prune import retain_flat_objects

class CheckpointPruneTests(unittest.TestCase):
    def test_verified_old_object_archived_and_compact_records_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'objects').mkdir()
            (root/'archive').mkdir()
            (root/'records').mkdir()
            (root/'objects/old.pt').write_bytes(b'old checkpoint')
            (root/'objects/current.pt').write_bytes(b'current checkpoint')
            (root/'records/old.json').write_bytes(b'permanent audit')
            mapping = retain_flat_objects(root, {'old.pt':False, 'current.pt':True})
            self.assertFalse((root/'objects/old.pt').exists())
            self.assertEqual((root/'archive/old.pt').read_bytes(),b'old checkpoint')
            self.assertEqual((root/'objects/current.pt').read_bytes(),b'current checkpoint')
            self.assertEqual((root/'records/old.json').read_bytes(),b'permanent audit')
            self.assertEqual(mapping['old.pt'],'archive/old.pt')

if __name__ == '__main__':
    unittest.main()
