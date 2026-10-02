import importlib.util
from pathlib import Path
import numpy as np
import torch

module_path = Path(__file__).resolve().parents[1] / 'scripts' / 'exp2' / 'exp2_run.py'
spec = importlib.util.spec_from_file_location('exp2_run', module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
root = np.array([1.0, 0.0], dtype=np.float64)
updates_a = {0: {'w': torch.tensor([100.0, 0.0])}, 1: {'w': torch.tensor([0.0, 1.0])}}
updates_b = {0: {'w': torch.tensor([2.0, 0.0])}, 1: {'w': torch.tensor([0.0, 1.0])}}
flat_a = {cid: mod.flatten_update(upd) for cid, upd in updates_a.items()}
flat_b = {cid: mod.flatten_update(upd) for cid, upd in updates_b.items()}
out_a = mod.fltrust_aggregate(flat_a, updates_a, torch.device('cpu'), root=root)['w']
out_b = mod.fltrust_aggregate(flat_b, updates_b, torch.device('cpu'), root=root)['w']
assert torch.allclose(out_a, torch.tensor([1.0, 0.0]), atol=1e-6), out_a
assert torch.allclose(out_a, out_b, atol=1e-6), (out_a, out_b)
print('FLTRUST_NORM_TEST_PASS', out_a.tolist())
