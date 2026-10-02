"""Call unchanged LASA author aggregation; isolate its in-place CUDA operations."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parent
UPSTREAM_COMMIT = '8477367a4e8708cde264f7572805040c650af59f'

def load_upstream():
    root = ROOT / 'LASA_upstream'
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('lasa_author', root/'algorithms/defense/lasa.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def upstream_args(n):
    # Author CLI defaults; fixed before inspecting comparison outcomes.
    return SimpleNamespace(num_selected_users=n, sparsity=0.3, lambda_n=1.0, lambda_s=1.0)

def lasa_update(updates, weights=None):
    if weights is not None:
        raise ValueError('LASA component does not take externally assigned trust weights')
    if not updates or not torch.cuda.is_available():
        raise ValueError('Nonempty CUDA LASA run required')
    if not all(v.dtype.is_floating_point and torch.isfinite(v).all()
               for d in updates.values() for v in d.values()):
        raise ValueError('This adapter is restricted to finite floating-point CNN updates')
    local = [{k: v.detach().clone().cuda() for k, v in d.items()} for d in updates.values()]
    global_zero = {k: torch.zeros_like(v) for k, v in local[0].items()}
    result = load_upstream().lasa(local, global_zero, upstream_args(len(local)))
    if not all(torch.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('Author LASA returned nonfinite update; no fallback applied')
    return {k: v.detach().clone().cpu() for k, v in result.items()}
