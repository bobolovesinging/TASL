"""GPU reference equivalence; the author's implementation is the oracle."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from lasa_adapter import lasa_update, load_upstream, upstream_args

def digest(updates):
    h = hashlib.sha256()
    for cid, state in updates.items():
        h.update(str(cid).encode())
        for key, val in state.items():
            h.update(key.encode()); h.update(val.cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    assert torch.cuda.is_available(), 'GPU required by unchanged author CUDA mask'
    torch.manual_seed(917)
    # Non-contiguous IDs, heterogeneous tensor shapes and a large-norm outlier.
    updates = {cid: {'conv.weight': torch.randn(3, 2, 3, 3),
                     'fc.weight': torch.randn(6, 8),
                     'fc.bias': torch.randn(6)} for cid in [2, 5, 7, 11, 13]}
    for value in updates[13].values(): value.mul_(15)
    before = digest(updates)
    states = [{k: v.clone().cuda() for k, v in d.items()} for d in updates.values()]
    zeros = {k: torch.zeros_like(v).cuda() for k, v in states[0].items()}
    oracle = load_upstream().lasa(states, zeros, upstream_args(len(states)))
    actual = lasa_update(updates)
    assert list(actual) == list(oracle), 'tensor ordering changed'
    for k in oracle:
        torch.testing.assert_close(actual[k], oracle[k].cpu(), rtol=0, atol=0)
    assert digest(updates) == before, 'upstream mutation escaped adapter'
    assert all(v.device.type == 'cpu' and torch.isfinite(v).all() for v in actual.values())
    print(json.dumps({'reference_equal': True, 'caller_unchanged': True,
                      'cases': 'non-contiguous IDs, clipping outlier, bias and layer order'}))

if __name__ == '__main__': main()
