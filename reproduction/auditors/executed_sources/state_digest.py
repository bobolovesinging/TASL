import hashlib
import numpy as np
import torch

def protocol_state_hash(model_state, ema, history, anchor, temporal, excluded, violations):
    """Hash the complete simulated protocol state for round-by-round evidence."""
    digest = hashlib.sha256()

    def visit(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            digest.update(str((value.dtype.str, value.shape)).encode())
            digest.update(value.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                visit(key)
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, set):
            for item in sorted(value):
                visit(item)
        else:
            digest.update(repr(value).encode())
        digest.update(b"\0")

    visit((model_state, ema, history, anchor, temporal, excluded, violations))
    return digest.hexdigest()

