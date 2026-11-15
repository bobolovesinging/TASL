import hashlib

import numpy as np
import torch


def protocol_state_hash(model_state, trust_state):
    """Hash the model and trust state for the round record."""
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
        else:
            digest.update(repr(value).encode())
        digest.update(b"\0")

    visit((model_state, *trust_state))
    return digest.hexdigest()

