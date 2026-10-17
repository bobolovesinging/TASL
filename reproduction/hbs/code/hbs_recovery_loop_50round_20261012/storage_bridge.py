"""Local persistence guards. These do not supply network consensus or signatures."""
import hashlib
import json
from pathlib import Path

def digest(blob):
    return hashlib.sha256(blob).hexdigest()

def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')

def verified_bytes(path, expected):
    blob = Path(path).read_bytes()
    if digest(blob) != expected:
        raise ValueError('payload digest mismatch: ' + Path(path).name)
    return blob

def validate_checkpoint(checkpoint, round_number):
    if checkpoint.get('schema') != 'tasl-full-state-v1':
        raise ValueError('checkpoint schema mismatch')
    if checkpoint.get('round') != round_number:
        raise ValueError('checkpoint round mismatch')
    if not isinstance(checkpoint.get('model'), dict) or not checkpoint['model']:
        raise ValueError('missing model state')
    trust = checkpoint.get('trust')
    if not isinstance(trust, (tuple, list)) or len(trust) != 6:
        raise ValueError('checkpoint must contain six trust-state components')
    if not isinstance(trust[0], dict) or not isinstance(trust[1], list) or \
            not isinstance(trust[4], set) or not isinstance(trust[5], dict):
        raise ValueError('trust-state component types invalid')
    return checkpoint

def commit_gate(path, *, round_number, parent, proposal, vs, vp, bound):
    if vs is not True or vp is not True or bound is not True:
        return False
    path = Path(path)
    try:
        # Exclusive creation is a local exactly-once guard, not a ledger commit.
        with path.open('xb') as stream:
            stream.write(json_bytes({'round': round_number, 'parent': parent,
                                     'proposal': proposal, 'vs': vs, 'vp': vp}))
    except FileExistsError:
        return False
    return True
