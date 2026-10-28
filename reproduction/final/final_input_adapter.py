"""Instrument the frozen runner only at its external client-update input boundary."""
import inspect
from pathlib import Path
from scripts.exp3 import exp3_governance_tasv2 as canonical

def build():
    source=inspect.getsource(canonical.run_experiment_group)
    source=source.replace('def run_experiment_group(', 'def run_transformed_group(',1)
    source=source.replace('    final_state_path: Optional[str] = None,',
        '    final_state_path: Optional[str] = None,\n    input_transform=None,',1)
    boundary='        chaos = chaos_schedule.get(r, RoundAttack(active=False))'
    assert source.count(boundary)==1
    source=source.replace(boundary,
        '        if input_transform is not None:\n'
        '            local_updates = input_transform(r, global_state, prior_trust_state, local_updates)\n'
        '            flat_updates = {cid: flatten_update(upd) for cid, upd in local_updates.items()}\n\n'+boundary)
    scope=dict(canonical.__dict__)
    exec(compile(source,'final_input_adapter.generated','exec'),scope)
    return scope['run_transformed_group'],source
