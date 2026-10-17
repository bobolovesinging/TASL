"""Retain real records, archive heavy state, and test recovery in new processes."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse
import copy
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from storage_bridge import digest, json_bytes
from fifty_store import recent_rounds
import torch

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / 'final_protocol_20260928'
sys.path.insert(0, str(FROZEN))
from run_table5_exact_v2 import m

def serialize(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()

def write_json(path, value):
    path.write_bytes(json_bytes(value))

def prepare(capture, store):
    manifest = json.loads((capture / 'manifest.json').read_text(encoding='utf-8'))
    defended = json.loads((capture / 'trust_v1v2.json').read_text(encoding='utf-8'))['round_logs']
    baseline = json.loads((capture / 'noexecutor.json').read_text(encoding='utf-8'))['round_logs']
    for sub in ('active', 'archive', 'headers', 'governance'):
        (store / sub).mkdir(parents=True)
    index = {'schema': 'local-hbs-replay-v1', 'window': 5,
             'configuration': manifest['configuration'], 'rounds': {}}
    measured = []
    for row in defended:
        r = row['round']
        snapshot_path = capture / 'snapshots' / f'round_{r:02d}.pt'
        snapshot = torch.load(snapshot_path, weights_only=False, map_location='cpu')
        checkpoint = {'schema': 'tasl-full-state-v1', 'round': r,
                      'model': snapshot['prior_model'], 'trust': snapshot['prior_trust_state']}
        inputs = {'local_updates': snapshot['local_updates'],
                  'submitted': m.weighted_average_updates(snapshot['local_updates'], snapshot['aggregate_weights']),
                  'published_weights': snapshot['published_weights']}
        cp_blob, input_blob = serialize(checkpoint), serialize(inputs)
        cp_path, input_path = f'active/checkpoint_{r:02d}.pt', f'active/inputs_{r:02d}.pt'
        (store / cp_path).write_bytes(cp_blob)
        (store / input_path).write_bytes(input_blob)
        header = {
            'round': r, 'parent_hash': snapshot['parent_hash'],
            'participants': sorted(snapshot['local_updates']),
            'checkpoint_sha': digest(cp_blob), 'inputs_sha': digest(input_blob),
            'update_hashes': {str(cid): m.hash_local_update(upd) for cid,upd in snapshot['local_updates'].items()},
            'original_proposal': row['proposal_commitment'],
            'frozen_protocol_sha256': manifest['source_sha256'],
        }
        governance = {
            'round': r, 'attack_type': row['attack_type'],
            'original_rejected': not bool(row['consensus_passed']),
            'original_vs_trust_reject': bool(row['v1_trust']),
            'original_vs_aggregate_reject': bool(row['v1_agg']),
            'original_vp_trust_reject': bool(row['v2_trust']),
            'original_vp_aggregate_reject': bool(row['v2_agg']),
            'captured_recovery_commitment': row['recovery_commitment'],
            'captured_recovery_vs_endorsed': bool(row['recovery_v1_endorsed']),
            'captured_recovery_vp_endorsed': bool(row['recovery_v2_endorsed']),
            'captured_final_protocol_hash': row['state_hash'],
            'events': json.loads(row['events']),
            'record_authentication': 'trusted local manifest; no digital signatures',
        }
        hp, gp = f'headers/round_{r:02d}.json', f'governance/round_{r:02d}.json'
        write_json(store / hp, header)
        write_json(store / gp, governance)
        entry = {'header': hp, 'header_sha': digest((store / hp).read_bytes()),
                 'governance': gp, 'governance_sha': digest((store / gp).read_bytes()),
                 'checkpoint': cp_path, 'inputs': input_path}
        index['rounds'][str(r)] = entry
        measured.append({'round': r, 'complete_checkpoint_bytes': len(cp_blob),
                         'model_only_bytes': len(serialize(snapshot['prior_model'])),
                         'trust_only_bytes': len(serialize(snapshot['prior_trust_state'])),
                         'inputs_plus_submitted_proposal_bytes': len(input_blob),
                         'header_bytes': (store / hp).stat().st_size,
                         'actual_governance_bytes': (store / gp).stat().st_size,
                         'history_length': len(snapshot['prior_trust_state'][1])})
    # Real archive-before-prune, rather than assigning two policies one formula.
    archive_events = []
    for r in sorted(set(range(1,len(defended)+1))-recent_rounds(len(defended),5)):
        entry = index['rounds'][str(r)]
        for key in ('checkpoint', 'inputs'):
            old = store / entry[key]
            target = store / 'archive' / old.name
            shutil.copy2(old, target)
            if digest(old.read_bytes()) != digest(target.read_bytes()):
                raise AssertionError('Archive byte-copy mismatch')
            old.unlink()
            entry[key] = 'archive/' + target.name
            archive_events.append({'round': r, 'kind': key, 'archive_sha256': digest(target.read_bytes()),
                                   'active_copy_removed_after_verify': not old.exists()})
    write_json(store / 'index.json', index)
    write_json(HERE / 'storage_measurements.json', {
        'schema': 'full-model-and-six-trust-fields; real governance records',
        'window': 5, 'captured_rounds': len(defended), 'rounds': measured,
        'archive_before_prune_events': archive_events,
        'active_payload_bytes': sum(p.stat().st_size for p in (store / 'active').iterdir()),
        'archive_payload_bytes': sum(p.stat().st_size for p in (store / 'archive').iterdir()),
        'permanent_header_governance_bytes': sum(p.stat().st_size for sub in ('headers','governance')
                                               for p in (store / sub).iterdir()),
        'not_comparable_to_old_aggregate_checkpoint_bytes': True,
    })
    return baseline

def mutate(store, case, r):
    index = json.loads((store / 'index.json').read_text())
    entry = index['rounds'][str(r)]
    header = json.loads((store / entry['header']).read_text())
    if case == 'missing_updates':
        (store / entry['inputs']).unlink()
    elif case in ('corrupt_updates', 'corrupt_checkpoint'):
        key = 'inputs' if case == 'corrupt_updates' else 'checkpoint'
        path = store / entry[key]
        blob = bytearray(path.read_bytes())
        blob[len(blob)//2] ^= 1
        path.write_bytes(blob)
    elif case in ('incomplete_trust', 'wrong_round'):
        path = store / entry['checkpoint']
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if case == 'incomplete_trust':
            checkpoint['trust'] = tuple(x for i,x in enumerate(checkpoint['trust']) if i != 1)
        else:
            checkpoint['round'] = r+1
        # A faithfully hashed incomplete/stale export must also be rejected.
        path.write_bytes(serialize(checkpoint))
        header['checkpoint_sha'] = digest(path.read_bytes())
    elif case == 'wrong_parent':
        header['parent_hash'] = '0'*64
    elif case == 'uniform_prune_no_archive':
        for path in (store / 'archive').iterdir():
            path.unlink()
    write_json(store / entry['header'], header)
    entry['header_sha'] = digest((store / entry['header']).read_bytes())
    write_json(store / 'index.json', index)
    return digest((store / 'index.json').read_bytes())
