"""Fresh-process local HBS -> frozen TAS recovery; no in-memory capture access."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import copy
import io
import json
from pathlib import Path
import secrets
import sys
import time
from storage_bridge import digest, json_bytes, verified_bytes, validate_checkpoint, commit_gate

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / 'final_protocol_20260928'
SOURCE = FROZEN / 'scripts/exp3/exp3_governance_tasv2.py'
EXPECTED_SOURCE = 'f71067a01c7af357c664306aed944a64c440e53df537a0f2f7e8f0311311e987'
if digest(SOURCE.read_bytes()) != EXPECTED_SOURCE:
    raise AssertionError('Frozen protocol SHA mismatch')
sys.path.insert(0, str(FROZEN))
from run_table5_exact_v2 import m
import numpy as np
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

def run(args):
    begin = time.perf_counter_ns()
    result = {'case': args.case, 'round': args.round, 'pid': os.getpid(),
              'worker_source_sha256': digest(Path(__file__).read_bytes()),
              'committed': False, 'commit_count': 0, 'events': [],
              'timing_scope': 'after Python/torch import; read+hash+trust+audit+local commit'}
    checkpoint = None
    try:
        index = json.loads(verified_bytes(args.store / 'index.json', args.index_sha))
        entry = index['rounds'][str(args.round)]
        header = json.loads(verified_bytes(args.store / entry['header'], entry['header_sha']))
        governance = json.loads(verified_bytes(args.store / entry['governance'], entry['governance_sha']))
        if header['round'] != args.round or governance['round'] != args.round:
            raise ValueError('header/governance round mismatch')
        if header['parent_hash'] != args.parent:
            raise ValueError('parent context mismatch')
        checkpoint_blob = verified_bytes(args.store / entry['checkpoint'], header['checkpoint_sha'])
        checkpoint = torch.load(io.BytesIO(checkpoint_blob), map_location='cpu', weights_only=False)
        validate_checkpoint(checkpoint, args.round)
        prior_hash = m.protocol_state_hash(checkpoint['model'], *checkpoint['trust'])
        result['restored_parent_hash'] = prior_hash
        result['history_length'] = len(checkpoint['trust'][1])
        if prior_hash != args.parent:
            raise ValueError('complete prior protocol-state hash mismatch')
        result['events'].append('full_checkpoint_restored')
        payload_blob = verified_bytes(args.store / entry['inputs'], header['inputs_sha'])
        payload = torch.load(io.BytesIO(payload_blob), map_location='cpu', weights_only=False)
        updates = payload['local_updates']
        if sorted(updates) != header['participants']:
            raise ValueError('participant set mismatch')
        hashes = {cid: m.hash_local_update(upd) for cid, upd in updates.items()}
        if {str(k):v for k,v in hashes.items()} != header['update_hashes']:
            raise ValueError('client update commitment mismatch')
        if not m.proposal_transcript_matches(header['original_proposal'],
                round_number=args.round, parent_hash=args.parent,
                aggregate=payload['submitted'], published_weights=payload['published_weights'],
                update_hashes=hashes):
            raise ValueError('original proposal binding mismatch')
        result['events'].append('round_inputs_verified')
        tas = index['configuration']['tas']
        num_clients = index['configuration']['experiment']['clients']
        if tas['exclusion'] is not False:
            raise ValueError('This narrow bridge does not implement alternate exclusion transitions')
        flat = {cid: m.flatten_update(update) for cid, update in updates.items()}
        def recompute():
            ema, history, anchor, temporal, _, _ = copy.deepcopy(checkpoint['trust'])
            return m.full_trust(flat, trust_power=tas['trust_power'],
                                max_weight_ratio=tas['max_weight_ratio'],
                                norm_penalty_strength=0.8, refine_anchor=True,
                                ema_weights=ema, ema_alpha=tas['ema_alpha'],
                                cos_history=history, prev_anchor=anchor,
                                temporal_anchor=temporal)
        weights, sims, anchor = recompute()
        independently_recomputed, _, _ = recompute()
        honest_aggregate = m.weighted_average_updates(updates, independently_recomputed)
        # Re-evaluate the stored original proposal rather than copying old verdicts.
        original_commit = header['original_proposal']
        original_seed = secrets.randbits(128)
        original_verdict = m.audit_recovery_candidate(
            payload['submitted'], payload['published_weights'], independently_recomputed, updates,
            challenge=m.make_postcommit_challenge(m.flatten_update(honest_aggregate).size, original_seed),
            num_clients=num_clients, use_v1=True, use_v2=True,
            v2_sample_ratio=tas['v2_sample_client_ratio'], v2_sample_seed=42+args.round*137)
        result['original_proposal_rejected'] = not original_verdict['endorsed']
        result['original_verdict'] = original_verdict
        result['original_challenge_seed'] = original_seed
        if original_verdict['endorsed']:
            raise ValueError('original attack not rejected')
        candidate = m.weighted_average_updates(updates, weights)
        recovery_commit = m.proposal_commitment(
            round_number=args.round, parent_hash=args.parent, aggregate=candidate,
            published_weights=weights, update_hashes=hashes)
        result['events'].append('replacement_committed_before_challenge')
        if args.case == 'corrupt_replacement':
            key = next(iter(candidate))
            candidate[key] = candidate[key].clone()
            candidate[key].reshape(-1)[0] += 1.0
        challenge_seed = secrets.randbits(128)
        result['recovery_challenge_seed'] = challenge_seed
        verdict = m.audit_recovery_candidate(
            candidate, weights, independently_recomputed, updates,
            challenge=m.make_postcommit_challenge(m.flatten_update(candidate).size, challenge_seed),
            num_clients=num_clients, use_v1=True, use_v2=True,
            v2_sample_ratio=tas['v2_sample_client_ratio'], v2_sample_seed=42+args.round*137)
        result['recovery_verdict_before_vote_fault'] = dict(verdict)
        if args.case == 'missing_vs_vote':
            verdict['v1_endorsed'] = False
        if args.case == 'missing_vp_vote':
            verdict['v2_endorsed'] = False
        binding = m.proposal_transcript_matches(
            recovery_commit, round_number=args.round, parent_hash=args.parent,
            aggregate=candidate, published_weights=weights, update_hashes=hashes)
        consistency = m.check_recovery_consistency(candidate, honest_aggregate)
        result.update({'recovery_commitment': recovery_commit,
                       'binding_valid': binding, 'recovery_consistency': consistency,
                       'vs_endorsed': verdict['v1_endorsed'], 'vp_endorsed': verdict['v2_endorsed'],
                       'recovery_commitment_matches_capture': recovery_commit == governance['captured_recovery_commitment']})
        result['events'].append('replacement_reaudited')
        commit = commit_gate(args.output.with_suffix('.commit.json'),
                             round_number=args.round, parent=args.parent, proposal=recovery_commit,
                             vs=verdict['v1_endorsed'], vp=verdict['v2_endorsed'],
                             bound=binding and consistency['consistent'])
        if commit:
            model = m.state_add(checkpoint['model'], candidate)
            prior = copy.deepcopy(checkpoint['trust'])
            history = prior[1]
            history.append(dict(sims))
            temporal = prior[3]
            flat_aggregate = m.flatten_update(candidate)
            if np.linalg.norm(flat_aggregate) > 1e-12:
                temporal = flat_aggregate
            trust = (dict(weights), history, anchor, temporal, prior[4], prior[5])
            result['final_state_hash'] = m.protocol_state_hash(model, *trust)
            final_path = args.output.with_suffix('.recovered.pt')
            torch.save({'model': model, 'trust': trust, 'round': args.round}, final_path)
            result['recovered_state_file_sha256'] = digest(final_path.read_bytes())
            result['recovered_state_bytes'] = final_path.stat().st_size
            result['committed'] = True
            result['commit_count'] = 1
            result['events'].append('committed_once')
            result['duplicate_commit_rejected'] = not commit_gate(
                args.output.with_suffix('.commit.json'), round_number=args.round,
                parent=args.parent, proposal=recovery_commit,
                vs=True, vp=True, bound=True)
        else:
            result['final_state_hash'] = prior_hash
            result['events'].append('recovery_rejected_prior_state_preserved')
    except (ValueError, FileNotFoundError, KeyError) as error:
        result['error'] = str(error)
        result['events'].append('recovery_blocked_before_commit')
        if checkpoint is not None and 'trust' in checkpoint and len(checkpoint['trust']) == 6:
            result['final_state_hash'] = m.protocol_state_hash(checkpoint['model'], *checkpoint['trust'])
    result['local_loop_ms'] = (time.perf_counter_ns() - begin) / 1e6
    args.output.write_bytes(json_bytes(result))
    print(json.dumps({k:result.get(k) for k in ('case','round','committed','error','local_loop_ms')}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--index-sha', required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
