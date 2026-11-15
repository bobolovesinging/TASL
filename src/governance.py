"""TASL training and dual-auditor governance."""

import hashlib
import hmac
import json
import secrets

import numpy as np
import torch
from torch import nn

from .core.model import MnistCNN
from .sampled_weight_audit import (
    WEIGHT_TOLERANCE,
    build_admissibility_bounds,
    flatten_update,
    verify_sampled_bounds,
)
from .state_digest import protocol_state_hash
from .trust import compute_tasl_final_weights


AGGREGATE_TOLERANCE = 1e-6
PROJECTION_TOLERANCE = 0.1


def evaluate(model: nn.Module, test_loader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += model(x).argmax(dim=1).eq(y).sum().item()
            total += y.size(0)
    return 100.0 * correct / max(total, 1)


def state_sub(local_state, global_state):
    return {k: (local_state[k] - global_state[k]).detach()
            for k in global_state
            if isinstance(global_state[k], torch.Tensor) and global_state[k].is_floating_point()}


def state_add(global_state, update):
    return {k: (global_state[k] + update[k] if k in update else global_state[k])
            for k in global_state}


def make_postcommit_challenge(dimension: int, seed: int) -> np.ndarray:
    """Return one reproducible unit Rademacher challenge for audit replay."""
    if dimension <= 0:
        raise ValueError("Projection dimension must be positive")
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=dimension)
    return signs / np.sqrt(float(dimension))


def hash_local_update(update) -> str:
    """Hash a client update in tensor-key order."""
    digest = hashlib.sha256()
    for key in sorted(update):
        value = update[key].detach().cpu().contiguous().numpy()
        digest.update(json.dumps([key, value.dtype.str, value.shape]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def proposal_commitment(*, round_number, parent_hash, aggregate,
                        published_weights, update_hashes) -> str:
    """Bind every field needed to interpret an Executor's round proposal."""
    if set(published_weights) != set(update_hashes):
        raise ValueError("Published participant set differs from update identifiers")
    transcript = {
        "domain": "TASL-Executor-proposal-v2",
        "round": int(round_number), "parent_state": str(parent_hash),
        "participants": sorted(int(cid) for cid in update_hashes),
        "updates": [[int(cid), update_hashes[cid]] for cid in sorted(update_hashes)],
        "weights": [[int(cid), float(published_weights[cid])]
                    for cid in sorted(published_weights)],
        "aggregate_sha256": hash_local_update(aggregate),
    }
    return hashlib.sha256(json.dumps(transcript, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def proposal_transcript_matches(committed_hash, **transcript) -> bool:
    return hmac.compare_digest(committed_hash, proposal_commitment(**transcript))


def audit_input_key(round_number, parent_hash, local_updates, prior_trust_state):
    payload = {
        "round": round_number,
        "parent": parent_hash,
        "updates": {cid: hash_local_update(update) for cid, update in local_updates.items()},
        "history": protocol_state_hash({}, prior_trust_state),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def aggregate_relative_error(actual, expected):
    if set(actual) != set(expected):
        return float("inf")
    if any(actual[key].shape != expected[key].shape for key in expected):
        return float("inf")
    actual_flat = flatten_update(actual)
    expected_flat = flatten_update(expected)
    if not np.isfinite(actual_flat).all() or not np.isfinite(expected_flat).all():
        return float("inf")
    return float(np.linalg.norm(actual_flat - expected_flat) /
                 max(np.linalg.norm(expected_flat), 1e-12))


def weighted_average_updates(updates, weights):
    valid = [(cid, float(weights.get(cid, 0.0))) for cid in updates
             if float(weights.get(cid, 0.0)) > 0]
    if not valid:
        raise ValueError("No eligible updates: skip without changing protocol state")
    w_sum = sum(w for _, w in valid)
    normed = {cid: w / max(w_sum, 1e-12) for cid, w in valid}
    agg = {k: torch.zeros_like(v) for k, v in updates[next(iter(normed))].items()}
    for cid, w in normed.items():
        for k in agg:
            agg[k] += updates[cid][k] * w
    return agg


def train_one_client(global_state, loader, device):
    model = MnistCNN().to(device)
    model.load_state_dict(global_state)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    for _ in range(2):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def v1_verify_trust(published_weights, reference_weights):
    if published_weights.keys() != reference_weights.keys():
        return True
    return any(abs(weight - reference_weights[cid]) > WEIGHT_TOLERANCE
               for cid, weight in published_weights.items())


def _projected_reference(local_updates, published_weights, challenge, projection_context):
    """Project bound inputs without materializing a reference model.

    The error envelope covers the original tensor multiply/add rounding and
    the float64 scalar projections. Ambiguous comparisons still use the
    original tensor aggregation, including its numerical boundary.
    """
    if projection_context is None:
        return None
    prepared = projection_context['flat_updates']
    norms = projection_context['norms']
    valid = [(cid, float(published_weights.get(cid, 0.0)))
             for cid in local_updates if float(published_weights.get(cid, 0.0)) > 0]
    if not valid:
        return None
    weight_sum = sum(w for _, w in valid)
    if not np.isfinite(weight_sum):
        return None
    template = local_updates[valid[0][0]]
    keys = [k for k, value in template.items()
            if isinstance(value, torch.Tensor) and value.is_floating_point()]
    if not keys:
        return None
    if any(not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
           or value.dtype not in (torch.float32, torch.float64) for value in template.values()):
        return None
    dimension = sum(template[k].numel() for k in keys)
    if dimension != challenge.size:
        return None
    unit_roundoff = max(torch.finfo(value.dtype).eps / 2. for value in template.values())
    rhs = 0.
    weighted_norm = 0.
    for cid, weight in valid:
        update = local_updates[cid]
        if tuple(update) != tuple(template) or cid not in prepared or cid not in norms or any(
            not isinstance(update[k], torch.Tensor)
            or update[k].device.type != 'cpu'
            or update[k].shape != template[k].shape
            or update[k].dtype != template[k].dtype for k in template):
            return None
        flat = prepared[cid]
        if flat.shape != challenge.shape or flat.dtype != np.float64:
            return None
        norm = norms[cid]
        max_normal = min(torch.finfo(value.dtype).max for value in template.values())
        if not np.isfinite(norm) or norm > max_normal / (2 * max(1, len(valid))):
            return None
        normalized = weight / max(weight_sum, 1e-12)
        rhs += normalized * float(np.dot(challenge, flat))
        weighted_norm += abs(normalized) * norm
    tensor_terms = (3 * len(valid) + 4) * unit_roundoff
    scalar_terms = (dimension + len(valid) + 8) * (np.finfo(np.float64).eps / 2.)
    if tensor_terms >= .01 or scalar_terms >= .01:
        return None
    gamma_tensor = tensor_terms / (1. - tensor_terms)
    gamma_scalar = scalar_terms / (1. - scalar_terms)
    error = 8. * (gamma_tensor + gamma_scalar) * weighted_norm * float(np.linalg.norm(challenge))
    # Include absolute underflow error as well as the relative-error envelope.
    error += 8. * dimension * len(valid) * np.finfo(np.float32).tiny
    return rhs, error


def v1_verify_agg(
    agg_update,
    published_weights,
    local_updates,
    challenge,
    projection_context,
):
    """Check the aggregate against a post-commit projection."""
    flat_agg = flatten_update(agg_update)
    challenge = np.asarray(challenge, dtype=np.float64).reshape(-1)
    if challenge.shape != flat_agg.shape or not np.isfinite(challenge).all():
        raise ValueError("V1 projection challenge does not match aggregate shape")
    tolerance = PROJECTION_TOLERANCE
    lhs = float(np.dot(challenge, flat_agg))
    projected = _projected_reference(local_updates, published_weights, challenge, projection_context)
    if projected is not None and np.isfinite(lhs):
        rhs, error = projected
        margin = abs(lhs - rhs) - tolerance * max(abs(rhs), 1e-12)
        envelope = (1. + tolerance) * error
        envelope += 16. * np.finfo(np.float64).eps * (abs(lhs) + abs(rhs) + 1e-12)
        if np.isfinite(margin) and abs(margin) > envelope:
            residual = abs(lhs - rhs) / max(abs(rhs), 1e-12)
            return bool(residual > tolerance), float(residual)
    expected_flat = flatten_update(
        weighted_average_updates(local_updates, published_weights))
    rhs = float(np.dot(challenge, expected_flat))
    residual = abs(lhs - rhs) / max(abs(rhs), 1e-12)
    return bool(residual > tolerance), float(residual)


def v2_verify_agg(
    agg_update,
    published_weights,
    local_updates,
) -> bool:
    """Verify the complete aggregate independently of client sampling."""
    if set(published_weights) != set(local_updates):
        return True
    if (not all(np.isfinite(weight) and weight >= 0.0
                for weight in published_weights.values()) or
            sum(published_weights.values()) <= 1e-12):
        return True
    expected = weighted_average_updates(local_updates, published_weights)
    return aggregate_relative_error(agg_update, expected) > AGGREGATE_TOLERANCE


def train(client_loaders, test_loader, rounds, device, seed, output_dir):
    torch.manual_seed(seed)
    global_model = MnistCNN().to(device)
    global_state = {key: value.detach().cpu() for key, value in global_model.state_dict().items()}
    trust_state = (None, [], None, None)
    num_clients = len(client_loaders)
    record_path = output_dir / "trust_v1v2.jsonl"

    for number in range(1, rounds + 1):
        parent_hash = protocol_state_hash(global_state, trust_state)
        local_updates = {}
        for cid, loader in enumerate(client_loaders):
            torch.manual_seed(seed + number * 10000 + cid)
            local_state = train_one_client(global_state, loader, device)
            local_updates[cid] = state_sub(local_state, global_state)

        flat_updates = {cid: flatten_update(update) for cid, update in local_updates.items()}
        weights, cosines, anchor, ema, _ = compute_tasl_final_weights(flat_updates, *trust_state)
        if sum(weights.values()) <= 1e-12:
            accuracy = evaluate(global_model, test_loader, device)
            row = {
                "round": number, "skipped": 1, "audit_executed": 0, "consensus_passed": 0,
                "accuracy": accuracy, "parent_hash": parent_hash, "state_hash": parent_hash,
                "final_weights": json.dumps(weights, sort_keys=True),
                "events": json.dumps(["no_eligible_updates", "round_skipped"]),
            }
            with record_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            continue

        proposal = weighted_average_updates(local_updates, weights)
        reference_weights, _, _, _, projection_inputs = compute_tasl_final_weights(
            flat_updates, *trust_state)
        bounds = build_admissibility_bounds(local_updates, trust_state)
        reference_key = audit_input_key(number, parent_hash, local_updates, trust_state)
        update_hashes = {cid: hash_local_update(update) for cid, update in local_updates.items()}
        published_weights = dict(weights)
        transcript = {
            "round_number": number, "parent_hash": parent_hash, "aggregate": proposal,
            "published_weights": published_weights, "update_hashes": update_hashes,
        }
        proposal_hash = proposal_commitment(**transcript)
        observed_hashes = {cid: hash_local_update(update) for cid, update in local_updates.items()}
        observed_transcript = {**transcript, "update_hashes": observed_hashes}
        binding_valid = proposal_transcript_matches(proposal_hash, **observed_transcript)
        # Draw a fresh challenge after committing the proposal.
        challenge_seed = secrets.randbits(128)
        challenge = make_postcommit_challenge(flatten_update(proposal).size, challenge_seed)
        weight_seed = seed + number * 137
        v1_trust_bad = v1_verify_trust(published_weights, reference_weights)
        v1_agg_bad, residual = v1_verify_agg(
            proposal, published_weights, local_updates, challenge, projection_inputs)
        v2_trust_bad, sampled = verify_sampled_bounds(published_weights, bounds, num_clients, weight_seed)
        v2_agg_bad = v2_verify_agg(proposal, published_weights, local_updates)
        consensus = binding_valid and not any((v1_trust_bad, v1_agg_bad, v2_trust_bad, v2_agg_bad))
        events = ["proposal_created", "proposal_committed", "v1_challenge_sampled",
                  "audit_accepted" if consensus else "audit_rejected"]

        final_aggregate = proposal
        recovery_hash = None
        recovery_error = 0.0
        recovery_consistent = v1_endorsed = v2_endorsed = healed = False
        if not consensus:
            if reference_key != audit_input_key(number, parent_hash, local_updates, trust_state):
                raise RuntimeError("Auditor inputs changed before recovery endorsement")
            final_aggregate = weighted_average_updates(local_updates, reference_weights)
            recovery_error = aggregate_relative_error(
                final_aggregate, weighted_average_updates(local_updates, weights))
            recovery_consistent = recovery_error <= 1e-7
            events.append("recovery_recomputed")
            replacement = {
                **transcript, "aggregate": final_aggregate, "published_weights": reference_weights,
            }
            recovery_hash = proposal_commitment(**replacement)
            recovery_challenge = make_postcommit_challenge(
                flatten_update(final_aggregate).size, secrets.randbits(128))
            recovery_agg_bad, _ = v1_verify_agg(
                final_aggregate, reference_weights, local_updates, recovery_challenge, projection_inputs)
            v1_endorsed = not recovery_agg_bad
            recovery_trust_bad, _ = verify_sampled_bounds(reference_weights, bounds, num_clients, weight_seed)
            recovery_exact_bad = v2_verify_agg(final_aggregate, reference_weights, local_updates)
            v2_endorsed = not (recovery_trust_bad or recovery_exact_bad)
            events.append("recovery_audited")
            healed = bool(recovery_consistent and v1_endorsed and v2_endorsed and
                          proposal_transcript_matches(recovery_hash, **replacement))
            events.append("recovery_verified" if healed else "recovery_failed")

        skipped = not (consensus or healed)
        if skipped:
            state_hash = parent_hash
            events.append("round_skipped")
        else:
            global_state = state_add(global_state, final_aggregate)
            aggregate_flat = flatten_update(final_aggregate)
            temporal_anchor = aggregate_flat if np.linalg.norm(aggregate_flat) > 1e-12 else trust_state[3]
            trust_state = (ema, [*trust_state[1], cosines], anchor, temporal_anchor)
            global_model.load_state_dict(global_state)
            state_hash = protocol_state_hash(global_state, trust_state)
            events.append("committed_once")

        accuracy = evaluate(global_model, test_loader, device)
        row = {
            "round": number,
            "audit_executed": 1,
            "v1_trust": int(v1_trust_bad),
            "v1_agg": int(v1_agg_bad),
            "v1_projection_residual": residual,
            "v1_challenge_seed": challenge_seed,
            "v2_trust": int(v2_trust_bad),
            "v2_agg": int(v2_agg_bad),
            "vp_sampled_clients": sorted(sampled),
            "vp_bound_intervals": bounds["intervals"],
            "audit_reference_input_key": reference_key,
            "proposal_commitment": proposal_hash,
            "proposal_binding_valid": int(binding_valid),
            "update_hashes": json.dumps(update_hashes, sort_keys=True),
            "recovery_attempted": int(not consensus),
            "recovery_consistent": int(recovery_consistent),
            "recovery_relative_error": recovery_error if np.isfinite(recovery_error) else None,
            "recovery_commitment": recovery_hash,
            "recovery_v1_endorsed": int(v1_endorsed),
            "recovery_v2_endorsed": int(v2_endorsed),
            "recovery_endorsed": int(v1_endorsed and v2_endorsed),
            "consensus_passed": int(consensus),
            "healed_applied": int(healed),
            "accuracy": accuracy,
            "skipped": int(skipped),
            "parent_hash": parent_hash,
            "state_hash": state_hash,
            "final_weights": json.dumps(reference_weights if healed else published_weights, sort_keys=True),
            "events": json.dumps(events),
        }
        with record_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

        if number % 10 == 0 or number == 1:
            print(f"Round {number}: accuracy={accuracy:.2f}% "
                  f"accepted={int(consensus)} recovered={int(healed)}")

    torch.save({
        "model": global_state,
        "trust_state": trust_state,
        "state_hash": protocol_state_hash(global_state, trust_state),
    }, output_dir / "model.pt")
    return accuracy
