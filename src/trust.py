"""TASL spatiotemporal trust scores and final aggregation weights."""

import numpy as np


EMA_ALPHA = 0.7
TRUST_POWER = 5.0
MAX_WEIGHT_RATIO = 1.5
NORM_GATE = 2.5
DIRECTION_THRESHOLD = 0.6


def direction_screening_scores(flat_updates, reference):
    """Lightweight direction gate used before multiplicative trust fusion."""
    ids = list(flat_updates.keys())
    if reference is None or np.linalg.norm(reference) < 1e-12:
        return {cid: 1.0 for cid in ids}

    ref_norm = np.linalg.norm(reference) + 1e-12
    scores = {}
    for cid, v in flat_updates.items():
        v_norm = np.linalg.norm(v) + 1e-12
        c = float(np.dot(v, reference) / (v_norm * ref_norm))
        if c <= 0:
            scores[cid] = 0.0
        elif c >= DIRECTION_THRESHOLD:
            scores[cid] = 1.0
        else:
            scores[cid] = c / DIRECTION_THRESHOLD
    return scores


def _trust_scores(flat_updates, norms, ema_weights, cos_history, prev_anchor, temporal_anchor):
    n = len(flat_updates)
    ids = list(flat_updates.keys())
    median_norm = np.median(list(norms.values()))
    norm_gate = median_norm * NORM_GATE

    stacked = np.stack(list(flat_updates.values()), axis=0)
    anchor = np.median(stacked, axis=0)
    anchor_norm = np.linalg.norm(anchor) + 1e-12

    pairwise_medians = {}
    for i in ids:
        vi_norm = norms[i] + 1e-12
        sims = []
        for j in ids:
            if i != j:
                vj_norm = norms[j] + 1e-12
                sim = float(np.dot(flat_updates[i], flat_updates[j]) / (vi_norm * vj_norm))
                sims.append(sim)
        pairwise_medians[i] = float(np.median(sims)) if sims else 0.5

    consensus_vals = list(pairwise_medians.values())
    median_consensus = np.median(consensus_vals)
    high_consensus_ids = [cid for cid in ids if pairwise_medians[cid] >= median_consensus]

    if len(high_consensus_ids) >= max(2, n // 2):
        consensus_stacked = np.stack([flat_updates[cid] for cid in high_consensus_ids])
        consensus_anchor = np.median(consensus_stacked, axis=0)
        anchor = 0.6 * anchor + 0.4 * consensus_anchor
        anchor_norm = np.linalg.norm(anchor) + 1e-12

    first_pass_sims = {}
    for cid, v in flat_updates.items():
        v_norm = norms[cid] + 1e-12
        cos_sim = float(np.dot(v, anchor) / (v_norm * anchor_norm))
        first_pass_sims[cid] = cos_sim

    trusted_ids = [cid for cid in ids if first_pass_sims[cid] > 0]
    if len(trusted_ids) >= max(2, n // 2):
        refined_stacked = np.stack([flat_updates[cid] for cid in trusted_ids])
        refined_anchor = np.median(refined_stacked, axis=0)
        if prev_anchor is not None:
            anchor = 0.6 * refined_anchor + 0.4 * prev_anchor
        else:
            anchor = refined_anchor
        anchor_norm = np.linalg.norm(anchor) + 1e-12

    if temporal_anchor is not None:
        temporal_norm = np.linalg.norm(temporal_anchor) + 1e-12
        direction_agreement = float(np.dot(anchor, temporal_anchor) / (anchor_norm * temporal_norm))
        if direction_agreement > 0.3:
            anchor = 0.6 * anchor + 0.4 * temporal_anchor
            anchor_norm = np.linalg.norm(anchor) + 1e-12

    cos_sims = {}
    for cid, v in flat_updates.items():
        v_norm = norms[cid] + 1e-12
        cos_sim = float(np.dot(v, anchor) / (v_norm * anchor_norm))
        cos_sims[cid] = cos_sim

    cos_vals = sorted(cos_sims.values())
    if len(cos_vals) >= 4:
        q1 = np.percentile(cos_vals, 25)
        q3 = np.percentile(cos_vals, 75)
        iqr = q3 - q1
        adaptive_threshold = max(0.2, q1 - 1.0 * iqr)
        adaptive_threshold = min(adaptive_threshold, 0.4)
    else:
        adaptive_threshold = 0.2

    raw_scores = {}
    for cid in ids:
        cos_sim = cos_sims[cid]
        if norms[cid] > norm_gate:
            raw_scores[cid] = 0.0
            continue
        if cos_sim < adaptive_threshold:
            raw_scores[cid] = 0.0
            continue
        norm_ratio = norms[cid] / (median_norm + 1e-12)
        norm_penalty = np.exp(-0.8 * abs(norm_ratio - 1.0))
        consensus = pairwise_medians.get(cid, 0.5)
        consensus_factor = max(0.01, consensus)
        raw_scores[cid] = (cos_sim ** TRUST_POWER) * norm_penalty * consensus_factor

    if cos_history is not None and len(cos_history) >= 2:
        for cid in ids:
            recent_cos = [ch.get(cid, 0.5) for ch in cos_history[-2:]]
            if all(c < 0.3 for c in recent_cos):
                raw_scores[cid] *= 0.1

    sum_w = sum(raw_scores.values())
    if sum_w > 1e-12:
        trust_weights = {cid: w / sum_w for cid, w in raw_scores.items()}
    else:
        trust_weights = {cid: 1.0 / n for cid in ids}

    max_w = MAX_WEIGHT_RATIO / n
    capped = {cid: min(w, max_w) for cid, w in trust_weights.items()}
    cap_sum = sum(capped.values())
    if cap_sum > 1e-12:
        trust_weights = {cid: w / cap_sum for cid, w in capped.items()}

    if ema_weights is not None:
        smoothed = {}
        for cid in ids:
            current = trust_weights.get(cid, 0.0)
            history = ema_weights.get(cid, 1.0 / n)
            smoothed[cid] = EMA_ALPHA * current + (1 - EMA_ALPHA) * history
        s_sum = sum(smoothed.values())
        if s_sum > 1e-12:
            trust_weights = {cid: w / s_sum for cid, w in smoothed.items()}

    return trust_weights, cos_sims, anchor

def compute_tasl_final_weights(flat_updates, ema_weights, cos_history, prev_anchor, temporal_anchor):
    """Return final weights, similarities, anchor, pre-fusion EMA and projection inputs."""
    updates = {cid: np.asarray(value, dtype=np.float64) for cid, value in flat_updates.items()}
    zero = dict.fromkeys(updates, 0.0)
    if not updates or any(not np.isfinite(value).all() for value in updates.values()):
        return zero, dict(zero), None, dict(zero), None
    norms = {cid: float(np.linalg.norm(value)) for cid, value in updates.items()}
    if not all(np.isfinite(norm) for norm in norms.values()):
        return zero, dict(zero), None, dict(zero), None

    projection_inputs = {"flat_updates": updates, "norms": norms}
    scores, cosines, anchor = _trust_scores(
        updates, norms, ema_weights, cos_history, prev_anchor, temporal_anchor)
    reference = temporal_anchor if temporal_anchor is not None else prev_anchor
    direction = direction_screening_scores(updates, reference)
    norm_cutoff = NORM_GATE * float(np.median(list(norms.values())))
    masked = {cid: scores[cid] if norms[cid] <= norm_cutoff else 0.0 for cid in updates}
    fused = {cid: masked[cid] * direction[cid] for cid in updates}
    total = sum(fused.values())
    if total > 1e-12 and np.isfinite(total):
        return {cid: weight / total for cid, weight in fused.items()}, cosines, anchor, scores, projection_inputs

    # The fallback retains only norm-admissible weights.
    total = sum(masked.values())
    if total > 1e-12 and np.isfinite(total):
        return {cid: weight / total for cid, weight in masked.items()}, cosines, anchor, scores, projection_inputs
    return zero, cosines, anchor, scores, projection_inputs
