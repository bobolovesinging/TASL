def compute_tasl_trust_weights(
    flat_updates,
    trust_power=5.0,
    max_weight_ratio=1.5,
    min_cos_threshold=0.2,
    norm_penalty_strength=0.8,
    refine_anchor=True,
    ema_weights=None,
    ema_alpha=0.7,
    cos_history=None,
    prev_anchor=None,
    temporal_anchor=None,
):
    n = len(flat_updates)
    ids = list(flat_updates.keys())
    if n == 0:
        return {}, {}, None

    # Norm gating
    norms = {cid: np.linalg.norm(v) for cid, v in flat_updates.items()}
    median_norm = np.median(list(norms.values()))
    norm_gate = median_norm * 3.0

    # Spatial median anchor
    stacked = np.stack(list(flat_updates.values()), axis=0)
    anchor = np.median(stacked, axis=0)
    anchor_norm = np.linalg.norm(anchor) + 1e-12

    # Pairwise consensus
    pairwise_medians = {}
    for i in ids:
        vi_norm = np.linalg.norm(flat_updates[i]) + 1e-12
        sims = []
        for j in ids:
            if i != j:
                vj_norm = np.linalg.norm(flat_updates[j]) + 1e-12
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

    # Iterative anchor refinement
    if refine_anchor:
        first_pass_sims = {}
        for cid, v in flat_updates.items():
            v_norm = np.linalg.norm(v) + 1e-12
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

    # Temporal anchor blending (v3)
    if temporal_anchor is not None:
        temporal_norm = np.linalg.norm(temporal_anchor) + 1e-12
        direction_agreement = float(np.dot(anchor, temporal_anchor) / (anchor_norm * temporal_norm))
        if direction_agreement > 0.3:
            anchor = 0.6 * anchor + 0.4 * temporal_anchor
            anchor_norm = np.linalg.norm(anchor) + 1e-12

    # Trust scores with adaptive threshold
    cos_sims = {}
    for cid, v in flat_updates.items():
        v_norm = np.linalg.norm(v) + 1e-12
        cos_sim = float(np.dot(v, anchor) / (v_norm * anchor_norm))
        cos_sims[cid] = cos_sim

    cos_vals = sorted(cos_sims.values())
    if len(cos_vals) >= 4:
        q1 = np.percentile(cos_vals, 25)
        q3 = np.percentile(cos_vals, 75)
        iqr = q3 - q1
        adaptive_threshold = max(min_cos_threshold, q1 - 1.0 * iqr)
        adaptive_threshold = min(adaptive_threshold, 0.4)
    else:
        adaptive_threshold = min_cos_threshold

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
        norm_penalty = np.exp(-norm_penalty_strength * abs(norm_ratio - 1.0))
        consensus = pairwise_medians.get(cid, 0.5)
        consensus_factor = max(0.01, consensus)
        raw_scores[cid] = (cos_sim ** trust_power) * norm_penalty * consensus_factor

    # Cos history penalty
    if cos_history is not None and len(cos_history) >= 2:
        for cid in ids:
            recent_cos = [ch.get(cid, 0.5) for ch in cos_history[-2:]]
            if all(c < 0.3 for c in recent_cos):
                raw_scores[cid] *= 0.1

    # Normalize
    sum_w = sum(raw_scores.values())
    if sum_w > 1e-12:
        trust_weights = {cid: w / sum_w for cid, w in raw_scores.items()}
    else:
        trust_weights = {cid: 1.0 / n for cid in ids}

    # Weight cap
    max_w = max_weight_ratio / n
    capped = {cid: min(w, max_w) for cid, w in trust_weights.items()}
    cap_sum = sum(capped.values())
    if cap_sum > 1e-12:
        trust_weights = {cid: w / cap_sum for cid, w in capped.items()}

    # EMA smoothing
    if ema_weights is not None:
        smoothed = {}
        for cid in ids:
            current = trust_weights.get(cid, 0.0)
            history = ema_weights.get(cid, 1.0 / n)
            smoothed[cid] = ema_alpha * current + (1 - ema_alpha) * history
        s_sum = sum(smoothed.values())
        if s_sum > 1e-12:
            trust_weights = {cid: w / s_sum for cid, w in smoothed.items()}

    return trust_weights, cos_sims, anchor
