"""
TASL + FedPure Fusion Module.
Combines FedPure data-level detection with TASL gradient-level trust scoring.
"""
import numpy as np


def fedpure_style_detection(flat_updates, corr_threshold=0.3):
    """
    Simplified FedPure-style anomaly detection applied to gradient vectors.
    
    Core idea from FedPure: compare each client's "prototype" (gradient)
    against others via pairwise correlation. Low-correlation clients
    are flagged as potentially malicious.
    
    Args:
        flat_updates: dict {client_id: np.array}
        corr_threshold: correlation below which client is suspicious
        
    Returns:
        dict {client_id: float} — FedPure confidence score (1.0 = clean, 0.0 = malicious)
    """
    ids = list(flat_updates.keys())
    n = len(ids)
    if n <= 2:
        return {cid: 1.0 for cid in ids}
    
    # ── Pairwise cosine similarity matrix ──
    sims = {}
    for i in ids:
        vi = flat_updates[i]
        vi_norm = np.linalg.norm(vi) + 1e-12
        sims[i] = {}
        for j in ids:
            if i != j:
                vj = flat_updates[j]
                vj_norm = np.linalg.norm(vj) + 1e-12
                sims[i][j] = float(np.dot(vi, vj) / (vi_norm * vj_norm))
    
    # ── Per-client median similarity to others ──
    median_sims = {}
    for i in ids:
        values = list(sims[i].values())
        median_sims[i] = float(np.median(values))
    
    # ── Adaptive threshold: clients with median_sim in bottom 25% are suspicious ──
    sim_values = sorted(median_sims.values())
    q25 = np.percentile(sim_values, 25)
    
    scores = {}
    for cid in ids:
        ms = median_sims[cid]
        if ms < corr_threshold:
            scores[cid] = 0.0  # Clearly malicious
        elif ms < q25:
            scores[cid] = ms / max(q25, 0.01)  # Scaled down
        else:
            scores[cid] = 1.0  # Clean
    
    return scores


def compute_fused_trust_weights(
    flat_updates,
    compute_tasl_fn,
    fedpure_weight=0.3,
    corr_threshold=0.3,
    **tasl_kwargs,
):
    """
    Fusion: combine FedPure detection scores with TASL trust weights.
    
    Pipeline:
        1. FedPure detection → per-client confidence scores
        2. TASL trust scoring → per-client direction+norm scores
        3. Fusion: final = fedpure_factor * tasl_trust (multiplicative)
    
    Multiplicative fusion is stricter than additive:
    - If either detector says "malicious", the client gets zero weight.
    - This avoids the "weak signal from both, but arithmetic mean looks OK" problem.
    
    Args:
        flat_updates: dict {client_id: np.array}
        compute_tasl_fn: TASL trust scoring function (from exp2_robustness_matrix)
        fedpure_weight: how much to weight FedPure detection (0-1)
        corr_threshold: FedPure correlation threshold
        **tasl_kwargs: passed through to compute_tasl_fn
        
    Returns:
        trust_weights: dict {client_id: float}
        cos_sims: dict {client_id: float}
        anchor: np.array
        fusion_info: dict with detection details
    """
    n = len(flat_updates)
    ids = list(flat_updates.keys())
    
    # Step 1: FedPure detection
    fedpure_scores = fedpure_style_detection(flat_updates, corr_threshold=corr_threshold)
    
    # Step 2: TASL trust scoring
    tasl_weights, cos_sims, anchor = compute_tasl_fn(
        flat_updates, **tasl_kwargs
    )
    # Default to equal weights if TASL returns empty
    if not tasl_weights:
        tasl_weights = {cid: 1.0 / n for cid in ids}
    
    # Step 3: Multiplicative fusion
    fused_raw = {}
    for cid in ids:
        fp_score = fedpure_scores.get(cid, 1.0)
        tasl_score = tasl_weights.get(cid, 0.0)
        # Convert TASL weight back to raw score (approx)
        # TASL weight = normalized(raw_score), so raw ≈ weight * n
        tasl_raw = max(0.0, tasl_score * n)
        # Fuse: if FedPure says malicious, zero out
        if fp_score < 0.1:
            fused_raw[cid] = 0.0
        else:
            fused_raw[cid] = (1 - fedpure_weight) * tasl_raw + fedpure_weight * fp_score * tasl_raw
    
    # Normalize
    sum_w = sum(fused_raw.values())
    if sum_w > 1e-12:
        fused_weights = {cid: w / sum_w for cid, w in fused_raw.items()}
    else:
        fused_weights = {cid: 1.0 / n for cid in ids}
    
    # Step 4: Apply weight cap (1.5/N)
    max_w = 1.5 / n
    capped = {cid: min(w, max_w) for cid, w in fused_weights.items()}
    cap_sum = sum(capped.values())
    if cap_sum > 1e-12:
        fused_weights = {cid: w / cap_sum for cid, w in capped.items()}
    
    fusion_info = {
        'fedpure_scores': fedpure_scores,
        'detected_malicious': [cid for cid in ids if fedpure_scores[cid] < 0.1],
        'tasl_weights': tasl_weights,
        'fused_weights': fused_weights,
    }
    
    return fused_weights, cos_sims, anchor, fusion_info
