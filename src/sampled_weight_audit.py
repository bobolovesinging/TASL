"""V_P sampled trust-weight admissibility audit."""

import numpy as np
import torch

from .trust import DIRECTION_THRESHOLD, EMA_ALPHA, NORM_GATE


WEIGHT_TOLERANCE = 1e-6
SAMPLE_RATIO = 0.3


def flatten_update(update):
    chunks = [value.detach().cpu().reshape(-1).numpy() for value in update.values()
              if isinstance(value, torch.Tensor) and value.is_floating_point()]
    return np.concatenate(chunks).astype(np.float64) if chunks else np.zeros(1, dtype=np.float64)


def _normalized_intervals(lower, upper, factor):
    low_n = lower * factor
    high_n = upper * factor
    low = np.zeros(len(lower))
    high = np.zeros(len(lower))
    for i in range(len(lower)):
        if factor[i] <= 0:
            continue
        other_high = sum(float(high_n[j]) for j in range(len(lower)) if j != i)
        other_low = sum(float(low_n[j]) for j in range(len(lower)) if j != i)
        denominator = float(low_n[i]) + other_high
        low[i] = float(low_n[i]) / denominator if denominator > 0 else 0.0
        denominator = float(high_n[i]) + other_low
        high[i] = float(high_n[i]) / denominator if denominator > 0 else float(high_n[i] > 0)
    return low, high


def build_admissibility_bounds(local_updates, prior_trust_state):
    flat = {cid: flatten_update(update) for cid, update in local_updates.items()}
    ids = list(flat)
    n = len(ids)
    if not n or any(not np.isfinite(vector).all() for vector in flat.values()):
        return {"valid": False, "intervals": {}}
    norms = np.array([np.linalg.norm(flat[cid]) for cid in ids], dtype=np.float64)
    if not np.isfinite(norms).all():
        return {"valid": False, "intervals": {}}

    eligible = norms <= NORM_GATE * float(np.median(norms))
    ema, _, previous, temporal = prior_trust_state
    reference = temporal if temporal is not None else previous
    direction = np.ones(n)
    if reference is not None and np.linalg.norm(reference) >= 1e-12:
        reference_norm = float(np.linalg.norm(reference)) + 1e-12
        for i, cid in enumerate(ids):
            cosine = float(np.dot(flat[cid], reference) / ((norms[i] + 1e-12) * reference_norm))
            direction[i] = min(1.0, max(0.0, cosine / DIRECTION_THRESHOLD))

    pad = 64.0 * (n + 1) * np.finfo(np.float64).eps
    lower = np.zeros(n)
    upper = np.ones(n)
    if ema is not None:
        history = np.array([ema.get(cid, 1.0 / n) for cid in ids], dtype=np.float64)
        if not np.isfinite(history).all() or (history < 0).any():
            return {"valid": False, "intervals": {}}
        denominator = EMA_ALPHA + (1.0 - EMA_ALPHA) * float(np.sum(history))
        if denominator > 1e-12:
            lower = np.maximum(0.0, (1.0 - EMA_ALPHA) * history / denominator - pad)
            upper = np.minimum(1.0, (EMA_ALPHA + (1.0 - EMA_ALPHA) * history) / denominator + pad)

    mask = eligible.astype(np.float64)
    fused = mask * direction
    lower_total = float(np.sum(lower * fused))
    upper_total = float(np.sum(upper * fused))
    candidates = []
    if upper_total > 1e-12 - pad:
        candidates.append(_normalized_intervals(lower, upper, fused))
    if lower_total <= 1e-12 + pad:
        if float(np.sum(upper * mask)) <= 1e-12:
            candidates.append((np.zeros(n), np.zeros(n)))
        else:
            fallback_low, fallback_high = _normalized_intervals(lower, upper, mask)
            if float(np.sum(lower * mask)) <= 1e-12 + pad:
                fallback_low = np.zeros(n)
            candidates.append((fallback_low, fallback_high))
    if not candidates:
        candidates.append((np.zeros(n), np.zeros(n)))

    lows = np.min(np.stack([low for low, _ in candidates]), axis=0)
    highs = np.max(np.stack([high for _, high in candidates]), axis=0)
    intervals = {
        cid: (max(0.0, float(lows[i]) - pad), min(1.0, float(highs[i]) + pad))
        if eligible[i] else (0.0, 0.0)
        for i, cid in enumerate(ids)
    }
    return {"valid": True, "intervals": intervals}


def verify_sampled_bounds(published_weights, bounds, num_clients, seed):
    count = max(1, int(num_clients * SAMPLE_RATIO))
    sampled = set(np.random.RandomState(seed).choice(num_clients, size=count, replace=False).tolist())
    values = list(published_weights.values())
    if not bounds["valid"] or any(not np.isfinite(value) or value < 0 for value in values):
        return True, sampled
    if abs(sum(values) - 1.0) > max(1, num_clients) * WEIGHT_TOLERANCE:
        return True, sampled
    for cid in sampled:
        low, high = bounds["intervals"].get(cid, (0.0, 0.0))
        value = published_weights.get(cid, 0.0)
        if value < low - WEIGHT_TOLERANCE or value > high + WEIGHT_TOLERANCE:
            return True, sampled
    return False, sampled
