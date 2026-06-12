# -*- coding: utf-8 -*-
"""
TASL Unified Data-Layer Trust Module
=====================================
General data-layer detection using training statistics (loss + gradient norm).
Works across ALL datasets (MNIST, CIFAR-10, NTU-60) and ALL model architectures.

Core insight: the local training process itself produces two universal signals:
  1. avg_train_loss   — poisoned data → high/abnormal loss
  2. avg_grad_norm    — poisoned data → unusual gradient magnitude

These are dataset-agnostic and model-agnostic.
"""

import numpy as np

# ── Training statistics collection ────────────────────────────

class TrainStatsCollector:
    """
    Lightweight collector for per-client training statistics.
    Reset at the start of each round, read after all clients finish.
    """
    def __init__(self):
        self.stats = {}  # {cid: {'loss_sum': float, 'loss_count': int, 'grad_norms': []}}

    def reset(self):
        self.stats = {}

    def record_batch(self, cid, loss_value, model_params):
        """Record one batch's loss and gradient norms."""
        if cid not in self.stats:
            self.stats[cid] = {'loss_sum': 0.0, 'loss_count': 0, 'grad_norms': []}
        self.stats[cid]['loss_sum'] += float(loss_value)
        self.stats[cid]['loss_count'] += 1
        # Compute total gradient L2 norm for this batch
        total_norm = 0.0
        for p in model_params:
            if p.grad is not None:
                total_norm += float((p.grad.data.norm(2).item()) ** 2)
        self.stats[cid]['grad_norms'].append(total_norm ** 0.5)

    def get_stats(self):
        """Return aggregated per-client statistics."""
        result = {}
        for cid, data in self.stats.items():
            avg_loss = data['loss_sum'] / max(data['loss_count'], 1)
            avg_grad_norm = float(np.mean(data['grad_norms'])) if data['grad_norms'] else 0.0
            result[cid] = {'loss': avg_loss, 'grad_norm': avg_grad_norm}
        return result


# ── Unified data trust computation ─────────────────────────────

def compute_unified_data_trust(stats, n_clients, iqr_multiplier=2.0):
    """
    Compute data-layer trust scores from training statistics.

    Uses TWO signals (loss anomaly + grad_norm anomaly), fused by min().
    IQR-based thresholding tolerates Non-IID natural variation.

    Args:
      stats: {cid: {'loss': float, 'grad_norm': float}}
      n_clients: total number of clients
      iqr_multiplier: IQR multiplier for outlier fence (higher = more tolerant)

    Returns:
      data_trust: {cid: float} — trust scores in [0, 1]
      diagnosis: dict — diagnostic information about which signal triggered
    """
    if not stats or len(stats) < 2:
        return {c: 1.0 for c in range(n_clients)}, {}

    losses = np.array([stats.get(c, {'loss': 0}).get('loss', 0) for c in range(n_clients)])
    grad_norms = np.array([stats.get(c, {'grad_norm': 0}).get('grad_norm', 0) for c in range(n_clients)])

    def iqr_trust(values):
        """Convert IQR-based z-scores to trust [0, 1]. Only flag significant outliers."""
        med = float(np.median(values))
        q25 = float(np.percentile(values, 25))
        q75 = float(np.percentile(values, 75))
        iqr = q75 - q25
        if iqr < 1e-8:
            return np.ones_like(values)
        # Wait - we need to be careful about direction:
        # - Higher loss → more suspicious → lower trust
        # - Higher grad_norm → more suspicious → lower trust
        # So for values above upper fence: trust → 0
        # For values below: trust → 1
        upper_fence = q75 + iqr_multiplier * iqr
        scores = np.ones_like(values)
        for i, v in enumerate(values):
            if v > upper_fence:
                # Linear degradation from fence to 2x fence → 0
                scores[i] = max(0.0, 1.0 - (v - upper_fence) / max(upper_fence, 1e-8))
            # Also check: extremely LOW values could indicate gradient suppression
            # but this is rare, skip for MVP
        return scores

    loss_trust = iqr_trust(losses)
    grad_trust = iqr_trust(grad_norms)

    # Conservative fusion: min() — any signal triggers suspicion
    fused_trust = np.minimum(loss_trust, grad_trust)

    data_trust = {c: float(fused_trust[i]) for i, c in enumerate(range(n_clients))}

    diagnosis = {
        'median_loss': float(np.median(losses)),
        'median_grad_norm': float(np.median(grad_norms)),
        'loss_trust_min': float(np.min(loss_trust)),
        'grad_trust_min': float(np.min(grad_trust)),
    }

    return data_trust, diagnosis


# ── Integration helper ─────────────────────────────────────────

def fuse_trust_with_gradient(data_trust, gradient_weights, n_clients):
    """
    Multiplicative fusion: data_trust[c] * gradient_weight[c].
    Then re-normalize.
    """
    fused = {}
    for cid in range(n_clients):
        dt = data_trust.get(cid, 1.0)
        gw = gradient_weights.get(cid, 0.0)
        fused[cid] = dt * gw

    total = sum(fused.values())
    if total > 1e-12:
        fused = {c: w / total for c, w in fused.items()}
    else:
        fused = {c: 1.0 / n_clients for c in range(n_clients)}

    return fused


# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate 10 clients: 4 Byzantine (label_flip) have higher loss
    np.random.seed(42)
    stats = {}
    for c in range(10):
        if c < 4:  # Byzantine
            stats[c] = {'loss': np.random.normal(2.5, 0.3),
                        'grad_norm': np.random.normal(15.0, 3.0)}
        else:  # Honest
            stats[c] = {'loss': np.random.normal(0.3, 0.1),
                        'grad_norm': np.random.normal(8.0, 1.0)}

    trust, diag = compute_unified_data_trust(stats, 10)
    print("Simulated data trust (4 Byzantine clients 0-3):")
    for c in range(10):
        flag = " [BYZ]" if c < 4 else ""
        print(f"  Client {c}: trust={trust[c]:.4f}  "
              f"loss={stats[c]['loss']:.2f}  "
              f"grad_norm={stats[c]['grad_norm']:.2f}{flag}")
    print(f"\nDiagnosis: {diag}")
    print(f"Byzantine avg trust: {np.mean([trust[c] for c in range(4)]):.4f}")
    print(f"Honest avg trust: {np.mean([trust[c] for c in range(6)]):.4f}")
