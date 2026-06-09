# -*- coding: utf-8 -*-
"""
TASL Data-Layer Trust Module (MVP)
===================================
Extracts per-class skeleton prototypes from raw NTU-60 data,
computes cross-client consistency scores as data-layer trust priors.

Design:
  - Class-aware centroid extraction (from raw .npz, no augmentation)
  - Cross-client pairwise cosine similarity per shared class
  - Data trust = normalized median similarity → [0, 1]
  - For data-poisoning attacks (label_flip), trust → 0
  - For gradient-only attacks (sign_flip, gaussian_noise), trust → 1.0
"""

import os
import numpy as np


def load_raw_client_data(data_dir, cid):
    """
    Load raw (unaugmented) client data from .npz files.
    Returns: x (N,C,T,V,M), y (N,)
    """
    raw = np.load(
        os.path.join(data_dir, 'fedtrain', f'{cid}.npz'),
        allow_pickle=True
    )['data'].item()
    return raw['x'], raw['y'].flatten()


def extract_per_class_prototypes(data_dir, n_clients):
    """
    Extract per-class mean skeleton centroids from raw client data.

    For each client and each class it owns:
      - Compute mean skeleton tensor (C, T, V, M)
      - Flatten to vector

    Returns:
      prototypes: {cid: {class_id: np.ndarray}}
      class_counts: {cid: {class_id: int}}
    """
    prototypes = {}
    class_counts = {}

    for cid in range(n_clients):
        x, y = load_raw_client_data(data_dir, cid)
        # x shape: (N, C, T, V, M) = (N, 3, 50, 25, 2)
        prototypes[cid] = {}
        class_counts[cid] = {}

        unique_classes = np.unique(y)
        for cls in unique_classes:
            cls = int(cls)
            mask = y == cls
            if mask.sum() == 0:
                continue
            # Mean skeleton for this class
            mean_skeleton = np.mean(x[mask], axis=0).astype(np.float32)
            prototypes[cid][cls] = mean_skeleton.flatten()
            class_counts[cid][cls] = int(mask.sum())

    return prototypes, class_counts


def compute_data_trust(prototypes, class_counts, n_clients, iqr_multiplier=2.0):
    """
    Compute data-layer trust scores via global-reference prototype comparison.

    Steps:
      1. Build a global reference prototype per class (average over all clients)
      2. Per client: avg cosine similarity to global reference per class
      3. IQR-based threshold: only clients below Q1 - k*IQR get low trust
         → handles Non-IID variance gracefully

    Args:
      prototypes: {cid: {class_id: flat_vector}}
      class_counts: {cid: {class_id: count}}
      n_clients: total number of clients
      iqr_multiplier: IQR multiplier for outlier fence (default 2.0, Tukey=1.5)

    Returns:
      data_trust: {cid: float}  trust scores in [0, 1]
    """
    # ── Step 1: Build global reference prototypes ──
    all_classes = set()
    for cid in range(n_clients):
        for cls in prototypes.get(cid, {}):
            all_classes.add(cls)

    global_proto = {}
    for cls in all_classes:
        cls_protos = []
        for cid in range(n_clients):
            if cls in prototypes.get(cid, {}):
                cls_protos.append(prototypes[cid][cls].astype(np.float64))
        if cls_protos:
            global_proto[cls] = np.mean(cls_protos, axis=0)

    # ── Step 2: Per-client similarity to global reference ──
    trust_raw = {}
    for cid in range(n_clients):
        sims = []
        for cls, proto_vec in prototypes.get(cid, {}).items():
            if cls not in global_proto:
                continue
            vi = proto_vec.astype(np.float64)
            v_ref = global_proto[cls]
            vi_norm = np.linalg.norm(vi) + 1e-12
            ref_norm = np.linalg.norm(v_ref) + 1e-12
            sim = float(np.dot(vi, v_ref) / (vi_norm * ref_norm))
            sims.append(sim)
        trust_raw[cid] = float(np.mean(sims)) if sims else 1.0

    # ── Step 3: IQR-based threshold (robust to Non-IID) ──
    scores = np.array(list(trust_raw.values()))
    q25 = float(np.percentile(scores, 25))
    q75 = float(np.percentile(scores, 75))
    iqr = q75 - q25
    lower_fence = q25 - iqr_multiplier * iqr

    data_trust = {}
    for cid in range(n_clients):
        score = trust_raw[cid]
        if score >= lower_fence:
            data_trust[cid] = 1.0  # In-distribution → full trust
        elif score <= 0:
            data_trust[cid] = 0.0  # Completely different → zero trust
        else:
            # Linear degradation below fence
            data_trust[cid] = score / max(0.01, lower_fence)

    return data_trust


def compute_data_trust_with_byz(data_trust, n_byz, honest_trust_threshold=0.5):
    """
    Adjust data trust scores based on Byzantine client set knowledge.
    Used only for logging/analysis — actual TASL fusion does NOT know n_byz.

    Returns diagnostic info about detection quality.
    """
    n_clients = len(data_trust)
    if n_byz == 0:
        return {"detection_rate": 0.0, "false_positive": 0.0}

    # Byzantine clients are first n_byz (by convention in our setup)
    byz_ids = set(range(n_byz))
    honest_ids = set(range(n_byz, n_clients))

    detected = sum(1 for c in byz_ids if data_trust.get(c, 1.0) < honest_trust_threshold)
    fp = sum(1 for c in honest_ids if data_trust.get(c, 1.0) < honest_trust_threshold)

    return {
        "detection_rate": detected / n_byz,
        "false_positive": fp / max(1, len(honest_ids)),
    }


# ════════════════ Quick test ════════════════
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--n-clients", type=int, default=10)
    p.add_argument("--n-byz", type=int, default=0)
    args = p.parse_args()

    print(f"Extracting prototypes from {args.data_dir}/fedtrain/...")
    protos, counts = extract_per_class_prototypes(args.data_dir, args.n_clients)

    # Report per-client class coverage
    for cid in range(args.n_clients):
        n_classes = len(counts.get(cid, {}))
        n_samples = sum(counts.get(cid, {}).values())
        print(f"  Client {cid}: {n_classes} classes, {n_samples} samples")

    print(f"\nComputing data-layer trust...")
    trust = compute_data_trust(protos, counts, args.n_clients)

    print("\nData Trust Scores:")
    for cid in range(args.n_clients):
        flag = " [BYZ]" if cid < args.n_byz else ""
        print(f"  Client {cid}: {trust.get(cid, 1.0):.4f}{flag}")

    if args.n_byz > 0:
        diag = compute_data_trust_with_byz(trust, args.n_byz)
        print(f"\nDetection: recall={diag['detection_rate']:.2f}, "
              f"FPR={diag['false_positive']:.2f}")
