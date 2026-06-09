"""
实验二 NTU-60: Byzantine Resilience Matrix — ST-GCN

5种聚合算法 × 3种攻击
模型: ST-GCN (3.1M params, 11.82 MB)
数据: NTU-RGB+D 60 (Non-IID Dirichlet α=0.1, 10客户端)
攻击: label_flip, sign_flip, gaussian_noise

与 MNIST/CIFAR-10 实验二配置一致:
- 10 客户端, 4 Byzantine (40%)
- TASL: trust_power=5, weight_cap=1.5/N, ema_alpha=0.7, dual_anchor+norm_gate
- ST-GCN: in_channels=3, num_class=60, layout=ntu-rgb+d, strategy=spatial
"""
import csv
import os
import random
import sys
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCKCHAIN_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CORE_DIR = os.path.join(BLOCKCHAIN_DIR, "core")
STGCN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))), "st-gcn")
sys.path.insert(0, BLOCKCHAIN_DIR)
sys.path.insert(0, CORE_DIR)
sys.path.insert(0, STGCN_DIR)

from net.st_gcn import Model as STGCN


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

NUM_CLASSES = 60
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))), "datasets", "dir0.1")


# ═══════════════════════════════════════════════════════════════════════
# Seed & Device
# ═══════════════════════════════════════════════════════════════════════

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════
# NTU-60 Data Loading
# ═══════════════════════════════════════════════════════════════════════

def prepare_data_loaders(n_clients=10, batch_size=64, seed=42):
    """Load pre-partitioned NTU-60 federated data from dir0.1"""
    client_loaders = []
    total_train = 0

    for cid in range(n_clients):
        npz_path = os.path.join(DATA_DIR, "fedtrain", f"{cid}.npz")
        raw = np.load(npz_path, allow_pickle=True)["data"].item()
        x = torch.from_numpy(raw["x"])  # (N, 3, 50, 25, 2) float32
        y = torch.from_numpy(raw["y"].flatten())  # (N,) int64
        ds = TensorDataset(x, y)
        client_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))
        total_train += len(ds)

    # Test data (use client 0's test set — all share the same global test)
    npz_path = os.path.join(DATA_DIR, "fedtest", "0.npz")
    raw = np.load(npz_path, allow_pickle=True)["data"].item()
    test_x = torch.from_numpy(raw["x"])
    test_y = torch.from_numpy(raw["y"].flatten())
    test_ds = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"  Data loaded: {total_train} train, {len(test_ds)} test, {n_clients} clients")
    return client_loaders, test_loader


# ═══════════════════════════════════════════════════════════════════════
# Model helpers
# ═══════════════════════════════════════════════════════════════════════

def create_model(device):
    """Create ST-GCN model for NTU-60"""
    model = STGCN(
        in_channels=3,
        num_class=NUM_CLASSES,
        graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
        edge_importance_weighting=True,
        dropout=0.5,
    )
    return model.to(device)


def evaluate(model, test_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += model(x).argmax(dim=1).eq(y).sum().item()
            total += y.size(0)
    return 100.0 * correct / max(total, 1)


def evaluate_full(model, test_loader, device, attack="none", num_classes=NUM_CLASSES):
    """返回 accuracy 和 ASR (label_flip专用)"""
    model.eval()
    correct = total = 0
    flip_correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += preds.eq(y).sum().item()
            if attack == "label_flip":
                flipped_targets = num_classes - 1 - y
                flip_correct += preds.eq(flipped_targets).sum().item()
            total += y.size(0)
    acc = 100.0 * correct / max(total, 1)
    asr = 100.0 * flip_correct / max(total, 1) if attack == "label_flip" else 0.0
    return acc, asr


def state_sub(local_state, global_state):
    return {k: (local_state[k].cpu() - global_state[k].cpu()).detach()
            for k in global_state
            if isinstance(global_state[k], torch.Tensor) and global_state[k].is_floating_point()}


def state_add(global_state, update):
    result = {}
    for k in global_state:
        if k in update and isinstance(global_state[k], torch.Tensor) and global_state[k].is_floating_point():
            result[k] = global_state[k] + update[k]
        else:
            result[k] = global_state[k]
    return result


def flatten_update(update):
    chunks = [v.detach().cpu().reshape(-1).numpy() for v in update.values()
              if isinstance(v, torch.Tensor) and v.is_floating_point()]
    return np.concatenate(chunks).astype(np.float64) if chunks else np.zeros(1)


# ═══════════════════════════════════════════════════════════════════════
# Local Training with Attacks
# ═══════════════════════════════════════════════════════════════════════

def train_honest(global_state, loader, device, local_epochs=5, lr=0.01, momentum=0.9, weight_decay=1e-4):
    """正常训练"""
    model = create_model(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
    return {k: v.cpu() for k, v in model.state_dict().items()}


def train_label_flip(global_state, loader, device, local_epochs=5, lr=0.01, momentum=0.9, weight_decay=1e-4, num_classes=NUM_CLASSES):
    """标签翻转攻击: y → num_classes - 1 - y"""
    model = create_model(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_flip = num_classes - 1 - y
            optimizer.zero_grad()
            criterion(model(x), y_flip).backward()
            optimizer.step()
    return {k: v.cpu() for k, v in model.state_dict().items()}


def train_sign_flip(global_state, loader, device, local_epochs=5, lr=0.01, momentum=0.9, weight_decay=1e-4):
    """符号翻转攻击: 正常训练后反转更新方向 (Δ → -Δ)"""
    model = create_model(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
    local_state = model.state_dict()
    # 反转: local = global - (local - global) = 2*global - local
    return {k: (2 * global_state[k].to(device) - v) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
            for k, v in local_state.items()}


def train_gaussian_noise(global_state, loader, device, local_epochs=5, lr=0.01, momentum=0.9, weight_decay=1e-4, noise_scale=5.0):
    """高斯噪声攻击: 正常训练后添加与更新范数等比例的噪声"""
    model = create_model(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
    local_state = model.state_dict()

    # 计算诚实更新L2范数
    total_sq = 0.0
    n_params = 0
    for k, v in local_state.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            delta = v - global_state[k].to(device)
            total_sq += delta.detach().norm().item() ** 2
            n_params += v.numel()
    honest_update_norm = total_sq ** 0.5
    noise_std_per_param = noise_scale * honest_update_norm / (n_params ** 0.5)

    noisy_state = {}
    for k, v in local_state.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            noise = torch.randn_like(v) * noise_std_per_param
            noisy_state[k] = (v + noise).cpu()
        else:
            noisy_state[k] = v
    return noisy_state


ATTACK_TRAIN_FNS = {
    "label_flip": train_label_flip,
    "sign_flip": train_sign_flip,
    "gaussian_noise": train_gaussian_noise,
}


# ═══════════════════════════════════════════════════════════════════════
# Aggregation Algorithms (与 MNIST exp2 完全一致)
# ═══════════════════════════════════════════════════════════════════════

def fedavg_aggregate(updates, weights=None):
    n = len(updates)
    if weights is None:
        weights = {cid: 1.0 / n for cid in updates}
    w_sum = sum(weights.values())
    if w_sum < 1e-12:
        weights = {cid: 1.0 / n for cid in updates}
        w_sum = 1.0
    normed = {cid: w / w_sum for cid, w in weights.items()}
    agg = {k: torch.zeros_like(v) for k, v in updates[next(iter(normed))].items()}
    for cid, w in normed.items():
        for k in agg:
            agg[k] += updates[cid][k] * w
    return agg


def multi_krum_aggregate(flat_updates_dict, updates, f):
    ids = list(flat_updates_dict.keys())
    n = len(ids)
    m = max(1, n - f - 2) if n > 2 * f + 2 else max(1, n - f)
    k = max(1, n - f - 2)

    dists = {i: {} for i in ids}
    for i in ids:
        for j in ids:
            if i != j:
                dists[i][j] = float(np.linalg.norm(flat_updates_dict[i] - flat_updates_dict[j]))

    scores = {}
    for i in ids:
        sorted_d = sorted(dists[i].values())
        scores[i] = sum(sorted_d[:k])

    best_ids = sorted(scores, key=scores.get)[:m]
    selected = {cid: updates[cid] for cid in best_ids}
    return fedavg_aggregate(selected)


def trimmed_mean_aggregate(updates, beta=0.2):
    first = list(updates.values())[0]
    n = len(updates)
    trim = max(0, min(int(n * beta), (n - 1) // 2))
    agg = {}
    for k in first:
        if isinstance(first[k], torch.Tensor) and first[k].is_floating_point():
            stacked = torch.stack([updates[cid][k] for cid in updates])
            sorted_v, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_v[trim:n - trim] if trim > 0 else sorted_v
            agg[k] = torch.mean(trimmed, dim=0)
        else:
            agg[k] = first[k]
    return agg


def fltrust_aggregate(flat_updates_dict, updates, device):
    ids = list(flat_updates_dict.keys())
    n = len(ids)
    stacked = np.stack([flat_updates_dict[cid] for cid in ids])
    root = np.mean(stacked, axis=0)
    root_norm = np.linalg.norm(root) + 1e-12

    trust_scores = {}
    for cid in ids:
        v = flat_updates_dict[cid]
        v_norm = np.linalg.norm(v) + 1e-12
        cos_sim = float(np.dot(v, root) / (v_norm * root_norm))
        trust_scores[cid] = max(0.0, cos_sim)

    sum_t = sum(trust_scores.values())
    trust_weights = {cid: t / sum_t for cid, t in trust_scores.items()} if sum_t > 1e-12 else {cid: 1.0 / n for cid in ids}
    return fedavg_aggregate(updates, trust_weights)


# ═══════════════════════════════════════════════════════════════════════
# TASL Trust Scoring (v3: 双锚点+范数门控+自适应阈值+temporal anchor)
# ═══════════════════════════════════════════════════════════════════════

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

    # Step 1: Norm gating
    norms = {cid: np.linalg.norm(v) for cid, v in flat_updates.items()}
    median_norm = np.median(list(norms.values()))
    norm_gate = median_norm * 2.5

    # Step 2: Spatial median anchor
    stacked = np.stack(list(flat_updates.values()), axis=0)
    anchor = np.median(stacked, axis=0)
    anchor_norm = np.linalg.norm(anchor) + 1e-12

    # Step 3: Pairwise consensus
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

    # Step 4: Iterative anchor refinement
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

    # Step 4.5: Temporal anchor blending (v3)
    if temporal_anchor is not None:
        temporal_norm = np.linalg.norm(temporal_anchor) + 1e-12
        direction_agreement = float(np.dot(anchor, temporal_anchor) / (anchor_norm * temporal_norm))
        if direction_agreement > 0.3:
            anchor = 0.6 * anchor + 0.4 * temporal_anchor
            anchor_norm = np.linalg.norm(anchor) + 1e-12

    # Step 5: Trust scores with adaptive threshold
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

    # Step 6: Cos history penalty
    if cos_history is not None and len(cos_history) >= 2:
        for cid in ids:
            recent_cos = [ch.get(cid, 0.5) for ch in cos_history[-2:]]
            if all(c < 0.3 for c in recent_cos):
                raw_scores[cid] *= 0.1

    # Step 7: Normalize
    sum_w = sum(raw_scores.values())
    trust_weights = {cid: w / sum_w for cid, w in raw_scores.items()} if sum_w > 1e-12 else {cid: 1.0 / n for cid in ids}

    # Step 8: Weight cap
    max_w = max_weight_ratio / n
    capped = {cid: min(w, max_w) for cid, w in trust_weights.items()}
    cap_sum = sum(capped.values())
    if cap_sum > 1e-12:
        trust_weights = {cid: w / cap_sum for cid, w in capped.items()}

    # Step 9: EMA smoothing
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


# ═══════════════════════════════════════════════════════════════════════
# Single experiment run
# ═══════════════════════════════════════════════════════════════════════

def run_single(
    algo, attack, client_loaders, test_loader,
    n_clients, n_byz, rounds, device,
    local_epochs=5, lr=0.01, momentum=0.9, weight_decay=1e-4,
    seed=42, noise_scale=5.0,
):
    set_seed(seed)
    byzantine_set = set(range(n_byz))
    global_model = create_model(device)
    global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}

    acc_history = []
    asr_history = []
    ema_weights = None
    cos_history = []
    prev_anchor = None
    temporal_anchor = None

    for r in range(1, rounds + 1):
        local_updates = {}
        flat_updates = {}

        for cid in range(n_clients):
            if cid in byzantine_set and attack != "none":
                train_fn = ATTACK_TRAIN_FNS.get(attack, train_honest)
                if attack == "gaussian_noise":
                    local_state = train_fn(global_state, client_loaders[cid], device,
                                           local_epochs=local_epochs, lr=lr,
                                           momentum=momentum, weight_decay=weight_decay,
                                           noise_scale=noise_scale)
                else:
                    local_state = train_fn(global_state, client_loaders[cid], device,
                                           local_epochs=local_epochs, lr=lr,
                                           momentum=momentum, weight_decay=weight_decay)
            else:
                local_state = train_honest(global_state, client_loaders[cid], device,
                                           local_epochs=local_epochs, lr=lr,
                                           momentum=momentum, weight_decay=weight_decay)

            upd = state_sub(local_state, global_state)
            local_updates[cid] = upd
            flat_updates[cid] = flatten_update(upd)

        # Aggregation
        if algo == "fedavg":
            agg_update = fedavg_aggregate(local_updates)
        elif algo == "multi_krum":
            agg_update = multi_krum_aggregate(flat_updates, local_updates, f=n_byz)
        elif algo == "trimmed_mean":
            agg_update = trimmed_mean_aggregate(local_updates, beta=0.2)
        elif algo == "fltrust":
            agg_update = fltrust_aggregate(flat_updates, local_updates, device)
        elif algo == "tasl":
            trust_weights, cos_sims, new_anchor = compute_tasl_trust_weights(
                flat_updates, trust_power=5.0, max_weight_ratio=1.5,
                min_cos_threshold=0.2, norm_penalty_strength=0.8,
                refine_anchor=True, ema_weights=ema_weights, ema_alpha=0.7,
                cos_history=cos_history, prev_anchor=prev_anchor,
                temporal_anchor=temporal_anchor,
            )
            ema_weights = dict(trust_weights)
            cos_history.append(dict(cos_sims))
            prev_anchor = new_anchor
            agg_update = fedavg_aggregate(local_updates, trust_weights)

            agg_flat = flatten_update(agg_update)
            if np.linalg.norm(agg_flat) > 1e-12:
                temporal_anchor = agg_flat
        else:
            raise ValueError(f"Unknown algo: {algo}")

        global_state = state_add(global_state, agg_update)
        global_model.load_state_dict(global_state)
        acc, asr = evaluate_full(global_model, test_loader, device, attack=attack)
        acc_history.append(acc)
        asr_history.append(asr)

        if r % 10 == 0 or r == 1:
            byz_avg_w = 0.0
            if algo == "tasl" and ema_weights:
                byz_avg_w = np.mean([ema_weights.get(c, 0) for c in byzantine_set])
            asr_str = f" asr={asr:.2f}%" if attack == "label_flip" else ""
            print(f"  [{algo}/{attack}] R{r:02d} acc={acc:.2f}%"
                  + (f" byz_w={byz_avg_w:.4f}" if algo == "tasl" else "") + asr_str)

        # Free GPU memory
        if r % 5 == 0:
            torch.cuda.empty_cache()

    best_acc = max(acc_history)
    avg_acc = np.mean(acc_history)
    last_acc = acc_history[-1]
    best_asr = max(asr_history) if asr_history else 0.0
    avg_asr = float(np.mean(asr_history)) if asr_history else 0.0

    return {
        "best_acc": best_acc, "avg_acc": avg_acc, "last_acc": last_acc,
        "acc_history": acc_history, "asr_history": asr_history,
        "best_asr": best_asr, "avg_asr": avg_asr,
    }


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════

def plot_heatmap(save_path, results_df, metric="best_acc"):
    import pandas as pd
    algos = ["fedavg", "multi_krum", "trimmed_mean", "fltrust", "tasl"]
    algo_labels = ["FedAvg", "Multi-Krum", "Trimmed Mean", "FLTrust", "TASL (Ours)"]
    attacks = ["label_flip", "sign_flip", "gaussian_noise"]
    attack_labels = ["Label Flip", "Sign Flip", "Gaussian Noise"]

    fig, ax = plt.subplots(figsize=(10, 5))
    data = np.zeros((len(algos), len(attacks)))
    for i, algo in enumerate(algos):
        for j, atk in enumerate(attacks):
            row = results_df[(results_df["algo"] == algo) & (results_df["attack"] == atk)]
            if not row.empty:
                data[i, j] = row[metric].values[0]

    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels(attack_labels, fontsize=12)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algo_labels, fontsize=12)

    for i in range(len(algos)):
        for j in range(len(attacks)):
            val = data[i, j]
            color = "white" if val < 40 or val > 80 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Best Accuracy (%)", fontsize=12)
    metric_label = "Best Accuracy" if metric == "best_acc" else "Avg Accuracy"
    ax.set_title(f"NTU-60 ST-GCN — {metric_label} (40% Byzantine, Non-IID α=0.1)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved heatmap: {save_path}")


def plot_bar_comparison(save_path, results_df, metric="best_acc"):
    import pandas as pd
    algos = ["fedavg", "multi_krum", "trimmed_mean", "fltrust", "tasl"]
    algo_labels = ["FedAvg", "Multi-\nKrum", "Trimmed\nMean", "FLTrust", "TASL\n(Ours)"]
    attacks = ["label_flip", "sign_flip", "gaussian_noise"]
    attack_labels = ["Label Flip", "Sign Flip", "Gaussian Noise"]
    algo_colors = ["#E53935", "#1E88E5", "#FB8C00", "#8E24AA", "#43A047"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for j, atk in enumerate(attacks):
        ax = axes[j]
        vals = []
        for algo in algos:
            row = results_df[(results_df["algo"] == algo) & (results_df["attack"] == atk)]
            vals.append(row[metric].values[0] if not row.empty else 0)

        bars = ax.bar(range(len(algos)), vals, color=algo_colors, edgecolor="white", linewidth=1.5)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algo_labels, fontsize=9)
        ax.set_title(attack_labels[j], fontsize=13, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    metric_label = "Best Accuracy (%)" if metric == "best_acc" else "Avg Accuracy (%)"
    axes[0].set_ylabel(metric_label, fontsize=12)
    fig.suptitle(f"NTU-60 ST-GCN — {metric_label} (40% Byzantine, Non-IID α=0.1)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp2 NTU-60: Byzantine Resilience Matrix (ST-GCN)")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--byzantine-ratio", type=float, default=0.4)
    parser.add_argument("--noise-scale", type=float, default=5.0)
    parser.add_argument("--attack", type=str, default=None,
                        choices=["label_flip", "sign_flip", "gaussian_noise"])
    parser.add_argument("--quick", action="store_true", help="Quick test: 5 clients, 10 rounds")
    args = parser.parse_args()

    if args.quick:
        args.clients = 5
        args.rounds = 10
        args.local_epochs = 1
        args.byzantine_ratio = 0.4

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_clients = int(args.clients)
    n_byz = max(1, int(n_clients * args.byzantine_ratio))
    rounds = int(args.rounds)

    algos = ["fedavg", "multi_krum", "trimmed_mean", "fltrust", "tasl"]
    attacks = ["label_flip", "sign_flip", "gaussian_noise"]
    if args.attack:
        attacks = [args.attack]

    print("=" * 70)
    print(f"[Exp2 NTU-60] Byzantine Resilience Matrix — ST-GCN")
    print(f"  {n_clients} clients, {n_byz} Byzantine ({args.byzantine_ratio:.0%})")
    print(f"  Non-IID α=0.1 (Dirichlet)")
    print(f"  {rounds} rounds, local_epochs={args.local_epochs}, lr={args.lr}")
    print(f"  Algorithms: {algos}")
    print(f"  Attacks: {attacks}")
    print(f"  Model: ST-GCN (3.1M params, 11.82 MB)")
    print(f"  TASL: trust_power=5, weight_cap=1.5/n, ema_alpha=0.7, dual_anchor+norm_gate")
    print(f"  Device: {device}")
    print("=" * 70)

    # Load data
    client_loaders, test_loader = prepare_data_loaders(
        n_clients=n_clients, batch_size=int(args.batch_size), seed=args.seed,
    )

    # Clean baseline
    print("\n>>> Clean Baseline (no attack) <<<")
    clean_result = run_single(
        "fedavg", "none", client_loaders, test_loader,
        n_clients=n_clients, n_byz=0, rounds=rounds, device=device,
        local_epochs=int(args.local_epochs), lr=float(args.lr),
        momentum=float(args.momentum), weight_decay=float(args.weight_decay),
        seed=args.seed,
    )
    print(f"  Clean: best={clean_result['best_acc']:.2f}%, avg={clean_result['avg_acc']:.2f}%")

    # Run all combinations
    all_results = {algo: {} for algo in algos}
    csv_rows = []
    total_runs = len(algos) * len(attacks)
    run_idx = 0

    for algo in algos:
        for attack in attacks:
            run_idx += 1
            print(f"\n>>> [{run_idx}/{total_runs}] {algo} × {attack} <<<")

            result = run_single(
                algo, attack, client_loaders, test_loader,
                n_clients=n_clients, n_byz=n_byz, rounds=rounds, device=device,
                local_epochs=int(args.local_epochs), lr=float(args.lr),
                momentum=float(args.momentum), weight_decay=float(args.weight_decay),
                seed=args.seed + hash(algo + attack) % 1000,
                noise_scale=float(args.noise_scale),
            )

            all_results[algo][attack] = result
            if attack == "label_flip":
                final_asr = result["avg_asr"]
            else:
                final_asr = max(0, 100.0 * (1 - result["avg_acc"] / max(clean_result["avg_acc"], 1e-6)))

            csv_rows.append({
                "algo": algo, "attack": attack, "split": "non_iid",
                "byzantine_ratio": args.byzantine_ratio,
                "best_acc": result["best_acc"], "avg_acc": result["avg_acc"],
                "last_acc": result["last_acc"], "asr": final_asr,
            })

            asr_str = f", asr={final_asr:.2f}%" if attack != "none" else ""
            print(f"  => best={result['best_acc']:.2f}%, avg={result['avg_acc']:.2f}%, last={result['last_acc']:.2f}%{asr_str}")

    # Save CSV
    results_dir = os.path.join(BLOCKCHAIN_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "exp2_ntu60_robustness_data.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "attack", "split", "byzantine_ratio",
                                                "best_acc", "avg_acc", "last_acc", "asr"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Add clean baseline
    csv_rows.append({
        "algo": "clean_baseline", "attack": "none", "split": "non_iid",
        "byzantine_ratio": 0.0, "best_acc": clean_result["best_acc"],
        "avg_acc": clean_result["avg_acc"], "last_acc": clean_result["last_acc"],
        "asr": 0.0,
    })

    # Summary table
    print("\n" + "=" * 70)
    print(f"[Exp2 NTU-60] Summary — Non-IID (α=0.1), {args.byzantine_ratio:.0%} Byzantine")
    print("=" * 70)
    print(f"{'Attack':<18s} {'FedAvg':>8s} {'M-Krum':>8s} {'T-Mean':>8s} {'FLTrust':>8s} {'TASL':>8s}")
    print("-" * 70)
    for attack in attacks:
        row_str = f"{attack:<18s}"
        for algo in algos:
            val = all_results[algo][attack]["best_acc"]
            all_vals = [all_results[a][attack]["best_acc"] for a in algos]
            is_best = val == max(all_vals)
            marker = " *" if is_best else ""
            row_str += f" {val:>6.2f}{marker:<2s}"
        print(row_str)
    print(f"\nClean Baseline: best={clean_result['best_acc']:.2f}%, avg={clean_result['avg_acc']:.2f}%")

    # Plots
    import pandas as pd
    results_df = pd.DataFrame(csv_rows)

    plot_heatmap(os.path.join(results_dir, "exp2_ntu60_heatmap.png"), results_df, "best_acc")
    plot_bar_comparison(os.path.join(results_dir, "exp2_ntu60_bar.png"), results_df, "best_acc")

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Plots: {results_dir}")
    print("DONE.")


if __name__ == "__main__":
    main()
