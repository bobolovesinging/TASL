"""
实验二 v3: Byzantine Resilience Matrix + Unified Data-Layer Detection
====================================================================

6种聚合算法 × 6种攻击 × 2种数据分布(Non-IID/IID)

聚合算法: FedAvg, Multi-Krum, Trimmed Mean, FLTrust, TASL(Ours), TASL+FedPure
攻击类型: label_flip, sign_flip, gaussian_noise, fgsm, pgd, cw
数据分布: Non-IID (Dirichlet α=0.3), IID

v3 新增:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Unified data-layer detection (TrainStatsCollector + compute_unified_data_trust)
2. 三种新对抗攻击 (FGSM, PGD, CW)
3. 数据层信任评分与TASL梯度信任评分的乘法融合

TASL优化（vs旧版）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. trust_power=5 (旧版3) → 更锋利地压制低相似度更新
2. 权重帽 weight_cap=2.0/N → 防止单客户端权重飙升
3. EMA信任平滑 α=0.6 → 防止信任评分单轮突变
4. 范数偏差惩罚 → 检测异常L2范数更新
5. 迭代锚点精化 → 排除负相似度更新后重算锚点，更可靠
6. 最低方向阈值 min_cos=0.2 → 要求最低方向对齐才给权重
7. Gaussian Noise v2 → 更新范数等比例噪声(noise_scale=5.0)
   → 高维随机噪声在聚合时相互抵消，所有鲁棒方法表现接近（预期行为）
8. 时间一致性锚点(v3) → 前一轮聚合方向作为第三锚点，提高anchor稳健性
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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCKCHAIN_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CORE_DIR = os.path.join(BLOCKCHAIN_DIR, "core")
sys.path.insert(0, BLOCKCHAIN_DIR)
sys.path.insert(0, CORE_DIR)

from model import FedAvgCNN

# ── v3: Unified data-layer detection + advanced attacks ────────────
from unified_data_layer import TrainStatsCollector, compute_unified_data_trust, fuse_trust_with_gradient
from attacks_advanced import ADVANCED_ATTACK_CONFIGS, train_with_data_attack


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
# Non-IID Data Partitioning (Dirichlet)
# ═══════════════════════════════════════════════════════════════════════

class FederatedDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def dirichlet_partition(labels, n_clients, alpha=0.3, seed=42):
    """Dirichlet Non-IID partitioning"""
    rng = np.random.RandomState(seed)
    n_classes = len(np.unique(labels))
    client_indices = {i: [] for i in range(n_clients)}

    for c in range(n_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.repeat(alpha, n_clients))
        # Normalize
        proportions = proportions / proportions.sum()
        splits = (proportions * len(class_idx)).astype(int)
        # Distribute remainder
        remainder = len(class_idx) - splits.sum()
        for r in range(remainder):
            splits[r % n_clients] += 1

        start = 0
        for i in range(n_clients):
            client_indices[i].extend(class_idx[start:start + splits[i]].tolist())
            start += splits[i]

    return client_indices


def iid_partition(labels, n_clients, seed=42):
    """IID partitioning"""
    rng = np.random.RandomState(seed)
    n_classes = len(np.unique(labels))
    client_indices = {i: [] for i in range(n_clients)}

    for c in range(n_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        splits = np.array_split(class_idx, n_clients)
        for i in range(n_clients):
            client_indices[i].extend(splits[i].tolist())

    return client_indices


def prepare_data_loaders(n_clients=10, batch_size=128, alpha=0.3, split='non_iid', seed=42):
    """Prepare MNIST data loaders with specified partitioning"""
    temp_dir = os.path.join(BLOCKCHAIN_DIR, "data", "temp")
    os.makedirs(temp_dir, exist_ok=True)

    train_dataset = datasets.MNIST(root=temp_dir, train=True, download=True,
                                    transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root=temp_dir, train=False, download=True,
                                   transform=transforms.ToTensor())

    train_data = train_dataset.data.numpy().astype(np.float32) / 255.0
    train_labels = train_dataset.targets.numpy()
    if len(train_data.shape) == 3:
        train_data = np.expand_dims(train_data, axis=1)

    test_data = test_dataset.data.numpy().astype(np.float32) / 255.0
    if len(test_data.shape) == 3:
        test_data = np.expand_dims(test_data, axis=1)
    test_labels = test_dataset.targets.numpy()

    # Partition
    if split == 'non_iid':
        client_indices = dirichlet_partition(train_labels, n_clients, alpha=alpha, seed=seed)
    else:
        client_indices = iid_partition(train_labels, n_clients, seed=seed)

    client_loaders = []
    for i in range(n_clients):
        idx = client_indices[i]
        ds = FederatedDataset(train_data[idx], train_labels[idx])
        client_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))

    test_loader = DataLoader(FederatedDataset(test_data, test_labels),
                             batch_size=batch_size, shuffle=False)

    return client_loaders, test_loader


# ═══════════════════════════════════════════════════════════════════════
# Model helpers
# ═══════════════════════════════════════════════════════════════════════

def evaluate(model, test_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += model(x).argmax(dim=1).eq(y).sum().item()
            total += y.size(0)
    return 100.0 * correct / max(total, 1)


def evaluate_full(model, test_loader, device, attack='none', num_classes=10):
    """返回 accuracy 和 attack-specific ASR.
    ASR definition:
      - label_flip: fraction of samples predicted as flipped label (9-y)
      - sign_flip / gaussian_noise / fgsm / pgd / cw: ASR computed post-hoc via accuracy drop
    """
    model.eval()
    correct = total = 0
    flip_correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += preds.eq(y).sum().item()
            if attack == 'label_flip':
                flipped_targets = num_classes - 1 - y
                flip_correct += preds.eq(flipped_targets).sum().item()
            total += y.size(0)
    acc = 100.0 * correct / max(total, 1)
    asr = 100.0 * flip_correct / max(total, 1) if attack == 'label_flip' else 0.0
    return acc, asr


def state_sub(local_state, global_state):
    return {k: (local_state[k] - global_state[k]).detach()
            for k in global_state
            if isinstance(global_state[k], torch.Tensor) and global_state[k].is_floating_point()}


def state_add(global_state, update):
    return {k: (global_state[k] + update[k] if k in update else global_state[k])
            for k in global_state}


def flatten_update(update):
    chunks = [v.detach().cpu().reshape(-1).numpy() for v in update.values()
              if isinstance(v, torch.Tensor) and v.is_floating_point()]
    return np.concatenate(chunks).astype(np.float64) if chunks else np.zeros(1)


# ═══════════════════════════════════════════════════════════════════════
# Local Training with Attacks
# ═══════════════════════════════════════════════════════════════════════

def train_honest(global_state, loader, device, local_epochs=1, lr=0.02,
                 stats_collector=None, cid=None):
    """正常训练"""
    model = FedAvgCNN().to(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss_val = loss.item()
            loss.backward()
            optimizer.step()
            if stats_collector is not None and cid is not None:
                stats_collector.record_batch(cid, loss_val, model.parameters())
    return model.state_dict()


def train_label_flip(global_state, loader, device, local_epochs=1, lr=0.02,
                     num_classes=10, stats_collector=None, cid=None):
    """标签翻转攻击: y → num_classes - 1 - y"""
    model = FedAvgCNN().to(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_flip = num_classes - 1 - y
            optimizer.zero_grad()
            loss = criterion(model(x), y_flip)
            loss_val = loss.item()
            loss.backward()
            optimizer.step()
            if stats_collector is not None and cid is not None:
                stats_collector.record_batch(cid, loss_val, model.parameters())
    return model.state_dict()


def train_sign_flip(global_state, loader, device, local_epochs=1, lr=0.02,
                    stats_collector=None, cid=None):
    """符号翻转攻击: 正常训练后反转更新方向 (Δ → -Δ)"""
    model = FedAvgCNN().to(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss_val = loss.item()
            loss.backward()
            optimizer.step()
            if stats_collector is not None and cid is not None:
                stats_collector.record_batch(cid, loss_val, model.parameters())
    local_state = model.state_dict()
    # 反转更新方向: local_state = global_state - (local_state - global_state) = 2*global - local
    return {k: (2 * global_state[k].to(device) - v) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
            for k, v in local_state.items()}


def train_gaussian_noise(global_state, loader, device, local_epochs=1, lr=0.02,
                         noise_scale=5.0, stats_collector=None, cid=None):
    """高斯噪声攻击: 正常训练后添加与更新范数等比例的噪声

    噪声L2范数 = noise_scale × 诚实更新L2范数
    noise_scale=5.0 表示噪声范数是诚实更新的5倍

    注: 在高维空间中，随机噪声方向与梯度方向近似正交，
    因此即使大幅噪声对模型精度影响有限。这是数学性质，
    非参数问题 — 所有鲁棒聚合方法在Gaussian Noise下表现接近。
    """
    model = FedAvgCNN().to(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss_val = loss.item()
            loss.backward()
            optimizer.step()
            if stats_collector is not None and cid is not None:
                stats_collector.record_batch(cid, loss_val, model.parameters())
    local_state = model.state_dict()

    # 计算诚实更新的L2范数和参数总数
    total_sq = 0.0
    n_params = 0
    for k, v in local_state.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            delta = v - global_state[k].to(device)
            total_sq += delta.detach().norm().item() ** 2
            n_params += v.numel()
    honest_update_norm = total_sq ** 0.5

    # 噪声标准差: 使 E[||noise||²] = (noise_scale * honest_update_norm)²
    noise_std_per_param = noise_scale * honest_update_norm / (n_params ** 0.5)

    noisy_state = {}
    for k, v in local_state.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            noise = torch.randn_like(v) * noise_std_per_param
            noisy_state[k] = v + noise
        else:
            noisy_state[k] = v
    return noisy_state


# ── v3: 新的对抗攻击训练函数 ─────────────────────────────────────────

def train_fgsm(global_state, loader, device, local_epochs=1, lr=0.02,
               stats_collector=None, cid=None):
    """FGSM攻击: 单步快速梯度符号攻击"""
    model = FedAvgCNN().to(device)
    model.load_state_dict(global_state)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    avg_loss, avg_acc = train_with_data_attack(
        model, loader, opt, crit, device,
        attack_type='fgsm', attack_config={'epsilon': 0.1},
        stats_collector=stats_collector, cid=cid)
    return model.state_dict()


def train_pgd(global_state, loader, device, local_epochs=1, lr=0.02,
              stats_collector=None, cid=None):
    """PGD攻击: 多步迭代投影梯度下降"""
    model = FedAvgCNN().to(device)
    model.load_state_dict(global_state)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    avg_loss, avg_acc = train_with_data_attack(
        model, loader, opt, crit, device,
        attack_type='pgd', attack_config={'epsilon': 0.1, 'alpha': 0.01, 'steps': 10, 'random_start': True},
        stats_collector=stats_collector, cid=cid)
    return model.state_dict()


def train_cw(global_state, loader, device, local_epochs=1, lr=0.02,
             stats_collector=None, cid=None):
    """CW攻击: Carlini & Wagner L2攻击"""
    model = FedAvgCNN().to(device)
    model.load_state_dict(global_state)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    avg_loss, avg_acc = train_with_data_attack(
        model, loader, opt, crit, device,
        attack_type='cw', attack_config={'c': 1.0, 'kappa': 0.0, 'steps': 20, 'lr': 0.01},
        stats_collector=stats_collector, cid=cid)
    return model.state_dict()



def train_min_max(global_state, loader, device, local_epochs=1, lr=0.02,
                  stats_collector=None, cid=None):
    model = FedAvgCNN().to(device)
    model.load_state_dict(global_state)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss_item = loss.item()
            loss.backward()
            opt.step()
            if stats_collector is not None and cid is not None:
                stats_collector.record_batch(cid, loss_item, model.parameters())
    state = model.state_dict()
    for k, v in state.items():
        if v.is_floating_point():
            perturbation = torch.randn_like(v) * 0.3 * v.std()
            state[k] = global_state[k].to(device) + perturbation
    return state


def train_scaling(global_state, loader, device, local_epochs=1, lr=0.02,
                   stats_collector=None, cid=None):
    model = FedAvgCNN().to(device)
    model.load_state_dict(global_state)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss_item = loss.item()
            loss.backward()
            opt.step()
            if stats_collector is not None and cid is not None:
                stats_collector.record_batch(cid, loss_item, model.parameters())
    state = model.state_dict()
    lambda_scale = 5.0
    for k, v in state.items():
        if v.is_floating_point():
            delta = v - global_state[k].to(device)
            state[k] = global_state[k].to(device) + lambda_scale * delta
    return state


ATTACK_TRAIN_FNS = {
    'label_flip': train_label_flip,
    'sign_flip': train_sign_flip,
    'gaussian_noise': train_gaussian_noise,
    'fgsm': train_fgsm,
    'pgd': train_pgd,
    'cw': train_cw,
    'min_max': train_min_max,
    'scaling': train_scaling,
}


# ═══════════════════════════════════════════════════════════════════════
# Aggregation Algorithms
# ═══════════════════════════════════════════════════════════════════════

def fedavg_aggregate(updates, weights=None):
    """FedAvg: 加权平均"""
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
    """Multi-Krum: 选择得分最低的m个梯度取平均"""
    ids = list(flat_updates_dict.keys())
    n = len(ids)
    if n <= 2 * f + 2:
        m = max(1, n - f)
    else:
        m = n - f - 2

    # Pairwise distances
    vecs = {cid: flat_updates_dict[cid] for cid in ids}
    dists = {}
    for i in ids:
        dists[i] = {}
        for j in ids:
            if i != j:
                dists[i][j] = float(np.linalg.norm(vecs[i] - vecs[j]))

    # Scores: sum of k nearest distances
    k = max(1, n - f - 2)
    scores = {}
    for i in ids:
        sorted_d = sorted(dists[i].values())
        scores[i] = sum(sorted_d[:k])

    # Select top-m
    best_ids = sorted(scores, key=scores.get)[:m]
    selected = {cid: updates[cid] for cid in best_ids}
    return fedavg_aggregate(selected)


def trimmed_mean_aggregate(updates, beta=0.2):
    """Trimmed Mean: 每个维度去掉两端beta比例后取平均"""
    first = list(updates.values())[0]
    n = len(updates)
    trim = max(0, min(int(n * beta), (n - 1) // 2))
    agg = {}
    for k in first:
        if isinstance(first[k], torch.Tensor) and first[k].is_floating_point():
            stacked = torch.stack([updates[cid][k] for cid in updates])
            sorted_v, _ = torch.sort(stacked, dim=0)
            if trim > 0:
                trimmed = sorted_v[trim:n - trim]
            else:
                trimmed = sorted_v
            agg[k] = torch.mean(trimmed, dim=0)
        else:
            agg[k] = first[k]
    return agg


def fltrust_aggregate(flat_updates_dict, updates, device):
    """FLTrust: 基于root dataset的信任评分"""
    ids = list(flat_updates_dict.keys())
    n = len(ids)

    # Compute root gradient (mean of all updates)
    stacked = np.stack([flat_updates_dict[cid] for cid in ids])
    root = np.mean(stacked, axis=0)
    root_norm = np.linalg.norm(root) + 1e-12
    root_unit = root / root_norm

    # Compute trust scores
    trust_scores = {}
    for cid in ids:
        v = flat_updates_dict[cid]
        v_norm = np.linalg.norm(v) + 1e-12
        cos_sim = float(np.dot(v, root) / (v_norm * root_norm))
        trust_scores[cid] = max(0.0, cos_sim)

    # Normalize
    sum_t = sum(trust_scores.values())
    if sum_t > 1e-12:
        trust_weights = {cid: t / sum_t for cid, t in trust_scores.items()}
    else:
        trust_weights = {cid: 1.0 / n for cid in ids}

    return fedavg_aggregate(updates, trust_weights)


# ═══════════════════════════════════════════════════════════════════════
# TASL Trust Scoring (v2: 改进版 — 双锚点+范数门控+自适应阈值)
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# FedPure-style Detection for TASL Fusion
# ═══════════════════════════════════════════════════════════════════════

def fedpure_style_detection(flat_updates, corr_threshold=0.3):
    """
    Simplified FedPure anomaly detection on gradient vectors.
    Computes pairwise cosine correlation; low-consensus clients flagged.
    Returns {cid: score} where 1.0=clean, 0.0=malicious.
    """
    ids = list(flat_updates.keys())
    n = len(ids)
    if n <= 2:
        return {cid: 1.0 for cid in ids}

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

    median_sims = {}
    for i in ids:
        median_sims[i] = float(np.median(list(sims[i].values())))

    sim_values = sorted(median_sims.values())
    q25 = np.percentile(sim_values, 25)

    scores = {}
    for cid in ids:
        ms = median_sims[cid]
        if ms < corr_threshold:
            scores[cid] = 0.0
        elif ms < q25:
            scores[cid] = ms / max(q25, 0.01)
        else:
            scores[cid] = 1.0
    return scores


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
    data_trust=None,
):
    """
    TASL v3 改进信任评分 (在v2基础上增加时间一致性锚点 + 数据层信任融合):
    1. 双锚点: spatial median + pairwise median consensus
    2. 范数门控: L2范数超过median 2.5倍直接清零
    3. 迭代锚点精化 (排除负相似度更新后重算)
    4. 自适应cos阈值: 用cos分布的IQR自适应
    5. Cubic Trust Scoring (power=5)
    6. 权重硬帽 1.5/N
    7. EMA平滑 α=0.7
    8. Cos历史惩罚: 连续低cos客户端降权
    9. [v3新增] 时间一致性锚点: 使用前一轮聚合更新方向作为第三锚点
   10. [v3新增] 数据层信任融合: 将unified_data_layer检测结果乘入信任评分
    """
    n = len(flat_updates)
    ids = list(flat_updates.keys())
    if n == 0:
        return {}, {}, None

    # ── Step 1: Compute norms & norm gating ────────────────────────────
    norms = {cid: np.linalg.norm(v) for cid, v in flat_updates.items()}
    median_norm = np.median(list(norms.values()))
    norm_gate = median_norm * 2.5  # Hard gate: norm > 2.5x median = zero trust

    # ── Step 2: Spatial median anchor ──────────────────────────────────
    stacked = np.stack(list(flat_updates.values()), axis=0)
    anchor = np.median(stacked, axis=0)
    anchor_norm = np.linalg.norm(anchor) + 1e-12

    # ── Step 3: Pairwise consensus scoring ─────────────────────────────
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

    # Use median consensus to create a second anchor
    consensus_vals = list(pairwise_medians.values())
    median_consensus = np.median(consensus_vals)
    high_consensus_ids = [cid for cid in ids if pairwise_medians[cid] >= median_consensus]

    if len(high_consensus_ids) >= max(2, n // 2):
        consensus_stacked = np.stack([flat_updates[cid] for cid in high_consensus_ids])
        consensus_anchor = np.median(consensus_stacked, axis=0)
        anchor = 0.6 * anchor + 0.4 * consensus_anchor
        anchor_norm = np.linalg.norm(anchor) + 1e-12

    # ── Step 4: Iterative anchor refinement ────────────────────────────
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

    # ── Step 4.5: Temporal anchor blending (v3 key improvement) ──────
    if temporal_anchor is not None:
        temporal_norm = np.linalg.norm(temporal_anchor) + 1e-12
        direction_agreement = float(np.dot(anchor, temporal_anchor) / (anchor_norm * temporal_norm))
        if direction_agreement > 0.3:
            anchor = 0.6 * anchor + 0.4 * temporal_anchor
            anchor_norm = np.linalg.norm(anchor) + 1e-12

    # ── Step 5: Compute trust scores with adaptive threshold ───────────
    cos_sims = {}
    for cid, v in flat_updates.items():
        v_norm = np.linalg.norm(v) + 1e-12
        cos_sim = float(np.dot(v, anchor) / (v_norm * anchor_norm))
        cos_sims[cid] = cos_sim

    # Adaptive threshold: use IQR of cos distribution
    cos_vals = sorted(cos_sims.values())
    if len(cos_vals) >= 4:
        q1 = np.percentile(cos_vals, 25)
        q3 = np.percentile(cos_vals, 75)
        iqr = q3 - q1
        adaptive_threshold = max(min_cos_threshold, q1 - 1.0 * iqr)
        adaptive_threshold = min(adaptive_threshold, 0.4)
    else:
        adaptive_threshold = min_cos_threshold

    # Compute raw trust scores
    raw_scores = {}
    for cid in ids:
        cos_sim = cos_sims[cid]

        # Hard gate: norm too large
        if norms[cid] > norm_gate:
            raw_scores[cid] = 0.0
            continue

        # Hard threshold: direction too different
        if cos_sim < adaptive_threshold:
            raw_scores[cid] = 0.0
            continue

        # Norm deviation penalty
        norm_ratio = norms[cid] / (median_norm + 1e-12)
        norm_penalty = np.exp(-norm_penalty_strength * abs(norm_ratio - 1.0))

        # Pairwise consensus bonus
        consensus = pairwise_medians.get(cid, 0.5)
        consensus_factor = max(0.01, consensus)

        # Combined: direction × norm × consensus
        raw_scores[cid] = (cos_sim ** trust_power) * norm_penalty * consensus_factor

    # ── Step 6: Cos history penalty ────────────────────────────────────
    if cos_history is not None and len(cos_history) >= 2:
        for cid in ids:
            recent_cos = [ch.get(cid, 0.5) for ch in cos_history[-2:]]
            if all(c < 0.3 for c in recent_cos):
                raw_scores[cid] *= 0.1

    # ── Step 7: Data-layer trust fusion (v3新增) ───────────────────
    # 将unified_data_layer检测结果乘入raw信任评分
    if data_trust is not None:
        for cid in ids:
            raw_scores[cid] *= data_trust.get(cid, 1.0)

    # ── Step 8: Normalize ──────────────────────────────────────────────
    sum_w = sum(raw_scores.values())
    if sum_w > 1e-12:
        trust_weights = {cid: w / sum_w for cid, w in raw_scores.items()}
    else:
        trust_weights = {cid: 1.0 / n for cid in ids}

    # ── Step 9: Weight cap (hard) ──────────────────────────────────────
    max_w = max_weight_ratio / n
    capped = {cid: min(w, max_w) for cid, w in trust_weights.items()}
    cap_sum = sum(capped.values())
    if cap_sum > 1e-12:
        trust_weights = {cid: w / cap_sum for cid, w in capped.items()}

    # ── Step 10: EMA smoothing ─────────────────────────────────────────
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
    algo: str,
    attack: str,
    client_loaders,
    test_loader,
    n_clients: int,
    n_byz: int,
    rounds: int,
    device: torch.device,
    local_epochs: int = 2,
    lr: float = 0.02,
    seed: int = 42,
    ema_alpha: float = 0.6,
    noise_scale: float = 5.0,
) -> Dict:
    """Run a single experiment configuration.

    Returns dict with: best_acc, avg_acc, last_acc, acc_history
    """
    set_seed(seed)
    byzantine_set = set(range(n_byz))
    global_model = FedAvgCNN().to(device)
    global_state = global_model.state_dict()

    acc_history = []
    asr_history = []
    ema_weights = None  # For TASL EMA
    cos_history = []   # For TASL cos history penalty
    prev_anchor = None # For TASL anchor stability
    temporal_anchor = None  # For TASL temporal consistency (v3)

    # ── v3: Unified data-layer stats collector ──────────────────────
    stats_collector = TrainStatsCollector()

    for r in range(1, rounds + 1):
        stats_collector.reset()
        local_updates = {}
        flat_updates = {}

        for cid in range(n_clients):
            if cid in byzantine_set and attack != 'none':
                train_fn = ATTACK_TRAIN_FNS.get(attack, train_honest)
                if attack == 'gaussian_noise':
                    local_state = train_fn(global_state, client_loaders[cid], device,
                                           local_epochs=local_epochs, lr=lr,
                                           noise_scale=noise_scale,
                                           stats_collector=stats_collector, cid=cid)
                else:
                    local_state = train_fn(global_state, client_loaders[cid], device,
                                           local_epochs=local_epochs, lr=lr,
                                           stats_collector=stats_collector, cid=cid)
            else:
                local_state = train_honest(global_state, client_loaders[cid], device,
                                            local_epochs=local_epochs, lr=lr,
                                            stats_collector=stats_collector, cid=cid)

            upd = state_sub(local_state, global_state)
            local_updates[cid] = upd
            flat_updates[cid] = flatten_update(upd)

        # ── v3: Data-layer trust computation ────────────────────────
        round_stats = stats_collector.get_stats()
        data_trust, diag = compute_unified_data_trust(round_stats, n_clients)

        # ── Aggregation ────────────────────────────────────────────────
        if algo == 'fedavg':
            agg_update = fedavg_aggregate(local_updates)

        elif algo == 'multi_krum':
            agg_update = multi_krum_aggregate(flat_updates, local_updates, f=n_byz)

        elif algo == 'trimmed_mean':
            agg_update = trimmed_mean_aggregate(local_updates, beta=0.2)

        elif algo == 'fltrust':
            agg_update = fltrust_aggregate(flat_updates, local_updates, device)

        elif algo == 'tasl_fused':
            # Layer 0: FedPure detection
            fedpure_scores = fedpure_style_detection(flat_updates, corr_threshold=0.3)
            # Layer 1: TASL trust scoring (with data-layer fusion)
            trust_weights, cos_sims, new_anchor = compute_tasl_trust_weights(
                flat_updates, trust_power=5.0, max_weight_ratio=1.5,
                min_cos_threshold=0.2, norm_penalty_strength=0.8,
                refine_anchor=True, ema_weights=ema_weights, ema_alpha=0.7,
                cos_history=cos_history, prev_anchor=prev_anchor,
                temporal_anchor=temporal_anchor,
                data_trust=data_trust,
            )
            # Layer 2: Multiplicative fusion (fp_score=0 → weight=0)
            n_cl = n_clients
            fused_raw = {}
            for cid in range(n_clients):
                fp = fedpure_scores.get(cid, 1.0)
                tsl = trust_weights.get(cid, 0.0) * n_cl  # denormalize
                fused_raw[cid] = fp * max(0.0, tsl)
            sum_f = sum(fused_raw.values())
            if sum_f > 1e-12:
                trust_weights = {cid: w / sum_f for cid, w in fused_raw.items()}
            max_w = 1.5 / n_cl
            capped = {cid: min(w, max_w) for cid, w in trust_weights.items()}
            cap_sum = sum(capped.values())
            if cap_sum > 1e-12:
                trust_weights = {cid: w / cap_sum for cid, w in capped.items()}
            ema_weights = dict(trust_weights)
            cos_history.append(dict(cos_sims))
            prev_anchor = new_anchor
            agg_update = fedavg_aggregate(local_updates, trust_weights)
            agg_flat = flatten_update(agg_update)
            if np.linalg.norm(agg_flat) > 1e-12:
                temporal_anchor = agg_flat

        elif algo == 'tasl':
            # v3: TASL with data-layer trust fusion
            trust_weights, cos_sims, new_anchor = compute_tasl_trust_weights(
                flat_updates,
                trust_power=5.0,
                max_weight_ratio=1.5,
                min_cos_threshold=0.2,
                norm_penalty_strength=0.8,
                refine_anchor=True,
                ema_weights=ema_weights,
                ema_alpha=0.7,
                cos_history=cos_history,
                prev_anchor=prev_anchor,
                temporal_anchor=temporal_anchor,
                data_trust=data_trust,
            )
            ema_weights = dict(trust_weights)
            cos_history.append(dict(cos_sims))
            prev_anchor = new_anchor
            agg_update = fedavg_aggregate(local_updates, trust_weights)

            # Update temporal anchor
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
            if algo == 'tasl' and ema_weights:
                byz_avg_w = np.mean([ema_weights.get(c, 0) for c in byzantine_set])
            asr_str = f" asr={asr:.2f}%" if attack == 'label_flip' else ""
            print(f"  [{algo}/{attack}] R{r:02d} acc={acc:.2f}%"
                  + (f" byz_w={byz_avg_w:.4f}" if algo == 'tasl' else "") + asr_str)

        # ── v3: Print data-layer diagnostics ────────────────────────
        if r % 10 == 0:
            byz_trust_avg = np.mean([data_trust.get(c, 1.0) for c in byzantine_set])
            honest_ids = [c for c in range(n_clients) if c not in byzantine_set]
            honest_trust_avg = np.mean([data_trust.get(c, 1.0) for c in honest_ids]) if honest_ids else 1.0
            print(f"    [DataLayer] loss_median={diag.get('median_loss', 0):.4f} "
                  f"grad_median={diag.get('median_grad_norm', 0):.4f} "
                  f"loss_trust_min={diag.get('loss_trust_min', 1):.4f} "
                  f"grad_trust_min={diag.get('grad_trust_min', 1):.4f} "
                  f"byz_trust={byz_trust_avg:.4f} "
                  f"honest_trust={honest_trust_avg:.4f}")

    best_acc = max(acc_history)
    avg_acc = np.mean(acc_history)
    last_acc = acc_history[-1]
    best_asr = max(asr_history) if asr_history else 0.0
    avg_asr = float(np.mean(asr_history)) if asr_history else 0.0

    return {
        'best_acc': best_acc,
        'avg_acc': avg_acc,
        'last_acc': last_acc,
        'acc_history': acc_history,
        'asr_history': asr_history,
        'best_asr': best_asr,
        'avg_asr': avg_asr,
    }


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════

def plot_heatmap(save_path, results_df, metric='best_acc'):
    """Plot MA/ASR heatmap: algorithms × attacks"""
    import pandas as pd
    algos = ['fedavg', 'multi_krum', 'trimmed_mean', 'fltrust', 'tasl', 'tasl_fused']
    algo_labels = ['FedAvg', 'Multi-Krum', 'Trimmed Mean', 'FLTrust', 'TASL (Ours)', 'TASL+FedPure']
    attacks = ['label_flip', 'sign_flip', 'gaussian_noise', 'fgsm', 'pgd', 'cw', 'min_max', 'scaling']
    attack_labels = ['Label Flip', 'Sign Flip', 'Gaussian Noise', 'FGSM', 'PGD', 'CW']

    fig, ax = plt.subplots(figsize=(14, 5))
    data = np.zeros((len(algos), len(attacks)))
    for i, algo in enumerate(algos):
        for j, atk in enumerate(attacks):
            row = results_df[(results_df['algo'] == algo) & (results_df['attack'] == atk)]
            if not row.empty:
                data[i, j] = row[metric].values[0]

    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels(attack_labels, fontsize=10)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algo_labels, fontsize=11)

    for i in range(len(algos)):
        for j in range(len(attacks)):
            val = data[i, j]
            color = 'white' if val < 40 or val > 80 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Best Accuracy (%)', fontsize=12)
    metric_label = 'Best Accuracy' if metric == 'best_acc' else 'Avg Accuracy'
    ax.set_title(f'Byzantine Resilience Matrix — {metric_label} (40% Byzantine, Non-IID α=0.3)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved heatmap: {save_path}")


def plot_curves(save_path, all_results, attack, rounds):
    """Plot accuracy curves for one attack type"""
    algo_colors = {
        'fedavg': '#E53935',
        'multi_krum': '#1E88E5',
        'trimmed_mean': '#FB8C00',
        'fltrust': '#8E24AA',
        'tasl': '#43A047',
        'tasl_fused': '#FF6F00',
    }
    algo_labels = {
        'fedavg': 'FedAvg',
        'multi_krum': 'Multi-Krum',
        'trimmed_mean': 'Trimmed Mean',
        'fltrust': 'FLTrust',
        'tasl': 'TASL (Ours)',
        'tasl_fused': 'TASL+FedPure',
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    for algo, res in all_results.items():
        if attack in res:
            history = res[attack]['acc_history']
            ax.plot(range(1, rounds + 1), history,
                    label=algo_labels[algo],
                    color=algo_colors[algo],
                    linewidth=2.0, alpha=0.9)

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    attack_label = {'label_flip': 'Label Flip', 'sign_flip': 'Sign Flip',
                    'gaussian_noise': 'Gaussian Noise', 'fgsm': 'FGSM',
                    'pgd': 'PGD', 'cw': 'CW'}
    ax.set_title(f'Convergence under {attack_label.get(attack, attack)} Attack (40% Byzantine, Non-IID α=0.3)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved curves: {save_path}")


def plot_bar_comparison(save_path, results_df, metric='best_acc'):
    """Bar chart comparing all algorithms across all attacks"""
    import pandas as pd
    algos = ['fedavg', 'multi_krum', 'trimmed_mean', 'fltrust', 'tasl', 'tasl_fused']
    algo_labels = ['FedAvg', 'Multi-\nKrum', 'Trimmed\nMean', 'FLTrust', 'TASL\n(Ours)', 'TASL+\nFedPure']
    attacks = ['label_flip', 'sign_flip', 'gaussian_noise', 'fgsm', 'pgd', 'cw', 'min_max', 'scaling']
    attack_labels = ['Label Flip', 'Sign Flip', 'Gaussian Noise', 'FGSM', 'PGD', 'CW']
    algo_colors = ['#E53935', '#1E88E5', '#FB8C00', '#8E24AA', '#43A047', '#FF6F00']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes = axes.flatten()
    for j, atk in enumerate(attacks):
        ax = axes[j]
        vals = []
        for algo in algos:
            row = results_df[(results_df['algo'] == algo) & (results_df['attack'] == atk)]
            vals.append(row[metric].values[0] if not row.empty else 0)

        bars = ax.bar(range(len(algos)), vals, color=algo_colors, edgecolor='white', linewidth=1.5)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algo_labels, fontsize=8)
        ax.set_title(attack_labels[j], fontsize=12, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    metric_label = 'Best Accuracy (%)' if metric == 'best_acc' else 'Avg Accuracy (%)'
    axes[0].set_ylabel(metric_label, fontsize=12)
    fig.suptitle(f'Byzantine Resilience Comparison — {metric_label} (40% Byzantine, Non-IID α=0.3)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved bar chart: {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp2 v3: Byzantine Resilience Matrix + Data-Layer Detection")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--byzantine-ratio", type=float, default=0.4)
    parser.add_argument("--ema-alpha", type=float, default=0.6)
    parser.add_argument("--alpha", type=float, default=0.3, help="Dirichlet alpha for Non-IID")
    parser.add_argument("--split", type=str, default='non_iid', choices=['iid', 'non_iid'])
    parser.add_argument("--noise-scale", type=float, default=5.0, help="Gaussian noise scale (noise L2 = N * honest update L2)")
    parser.add_argument("--attack", type=str, default=None,
                        choices=['label_flip', 'sign_flip', 'gaussian_noise', 'fgsm', 'pgd', 'cw', 'scaling', 'min_max'],
                        help="Only run specified attack (default: run all)")
    # Quick mode for testing
    parser.add_argument("--quick", action='store_true', help="Quick test: 5 clients, 10 rounds")
    parser.add_argument("--skip-clean", action='store_true', help="Skip clean baseline")
    parser.add_argument("--output", type=str, default=None, help="Output directory override")
    args = parser.parse_args()

    if args.quick:
        args.clients = 5
        args.rounds = 10
        args.local_epochs = 1
        args.byzantine_ratio = 0.4

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_clients = int(args.clients)
    n_byz = max(1, int(n_clients * args.byzantine_ratio))
    rounds = int(args.rounds)

    algos = ['fedavg', 'multi_krum', 'trimmed_mean', 'fltrust', 'tasl', 'tasl_fused']
    attacks = ['label_flip', 'sign_flip', 'gaussian_noise', 'fgsm', 'pgd', 'cw', 'min_max', 'scaling']
    if args.attack:
        attacks = [args.attack]

    print("=" * 70)
    print(f"[Exp2 v3] Byzantine Resilience Matrix + Data-Layer Detection")
    print(f"  {n_clients} clients, {n_byz} Byzantine ({args.byzantine_ratio:.0%})")
    print(f"  Split: {args.split} (alpha={args.alpha})")
    print(f"  {rounds} rounds, local_epochs={args.local_epochs}")
    print(f"  Algorithms: {algos}")
    print(f"  Attacks: {attacks}")
    print(f"  TASL: trust_power=5, weight_cap=1.5/n, ema_alpha=0.7")
    print(f"  TASL: norm_penalty=0.8, min_cos=0.2(adaptive), dual_anchor+norm_gate")
    print(f"  TASL v3: temporal_anchor + data_layer_fusion (unified loss+grad_norm detection)")
    print(f"  Gaussian Noise: noise_scale={args.noise_scale} (noise L2 = {args.noise_scale}x honest update L2)")
    print(f"  Device: {device}")
    print("=" * 70)

    # Load data
    client_loaders, test_loader = prepare_data_loaders(
        n_clients=n_clients,
        batch_size=int(args.batch_size),
        alpha=float(args.alpha),
        split=args.split,
        seed=args.seed,
    )

    # ── Run clean baseline first (skip if --skip-clean) ────────────────
    if args.skip_clean:
        print("\n>>> Clean Baseline skipped (--skip-clean) <<<")
        clean_result = None
    else:
        print("\n>>> Clean Baseline (no attack) <<<")
        clean_result = run_single(
            'fedavg', 'none', client_loaders, test_loader,
            n_clients=n_clients, n_byz=0, rounds=rounds, device=device,
            local_epochs=int(args.local_epochs), lr=float(args.lr),
            seed=args.seed,
        )
        print(f"  Clean: best={clean_result['best_acc']:.2f}%, avg={clean_result['avg_acc']:.2f}%")

    # ── Run all combinations ───────────────────────────────────────────
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
                seed=args.seed + hash(algo + attack) % 1000,
                ema_alpha=float(args.ema_alpha),
                noise_scale=float(args.noise_scale),
            )

            all_results[algo][attack] = result
            # 计算 ASR: label_flip 直接用 asr; 其他攻击用精度下降比
            if attack == 'label_flip':
                final_asr = result['avg_asr']
            else:
                # ASR = 1 - (attacked_avg_acc / clean_avg_acc)，越高攻击越成功
                if clean_result is not None:
                    final_asr = max(0, 100.0 * (1 - result['avg_acc'] / max(clean_result['avg_acc'], 1e-6)))
                else:
                    final_asr = float("nan")
            csv_rows.append({
                'algo': algo,
                'attack': attack,
                'split': args.split,
                'byzantine_ratio': args.byzantine_ratio,
                'best_acc': result['best_acc'],
                'avg_acc': result['avg_acc'],
                'last_acc': result['last_acc'],
                'asr': final_asr,
            })

            asr_str = f", asr={final_asr:.2f}%" if attack != 'none' else ""
            print(f"  => best={result['best_acc']:.2f}%, avg={result['avg_acc']:.2f}%, last={result['last_acc']:.2f}%{asr_str}")

    # ── Save CSV ───────────────────────────────────────────────────────
    if args.output:
        results_dir = args.output
    else:
        results_dir = os.path.join(BLOCKCHAIN_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    attack_tag = args.attack if args.attack else "all"
    csv_path = os.path.join(results_dir, f"exp2_mnist_v3_seed{args.seed}_{attack_tag}.csv")

    # Add clean baseline row when it was actually computed.
    if clean_result is not None:
        csv_rows.append({
            'algo': 'clean_baseline', 'attack': 'none', 'split': args.split,
            'byzantine_ratio': 0.0, 'best_acc': clean_result['best_acc'],
            'avg_acc': clean_result['avg_acc'], 'last_acc': clean_result['last_acc'],
            'asr': 0.0,
        })

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['algo', 'attack', 'split', 'byzantine_ratio',
                                                'best_acc', 'avg_acc', 'last_acc', 'asr'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved CSV: {csv_path}")

    # ── Print summary table ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"[Exp2 v3] Summary — {args.split} (α={args.alpha}), {args.byzantine_ratio:.0%} Byzantine")
    print("=" * 70)
    print(f"{'Attack':<18s} {'FedAvg':>8s} {'M-Krum':>8s} {'T-Mean':>8s} {'FLTrust':>8s} {'TASL':>8s} {'TASL+FP':>8s}")
    print("-" * 85)
    for attack in attacks:
        row_str = f"{attack:<18s}"
        for algo in algos:
            val = all_results[algo][attack]['best_acc']
            # Highlight best
            all_vals = [all_results[a][attack]['best_acc'] for a in algos]
            is_best = val == max(all_vals)
            marker = " *" if is_best else ""
            row_str += f" {val:>6.2f}{marker:<2s}"
        print(row_str)
    if clean_result is not None:
        print(f"\nClean Baseline: best={clean_result['best_acc']:.2f}%, avg={clean_result['avg_acc']:.2f}%")
    else:
        print("\nClean Baseline: skipped")

    # ── Plots ──────────────────────────────────────────────────────────
    import pandas as pd
    results_df = pd.DataFrame(csv_rows)

    try:
        suffix = f"_{args.split}"
        plot_heatmap(os.path.join(results_dir, f"exp2_mnist_v3_heatmap{suffix}.png"), results_df, 'best_acc')
        plot_bar_comparison(os.path.join(results_dir, f"exp2_mnist_v3_bar{suffix}.png"), results_df, 'best_acc')

        for attack in attacks:
            plot_curves(os.path.join(results_dir, f"exp2_mnist_v3_curves_{attack}{suffix}.png"),
                        all_results, attack, rounds)

        print(f"Saved Plots: {results_dir}")
    except Exception as e:
        print(f"[WARN] Plot generation skipped: {e}")


if __name__ == "__main__":
    main()
