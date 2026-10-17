"""
实验二（CIFAR-10 补充）: Byzantine Resilience Matrix — CIFAR-10 鲁棒性全景对比

5种聚合算法 × 3种攻击 × CIFAR-10 (32×32 RGB, Non-IID α=0.3)

聚合算法: FedAvg, Multi-Krum, Trimmed Mean, FLTrust, TASL(Ours)
攻击类型: label_flip, sign_flip, gaussian_noise
数据集: CIFAR-10 (10类, 50000 train / 10000 test)
分区: Dirichlet Non-IID (α=0.3)

TASL 配置（与 MNIST 实验相同）:
  trust_power=5, weight_cap=1.5/N, ema_alpha=0.7
  dual_anchor (spatial median + pairwise median consensus)
  norm_gating (2.5x median), adaptive_cos_threshold (IQR)
  temporal_anchor (v3: 前一轮聚合方向作为第三锚点)

模型: CifarResNet18 (ResNet-18 + GroupNorm，适合 Non-IID 联邦学习)
  输入: 3×32×32
  4个 stage (64→64→128→256→512 channels), 各含 BasicBlock
  参数量: ~11M
  使用 GroupNorm 替代 BatchNorm（BN 统计量在 Non-IID 时不稳定）

训练配置:
  LR=0.1, CosineAnnealing scheduler, SGD+momentum+wd
  100 rounds, local_epochs=5
  目标 Clean Baseline ≥ 70%
"""
import csv
import math
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Path setup ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCKCHAIN_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, BLOCKCHAIN_DIR)


# ═══════════════════════════════════════════════════════════════════════
# CIFAR-10 ResNet-18 模型 (GroupNorm，适合 Non-IID 联邦学习)
# ═══════════════════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
    """ResNet-18 基本残差块: 2×(Conv → GN → ReLU) + skip connection"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(min(4, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(min(4, out_channels), out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(min(4, out_channels), out_channels),
            )

    def forward(self, x):
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class CifarCNN(nn.Module):
    """
    CIFAR-10 ResNet-18，专为联邦学习设计:
    - 使用 GroupNorm 代替 BatchNorm（BN 统计量在 Non-IID 时不稳定）
    - 标准残差连接加速收敛，尤其适合 FL 的 Non-IID 设置
    - 参数量 ~11M，与近年 FL 文献一致 (OptiGradTrust, FLTrust 等)
    - 输入: 3×32×32 → 输出: 10 类

    架构 (标准 ResNet-18 适配 CIFAR-10):
      Conv stem(3→64, 3×3) → Stage1(64, 2 blocks)
      → Stage2(128, 2 blocks, stride=2) → Stage3(256, 2 blocks, stride=2)
      → Stage4(512, 2 blocks, stride=2) → GAP → FC(512→10)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64

        # Stem: 3×3 conv (适配 CIFAR-10 的 32×32，不用 7×7)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.GroupNorm(4, 64),
            nn.ReLU(inplace=True),
        )
        # 不加 MaxPool（CIFAR-10 分辨率只有 32×32，池化后太小）

        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


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
# Data — CIFAR-10 Non-IID Partitioning
# ═══════════════════════════════════════════════════════════════════════

def dirichlet_partition(labels, n_clients, alpha=0.3, seed=42):
    """Dirichlet Non-IID 分区"""
    rng = np.random.RandomState(seed)
    n_classes = len(np.unique(labels))
    client_indices = {i: [] for i in range(n_clients)}
    for c in range(n_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.repeat(alpha, n_clients))
        proportions = proportions / proportions.sum()
        splits = (proportions * len(class_idx)).astype(int)
        remainder = len(class_idx) - splits.sum()
        for r in range(remainder):
            splits[r % n_clients] += 1
        start = 0
        for i in range(n_clients):
            client_indices[i].extend(class_idx[start:start + splits[i]].tolist())
            start += splits[i]
    return client_indices


def _load_cifar10_pickle(data_dir):
    """
    手动读取 CIFAR-10 pickle 文件，绕过 torchvision 在 Python 3.14+ / NumPy 2.4 的兼容性问题。
    自动下载（若尚未下载）。
    返回: (train_data, train_labels, test_data, test_labels)
      - train_data: (50000, 32, 32, 3), uint8
      - test_data:  (10000, 32, 32, 3), uint8
    """
    import pickle as pk
    import tarfile
    import urllib.request

    # ── 自动下载 ──────────────────────────────────────────────────────
    batches_dir = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.isdir(batches_dir):
        url     = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
        if not os.path.isfile(tar_path):
            print(f"[Data] Downloading CIFAR-10 from {url} ...")
            urllib.request.urlretrieve(url, tar_path)
            print("[Data] Download complete")
        print("[Data] Extracting ...")
        with tarfile.open(tar_path, "r:gz") as t:
            t.extractall(data_dir)
        print("[Data] Extraction complete")

    def _load_batch(fname):
        with open(fname, 'rb') as f:
            d = pk.load(f, encoding='bytes')
        data   = np.array(d[b'data']).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N, 32, 32, 3)
        labels = np.array(d[b'labels'])
        return data, labels

    # ── 训练集 (5 batches) ────────────────────────────────────────────
    train_data_list, train_label_list = [], []
    for i in range(1, 6):
        d, l = _load_batch(os.path.join(batches_dir, f"data_batch_{i}"))
        train_data_list.append(d)
        train_label_list.append(l)
    train_data   = np.concatenate(train_data_list, axis=0)   # (50000, 32, 32, 3)
    train_labels = np.concatenate(train_label_list, axis=0)  # (50000,)

    # ── 测试集 ────────────────────────────────────────────────────────
    test_data, test_labels = _load_batch(os.path.join(batches_dir, "test_batch"))

    return train_data, train_labels, test_data, test_labels


def _cifar10_normalize(img_uint8):
    """
    uint8 H×W×C [0,255] → float32 C×H×W, 已归一化
    归一化参数: mean=(0.4914,0.4822,0.4465), std=(0.2023,0.1994,0.2010)
    """
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
    img  = img_uint8.astype(np.float32) / 255.0   # H×W×C
    img  = (img - mean) / std                       # H×W×C
    img  = img.transpose(2, 0, 1)                  # C×H×W
    return img


class Cifar10TrainDataset(Dataset):
    """
    CIFAR-10 训练集，含随机裁剪 + 水平翻转增强（纯 numpy 实现，无 torchvision 依赖）
    """
    def __init__(self, data, labels):
        self.data   = data    # (N, 32, 32, 3), uint8
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx].copy()  # (32, 32, 3), uint8

        # 随机水平翻转
        if np.random.rand() > 0.5:
            img = img[:, ::-1, :].copy()

        # 随机裁剪 (padding=4)
        padded = np.pad(img, ((4, 4), (4, 4), (0, 0)), mode='reflect')  # (40, 40, 3)
        ty = np.random.randint(0, 8)
        tx = np.random.randint(0, 8)
        img = padded[ty:ty+32, tx:tx+32, :]  # (32, 32, 3)

        img_tensor = torch.from_numpy(_cifar10_normalize(img))  # (3, 32, 32)
        return img_tensor, self.labels[idx]


class Cifar10TestDataset(Dataset):
    """CIFAR-10 测试集，仅归一化"""
    def __init__(self, data, labels):
        # 预先归一化，加速 evaluation
        normalized = np.stack([_cifar10_normalize(data[i]) for i in range(len(data))], axis=0)
        self.data   = torch.from_numpy(normalized)  # (N, 3, 32, 32)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def prepare_cifar10_loaders(n_clients=10, batch_size=128, alpha=0.3, seed=42):
    """
    准备 CIFAR-10 数据加载器
    - 使用自定义 pickle 加载，兼容 Python 3.14+ / NumPy 2.4
    - 训练集: Dirichlet Non-IID 分区（α=0.3）+ 随机裁剪/翻转增强
    - 测试集: 完整 10000 样本，仅归一化
    """
    data_dir = os.path.join(BLOCKCHAIN_DIR, "data", "cifar10")
    os.makedirs(data_dir, exist_ok=True)

    # 加载数据
    train_data, train_labels, test_data, test_labels = _load_cifar10_pickle(data_dir)
    print(f"[Data] CIFAR-10 loaded: train={train_data.shape}, test={test_data.shape}")

    # Non-IID 分区
    client_indices = dirichlet_partition(train_labels, n_clients, alpha=alpha, seed=seed)

    client_loaders = []
    for i in range(n_clients):
        idx = np.array(client_indices[i])
        ds  = Cifar10TrainDataset(train_data[idx], train_labels[idx])
        client_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True,
                                          num_workers=0, pin_memory=False))

    test_ds     = Cifar10TestDataset(test_data, test_labels)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return client_loaders, test_loader


# ═══════════════════════════════════════════════════════════════════════
# Model helpers
# ═══════════════════════════════════════════════════════════════════════

def evaluate(model, test_loader, device):
    """返回准确率"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += model(x).argmax(dim=1).eq(y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / max(total, 1)


def evaluate_full(model, test_loader, device, attack='none', num_classes=10):
    """返回 accuracy 和 attack-specific ASR.
    ASR definition:
      - label_flip: fraction of samples predicted as flipped label (9-y)
      - sign_flip / gaussian_noise: ASR computed post-hoc via accuracy drop
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
    """计算 delta = local - global（仅浮点参数）"""
    return {k: (local_state[k] - global_state[k]).detach()
            for k in global_state
            if isinstance(global_state[k], torch.Tensor)
            and global_state[k].is_floating_point()}


def state_add(global_state, update):
    """global_state + update"""
    return {k: (global_state[k] + update[k] if k in update else global_state[k])
            for k in global_state}


def flatten_update(update):
    chunks = [v.detach().cpu().reshape(-1).numpy()
              for v in update.values()
              if isinstance(v, torch.Tensor) and v.is_floating_point()]
    return np.concatenate(chunks).astype(np.float64) if chunks else np.zeros(1)


def _get_lr_for_round(base_lr, current_round, total_rounds):
    """Cosine annealing: LR 按 cosine 曲线从 base_lr 衰减到 base_lr * 0.01"""
    if total_rounds <= 1:
        return base_lr
    progress = current_round / total_rounds
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════════════════
# Local Training with Attacks
# ═══════════════════════════════════════════════════════════════════════

def train_honest(global_state, loader, device, local_epochs=5, lr=0.1):
    """正常训练，使用 CosineAnnealing-adjusted LR"""
    model = CifarCNN().to(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
    return model.state_dict()


def train_label_flip(global_state, loader, device, local_epochs=5, lr=0.1, num_classes=10):
    """标签翻转攻击: y → num_classes - 1 - y"""
    model = CifarCNN().to(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y   = x.to(device), y.to(device)
            y_flip = num_classes - 1 - y
            optimizer.zero_grad()
            criterion(model(x), y_flip).backward()
            optimizer.step()
    return model.state_dict()


def train_sign_flip(global_state, loader, device, local_epochs=5, lr=0.1):
    """符号翻转攻击: 正常训练后反转更新方向 (Δ → -Δ)"""
    model = CifarCNN().to(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
    local_state = model.state_dict()
    # 反转更新: local = global - Δ = 2*global - local_trained
    return {k: (2 * global_state[k].to(device) - v)
            if isinstance(v, torch.Tensor) and v.is_floating_point() else v
            for k, v in local_state.items()}


def train_gaussian_noise(global_state, loader, device, local_epochs=5, lr=0.1, noise_scale=5.0):
    """
    高斯噪声攻击: 正常训练后添加与更新范数等比例的噪声
    noise_scale=5.0 → 噪声 L2 范数 = 5 × 诚实更新 L2 范数
    """
    model = CifarCNN().to(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
    local_state = model.state_dict()

    total_sq = 0.0
    n_params  = 0
    for k, v in local_state.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            delta      = v - global_state[k].to(device)
            total_sq  += delta.detach().norm().item() ** 2
            n_params   += v.numel()
    honest_norm        = total_sq ** 0.5
    noise_std_per_param = noise_scale * honest_norm / max(n_params ** 0.5, 1e-12)

    noisy_state = {}
    for k, v in local_state.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            noisy_state[k] = v + torch.randn_like(v) * noise_std_per_param
        else:
            noisy_state[k] = v
    return noisy_state


ATTACK_TRAIN_FNS = {
    'label_flip':    train_label_flip,
    'sign_flip':     train_sign_flip,
    'gaussian_noise': train_gaussian_noise,
}


# ═══════════════════════════════════════════════════════════════════════
# Aggregation Algorithms (与 MNIST 版本相同)
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
    n   = len(ids)
    m   = max(1, n - f - 2) if n > 2 * f + 2 else max(1, n - f)
    vecs = {cid: flat_updates_dict[cid] for cid in ids}
    dists = {i: {j: float(np.linalg.norm(vecs[i] - vecs[j]))
                 for j in ids if i != j}
             for i in ids}
    k = max(1, n - f - 2)
    scores = {i: sum(sorted(dists[i].values())[:k]) for i in ids}
    best_ids = sorted(scores, key=scores.get)[:m]
    return fedavg_aggregate({cid: updates[cid] for cid in best_ids})


def trimmed_mean_aggregate(updates, beta=0.2):
    first = list(updates.values())[0]
    n     = len(updates)
    trim  = max(0, min(int(n * beta), (n - 1) // 2))
    agg   = {}
    for k in first:
        if isinstance(first[k], torch.Tensor) and first[k].is_floating_point():
            stacked = torch.stack([updates[cid][k] for cid in updates])
            sorted_v, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_v[trim:n - trim] if trim > 0 else sorted_v
            agg[k]  = torch.mean(trimmed, dim=0)
        else:
            agg[k] = first[k]
    return agg


def fltrust_aggregate(flat_updates_dict, updates, device):
    ids   = list(flat_updates_dict.keys())
    n     = len(ids)
    stacked   = np.stack([flat_updates_dict[cid] for cid in ids])
    root      = np.mean(stacked, axis=0)
    root_norm = np.linalg.norm(root) + 1e-12
    trust_scores = {}
    for cid in ids:
        v      = flat_updates_dict[cid]
        v_norm = np.linalg.norm(v) + 1e-12
        trust_scores[cid] = max(0.0, float(np.dot(v, root) / (v_norm * root_norm)))
    s = sum(trust_scores.values())
    if s > 1e-12:
        trust_weights = {cid: t / s for cid, t in trust_scores.items()}
    else:
        trust_weights = {cid: 1.0 / n for cid in ids}
    return fedavg_aggregate(updates, trust_weights)


# ═══════════════════════════════════════════════════════════════════════
# TASL Trust Scoring v3 (完全复用 MNIST 版本算法)
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
    """
    TASL v3 信任评分:
    1. 双锚点: spatial median + pairwise median consensus
    2. 范数门控: L2范数超过 median 2.5倍直接清零
    3. 迭代锚点精化 (排除负相似度更新后重算)
    4. 自适应 cos 阈值: IQR 自适应
    5. Cubic Trust Scoring (power=5)
    6. 权重硬帽 1.5/N
    7. EMA 平滑 α=0.7
    8. Cos 历史惩罚: 连续低 cos 客户端降权
    9. 时间一致性锚点 (v3): 前一轮聚合方向作为第三锚点
    """
    n   = len(flat_updates)
    ids = list(flat_updates.keys())
    if n == 0:
        return {}, {}, None

    # Step 1: 计算范数 & 范数门控
    norms        = {cid: np.linalg.norm(v) for cid, v in flat_updates.items()}
    median_norm  = np.median(list(norms.values()))
    norm_gate    = median_norm * 2.5

    # Step 2: Spatial median anchor
    stacked     = np.stack(list(flat_updates.values()), axis=0)
    anchor      = np.median(stacked, axis=0)
    anchor_norm = np.linalg.norm(anchor) + 1e-12

    # Step 3: Pairwise consensus anchor
    pairwise_medians = {}
    for i in ids:
        vi_norm = np.linalg.norm(flat_updates[i]) + 1e-12
        sims = [float(np.dot(flat_updates[i], flat_updates[j]) /
                      (vi_norm * (np.linalg.norm(flat_updates[j]) + 1e-12)))
                for j in ids if i != j]
        pairwise_medians[i] = float(np.median(sims)) if sims else 0.5

    consensus_vals  = list(pairwise_medians.values())
    median_consensus = np.median(consensus_vals)
    high_consensus_ids = [cid for cid in ids if pairwise_medians[cid] >= median_consensus]

    if len(high_consensus_ids) >= max(2, n // 2):
        cons_stacked   = np.stack([flat_updates[cid] for cid in high_consensus_ids])
        consensus_anchor = np.median(cons_stacked, axis=0)
        anchor      = 0.6 * anchor + 0.4 * consensus_anchor
        anchor_norm = np.linalg.norm(anchor) + 1e-12

    # Step 4: 迭代锚点精化
    if refine_anchor:
        fp_sims = {cid: float(np.dot(v, anchor) /
                              ((np.linalg.norm(v) + 1e-12) * anchor_norm))
                   for cid, v in flat_updates.items()}
        trusted_ids = [cid for cid in ids if fp_sims[cid] > 0]
        if len(trusted_ids) >= max(2, n // 2):
            refined_stacked = np.stack([flat_updates[cid] for cid in trusted_ids])
            refined_anchor  = np.median(refined_stacked, axis=0)
            anchor = (0.6 * refined_anchor + 0.4 * prev_anchor
                      if prev_anchor is not None else refined_anchor)
            anchor_norm = np.linalg.norm(anchor) + 1e-12

    # Step 4.5: 时间一致性锚点 (v3)
    if temporal_anchor is not None:
        temporal_norm = np.linalg.norm(temporal_anchor) + 1e-12
        dir_agree     = float(np.dot(anchor, temporal_anchor) / (anchor_norm * temporal_norm))
        if dir_agree > 0.3:
            anchor      = 0.6 * anchor + 0.4 * temporal_anchor
            anchor_norm = np.linalg.norm(anchor) + 1e-12

    # Step 5: cos 相似度 + 自适应阈值
    cos_sims = {cid: float(np.dot(v, anchor) /
                            ((np.linalg.norm(v) + 1e-12) * anchor_norm))
                for cid, v in flat_updates.items()}
    cos_vals = sorted(cos_sims.values())
    if len(cos_vals) >= 4:
        q1 = np.percentile(cos_vals, 25)
        q3 = np.percentile(cos_vals, 75)
        iqr = q3 - q1
        adaptive_threshold = min(max(min_cos_threshold, q1 - 1.0 * iqr), 0.4)
    else:
        adaptive_threshold = min_cos_threshold

    # Step 6: 原始信任评分
    raw_scores = {}
    for cid in ids:
        if norms[cid] > norm_gate or cos_sims[cid] < adaptive_threshold:
            raw_scores[cid] = 0.0
            continue
        norm_ratio    = norms[cid] / (median_norm + 1e-12)
        norm_penalty  = np.exp(-norm_penalty_strength * abs(norm_ratio - 1.0))
        consensus     = max(0.01, pairwise_medians.get(cid, 0.5))
        raw_scores[cid] = (cos_sims[cid] ** trust_power) * norm_penalty * consensus

    # Step 7: Cos 历史惩罚
    if cos_history is not None and len(cos_history) >= 2:
        for cid in ids:
            recent = [ch.get(cid, 0.5) for ch in cos_history[-2:]]
            if all(c < 0.3 for c in recent):
                raw_scores[cid] *= 0.1

    # Step 8: 归一化
    sum_w = sum(raw_scores.values())
    if sum_w > 1e-12:
        trust_weights = {cid: w / sum_w for cid, w in raw_scores.items()}
    else:
        trust_weights = {cid: 1.0 / n for cid in ids}

    # Step 9: 权重硬帽
    max_w  = max_weight_ratio / n
    capped = {cid: min(w, max_w) for cid, w in trust_weights.items()}
    cap_sum = sum(capped.values())
    if cap_sum > 1e-12:
        trust_weights = {cid: w / cap_sum for cid, w in capped.items()}

    # Step 10: EMA 平滑
    if ema_weights is not None:
        smoothed = {cid: ema_alpha * trust_weights.get(cid, 0.0) +
                    (1 - ema_alpha) * ema_weights.get(cid, 1.0 / n)
                    for cid in ids}
        s = sum(smoothed.values())
        if s > 1e-12:
            trust_weights = {cid: w / s for cid, w in smoothed.items()}

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
    local_epochs: int = 5,
    lr: float = 0.1,
    seed: int = 42,
    noise_scale: float = 5.0,
) -> Dict:
    set_seed(seed)
    byzantine_set = set(range(n_byz))
    global_model  = CifarCNN().to(device)
    global_state  = global_model.state_dict()

    acc_history    = []
    asr_history    = []
    ema_weights    = None
    cos_history    = []
    prev_anchor    = None
    temporal_anchor = None

    for r in range(1, rounds + 1):
        # Cosine annealing LR
        current_lr = _get_lr_for_round(lr, r, rounds)

        local_updates = {}
        flat_updates  = {}

        for cid in range(n_clients):
            if cid in byzantine_set and attack != 'none':
                train_fn = ATTACK_TRAIN_FNS.get(attack, train_honest)
                kwargs   = dict(local_epochs=local_epochs, lr=current_lr)
                if attack == 'gaussian_noise':
                    kwargs['noise_scale'] = noise_scale
                local_state = train_fn(global_state, client_loaders[cid], device, **kwargs)
            else:
                local_state = train_honest(global_state, client_loaders[cid], device,
                                            local_epochs=local_epochs, lr=current_lr)

            upd = state_sub(local_state, global_state)
            local_updates[cid] = upd
            flat_updates[cid]  = flatten_update(upd)

        # ── Aggregation ────────────────────────────────────────────────
        if algo == 'fedavg':
            agg_update = fedavg_aggregate(local_updates)

        elif algo == 'multi_krum':
            agg_update = multi_krum_aggregate(flat_updates, local_updates, f=n_byz)

        elif algo == 'trimmed_mean':
            agg_update = trimmed_mean_aggregate(local_updates, beta=0.2)

        elif algo == 'fltrust':
            agg_update = fltrust_aggregate(flat_updates, local_updates, device)

        elif algo == 'tasl':
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
            )
            ema_weights  = dict(trust_weights)
            cos_history.append(dict(cos_sims))
            prev_anchor  = new_anchor
            agg_update   = fedavg_aggregate(local_updates, trust_weights)
            # 更新 temporal anchor
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

        if r % 5 == 0 or r == 1:
            byz_avg_w = (np.mean([ema_weights.get(c, 0) for c in byzantine_set])
                         if algo == 'tasl' and ema_weights else 0.0)
            asr_str = f" asr={asr:.2f}%" if attack == 'label_flip' else ""
            print(f"  [{algo}/{attack}] R{r:03d} acc={acc:.2f}% lr={current_lr:.5f}"
                  + (f" byz_w={byz_avg_w:.4f}" if algo == 'tasl' else "") + asr_str)

    # 对 sign_flip/gaussian_noise，ASR 用精度下降比: 1 - (avg_acc / clean_avg_acc)
    # 这里先返回原始数据，最终 ASR 在 main() 中用 clean baseline 计算
    return {
        'best_acc':    max(acc_history),
        'avg_acc':     float(np.mean(acc_history)),
        'last_acc':    acc_history[-1],
        'acc_history': acc_history,
        'asr_history': asr_history,   # label_flip 逐轮 ASR
        'best_asr':    max(asr_history) if asr_history else 0.0,
        'avg_asr':     float(np.mean(asr_history)) if asr_history else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════

def plot_heatmap(save_path, results_df, metric='best_acc'):
    import pandas as pd
    algos        = ['fedavg', 'multi_krum', 'trimmed_mean', 'fltrust', 'tasl']
    algo_labels  = ['FedAvg', 'Multi-Krum', 'Trimmed Mean', 'FLTrust', 'TASL (Ours)']
    attacks      = ['label_flip', 'sign_flip', 'gaussian_noise']
    attack_labels = ['Label Flip', 'Sign Flip', 'Gaussian Noise']
    metric_label = 'Best Accuracy' if metric == 'best_acc' else 'Avg Accuracy'

    fig, ax = plt.subplots(figsize=(10, 5))
    data = np.zeros((len(algos), len(attacks)))
    for i, algo in enumerate(algos):
        for j, atk in enumerate(attacks):
            row = results_df[(results_df['algo'] == algo) & (results_df['attack'] == atk)]
            if not row.empty:
                data[i, j] = row[metric].values[0]

    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(attacks)));  ax.set_xticklabels(attack_labels, fontsize=12)
    ax.set_yticks(range(len(algos)));   ax.set_yticklabels(algo_labels, fontsize=12)
    for i in range(len(algos)):
        for j in range(len(attacks)):
            val   = data[i, j]
            color = 'white' if val < 40 or val > 80 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=14, fontweight='bold', color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f'{metric_label} (%)', fontsize=12)
    ax.set_title(f'CIFAR-10 Byzantine Resilience Matrix — {metric_label}\n'
                 f'(ResNet-18, 40% Byzantine, Non-IID α=0.3)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved heatmap: {save_path}")


def plot_curves(save_path, all_results, attack, rounds):
    algo_colors = {
        'fedavg':       '#E53935',
        'multi_krum':   '#1E88E5',
        'trimmed_mean': '#FB8C00',
        'fltrust':      '#8E24AA',
        'tasl':         '#43A047',
    }
    algo_labels = {
        'fedavg':       'FedAvg',
        'multi_krum':   'Multi-Krum',
        'trimmed_mean': 'Trimmed Mean',
        'fltrust':      'FLTrust',
        'tasl':         'TASL (Ours)',
    }
    attack_label = {'label_flip': 'Label Flip', 'sign_flip': 'Sign Flip',
                    'gaussian_noise': 'Gaussian Noise'}

    fig, ax = plt.subplots(figsize=(12, 6))
    for algo, res in all_results.items():
        if attack in res:
            ax.plot(range(1, rounds + 1), res[attack]['acc_history'],
                    label=algo_labels[algo], color=algo_colors[algo],
                    linewidth=2.0, alpha=0.9)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'CIFAR-10 Convergence under {attack_label.get(attack, attack)} Attack\n'
                 f'(ResNet-18, 40% Byzantine, Non-IID α=0.3)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved curves: {save_path}")


def plot_bar_comparison(save_path, results_df, metric='best_acc'):
    import pandas as pd
    algos        = ['fedavg', 'multi_krum', 'trimmed_mean', 'fltrust', 'tasl']
    algo_labels  = ['FedAvg', 'Multi-\nKrum', 'Trimmed\nMean', 'FLTrust', 'TASL\n(Ours)']
    attacks      = ['label_flip', 'sign_flip', 'gaussian_noise']
    attack_labels = ['Label Flip', 'Sign Flip', 'Gaussian Noise']
    algo_colors  = ['#E53935', '#1E88E5', '#FB8C00', '#8E24AA', '#43A047']
    metric_label = 'Best Accuracy (%)' if metric == 'best_acc' else 'Avg Accuracy (%)'

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for j, atk in enumerate(attacks):
        ax   = axes[j]
        vals = [results_df[(results_df['algo'] == algo) &
                            (results_df['attack'] == atk)][metric].values[0]
                if not results_df[(results_df['algo'] == algo) &
                                   (results_df['attack'] == atk)].empty else 0
                for algo in algos]
        bars = ax.bar(range(len(algos)), vals, color=algo_colors,
                      edgecolor='white', linewidth=1.5)
        ax.set_xticks(range(len(algos)));  ax.set_xticklabels(algo_labels, fontsize=9)
        ax.set_title(attack_labels[j], fontsize=13, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    axes[0].set_ylabel(metric_label, fontsize=12)
    fig.suptitle(f'CIFAR-10 Byzantine Resilience — {metric_label} (ResNet-18, 40% Byzantine, Non-IID α=0.3)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved bar chart: {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp2 CIFAR-10: Byzantine Resilience Matrix (ResNet-18)")
    parser.add_argument("--rounds",          type=int,   default=100,  help="训练轮数")
    parser.add_argument("--clients",         type=int,   default=10)
    parser.add_argument("--batch-size",      type=int,   default=64)
    parser.add_argument("--local-epochs",    type=int,   default=5)
    parser.add_argument("--lr",              type=float, default=0.1)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--byzantine-ratio", type=float, default=0.4)
    parser.add_argument("--alpha",           type=float, default=0.3,  help="Dirichlet α")
    parser.add_argument("--noise-scale",     type=float, default=5.0)
    parser.add_argument("--attack",          type=str,   default=None,
                        choices=['label_flip', 'sign_flip', 'gaussian_noise'])
    parser.add_argument("--quick",           action='store_true',
                        help="快速测试: 5客户端, 10轮, 1个local_epoch")
    args = parser.parse_args()

    if args.quick:
        args.clients      = 5
        args.rounds       = 10
        args.local_epochs = 1

    set_seed(args.seed)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_clients = int(args.clients)
    n_byz     = max(1, int(n_clients * args.byzantine_ratio))
    rounds    = int(args.rounds)

    algos   = ['fedavg', 'multi_krum', 'trimmed_mean', 'fltrust', 'tasl']
    attacks = ['label_flip', 'sign_flip', 'gaussian_noise']
    if args.attack:
        attacks = [args.attack]

    # 打印模型信息
    model_info = CifarCNN()
    n_params = sum(p.numel() for p in model_info.parameters())
    del model_info

    print("=" * 70)
    print(f"[Exp2-CIFAR10] Byzantine Resilience Matrix")
    print(f"  Dataset: CIFAR-10 (10 classes, 3×32×32)")
    print(f"  Model: ResNet-18 (GroupNorm), ~{n_params/1e6:.1f}M params")
    print(f"  {n_clients} clients, {n_byz} Byzantine ({args.byzantine_ratio:.0%})")
    print(f"  Non-IID Dirichlet α={args.alpha}")
    print(f"  {rounds} rounds, local_epochs={args.local_epochs}, lr={args.lr}")
    print(f"  LR scheduler: CosineAnnealing ({args.lr} → {args.lr*0.01:.4f})")
    print(f"  Algorithms: {algos}")
    print(f"  Attacks: {attacks}")
    print(f"  Device: {device}")
    print("=" * 70)

    # 准备数据
    print("\n[Data] Downloading/loading CIFAR-10 ...")
    client_loaders, test_loader = prepare_cifar10_loaders(
        n_clients=n_clients,
        batch_size=int(args.batch_size),
        alpha=float(args.alpha),
        seed=args.seed,
    )
    print(f"[Data] {n_clients} client loaders ready")

    # ── Clean Baseline ─────────────────────────────────────────────────
    print("\n>>> Clean Baseline (no attack) <<<")
    clean_result = run_single(
        'fedavg', 'none', client_loaders, test_loader,
        n_clients=n_clients, n_byz=0,
        rounds=rounds, device=device,
        local_epochs=int(args.local_epochs), lr=float(args.lr),
        seed=args.seed,
    )
    print(f"  Clean: best={clean_result['best_acc']:.2f}%, avg={clean_result['avg_acc']:.2f}%")
    # ── All combinations ───────────────────────────────────────────────
    all_results = {algo: {} for algo in algos}
    csv_rows    = []
    total_runs  = len(algos) * len(attacks)
    run_idx     = 0

    for algo in algos:
        for attack in attacks:
            run_idx += 1
            print(f"\n>>> [{run_idx}/{total_runs}] {algo} × {attack} <<<")
            result = run_single(
                algo, attack, client_loaders, test_loader,
                n_clients=n_clients, n_byz=n_byz,
                rounds=rounds, device=device,
                local_epochs=int(args.local_epochs), lr=float(args.lr),
                seed=args.seed + hash(algo + attack) % 1000,
                noise_scale=float(args.noise_scale),
            )
            all_results[algo][attack] = result
            # 计算 ASR: label_flip 直接用 asr; sign_flip/gaussian_noise 用精度下降比
            if attack == 'label_flip':
                final_asr = result['avg_asr']
            else:
                # ASR = 1 - (attacked_avg_acc / clean_avg_acc)，越高攻击越成功
                final_asr = max(0, 100.0 * (1 - result['avg_acc'] / max(clean_result['avg_acc'], 1e-6)))
            csv_rows.append({
                'algo': algo, 'attack': attack,
                'dataset': 'cifar10', 'alpha': args.alpha,
                'byzantine_ratio': args.byzantine_ratio,
                'best_acc': result['best_acc'],
                'avg_acc':  result['avg_acc'],
                'last_acc': result['last_acc'],
                'asr':      final_asr,
            })
            asr_str = f", asr={final_asr:.2f}%" if attack != 'none' else ""
            print(f"  => best={result['best_acc']:.2f}%, avg={result['avg_acc']:.2f}%, last={result['last_acc']:.2f}%{asr_str}")

    # ── Save CSV ───────────────────────────────────────────────────────
    results_dir = os.path.join(BLOCKCHAIN_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path    = os.path.join(results_dir, "exp2_cifar10_robustness_data.csv")
    fieldnames  = ['algo', 'attack', 'dataset', 'alpha', 'byzantine_ratio',
                   'best_acc', 'avg_acc', 'last_acc', 'asr']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Add clean baseline
    csv_rows.append({
        'algo': 'clean_baseline', 'attack': 'none',
        'dataset': 'cifar10', 'alpha': args.alpha,
        'byzantine_ratio': 0.0,
        'best_acc': clean_result['best_acc'],
        'avg_acc':  clean_result['avg_acc'],
        'last_acc': clean_result['last_acc'],
        'asr':      0.0,
    })

    # ── Summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"[Exp2-CIFAR10] Summary — ResNet-18, Non-IID α={args.alpha}, {args.byzantine_ratio:.0%} Byzantine")
    print(f"Clean Baseline: best={clean_result['best_acc']:.2f}%, avg={clean_result['avg_acc']:.2f}%")
    print("=" * 70)
    print(f"{'Attack':<18s} {'FedAvg':>8s} {'M-Krum':>8s} {'T-Mean':>8s} {'FLTrust':>8s} {'TASL':>8s}")
    print("-" * 70)
    for attack in attacks:
        row_str = f"{attack:<18s}"
        all_vals = [all_results[a][attack]['best_acc'] for a in algos]
        for algo, val in zip(algos, all_vals):
            marker   = " *" if val == max(all_vals) else "  "
            row_str += f" {val:>6.2f}{marker}"
        print(row_str)
    print()
    print(f"{'Attack (avg)':<18s} {'FedAvg':>8s} {'M-Krum':>8s} {'T-Mean':>8s} {'FLTrust':>8s} {'TASL':>8s}")
    print("-" * 70)
    for attack in attacks:
        row_str = f"{attack:<18s}"
        all_vals = [all_results[a][attack]['avg_acc'] for a in algos]
        for algo, val in zip(algos, all_vals):
            marker   = " *" if val == max(all_vals) else "  "
            row_str += f" {val:>6.2f}{marker}"
        print(row_str)

    # ASR table
    print()
    print(f"{'ASR (lower=better)':<18s} {'FedAvg':>8s} {'M-Krum':>8s} {'T-Mean':>8s} {'FLTrust':>8s} {'TASL':>8s}")
    print("-" * 70)
    for attack in attacks:
        row_str = f"{attack:<18s}"
        all_asr = []
        for algo in algos:
            if attack == 'label_flip':
                asr_val = all_results[algo][attack]['avg_asr']
            else:
                asr_val = max(0, 100.0 * (1 - all_results[algo][attack]['avg_acc'] / max(clean_result['avg_acc'], 1e-6)))
            all_asr.append(asr_val)
        for algo, asr_val in zip(algos, all_asr):
            marker   = " *" if asr_val == min(all_asr) else "  "
            row_str += f" {asr_val:>6.2f}{marker}"
        print(row_str)

    # ── Plots ──────────────────────────────────────────────────────────
    import pandas as pd
    results_df = pd.DataFrame(csv_rows)

    plot_heatmap(os.path.join(results_dir, "exp2_cifar10_heatmap_best.png"),
                 results_df, 'best_acc')
    plot_heatmap(os.path.join(results_dir, "exp2_cifar10_heatmap_avg.png"),
                 results_df, 'avg_acc')
    plot_bar_comparison(os.path.join(results_dir, "exp2_cifar10_bar_best.png"),
                        results_df, 'best_acc')
    plot_bar_comparison(os.path.join(results_dir, "exp2_cifar10_bar_avg.png"),
                        results_df, 'avg_acc')
    for attack in attacks:
        plot_curves(os.path.join(results_dir, f"exp2_cifar10_curves_{attack}.png"),
                    all_results, attack, rounds)

    print(f"\nSaved CSV  : {csv_path}")
    print(f"Saved Plots: {results_dir}")
    print(f"Files: exp2_cifar10_heatmap_best.png, exp2_cifar10_heatmap_avg.png,")
    print(f"       exp2_cifar10_bar_best.png, exp2_cifar10_bar_avg.png,")
    for attack in attacks:
        print(f"       exp2_cifar10_curves_{attack}.png")


if __name__ == "__main__":
    main()
