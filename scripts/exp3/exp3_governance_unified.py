"""
实验二: Byzantine Resilience Matrix — YAML配置驱动版

支持:
  - 5种攻击: label_flip, sign_flip, gaussian_noise, min_max, scaling
  - 5种聚合: FedAvg, Multi-Krum, Trimmed Mean, FLTrust, TASL(Ours)
  - 可配置模型: ST-GCN(NTU-60) / FedAvgCNN(MNIST) / ResNet-18(CIFAR-10)
  - CLI覆盖: --algorithms, --attacks, --quick

用法:
  # 完整矩阵
  python exp2_run.py --config configs/exp2_ntu60.yaml

  # 只跑特定算法+攻击
  python exp2_run.py --config configs/exp2_ntu60.yaml --algorithms tasl,fedavg --attacks label_flip,min_max

  # 快速测试
  python exp2_run.py --config configs/exp2_ntu60.yaml --quick
"""
import argparse
import random
import copy
import csv
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCKCHAIN_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CORE_DIR = os.path.join(BLOCKCHAIN_DIR, "core")
sys.path.insert(0, BLOCKCHAIN_DIR)
sys.path.insert(0, CORE_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
# Config Loader
# ═══════════════════════════════════════════════════════════════════════

def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def deep_merge(base, override):
    """Merge override into base (override wins on conflicts)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════

class FederatedDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = torch.from_numpy(data).float() if isinstance(data, np.ndarray) else data.float()
        self.labels = torch.from_numpy(labels).long() if isinstance(labels, np.ndarray) else labels.long()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def dirichlet_partition(labels, n_clients, alpha=0.3, seed=42):
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


def load_ntu60_data(cfg):
    """Load NTU-60 federated pre-split data."""
    data_dir = cfg['data']['data_dir']
    batch_size = cfg['data']['batch_size']
    n_clients = cfg['data']['n_clients']

    client_loaders = []
    for cid in range(n_clients):
        raw = np.load(os.path.join(data_dir, 'fedtrain', f'{cid}.npz'),
                      allow_pickle=True)['data'].item()
        x, y = raw['x'], raw['y'].flatten()
        ds = FederatedDataset(x, y)
        client_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))

    raw_test = np.load(os.path.join(data_dir, 'fedtest', '0.npz'),
                       allow_pickle=True)['data'].item()
    test_ds = FederatedDataset(raw_test['x'], raw_test['y'].flatten())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return client_loaders, test_loader



import torchvision.transforms as _T_proxy

def create_proxy_loader(dataset_name, n_per_class=50, batch_size=64):
    from os import makedirs as _makedirs
    _makedirs(BLOCKCHAIN_DIR + "/data/temp", exist_ok=True)
    tdir = BLOCKCHAIN_DIR + "/data/temp"
    from torchvision import datasets as _tds

    if dataset_name == "mnist":
        ds = _tds.MNIST(tdir, train=True, download=True)
        x = ds.data.numpy().astype("float32") / 255.0
        if len(x.shape) == 3:
            x = x[:, None, :, :]
        y = ds.targets.numpy()
        transform = None
    elif dataset_name in ("fashionmnist", "fashion_mnist"):
        ds = _tds.FashionMNIST(tdir, train=True, download=True)
        x = ds.data.numpy().astype("float32") / 255.0
        if len(x.shape) == 3:
            x = x[:, None, :, :]
        y = ds.targets.numpy()
        transform = None
    elif dataset_name == "cifar10":
        ds = _tds.CIFAR10(tdir, train=True, download=True)
        x = ds.data.astype("float32") / 255.0
        x = np.transpose(x, (0, 3, 1, 2))
        cm = np.array([0.4914, 0.4822, 0.4465], dtype="float32").reshape(1, 3, 1, 1)
        cs = np.array([0.247, 0.243, 0.261], dtype="float32").reshape(1, 3, 1, 1)
        x = (x - cm) / cs
        y = np.array(ds.targets)
        transform = _T_proxy.Compose([
            _T_proxy.RandomCrop(32, padding=4),
            _T_proxy.RandomHorizontalFlip(),
        ])
    else:
        raise ValueError("Unknown proxy dataset: " + dataset_name)

    nc = int(y.max() + 1)
    indices = []
    for c in range(nc):
        ci = np.where(y == c)[0]
        ns = min(n_per_class, len(ci))
        pick = np.random.RandomState(42).choice(ci, size=ns, replace=False)
        indices.extend(pick)

    px = x[indices]
    py = y[indices]
    pds = FederatedDataset(px, py, transform=transform)
    pl = DataLoader(pds, batch_size=min(batch_size, len(px)), shuffle=True)
    print("  Proxy dataset: %d samples (%d/class x %d classes)" % (len(px), n_per_class, nc))
    return pl


def compute_proxy_root(global_state, proxy_loader, device, model_fn, lr=0.01, epochs=1):
    model = model_fn().to(device)
    sd = {k: v.to(device) for k, v in global_state.items()}
    model.load_state_dict(sd)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    for _ in range(epochs):
        for xb, yb in proxy_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
    fs = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    return flatten_update(state_sub(fs, global_state))


def load_mnist_data(cfg):
    """Load MNIST data with Dirichlet partitioning."""
    from torchvision import datasets, transforms

    batch_size = cfg['data']['batch_size']
    n_clients = cfg['data']['n_clients']
    alpha = cfg['data']['alpha']
    split = cfg['data'].get('split', 'non_iid')
    seed = cfg['experiment']['seed']

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

    if split == 'non_iid':
        client_indices = dirichlet_partition(train_labels, n_clients, alpha=alpha, seed=seed)
    else:
        rng = np.random.RandomState(seed)
        client_indices = {i: [] for i in range(n_clients)}
        for c in range(10):
            idx = np.where(train_labels == c)[0]
            rng.shuffle(idx)
            splits = np.array_split(idx, n_clients)
            for i in range(n_clients):
                client_indices[i].extend(splits[i].tolist())

    client_loaders = []
    for i in range(n_clients):
        idx = client_indices[i]
        ds = FederatedDataset(train_data[idx], train_labels[idx])
        client_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))

    test_loader = DataLoader(FederatedDataset(test_data, test_labels),
                             batch_size=batch_size, shuffle=False)
    return client_loaders, test_loader


def load_fashionmnist_data(cfg):
    """Load Fashion-MNIST data with Dirichlet partitioning."""
    from torchvision import datasets, transforms

    batch_size = cfg['data']['batch_size']
    n_clients = cfg['data']['n_clients']
    alpha = cfg['data']['alpha']
    split = cfg['data'].get('split', 'non_iid')
    seed = cfg['experiment']['seed']

    temp_dir = os.path.join(BLOCKCHAIN_DIR, "data", "temp")
    os.makedirs(temp_dir, exist_ok=True)

    train_dataset = datasets.FashionMNIST(root=temp_dir, train=True, download=True,
                                           transform=transforms.ToTensor())
    test_dataset = datasets.FashionMNIST(root=temp_dir, train=False, download=True,
                                          transform=transforms.ToTensor())

    train_data = train_dataset.data.numpy().astype(np.float32) / 255.0
    train_labels = train_dataset.targets.numpy()
    if len(train_data.shape) == 3:
        train_data = np.expand_dims(train_data, axis=1)

    test_data = test_dataset.data.numpy().astype(np.float32) / 255.0
    if len(test_data.shape) == 3:
        test_data = np.expand_dims(test_data, axis=1)
    test_labels = test_dataset.targets.numpy()

    if split == 'non_iid':
        client_indices = dirichlet_partition(train_labels, n_clients, alpha=alpha, seed=seed)
    else:
        rng = np.random.RandomState(seed)
        client_indices = {i: [] for i in range(n_clients)}
        for c in range(10):
            idx = np.where(train_labels == c)[0]
            rng.shuffle(idx)
            splits = np.array_split(idx, n_clients)
            for i in range(n_clients):
                client_indices[i].extend(splits[i].tolist())

    client_loaders = []
    for i in range(n_clients):
        idx = client_indices[i]
        ds = FederatedDataset(train_data[idx], train_labels[idx])
        client_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))

    test_loader = DataLoader(FederatedDataset(test_data, test_labels),
                             batch_size=batch_size, shuffle=False)
    return client_loaders, test_loader


def load_cifar10_data(cfg):
    """Load CIFAR-10 data with Dirichlet partitioning."""
    from torchvision import datasets, transforms

    batch_size = cfg['data']['batch_size']
    n_clients = cfg['data']['n_clients']
    alpha = cfg['data']['alpha']
    split = cfg['data'].get('split', 'non_iid')
    seed = cfg['experiment']['seed']

    temp_dir = os.path.join(BLOCKCHAIN_DIR, "data", "temp")
    os.makedirs(temp_dir, exist_ok=True)

    train_dataset = datasets.CIFAR10(root=temp_dir, train=True, download=True,
                                     transform=transforms.ToTensor())
    test_dataset = datasets.CIFAR10(root=temp_dir, train=False, download=True,
                                    transform=transforms.ToTensor())

    train_data = train_dataset.data.astype(np.float32) / 255.0
    train_labels = np.array(train_dataset.targets)
    if len(train_data.shape) == 3:
        train_data = np.expand_dims(train_data, axis=1)
    else:
        # CIFAR-10: NHWC -> NCHW
        train_data = np.transpose(train_data, (0, 3, 1, 2))

    test_data = test_dataset.data.astype(np.float32) / 255.0
    if len(test_data.shape) == 3:
        test_data = np.expand_dims(test_data, axis=1)
    else:
        # CIFAR-10: NHWC -> NCHW
        test_data = np.transpose(test_data, (0, 3, 1, 2))
    test_labels = np.array(test_dataset.targets)

    if split == 'non_iid':
        client_indices = dirichlet_partition(train_labels, n_clients, alpha=alpha, seed=seed)
    else:
        rng = np.random.RandomState(seed)
        client_indices = {i: [] for i in range(n_clients)}
        for c in range(10):
            idx = np.where(train_labels == c)[0]
            rng.shuffle(idx)
            splits = np.array_split(idx, n_clients)
            for i in range(n_clients):
                client_indices[i].extend(splits[i].tolist())

    client_loaders = []
    for i in range(n_clients):
        idx = client_indices[i]
        ds = FederatedDataset(train_data[idx], train_labels[idx])
        client_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))

    test_loader = DataLoader(FederatedDataset(test_data, test_labels),
                             batch_size=batch_size, shuffle=False)
    return client_loaders, test_loader


DATA_LOADERS = {
    'ntu60': load_ntu60_data,
    'mnist': load_mnist_data,
    'fashionmnist': load_fashionmnist_data,
    'cifar10': load_cifar10_data,
}


# ═══════════════════════════════════════════════════════════════════════
# Model Factory
# ═══════════════════════════════════════════════════════════════════════

def create_model(cfg):
    """Create model based on config."""
    model_cfg = cfg['model']
    name = model_cfg['name']

    if name == 'stgcn':
        stgcn_dir = os.path.join(os.path.dirname(BLOCKCHAIN_DIR), 'st-gcn')
        if stgcn_dir not in sys.path:
            sys.path.insert(0, stgcn_dir)
        from net.st_gcn import Model as STGCN
        model = STGCN(
            in_channels=model_cfg.get('in_channels', 3),
            num_class=model_cfg['num_classes'],
            graph_args={'layout': model_cfg.get('graph_layout', 'ntu-rgb+d'),
                        'strategy': model_cfg.get('graph_strategy', 'spatial')},
            edge_importance_weighting=model_cfg.get('edge_importance_weighting', True),
            dropout=model_cfg.get('dropout', 0.5),
        )
    elif name == 'fedavgcnn':
        from model import FedAvgCNN
        model = FedAvgCNN(num_classes=model_cfg['num_classes'])
    elif name == 'resnet18_cifar':
        import torchvision.models as models
        model = models.resnet18(num_classes=model_cfg['num_classes'])
        # CIFAR-10 adaptation: 3x3 conv1, stride 1, no maxpool
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif name == 'resnet18':
        import torchvision.models as models
        model = models.resnet18(num_classes=model_cfg['num_classes'])

    else:
        raise ValueError(f"Unknown model: {name}")

    return model


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
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
    """Return accuracy and attack-specific ASR."""
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


# ═══════════════════════════════════════════════════════════════════════
# State helpers
# ═══════════════════════════════════════════════════════════════════════

def state_sub(local_state, global_state):
    return {k: (local_state[k].cpu() - global_state[k].cpu()).detach()
            for k in global_state
            if isinstance(global_state[k], torch.Tensor) and global_state[k].is_floating_point()}


def state_add(global_state, update):
    return {k: (global_state[k].cpu() + update[k].cpu() if k in update else global_state[k])
            for k in global_state}


def flatten_update(update):
    chunks = [v.detach().cpu().reshape(-1).numpy() for v in update.values()
              if isinstance(v, torch.Tensor) and v.is_floating_point()]
    return np.concatenate(chunks).astype(np.float64) if chunks else np.zeros(1)


# ═══════════════════════════════════════════════════════════════════════
# Attacks
# ═══════════════════════════════════════════════════════════════════════

def train_honest(global_state, loader, device, model_fn, local_epochs=1, lr=0.01,
                 momentum=0.9, weight_decay=1e-4, nesterov=False):
    """Normal local training."""
    model = model_fn().to(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay, nesterov=nesterov)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
    return {k: v.cpu() for k, v in model.state_dict().items()}


def train_fedprox(global_state, loader, device, model_fn, local_epochs=1, lr=0.01,
                  momentum=0.9, weight_decay=1e-4, mu=0.01, nesterov=False):
    """FedProx local training with proximal term: loss + (mu/2)||w - w_global||^2"""
    model = model_fn().to(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay, nesterov=nesterov)
    criterion = nn.CrossEntropyLoss()
    # Store global weights as detached tensor for proximal term
    global_params = {name: p.clone().detach() for name, p in model.named_parameters()}
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            # Proximal term: (mu/2) * sum((w_i - w_global_i)^2)
            prox = 0.0
            for name, p in model.named_parameters():
                if p.requires_grad:
                    prox += ((p - global_params[name]) ** 2).sum()
            loss = loss + (mu / 2) * prox
            loss.backward()
            optimizer.step()
    return {k: v.cpu() for k, v in model.state_dict().items()}



def get_lr_for_round(round_idx, base_lr, lr_schedule_cfg=None):
    """Compute learning rate for given round with warmup + step decay."""
    if lr_schedule_cfg is None:
        return base_lr
    warm_up = lr_schedule_cfg.get('warm_up_epoch', 0)
    step_milestones = lr_schedule_cfg.get('step', [])
    decay_rate = lr_schedule_cfg.get('lr_decay_rate', 0.1)
    if round_idx < warm_up:
        return base_lr * (round_idx + 1) / warm_up
    n_decays = sum(1 for s in step_milestones if round_idx >= s)
    return base_lr * (decay_rate ** n_decays)

def attack_label_flip(global_state, loader, device, model_fn, local_epochs=1, lr=0.01,
                      momentum=0.9, weight_decay=1e-4, num_classes=10, **kwargs):
    """Label flipping: y → num_classes - 1 - y"""
    model = model_fn().to(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_flip = num_classes - 1 - y
            optimizer.zero_grad()
            criterion(model(x), y_flip).backward()
            optimizer.step()
    return {k: v.cpu() for k, v in model.state_dict().items()}


def attack_sign_flip(global_state, loader, device, model_fn, local_epochs=1, lr=0.01,
                     momentum=0.9, weight_decay=1e-4, **kwargs):
    """Sign flipping: Δ → -Δ (equivalent to model = 2*global - local)"""
    model = model_fn().to(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
    local_state = {k: v.cpu() for k, v in model.state_dict().items()}
    # Invert: local = 2*global - local
    return {k: (2 * global_state[k].cpu() - v) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
            for k, v in local_state.items()}


def attack_gaussian_noise(global_state, loader, device, model_fn, local_epochs=1, lr=0.01,
                          momentum=0.9, weight_decay=1e-4, noise_scale=5.0, **kwargs):
    """Gaussian noise: add noise with L2 = noise_scale × honest update L2"""
    model = model_fn().to(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
    local_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Compute honest update L2 norm
    total_sq = 0.0
    n_params = 0
    for k, v in local_state.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            delta = v - global_state[k].cpu()
            total_sq += delta.detach().norm().item() ** 2
            n_params += v.numel()
    honest_update_norm = total_sq ** 0.5

    # Add calibrated noise
    noise_std_per_param = noise_scale * honest_update_norm / (n_params ** 0.5)
    noisy_state = {}
    for k, v in local_state.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            noise = torch.randn_like(v) * noise_std_per_param
            noisy_state[k] = v + noise
        else:
            noisy_state[k] = v
    return noisy_state


def attack_min_max(global_state, loader, device, model_fn, local_epochs=1, lr=0.01,
                   momentum=0.9, weight_decay=1e-4, z_max=2.0,
                   honest_flat_updates=None, **kwargs):
    """Min-Max attack (Baruch et al., 2019 "A Little Is Enough")

    Byzantine clients send updates at the edge of the honest distribution:
    malicious_update = mu_honest - z * sigma_honest
    where mu and sigma are computed from all honest client updates.

    This pushes the aggregate away from the true gradient while staying
    within the range that coordinate-wise robust aggregators would accept.

    Args:
        honest_flat_updates: dict {cid: flat_np_array} of honest updates
                            (provided by run_single after collecting honest updates)
    """
    if honest_flat_updates is None or len(honest_flat_updates) == 0:
        # Fallback: do honest training (no attack info available)
        return train_honest(global_state, loader, device, model_fn,
                           local_epochs, lr, momentum, weight_decay)

    # Compute honest statistics
    honest_arr = np.stack(list(honest_flat_updates.values()), axis=0)
    mu = honest_arr.mean(axis=0)
    sigma = honest_arr.std(axis=0)

    # Craft malicious vector: mu - z * sigma
    mal_flat = mu - z_max * sigma

    # Convert flat vector back to state_dict format
    # Use the first honest update's structure as template
    # First do honest training to get proper structure
    model = model_fn().to(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
    local_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Replace the update with the malicious one
    malicious_state = {}
    offset = 0
    for k, v in local_state.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            n = v.numel()
            mal_chunk = mal_flat[offset:offset + n].reshape(v.shape)
            malicious_state[k] = global_state[k].cpu() + torch.from_numpy(mal_chunk).float()
            offset += n
        else:
            malicious_state[k] = v

    return malicious_state


def attack_scaling(global_state, loader, device, model_fn, local_epochs=1, lr=0.01,
                   momentum=0.9, weight_decay=1e-4, scaling_factor=100.0, **kwargs):
    """Scaling attack: amplify malicious update by a large factor.

    Byzantine clients train honestly but then amplify their update by
    scaling_factor. In FedAvg, this means the malicious update dominates
    the aggregation: agg ≈ (n_byz/N) * λ * malicious_update

    Very effective against FedAvg, but detectable by norm-based defenses.
    """
    model = model_fn().to(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
    local_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Scale update: local = global + scaling_factor * (local - global)
    scaled_state = {}
    for k, v in local_state.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            delta = v - global_state[k].cpu()
            scaled_state[k] = global_state[k].cpu() + scaling_factor * delta
        else:
            scaled_state[k] = v
    return scaled_state


ATTACK_FNS = {
    'label_flip': attack_label_flip,
    'sign_flip': attack_sign_flip,
    'gaussian_noise': attack_gaussian_noise,
    'min_max': attack_min_max,
    'scaling': attack_scaling,
}


# ═══════════════════════════════════════════════════════════════════════
# Aggregation Algorithms
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
    if n <= 2 * f + 2:
        m = max(1, n - f)
    else:
        m = n - f - 2

    vecs = {cid: flat_updates_dict[cid] for cid in ids}
    dists = {}
    for i in ids:
        dists[i] = {}
        for j in ids:
            if i != j:
                dists[i][j] = float(np.linalg.norm(vecs[i] - vecs[j]))

    k = max(1, n - f - 2)
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


def fltrust_aggregate(flat_updates_dict, updates, device, root=None):
    ids = list(flat_updates_dict.keys())
    n = len(ids)

    # g0: server proxy dataset gradient (or fallback to client mean)
    if root is None:
        stacked = np.stack([flat_updates_dict[cid] for cid in ids])
        root = np.mean(stacked, axis=0)
    root_norm = np.linalg.norm(root) + 1e-12
    root_unit = root / root_norm

    trust_scores = {}
    for cid in ids:
        v = flat_updates_dict[cid]
        v_norm = np.linalg.norm(v) + 1e-12
        cos_sim = float(np.dot(v, root) / (v_norm * root_norm))
        trust_scores[cid] = max(0.0, cos_sim)

    sum_t = sum(trust_scores.values())
    if sum_t > 1e-12:
        trust_weights = {cid: t / sum_t for cid, t in trust_scores.items()}
    else:
        trust_weights = {cid: 1.0 / n for cid in ids}

    return fedavg_aggregate(updates, trust_weights)


# ═══════════════════════════════════════════════════════════════════════
# TASL Trust Scoring (v3)
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

    # Norm gating
    norms = {cid: np.linalg.norm(v) for cid, v in flat_updates.items()}
    median_norm = np.median(list(norms.values()))
    norm_gate = median_norm * 2.5

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


# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# BSP: Blockchain-Shared Pre-training (data layer)
# ═══════════════════════════════════════════════════════════════════════

def bsp_pretrain(global_state, shared_loader, device, model_fn, lr=0.01, epochs=1):
    """Pre-train on shared pool with identical seed/order/hyperparams.
    Returns the new global_state after pre-training."""
    model = model_fn().to(device)
    model.load_state_dict({k: v.to(device) for k, v in global_state.items()})
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in shared_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
    return {k: v.clone().cpu() for k, v in model.state_dict().items()}


# Single experiment run
# ═══════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════
# Governance: V1/V2 Audit + Corrupt Executor + Healing
# ═══════════════════════════════════════════════════════════════════════

EXECUTOR_ATTACKS = {
    "A1": lambda w, tw, byz, rng: {cid: 0.7 for cid in byz},
    "A2": lambda w, tw, byz, rng: {cid: min(0.50, w.get(cid, 0.0) + 0.25) for cid in byz},
    "A3": None,  # handled separately
}

def apply_executor_attack(tampered_w, trust_w, byz_set, rng, attack_type="A1"):
    """Corrupt Executor tampers with aggregation weights."""
    if attack_type == "A1":
        for cid in byz_set:
            tampered_w[cid] = 0.7
    elif attack_type == "A2":
        for cid in byz_set:
            old = tampered_w.get(cid, 0.0)
            tampered_w[cid] = min(0.50, old + 0.25)
    elif attack_type == "A3":
        for cid in byz_set:
            tampered_w[cid] = 0.7
        # Suppress top2 honest clients
        honest = {c: trust_w.get(c, 0.0) for c in trust_w if c not in byz_set}
        if honest:
            top2 = sorted(honest, key=honest.get, reverse=True)[:2]
            for cid in top2:
                tampered_w[cid] = 0.0
    return tampered_w

def v1_weight_audit(tampered_w, correct_w, num_clients, tolerance=2e-3):
    """V1: Full-scan weight deviation detection."""
    for cid in range(num_clients):
        if abs(tampered_w.get(cid, 0.0) - correct_w.get(cid, 0.0)) > tolerance:
            return True
    return False

def v2_rank_audit(trust_rank_history, num_clients, consecutive=3, low_threshold=0.3):
    """V2: Membership audit via rank persistence."""
    threshold_rank = int(num_clients * (1 - low_threshold))
    suspected = set()
    for cid in range(num_clients):
        ranks = trust_rank_history.get(cid, [])
        if len(ranks) < consecutive:
            continue
        recent = ranks[-consecutive:]
        if all(r >= threshold_rank for r in recent):
            suspected.add(cid)
    return len(suspected) > 0, suspected

def make_chaos_schedule(rounds, seed=2026, attacks_per_block=6):
    """6/10 rounds are attack rounds, random A1/A2/A3."""
    rng = random.Random(seed)
    schedule = {}
    block_start = 1
    while block_start <= rounds:
        block_end = min(block_start + 9, rounds)
        block_rounds = list(range(block_start, block_end + 1))
        atk_rounds = rng.sample(block_rounds, k=min(attacks_per_block, len(block_rounds)))
        for r in atk_rounds:
            schedule[r] = rng.choice(["A1", "A2", "A3"])
        block_start += 10
    return schedule

def run_single(algo, attack, client_loaders, test_loader, cfg,
               model_fn, n_clients, n_byz, rounds, device, proxy_loader=None,
               governance_mode=None, chaos_schedule=None):
    """Run a single algo × attack experiment."""
    seed = cfg['experiment']['seed']
    set_seed(seed)

    train_cfg = cfg['training']
    local_epochs = int(train_cfg['local_epochs'])
    lr = float(train_cfg['lr'])
    momentum = float(train_cfg.get('momentum', 0.9))
    weight_decay = float(train_cfg.get('weight_decay', 1e-4))
    nesterov = train_cfg.get('nesterov', False)
    lr_schedule_cfg = train_cfg.get('lr_schedule', None)
    num_classes = int(cfg['model']['num_classes'])
    algo_params = cfg.get('algorithm_params', {})

    byzantine_set = set(range(n_byz))
    global_model = model_fn().to(device)
    global_state = {k: v.clone().cpu() for k, v in global_model.state_dict().items()}

    # BSP: Blockchain-Shared Pre-training (data layer, TASL only)
    bsp_cfg = algo_params.get('bsp', {})
    if algo == 'tasl' and bsp_cfg.get('enabled', False):
        dataset_name = cfg['data']['dataset']
        n_per_class = int(bsp_cfg.get('n_per_class', 50))
        bsp_lr = float(bsp_cfg.get('lr', lr))
        bsp_epochs = int(bsp_cfg.get('epochs', 1))
        shared_loader = create_proxy_loader(dataset_name, n_per_class=n_per_class,
                                            batch_size=cfg['data'].get('batch_size', 64))
        global_state = bsp_pretrain(global_state, shared_loader, device, model_fn,
                                    lr=bsp_lr, epochs=bsp_epochs)
        global_model.load_state_dict({k: v.to(device) for k, v in global_state.items()})
        print("  [BSP] Pre-trained on %d shared samples" % (n_per_class * num_classes))

    # Trust-layer: progressive exclusion state (TASL only)
    excluded_clients = set()
    violation_counts = {cid: 0 for cid in range(n_clients)}
    max_exclusions = n_byz  # safety cap
    _tasl_cfg = cfg.get('algorithm_params', {}).get('tasl', {})
    if not _tasl_cfg.get('exclusion', True):
        max_exclusions = 0  # ablation: disable progressive exclusion

    # Governance state (exp3 only)
    gov_active = governance_mode is not None
    use_v1 = governance_mode in ("trust_v1", "tas")
    use_v2 = governance_mode in ("trust_v2", "tas")
    trust_rank_history = {cid: [] for cid in range(n_clients)}
    gov_v1_detections = 0
    gov_v2_detections = 0
    gov_blocked = 0
    gov_attack_rounds = 0
    gov_healed = 0

    acc_history = []
    asr_history = []
    ema_weights = None
    cos_history = []
    prev_anchor = None
    temporal_anchor = None

    # Attack-specific params
    attack_kwargs = {}
    if attack == 'gaussian_noise':
        noise_cfg = algo_params.get('gaussian_noise', {})
        attack_kwargs['noise_scale'] = noise_cfg.get('noise_scale', 5.0)
    elif attack == 'min_max':
        mm_cfg = algo_params.get('min_max', {})
        attack_kwargs['z_max'] = mm_cfg.get('z_max', 2.0)
    elif attack == 'scaling':
        sc_cfg = algo_params.get('scaling', {})
        attack_kwargs['scaling_factor'] = sc_cfg.get('scaling_factor', 100.0)

    for r in range(1, rounds + 1):
        current_lr = get_lr_for_round(r - 1, lr, lr_schedule_cfg)
        local_updates = {}
        flat_updates = {}

        # For min_max attack: collect honest updates first, then craft malicious
        if attack == 'min_max' and attack != 'none':
            # Phase 1: train all honest clients
            honest_flat = {}
            for cid in range(n_clients):
                if cid not in byzantine_set:
                    if algo == 'fedprox':
                        mu = algo_params.get('fedprox', {}).get('mu', 0.01)
                        local_state = train_fedprox(
                            global_state, client_loaders[cid], device, model_fn,
                            local_epochs=local_epochs, lr=current_lr, momentum=momentum,
                            weight_decay=weight_decay, mu=mu, nesterov=nesterov,
                        )
                    else:
                        local_state = train_honest(
                            global_state, client_loaders[cid], device, model_fn,
                            local_epochs=local_epochs, lr=current_lr, momentum=momentum,
                            weight_decay=weight_decay, nesterov=nesterov,
                        )
                    upd = state_sub(local_state, global_state)
                    local_updates[cid] = upd
                    flat_updates[cid] = flatten_update(upd)
                    honest_flat[cid] = flat_updates[cid]

            # Phase 2: craft malicious updates based on honest statistics
            for cid in range(n_clients):
                if cid in byzantine_set:
                    local_state = attack_min_max(
                        global_state, client_loaders[cid], device, model_fn,
                        local_epochs=local_epochs, lr=lr, momentum=momentum,
                        weight_decay=weight_decay, num_classes=num_classes,
                        z_max=attack_kwargs.get('z_max', 2.0),
                        honest_flat_updates=honest_flat,
                    )
                    upd = state_sub(local_state, global_state)
                    local_updates[cid] = upd
                    flat_updates[cid] = flatten_update(upd)
        else:
            # Standard attack flow
            for cid in range(n_clients):
                if cid in byzantine_set and attack != 'none':
                    attack_fn = ATTACK_FNS.get(attack, train_honest)
                    local_state = attack_fn(
                        global_state, client_loaders[cid], device, model_fn,
                        local_epochs=local_epochs, lr=lr, momentum=momentum,
                        weight_decay=weight_decay, num_classes=num_classes,
                        **attack_kwargs,
                    )
                else:
                    if algo == 'fedprox':
                        mu = algo_params.get('fedprox', {}).get('mu', 0.01)
                        local_state = train_fedprox(
                            global_state, client_loaders[cid], device, model_fn,
                            local_epochs=local_epochs, lr=current_lr, momentum=momentum,
                            weight_decay=weight_decay, mu=mu, nesterov=nesterov,
                        )
                    else:
                        local_state = train_honest(
                            global_state, client_loaders[cid], device, model_fn,
                            local_epochs=local_epochs, lr=current_lr, momentum=momentum,
                            weight_decay=weight_decay, nesterov=nesterov,
                        )

                upd = state_sub(local_state, global_state)
                local_updates[cid] = upd
                flat_updates[cid] = flatten_update(upd)

        # Remove excluded clients from this round (TASL progressive exclusion)
        if algo == 'tasl' and excluded_clients:
            for ecid in list(excluded_clients):
                local_updates.pop(ecid, None)
                flat_updates.pop(ecid, None)

        # Aggregation
        tasl_cfg = algo_params.get('tasl', {})
        if algo == 'fedavg':
            agg_update = fedavg_aggregate(local_updates)
        elif algo == 'fedprox':
            agg_update = fedavg_aggregate(local_updates)
        elif algo == 'multi_krum':
            agg_update = multi_krum_aggregate(flat_updates, local_updates, f=n_byz)
        elif algo == 'trimmed_mean':
            agg_update = trimmed_mean_aggregate(local_updates, beta=0.2)
        elif algo == 'fltrust':
            # FLTrust: compute g0 from proxy dataset
            if proxy_loader is not None:
                proxy_root = compute_proxy_root(global_state, proxy_loader, device, model_fn, lr=lr)
            else:
                proxy_root = None
            agg_update = fltrust_aggregate(flat_updates, local_updates, device, root=proxy_root)
        elif algo == 'tasl':
            fusion_mode = tasl_cfg.get('fusion', 'multiplicative')
            if fusion_mode == 'nograd':
                # Ablation: no gradient-layer trust, equal weights
                n_active = len(local_updates)
                trust_weights = {cid: 1.0 / n_active for cid in local_updates}
                cos_sims = {}
                new_anchor = prev_anchor
            else:
                trust_weights, cos_sims, new_anchor = compute_tasl_trust_weights(
                    flat_updates,
                    trust_power=tasl_cfg.get('trust_power', 5.0),
                    max_weight_ratio=tasl_cfg.get('weight_cap_ratio', 1.5),
                    min_cos_threshold=tasl_cfg.get('min_cos', 0.2),
                    norm_penalty_strength=tasl_cfg.get('norm_penalty', 0.8),
                    refine_anchor=True,
                    ema_weights=ema_weights,
                    ema_alpha=tasl_cfg.get('ema_alpha', 0.7),
                    cos_history=cos_history,
                    prev_anchor=prev_anchor,
                    temporal_anchor=temporal_anchor,
                )
            ema_weights = dict(trust_weights)
            cos_history.append(dict(cos_sims))
            prev_anchor = new_anchor
            if fusion_mode == 'additive':
                # Ablation: additive fusion (p+omega)/2 with equal data trust
                n_active = len(local_updates)
                eq_w = {cid: 1.0 / n_active for cid in local_updates}
                fused = {cid: (trust_weights.get(cid, 0.0) + eq_w.get(cid, 0.0)) / 2.0 for cid in local_updates}
                s = sum(fused.values()) or 1e-12
                trust_weights = {cid: v / s for cid, v in fused.items()}
            agg_update = fedavg_aggregate(local_updates, trust_weights)

            agg_flat = flatten_update(agg_update)
            if np.linalg.norm(agg_flat) > 1e-12:
                temporal_anchor = agg_flat

            # ── Governance: Corrupt Executor + V1/V2 audit ──
            if gov_active and chaos_schedule and r in chaos_schedule:
                gov_attack_rounds += 1
                atk_type = chaos_schedule[r]

                # Executor tampers with trust_weights
                tampered_w = dict(trust_weights)
                tampered_w = apply_executor_attack(tampered_w, trust_weights,
                                                   byzantine_set, None, atk_type)

                # V1: weight deviation audit (full scan)
                v1_det = False
                if use_v1:
                    v1_det = v1_weight_audit(tampered_w, trust_weights, n_clients)
                    if v1_det:
                        gov_v1_detections += 1

                # V2: rank persistence audit (independent of V1)
                v2_det = False
                v2_suspected = set()
                if use_v2 and r >= 3:
                    v2_det, v2_suspected = v2_rank_audit(trust_rank_history, n_clients)
                    if v2_det:
                        gov_v2_detections += 1

                # Healing: if detected, use healed weights
                if v1_det or v2_det:
                    gov_blocked += 1
                    healed_w = dict(trust_weights)
                    for cid in byzantine_set:
                        healed_w[cid] = 0.0
                    for cid in v2_suspected:
                        healed_w[cid] = 0.0
                    for cid in range(n_clients):
                        if tampered_w.get(cid, 0.0) > 0.3:
                            healed_w[cid] = 0.0
                    h_sum = sum(healed_w.values())
                    if h_sum > 1e-12:
                        healed_w = {c: w / h_sum for c, w in healed_w.items()}
                    else:
                        honest = [c for c in range(n_clients) if c not in byzantine_set]
                        healed_w = {c: 1.0/len(honest) for c in honest}
                    agg_update = fedavg_aggregate(local_updates, healed_w)
                    gov_healed += 1
                else:
                    # Not detected: use tampered weights
                    agg_update = fedavg_aggregate(local_updates, tampered_w)
                # Update rank history (after healing/tampering)
                sorted_cids = sorted(range(n_clients), key=lambda c: trust_weights.get(c, 0.0))
                rank_map = {cid: rank for rank, cid in enumerate(sorted_cids)}
                for cid in range(n_clients):
                    trust_rank_history[cid].append(rank_map[cid])
            elif gov_active:
                # Non-attack round: update rank history
                sorted_cids = sorted(range(n_clients), key=lambda c: trust_weights.get(c, 0.0))
                rank_map = {cid: rank for rank, cid in enumerate(sorted_cids)}
                for cid in range(n_clients):
                    trust_rank_history[cid].append(rank_map[cid])

            # Trust-layer: progressive exclusion (TASL only)
            if len(excluded_clients) < max_exclusions:
                # Compute per-client violation signals
                active_ids = [cid for cid in range(n_clients) if cid not in excluded_clients]
                if len(active_ids) >= 2:
                    active_norms = [np.linalg.norm(flat_updates.get(cid, np.zeros(1))) for cid in active_ids]
                    median_norm_r = np.median(active_norms) + 1e-12
                    for cid in active_ids:
                        if cid not in flat_updates:
                            continue
                        cs = cos_sims.get(cid, 0.5)
                        nr = np.linalg.norm(flat_updates[cid]) / median_norm_r
                        violated = (cs < -0.1) or (nr > 2.0)
                        if violated:
                            violation_counts[cid] += 1
                        else:
                            violation_counts[cid] = max(0, violation_counts[cid] - 1)
                        # Progressive exclusion: 2 consecutive violations => exclude
                        if violation_counts[cid] >= 2 and len(excluded_clients) < max_exclusions:
                            excluded_clients.add(cid)
                            violation_counts[cid] = 0
                            print("    [Exclusion] Client %d excluded (round %d)" % (cid, r))
        else:
            raise ValueError(f"Unknown algo: {algo}")

        global_state = state_add(global_state, agg_update)
        global_model.load_state_dict(global_state)
        acc, asr = evaluate_full(global_model, test_loader, device,
                                 attack=attack, num_classes=num_classes)
        acc_history.append(acc)
        asr_history.append(asr)

        if r % 10 == 0 or r == 1:
            byz_avg_w = 0.0
            if algo == 'tasl' and ema_weights:
                byz_avg_w = np.mean([ema_weights.get(c, 0) for c in byzantine_set])
            asr_str = f" asr={asr:.2f}%" if attack == 'label_flip' else ""
            print(f"  [{algo}/{attack}] R{r:02d} acc={acc:.2f}%"
                  + (f" byz_w={byz_avg_w:.4f}" if algo == 'tasl' else "") + asr_str)

    best_acc = max(acc_history)
    avg_acc = float(np.mean(acc_history))
    last_acc = acc_history[-1]
    best_asr = max(asr_history) if asr_history else 0.0
    avg_asr = float(np.mean(asr_history)) if asr_history else 0.0

    return {
        'best_acc': best_acc, 'avg_acc': avg_acc, 'last_acc': last_acc,
        'acc_history': acc_history, 'asr_history': asr_history,
        'best_asr': best_asr, 'avg_asr': avg_asr,
    }


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════

ATTACK_LABELS = {
    'label_flip': 'Label Flip', 'sign_flip': 'Sign Flip',
    'gaussian_noise': 'Gaussian Noise', 'min_max': 'Min-Max',
    'scaling': 'Scaling',
}

ALGO_LABELS = {
    'fedavg': 'FedAvg', 'fedprox': 'FedProx', 'multi_krum': 'Multi-Krum',
    'trimmed_mean': 'Trimmed Mean', 'fltrust': 'FLTrust',
    'tasl': 'TASL (Ours)',
}

ALGO_COLORS = {
    'fedavg': '#E53935', 'fedprox': '#00ACC1', 'multi_krum': '#1E88E5',
    'trimmed_mean': '#FB8C00', 'fltrust': '#8E24AA',
    'tasl': '#43A047',
}


def plot_heatmap(save_path, results_df, algos, attacks, metric='best_acc'):
    import pandas as pd
    fig, ax = plt.subplots(figsize=(10, 5))
    data = np.zeros((len(algos), len(attacks)))
    for i, algo in enumerate(algos):
        for j, atk in enumerate(attacks):
            row = results_df[(results_df['algo'] == algo) & (results_df['attack'] == atk)]
            if not row.empty:
                data[i, j] = row[metric].values[0]

    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels([ATTACK_LABELS.get(a, a) for a in attacks], fontsize=11)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels([ALGO_LABELS.get(a, a) for a in algos], fontsize=11)

    for i in range(len(algos)):
        for j in range(len(attacks)):
            val = data[i, j]
            color = 'white' if val < 40 or val > 80 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=13, fontweight='bold', color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    metric_label = 'Best Accuracy' if metric == 'best_acc' else 'Avg Accuracy'
    cbar.set_label(f'{metric_label} (%)', fontsize=12)
    ax.set_title(f'Byzantine Resilience Matrix — {metric_label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved heatmap: {save_path}")


def plot_curves(save_path, all_results, attack, rounds):
    fig, ax = plt.subplots(figsize=(12, 6))
    for algo, res in all_results.items():
        if attack in res:
            history = res[attack]['acc_history']
            ax.plot(range(1, rounds + 1), history,
                    label=ALGO_LABELS.get(algo, algo),
                    color=ALGO_COLORS.get(algo, 'gray'),
                    linewidth=2.0, alpha=0.9)

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Convergence under {ATTACK_LABELS.get(attack, attack)} Attack',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved curves: {save_path}")


def plot_bar_comparison(save_path, results_df, algos, attacks, metric='best_acc'):
    import pandas as pd
    fig, axes = plt.subplots(1, len(attacks), figsize=(5 * len(attacks), 5), sharey=True)
    if len(attacks) == 1:
        axes = [axes]
    for j, atk in enumerate(attacks):
        ax = axes[j]
        vals = []
        for algo in algos:
            row = results_df[(results_df['algo'] == algo) & (results_df['attack'] == atk)]
            vals.append(row[metric].values[0] if not row.empty else 0)

        colors = [ALGO_COLORS.get(a, 'gray') for a in algos]
        bars = ax.bar(range(len(algos)), vals, color=colors, edgecolor='white', linewidth=1.5)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels([ALGO_LABELS.get(a, a).replace(' ', '\n') for a in algos], fontsize=8)
        ax.set_title(ATTACK_LABELS.get(atk, atk), fontsize=13, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    metric_label = 'Best Accuracy (%)' if metric == 'best_acc' else 'Avg Accuracy (%)'
    axes[0].set_ylabel(metric_label, fontsize=12)
    fig.suptitle(f'Byzantine Resilience Comparison — {metric_label}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved bar chart: {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp2: Byzantine Resilience Matrix (YAML-config driven)")
    parser.add_argument("--config", type=str, required=True, help="YAML config file path")
    parser.add_argument("--algorithms", type=str, default=None,
                        help="Comma-separated algo list (overrides YAML). e.g. tasl,fedavg")
    parser.add_argument("--attacks", type=str, default=None,
                        help="Comma-separated attack list (overrides YAML). e.g. label_flip,min_max")
    parser.add_argument("--quick", action='store_true',
                        help="Quick test: fewer clients, fewer rounds")
    parser.add_argument("--rounds", type=int, default=None, help="Override number of rounds")
    parser.add_argument("--no-baseline", action='store_true',
                        help="Skip clean baseline (use when you already have baseline results)")
    parser.add_argument("--governance-mode", type=str, default=None,
                        choices=["fedavg", "trust_only", "trust_v1", "trust_v2", "tas"],
                        help="Governance mode for Executor threat ablation")
    parser.add_argument("--corrupt-executor", action='store_true',
                        help="Enable Corrupt Executor attacks")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides
    if args.algorithms:
        cfg['algorithms'] = [a.strip() for a in args.algorithms.split(',')]
    if args.attacks:
        cfg['byzantine']['attacks'] = [a.strip() for a in args.attacks.split(',')]
    if args.rounds:
        cfg['training']['rounds'] = args.rounds

    if args.quick:
        cfg['data']['n_clients'] = 5
        cfg['training']['rounds'] = 10
        cfg['training']['local_epochs'] = 1
        cfg['byzantine']['ratio'] = 0.4

    set_seed(cfg['experiment']['seed'])
    device = torch.device(cfg['experiment'].get('device', 'cuda') if torch.cuda.is_available() else 'cpu')

    n_clients = cfg['data']['n_clients']
    n_byz = int(n_clients * cfg['byzantine']['ratio'])
    rounds = cfg['training']['rounds']
    algos = cfg['algorithms']
    attacks = cfg['byzantine']['attacks']

    # Model factory
    model_fn = lambda: create_model(cfg)

    print("=" * 70)
    print(f"[Exp2] Byzantine Resilience Matrix — {cfg['experiment']['name']}")
    print(f"  Dataset: {cfg['data']['dataset']}, Model: {cfg['model']['name']}")
    print(f"  {n_clients} clients, {n_byz} Byzantine ({cfg['byzantine']['ratio']:.0%})")
    print(f"  Split: {cfg['data'].get('split', 'non_iid')} (alpha={cfg['data']['alpha']})")
    print(f"  {rounds} rounds, local_epochs={cfg['training']['local_epochs']}")
    print(f"  Algorithms: {algos}")
    print(f"  Attacks: {attacks}")
    print(f"  Device: {device}")
    print("=" * 70)

    # Load data
    dataset_name = cfg['data']['dataset']
    data_loader_fn = DATA_LOADERS.get(dataset_name)
    if data_loader_fn is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(DATA_LOADERS.keys())}")
    client_loaders, test_loader = data_loader_fn(cfg)
    print(f"  Data loaded: {n_clients} clients, test set ready")

    # Create proxy dataset for FLTrust g0 (50 images per class)
    if 'fltrust' in algos:
        proxy_loader = create_proxy_loader(dataset_name, n_per_class=50)
    else:
        proxy_loader = None

    # ── Clean baseline ──
    if args.no_baseline:
        clean_result = None
        print("\n>>> Clean Baseline: SKIPPED (--no-baseline) <<<")
    else:
        print("\n>>> Clean Baseline (no attack) <<<")
        clean_result = run_single('fedavg', 'none', client_loaders, test_loader, cfg,
                                  model_fn, n_clients, n_byz=0, rounds=rounds, device=device,
                                  proxy_loader=proxy_loader)
        print(f"  Clean: best={clean_result['best_acc']:.2f}%, avg={clean_result['avg_acc']:.2f}%")

    # ── Run all combinations ──
    all_results = {algo: {} for algo in algos}
    csv_rows = []
    total_runs = len(algos) * len(attacks)
    run_idx = 0

    for algo in algos:
        for attack in attacks:
            run_idx += 1
            print(f"\n>>> [{run_idx}/{total_runs}] {algo} × {attack} <<<")

            result = run_single(algo, attack, client_loaders, test_loader, cfg,
                                model_fn, n_clients, n_byz, rounds, device, proxy_loader=proxy_loader)

            all_results[algo][attack] = result

            # Compute ASR
            if attack == 'label_flip':
                final_asr = result['avg_asr']
            elif clean_result is not None:
                final_asr = max(0, 100.0 * (1 - result['avg_acc'] / max(clean_result['avg_acc'], 1e-6)))
            else:
                final_asr = 0.0

            csv_rows.append({
                'algo': algo, 'attack': attack,
                'split': cfg['data'].get('split', 'non_iid'),
                'byzantine_ratio': cfg['byzantine']['ratio'],
                'best_acc': result['best_acc'], 'avg_acc': result['avg_acc'],
                'last_acc': result['last_acc'], 'asr': final_asr,
            })

            asr_str = f", asr={final_asr:.2f}%" if attack != 'none' else ""
            print(f"  => best={result['best_acc']:.2f}%, avg={result['avg_acc']:.2f}%,"
                  f" last={result['last_acc']:.2f}%{asr_str}")

    # ── Save results ──
    results_dir = os.path.join(BLOCKCHAIN_DIR, cfg['experiment'].get('output_dir', 'results'),
                               cfg['experiment']['name'])
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "exp2_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['algo', 'attack', 'split', 'byzantine_ratio',
                                                'best_acc', 'avg_acc', 'last_acc', 'asr'])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Add clean baseline
    if clean_result is not None:
        csv_rows.append({
            'algo': 'clean_baseline', 'attack': 'none',
            'split': cfg['data'].get('split', 'non_iid'),
            'byzantine_ratio': 0.0, 'best_acc': clean_result['best_acc'],
            'avg_acc': clean_result['avg_acc'], 'last_acc': clean_result['last_acc'],
            'asr': 0.0,
        })

    # ── Summary table ──
    print("\n" + "=" * 70)
    print(f"[Exp2] Summary — {cfg['data']['dataset']}, {cfg['byzantine']['ratio']:.0%} Byzantine")
    print("=" * 70)
    header = f"{'Attack':<18s}" + "".join(f" {ALGO_LABELS.get(a, a):>10s}" for a in algos)
    print(header)
    print("-" * 70)
    for attack in attacks:
        row_str = f"{ATTACK_LABELS.get(attack, attack):<18s}"
        for algo in algos:
            val = all_results[algo][attack]['best_acc']
            all_vals = [all_results[a][attack]['best_acc'] for a in algos]
            is_best = val == max(all_vals)
            marker = " *" if is_best else ""
            row_str += f" {val:>8.2f}{marker:<2s}"
        print(row_str)
    if clean_result is not None:
        print(f"\nClean Baseline: best={clean_result['best_acc']:.2f}%, avg={clean_result['avg_acc']:.2f}%")
    else:
        print(f"\nClean Baseline: SKIPPED")

    # ── Plots ──
    import pandas as pd
    results_df = pd.DataFrame(csv_rows)

    plot_heatmap(os.path.join(results_dir, "exp2_heatmap.png"), results_df, algos, attacks, 'best_acc')
    plot_bar_comparison(os.path.join(results_dir, "exp2_bar.png"), results_df, algos, attacks, 'best_acc')

    for attack in attacks:
        plot_curves(os.path.join(results_dir, f"exp2_curves_{attack}.png"),
                    all_results, attack, rounds)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Plots: {results_dir}")


if __name__ == "__main__":
    main()
