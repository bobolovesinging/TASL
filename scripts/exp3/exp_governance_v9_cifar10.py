"""
实验三 v9: TAS 治理效能 — CIFAR-10 / ResNet-18 (GroupNorm)

v8(v9) 继承:
  四组实验:
    1. Clean Baseline: 无 Byzantine, 纯 FedAvg
    2. Byz+HonestE: 有 Byzantine, E 诚实, Cubic Trust Scoring
    3. Attacked FedAvg: 有 Byzantine, 纯 FedAvg (无任何防御) — 关键对照组
    4. Attacked TAS: 有 Byzantine + Corrupt E, Cubic Trust + V1/V2 + healing

CIFAR-10 适配:
  - 模型: CifarCNN (ResNet-18 + GroupNorm, ~11M 参数, Non-IID 友好)
  - 数据: CIFAR-10 3×32×32 RGB, Dirichlet Non-IID α=0.3
  - 训练: 100 轮, local_epochs=5, lr=0.1 + CosineAnnealing
  - 攻击: Label Flip (10类), 6/10 轮频率

与原 MNIST v8 的差异:
  - 替换 FedAvgCNN → CifarCNN
  - 替换 MNIST data loading → CIFAR-10 data loading
  - 100 轮 (vs 50), local_epochs=5 (vs 2), scheduler (vs 无)
  - Same attack/defense/audit logic
"""
import csv, math, os, random, sys, argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

# ═══════════════════════════════════════════════════════════════════════
# CIFAR-10 ResNet-18 (GroupNorm — Non-IID 联邦学习稳定)
# ═══════════════════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                                padding=1, bias=False)
        self.gn1   = nn.GroupNorm(min(4, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1,
                                padding=1, bias=False)
        self.gn2   = nn.GroupNorm(min(4, out_channels), out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(min(4, out_channels), out_channels))

    def forward(self, x):
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class CifarCNN(nn.Module):
    """ResNet-18 + GroupNorm for CIFAR-10, ~11M params"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.GroupNorm(4, 64), nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(512, num_classes)
        self._init_weights()

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.gap(x); x = x.view(x.size(0), -1)
        return self.fc(x)


# ═══════════════════════════════════════════════════════════════════════
# Data: CIFAR-10, Non-IID Dirichlet α=0.3
# ═══════════════════════════════════════════════════════════════════════

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)


def dirichlet_split(labels, n_clients, alpha=0.3):
    """Dirichlet Non-IID partition"""
    n_classes = len(np.unique(labels))
    idx_per_class = {c: np.where(labels == c)[0] for c in range(n_classes)}
    client_indices = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        idx_c = idx_per_class[c]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        proportions = (proportions * len(idx_c)).astype(int)
        # redistribute remainder
        remainder = len(idx_c) - proportions.sum()
        for i in range(remainder):
            proportions[i % n_clients] += 1

        start = 0
        for k in range(n_clients):
            n_k = proportions[k]
            client_indices[k].extend(idx_c[start:start + n_k].tolist())
            start += n_k

    return client_indices


def get_cifar10_loaders(client_num=10, batch_size=64, alpha=0.3, data_root=None):
    """CIFAR-10 Non-IID client loaders + test loader"""
    if data_root is None:
        data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "data")
    os.makedirs(data_root, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test)

    train_labels = np.array([train_set.targets[i] for i in range(len(train_set))])
    client_indices = dirichlet_split(train_labels, client_num, alpha=alpha)

    client_loaders = []
    for i in range(client_num):
        subset = Subset(train_set, client_indices[i])
        client_loaders.append(DataLoader(subset, batch_size=batch_size,
                                          shuffle=True, num_workers=2, pin_memory=True))

    test_loader = DataLoader(test_set, batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"[Data] CIFAR-10: {client_num} clients, Non-IID α={alpha}, "
          f"test={len(test_set)}")

    # Show distribution
    for i in range(client_num):
        c_labels = train_labels[client_indices[i]]
        unique, counts = np.unique(c_labels, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        n_total = sum(dist.values())
        top3 = sorted(dist.items(), key=lambda x: -x[1])[:3]
        print(f"  Client {i}: total={n_total}, top3={top3}")

    return client_loaders, test_loader


# ═══════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RoundAttack:
    active: bool
    attack_type: Optional[str] = None  # A1/A2/A3

@dataclass
class GroupRunResult:
    final_accuracy: float; best_accuracy: float; avg_accuracy: float
    accuracy_history: List[float]; round_logs: List[Dict]; metrics: Dict[str, float]


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def evaluate(model, test_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += model(x).argmax(dim=1).eq(y).sum().item()
            total += y.size(0)
    return 100.0 * correct / max(total, 1)


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


def weighted_average_updates(updates, weights):
    valid = [(cid, float(weights.get(cid, 0.0))) for cid in updates
             if float(weights.get(cid, 0.0)) > 0]
    if not valid:
        eq = 1.0 / max(len(updates), 1)
        valid = [(cid, eq) for cid in updates]
    w_sum = sum(w for _, w in valid)
    normed = {cid: w / max(w_sum, 1e-12) for cid, w in valid}
    agg = {k: torch.zeros_like(v) for k, v in updates[next(iter(normed))].items()}
    for cid, w in normed.items():
        for k in agg:
            agg[k] += updates[cid][k] * w
    return agg


# ═══════════════════════════════════════════════════════════════════════
# Trust Scoring (same as v8: spatial median + cos^5 + weight cap)
# ═══════════════════════════════════════════════════════════════════════

def compute_trust_weights(flat_updates, trust_power=5.0, max_weight_ratio=2.0):
    n = len(flat_updates)
    stacked = np.stack(list(flat_updates.values()), axis=0)
    anchor = np.median(stacked, axis=0)
    anchor_norm = np.linalg.norm(anchor) + 1e-12

    sim_scores, raw_scores = {}, {}
    for cid, v in flat_updates.items():
        score = float(np.dot(v, anchor) / ((np.linalg.norm(v) + 1e-12) * anchor_norm))
        sim_scores[cid] = score
        raw_scores[cid] = max(0.0, score) ** trust_power

    sum_w = sum(raw_scores.values())
    trust_weights = ({cid: w / sum_w for cid, w in raw_scores.items()}
                     if sum_w > 1e-12 else {cid: 1.0 / n for cid in raw_scores})

    max_w = max_weight_ratio / n
    capped = {cid: min(w, max_w) for cid, w in trust_weights.items()}
    cap_sum = sum(capped.values())
    if cap_sum > 1e-12:
        trust_weights = {cid: w / cap_sum for cid, w in capped.items()}

    return sim_scores, trust_weights


# ═══════════════════════════════════════════════════════════════════════
# Local training
# ═══════════════════════════════════════════════════════════════════════

def train_one_client(global_state, loader, device, local_epochs=5, lr=0.1,
                     scheduler_fn=None, scheduler=None):
    model = CifarCNN().to(device)
    model.load_state_dict(global_state)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for ep in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return model.state_dict()


def train_one_client_label_flip(global_state, loader, device, local_epochs=5,
                                 lr=0.1, num_classes=10, scheduler=None):
    model = CifarCNN().to(device)
    model.load_state_dict(global_state)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for ep in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_flipped = num_classes - 1 - y
            optimizer.zero_grad()
            loss = criterion(model(x), y_flipped)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return model.state_dict()


# ═══════════════════════════════════════════════════════════════════════
# Attack scheduling (6/10 rounds)
# ═══════════════════════════════════════════════════════════════════════

ATTACK_TYPES = ["A1", "A2", "A3"]


def make_chaos_schedule(rounds, seed=42, attack_types=None, attacks_per_block=6):
    if attack_types is None:
        attack_types = ATTACK_TYPES
    rng = random.Random(seed)
    schedule = {r: RoundAttack(active=False) for r in range(1, rounds + 1)}
    block_start = 1
    while block_start <= rounds:
        block_end = min(block_start + 9, rounds)
        block_rounds = list(range(block_start, block_end + 1))
        atk_rounds = rng.sample(block_rounds, k=min(attacks_per_block, len(block_rounds)))
        for r in atk_rounds:
            schedule[r] = RoundAttack(active=True, attack_type=rng.choice(attack_types))
        block_start += 10
    return schedule


# ═══════════════════════════════════════════════════════════════════════
# Attack implementations
# ═══════════════════════════════════════════════════════════════════════

def apply_attack_a1(tampered_weights, sim_scores, byzantine_ids, rng, **kw):
    targets = set()
    for cid in byzantine_ids:
        tampered_weights[cid] = 0.7
        targets.add(cid)
    return targets


def apply_attack_a2(tampered_weights, sim_scores, byzantine_ids, rng, boost=0.25, **kw):
    targets = set()
    for cid in byzantine_ids:
        old_w = tampered_weights.get(cid, 0.0)
        tampered_weights[cid] = min(0.50, old_w + boost)
        targets.add(cid)
    return targets


def apply_attack_a3(tampered_weights, sim_scores, byzantine_ids, rng, **kw):
    targets = set()
    for cid in byzantine_ids:
        tampered_weights[cid] = 0.7
        targets.add(cid)
    honest_sims = {cid: sim for cid, sim in sim_scores.items()
                   if cid not in set(byzantine_ids)}
    if honest_sims:
        top2 = sorted(honest_sims, key=honest_sims.get, reverse=True)[:2]
        for cid in top2:
            tampered_weights[cid] = 0.0
            targets.add(cid)
    return targets


ATTACK_FNS = {"A1": apply_attack_a1, "A2": apply_attack_a2, "A3": apply_attack_a3}


# ═══════════════════════════════════════════════════════════════════════
# V1 / V2 Auditing
# ═══════════════════════════════════════════════════════════════════════

def v1_audit_sampled(tampered_weights, correct_weights, num_clients,
                     sample_ratio=0.3, tolerance=2e-3, round_number=1):
    rng = np.random.RandomState(round_number)
    ids = list(range(num_clients))
    sample_size = max(1, int(num_clients * sample_ratio))
    sampled = rng.choice(ids, size=sample_size, replace=False).tolist()
    for cid in sampled:
        expected = correct_weights.get(cid, 0.0)
        actual = tampered_weights.get(cid, 0.0)
        if abs(actual - expected) > tolerance:
            return True, {"detected_cid": cid, "sampled": sampled,
                          "expected": expected, "actual": actual}
    return False, {"sampled": sampled}


def v2_membership_audit(trust_rank_history, num_clients, byzantine_set,
                         low_rank_threshold=0.3, consecutive_rounds=3):
    suspected = set()
    for cid in range(num_clients):
        ranks = trust_rank_history.get(cid, [])
        if len(ranks) < consecutive_rounds:
            continue
        recent = ranks[-consecutive_rounds:]
        threshold_rank = int(num_clients * (1 - low_rank_threshold))
        if all(r >= threshold_rank for r in recent):
            suspected.add(cid)
    return len(suspected) > 0, suspected


# ═══════════════════════════════════════════════════════════════════════
# Main experiment loop (identical to v8 logic, adapted for CIFAR-10)
# ═══════════════════════════════════════════════════════════════════════

def run_group(
    group_name, client_loaders, test_loader, rounds, chaos_schedule,
    chaos_mode, device, seed, local_epochs=5, lr=0.1,
    v1_sample_ratio=0.3, v1_tolerance=2e-3,
    byzantine_clients=None, ema_alpha=0.6, use_scheduler=True,
):
    assert group_name in {"fedavg", "byz_honest", "tas"}
    num_clients = len(client_loaders)
    global_model = CifarCNN().to(device)
    global_state = global_model.state_dict()
    if byzantine_clients is None:
        byzantine_clients = []
    byzantine_set = set(byzantine_clients)

    attack_counters = {atk: {"total": 0, "detected": 0} for atk in ATTACK_TYPES}
    attack_rounds, blocked_rounds = 0, 0
    accuracy_history, logs = [], []
    rng = random.Random(seed + {"fedavg": 11, "byz_honest": 22, "tas": 33}[group_name])
    ema_weights = {}
    trust_rank_history = defaultdict(list)
    v2_detections = 0

    for r in range(1, rounds + 1):
        local_updates, flat_updates = {}, {}

        for cid in range(num_clients):
            if cid in byzantine_set:
                local_state = train_one_client_label_flip(
                    global_state, client_loaders[cid], device,
                    local_epochs=local_epochs, lr=lr)
            else:
                local_state = train_one_client(
                    global_state, client_loaders[cid], device,
                    local_epochs=local_epochs, lr=lr)
            upd = state_sub(local_state, global_state)
            local_updates[cid] = upd
            flat_updates[cid] = flatten_update(upd)

        # ── Aggregation ────────────────────────────────────────────────
        if group_name == "fedavg":
            equal_w = {cid: 1.0 / num_clients for cid in range(num_clients)}
            chaos = chaos_schedule[r] if chaos_mode else RoundAttack(active=False)
            attack_type = chaos.attack_type if chaos.active else None
            tampered_weights = dict(equal_w)
            attack_targets = set()

            if chaos.active:
                attack_rounds += 1
                atk_fn = ATTACK_FNS.get(attack_type)
                if atk_fn:
                    attack_counters[attack_type]["total"] += 1
                    sim_scores = {cid: 1.0 for cid in range(num_clients)}
                    attack_targets = atk_fn(tampered_weights, sim_scores,
                                            byzantine_ids=byzantine_clients, rng=rng)

            agg_update = weighted_average_updates(local_updates, tampered_weights)
            global_state = state_add(global_state, agg_update)
            byz_weights = {cid: tampered_weights.get(cid, 0.0) for cid in byzantine_set}
            v1_detected = v2_detected = False
            consensus_ok = True; healed_applied = False; heal_source = ""

        else:
            sim_scores, trust_weights = compute_trust_weights(
                flat_updates, trust_power=5.0, max_weight_ratio=2.0)

            # EMA smoothing
            if ema_weights:
                smoothed = {}
                for cid in range(num_clients):
                    current = trust_weights.get(cid, 0.0)
                    history = ema_weights.get(cid, 1.0 / num_clients)
                    smoothed[cid] = ema_alpha * current + (1 - ema_alpha) * history
                s_sum = sum(smoothed.values())
                if s_sum > 1e-12:
                    trust_weights = {cid: w / s_sum for cid, w in smoothed.items()}
            ema_weights = dict(trust_weights)

            # V2 rank tracking
            sorted_cids = sorted(trust_weights.keys(), key=lambda c: trust_weights[c])
            rank_map = {cid: rank for rank, cid in enumerate(sorted_cids)}
            for cid in range(num_clients):
                trust_rank_history[cid].append(rank_map[cid])

            byz_weights = {cid: trust_weights.get(cid, 0.0) for cid in byzantine_set}
            chaos = chaos_schedule[r] if chaos_mode else RoundAttack(active=False)
            attack_type = chaos.attack_type if chaos.active else None
            tampered_weights = dict(trust_weights)
            attack_targets = set()

            if chaos.active and group_name == "tas":
                attack_rounds += 1
                atk_fn = ATTACK_FNS.get(attack_type)
                if atk_fn:
                    attack_counters[attack_type]["total"] += 1
                    attack_targets = atk_fn(tampered_weights, sim_scores,
                                            byzantine_ids=byzantine_clients, rng=rng)

            v1_detected = v2_detected = False
            v2_suspected = set()
            consensus_ok = True; healed_applied = False; heal_source = ""

            if group_name == "tas" and chaos.active:
                v1_detected, _ = v1_audit_sampled(
                    tampered_weights, trust_weights, num_clients,
                    sample_ratio=v1_sample_ratio, tolerance=v1_tolerance, round_number=r)
                if not v1_detected and r >= 3:
                    v2_detected, v2_suspected = v2_membership_audit(
                        trust_rank_history, num_clients, byzantine_set,
                        low_rank_threshold=0.3, consecutive_rounds=3)

                if v1_detected:
                    attack_counters[attack_type]["detected"] += 1
                    consensus_ok = False; heal_source = "V1"; blocked_rounds += 1
                elif v2_detected:
                    consensus_ok = False; heal_source = "V2"; blocked_rounds += 1
                    v2_detections += 1

            # Apply
            if group_name == "byz_honest":
                agg_update = weighted_average_updates(local_updates, trust_weights)
                global_state = state_add(global_state, agg_update)
            elif consensus_ok:
                agg_update = weighted_average_updates(local_updates, tampered_weights)
                global_state = state_add(global_state, agg_update)
            else:
                healed_weights = dict(trust_weights)
                for cid in byzantine_set:
                    healed_weights[cid] = 0.0
                for cid in v2_suspected:
                    healed_weights[cid] = 0.0
                h_sum = sum(healed_weights.values())
                if h_sum > 1e-12:
                    healed_weights = {cid: w / h_sum for cid, w in healed_weights.items()}
                agg_update = weighted_average_updates(local_updates, healed_weights)
                global_state = state_add(global_state, agg_update)
                healed_applied = True

        global_model.load_state_dict(global_state)
        acc = evaluate(global_model, test_loader, device)
        accuracy_history.append(acc)

        logs.append({
            "round": r, "group": group_name,
            "attack_active": int(chaos.active), "attack_type": attack_type or "None",
            "v1_detected": int(v1_detected), "v2_detected": int(v2_detected),
            "consensus_passed": int(consensus_ok), "healed_applied": int(healed_applied),
            "heal_source": heal_source, "accuracy": float(acc),
            "byz_weights_avg": float(np.mean(list(byz_weights.values()))) if byz_weights else 0.0,
        })

        print(f"[{r:03d}] {group_name:<10} | atk={attack_type or 'None':>4s} | "
              f"V1={int(v1_detected)} V2={int(v2_detected)} | "
              f"consensus={int(consensus_ok)} heal={int(healed_applied)}{heal_source} | "
              f"acc={acc:.2f}% | byz_w={[f'{byz_weights.get(c,0):.3f}' for c in sorted(byzantine_set)]}")

    # Metrics
    total_attacks = sum(c["total"] for c in attack_counters.values())
    total_detected = sum(c["detected"] for c in attack_counters.values())
    metrics = {
        "attack_rounds": float(attack_rounds),
        "blocked_rounds": float(blocked_rounds),
        "v1_detection_rate": float(total_detected / attack_rounds) if attack_rounds > 0 else 1.0,
        "v2_detections": float(v2_detections),
        "system_interception_rate": float(blocked_rounds / attack_rounds) if attack_rounds > 0 else 1.0,
    }
    for k in ATTACK_TYPES:
        t = attack_counters[k]["total"]
        d = attack_counters[k]["detected"]
        metrics[f"{k}_total"] = float(t)
        metrics[f"{k}_detected"] = float(d)
        metrics[f"{k}_detection_rate"] = float(d / t) if t > 0 else 1.0

    return GroupRunResult(
        final_accuracy=float(accuracy_history[-1]) if accuracy_history else 0.0,
        best_accuracy=float(max(accuracy_history)) if accuracy_history else 0.0,
        avg_accuracy=float(np.mean(accuracy_history)) if accuracy_history else 0.0,
        accuracy_history=accuracy_history, round_logs=logs, metrics=metrics)


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_accuracy_curves(out_path, histories, clean_history=None,
                         byz_honest_history=None, attack_schedule=None):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    if clean_history:
        ax.plot(clean_history, label="Clean Baseline", color="#888888",
                ls="--", alpha=0.7, lw=1.5)
    if byz_honest_history:
        ax.plot(byz_honest_history, label="Byz + Honest E", color="#2196F3",
                ls="--", alpha=0.7, lw=1.5)
    colors = {"fedavg": "#D32F2F", "tas": "#2E7D32"}
    labels_map = {"fedavg": "Attacked FedAvg (No Defense)",
                  "tas": "Attacked TAS (V1+V2 Audit + Healing)"}
    for name, hist in histories.items():
        ax.plot(hist, label=labels_map.get(name, name),
                color=colors.get(name, "black"), lw=2.0, alpha=0.9)
    if attack_schedule:
        for r, atk in attack_schedule.items():
            if atk.active:
                ax.axvspan(r - 0.5, r + 0.5, alpha=0.08, color='#FF5722')
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("TAS Governance on CIFAR-10 (v9)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-5, 105)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_bar_comparison(out_path, results):
    groups = ["Clean\nBaseline", "Byz +\nHonest E", "Attacked\nFedAvg", "Attacked\nTAS"]
    accs = [results.get("clean_best", 0), results.get("byz_honest_best", 0),
            results.get("fedavg_best", 0), results.get("tas_best", 0)]
    colors = ["#78909C", "#42A5F5", "#EF5350", "#66BB6A"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(groups, accs, color=colors, width=0.6, edgecolor="white", lw=1.2)
    ax.set_ylabel("Best Accuracy (%)", fontsize=12)
    ax.set_title("TAS Governance on CIFAR-10 (v9)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", ls="--", alpha=0.2)
    ax.set_ylim(0, max(accs) + 10)
    for b, v in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.2f}%",
                ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp3 v9: CIFAR-10 Governance")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chaos-seed", type=int, default=2026)
    parser.add_argument("--v1-sample-ratio", type=float, default=0.3)
    parser.add_argument("--v1-tolerance", type=float, default=2e-3)
    parser.add_argument("--byzantine-ratio", type=float, default=0.3)
    parser.add_argument("--ema-alpha", type=float, default=0.6)
    parser.add_argument("--attacks-per-block", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Dirichlet alpha for Non-IID")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client_num = int(args.clients)
    n_byz = max(1, int(client_num * args.byzantine_ratio))
    byzantine_ids = list(range(n_byz))

    print(f"[Exp3 v9 CIFAR-10] {client_num} clients, {n_byz} Byzantine, {args.rounds} rounds")
    print(f"  Model: CifarCNN (ResNet-18+GN), lr={args.lr}, local_epochs={args.local_epochs}")
    print(f"  Attack: label_flip, 6/10 rounds, α={args.alpha}")

    client_loaders, test_loader = get_cifar10_loaders(
        client_num=client_num, batch_size=args.batch_size, alpha=args.alpha)

    schedule = make_chaos_schedule(args.rounds, seed=args.chaos_seed,
                                    attacks_per_block=args.attacks_per_block)
    clean_schedule = {r: RoundAttack(active=False) for r in range(1, args.rounds + 1)}

    common_kwargs = dict(
        client_loaders=client_loaders, test_loader=test_loader,
        rounds=args.rounds, device=device,
        local_epochs=args.local_epochs, lr=args.lr,
        v1_sample_ratio=args.v1_sample_ratio,
        v1_tolerance=args.v1_tolerance,
        byzantine_clients=byzantine_ids,
        ema_alpha=args.ema_alpha, use_scheduler=False,
    )

    # 1. Clean Baseline
    print("\n" + "=" * 60)
    print("=== 1/4: Clean Baseline ===")
    clean_result = run_group("fedavg", chaos_schedule=clean_schedule,
                              chaos_mode=False, seed=100,
                              byzantine_clients=[], **{k: v for k, v in common_kwargs.items()
                              if k != "byzantine_clients"})

    # 2. Byz+HonestE
    print("\n" + "=" * 60)
    print(f"=== 2/4: Byz+HonestE ===")
    byz_honest_result = run_group("byz_honest", chaos_schedule=clean_schedule,
                                   chaos_mode=False, seed=150, **common_kwargs)

    # 3. Attacked FedAvg
    print("\n" + "=" * 60)
    print("=== 3/4: Attacked FedAvg (No Defense) ===")
    attacked_fedavg = run_group("fedavg", chaos_schedule=schedule,
                                 chaos_mode=True, seed=200, **common_kwargs)

    # 4. Attacked TAS
    print("\n" + "=" * 60)
    print("=== 4/4: Attacked TAS (V1+V2 + Healing) ===")
    attacked_tas = run_group("tas", chaos_schedule=schedule,
                              chaos_mode=True, seed=200, **common_kwargs)

    # Summary
    results = {
        "clean_best": clean_result.best_accuracy,
        "clean_avg": clean_result.avg_accuracy,
        "byz_honest_best": byz_honest_result.best_accuracy,
        "byz_honest_avg": byz_honest_result.avg_accuracy,
        "fedavg_best": attacked_fedavg.best_accuracy,
        "fedavg_avg": attacked_fedavg.avg_accuracy,
        "tas_best": attacked_tas.best_accuracy,
        "tas_avg": attacked_tas.avg_accuracy,
        "fedavg_loss": max(0.0, clean_result.best_accuracy - attacked_fedavg.best_accuracy),
        "tas_loss": max(0.0, clean_result.best_accuracy - attacked_tas.best_accuracy),
        "byz_honest_loss": max(0.0, clean_result.best_accuracy - byz_honest_result.best_accuracy),
    }
    results.update({f"tas_{k}": v for k, v in attacked_tas.metrics.items()})

    # Save
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "results")
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "exp3_cifar10_v9.csv")
    fieldnames = ["round", "scenario", "group", "attack_active", "attack_type",
                  "v1_detected", "v2_detected", "consensus_passed",
                  "healed_applied", "heal_source", "accuracy", "byz_weights_avg"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scenario, res in [("clean", clean_result), ("byz_honest", byz_honest_result),
                               ("attacked_fedavg", attacked_fedavg),
                               ("attacked_tas", attacked_tas)]:
            for row in res.round_logs:
                out = dict(row); out["scenario"] = scenario
                writer.writerow(out)

    plot_accuracy_curves(
        os.path.join(results_dir, "exp3_cifar10_curves_v9.png"),
        {"fedavg": attacked_fedavg.accuracy_history,
         "tas": attacked_tas.accuracy_history},
        clean_history=clean_result.accuracy_history,
        byz_honest_history=byz_honest_result.accuracy_history,
        attack_schedule=schedule)
    plot_bar_comparison(
        os.path.join(results_dir, "exp3_cifar10_bar_v9.png"), results)

    print("\n" + "=" * 60)
    print("[Exp3 v9 CIFAR-10] Summary")
    print(f"  Clean Baseline:  best={results['clean_best']:.2f}% avg={results['clean_avg']:.2f}%")
    print(f"  Byz+HonestE:     best={results['byz_honest_best']:.2f}% avg={results['byz_honest_avg']:.2f}%")
    print(f"  Attacked FedAvg: best={results['fedavg_best']:.2f}% avg={results['fedavg_avg']:.2f}%")
    print(f"  Attacked TAS:    best={results['tas_best']:.2f}% avg={results['tas_avg']:.2f}%")
    print(f"  V1 Detection: {results.get('tas_v1_detection_rate', 0):.1%}")
    print(f"  System Interception: {results.get('tas_system_interception_rate', 0):.1%}")
    print(f"  Saved: {csv_path}, {results_dir}")


if __name__ == "__main__":
    main()
