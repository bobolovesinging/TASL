"""
NTU-60 FedAvg/FedProx FedPure-augmented Experiment.
Runs FedAvg and FedProx with FedPure-style data augmentation targeting ~80% accuracy.
"""
import sys, os, json
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

TASL_DIR = "/media/njit5/39e16d05-05ba-46b4-bc9b-47b3c98f1d4f/tlf/TASL"
sys.path.insert(0, TASL_DIR)
sys.path.insert(0, os.path.join(TASL_DIR, "scripts", "exp2"))

from ntu60_aug_loader import load_ntu60_augmented

STGCN_DIR = os.path.join(os.path.dirname(TASL_DIR), "st-gcn")
sys.path.insert(0, STGCN_DIR)
from net.st_gcn import Model as STGCN


def train_one_epoch(model, loader, optimizer, criterion, device, fedprox_mu=0.0,
                    global_state=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        if fedprox_mu > 0 and global_state is not None:
            prox = 0.0
            for name, param in model.named_parameters():
                prox += ((param - global_state[name].to(device)) ** 2).sum()
            loss += (fedprox_mu / 2) * prox
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        _, pred = output.max(1)
        total += label.size(0)
        correct += pred.eq(label).sum().item()
    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            total_loss += loss.item() * data.size(0)
            _, pred = output.max(1)
            total += label.size(0)
            correct += pred.eq(label).sum().item()
    return total_loss / total, 100.0 * correct / total


def get_state_dict(model):
    return {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}


def fedavg_aggregate(local_states, weights=None):
    n = len(local_states)
    if weights is None:
        weights = {i: 1.0 / n for i in range(n)}
    agg = OrderedDict()
    for i, state in enumerate(local_states):
        w = weights.get(i, 1.0 / n)
        for k, v in state.items():
            if v.dtype in (torch.float32, torch.float64):
                if k not in agg:
                    agg[k] = w * v.clone()
                else:
                    agg[k] += w * v.clone()
    return agg


def run_fl(config):
    device = torch.device(config['device'])
    n_clients = config['n_clients']
    rounds = config['rounds']
    algo = config['algo']

    print(f"\n{'='*60}")
    print(f"[{algo}] Loading NTU-60 with FedPure augmentation...")
    client_loaders, test_loader = load_ntu60_augmented(
        data_dir=config['data_dir'], n_clients=n_clients,
        batch_size=config['batch_size'], window_size=64,
        random_rot=True, p_interval_train=[0.5, 1.0], test_p_interval=0.95,
    )
    print(f"[{algo}] {n_clients} clients, {len(test_loader.dataset)} test samples")

    criterion = nn.CrossEntropyLoss()
    base_lr = config['lr']
    warmup = config['warmup_epochs']

    def get_lr(r):
        if r <= warmup:
            return base_lr * r / warmup
        factor = 1.0
        for s in config['lr_step']:
            if r > s:
                factor *= config['lr_decay']
        return base_lr * factor

    # Pre-create models (once, not per-round)
    global_model = STGCN(num_class=60, in_channels=3,
                         graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                         edge_importance_weighting=True).to(device)
    client_models = []
    for _ in range(n_clients):
        m = STGCN(num_class=60, in_channels=3,
                  graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                  edge_importance_weighting=True).to(device)
        client_models.append(m)

    global_state = get_state_dict(global_model)
    best_acc, avg_acc = 0.0, 0.0
    acc_history = []

    print(f"[{algo}] Training {rounds} rounds...")
    for r in range(1, rounds + 1):
        local_states = []
        lr = get_lr(r)

        for cid in range(n_clients):
            client_models[cid].load_state_dict(
                {k: v.to(device) for k, v in global_state.items()})

            opt = torch.optim.SGD(client_models[cid].parameters(), lr=lr,
                                  momentum=0.9, weight_decay=config['weight_decay'],
                                  nesterov=True)

            if algo == 'fedprox':
                train_one_epoch(client_models[cid], client_loaders[cid], opt,
                                criterion, device, fedprox_mu=0.01,
                                global_state=global_state)
            else:
                train_one_epoch(client_models[cid], client_loaders[cid], opt,
                                criterion, device)

            local_states.append(get_state_dict(client_models[cid]))

        global_state = fedavg_aggregate(local_states)
        global_model.load_state_dict({k: v.to(device) for k, v in global_state.items()})

        if r % 2 == 0 or r == 1:
            _, acc = evaluate(global_model, test_loader, criterion, device)
            best_acc = max(best_acc, acc)
            if r > 10:
                avg_acc += acc
            acc_history.append((r, acc))
            print(f"  [{algo}] R{r:03d} lr={lr:.4f} test_acc={acc:.2f}% best={best_acc:.2f}%")

    _, last_acc = evaluate(global_model, test_loader, criterion, device)
    n_eval = max(1, len(acc_history) - 2)
    avg_acc = avg_acc / n_eval
    print(f"  [{algo}] Final: best={best_acc:.2f}% avg={avg_acc:.2f}% last={last_acc:.2f}%")
    return {'algo': algo, 'best': best_acc, 'avg': avg_acc, 'last': last_acc}


def main():
    DATA_DIR = "/media/njit5/39e16d05-05ba-46b4-bc9b-47b3c98f1d4f/tlf/datasets/dir0.1"

    configs = [
        {'algo': 'fedavg', 'data_dir': DATA_DIR, 'n_clients': 10, 'batch_size': 64,
         'rounds': 65, 'lr': 0.1, 'warmup_epochs': 5, 'lr_step': [35, 55],
         'lr_decay': 0.1, 'weight_decay': 0.0004, 'device': 'cuda:0'},
        {'algo': 'fedprox', 'data_dir': DATA_DIR, 'n_clients': 10, 'batch_size': 64,
         'rounds': 65, 'lr': 0.1, 'warmup_epochs': 5, 'lr_step': [35, 55],
         'lr_decay': 0.1, 'weight_decay': 0.0004, 'device': 'cuda:0'},
    ]

    results = []
    for cfg in configs:
        results.append(run_fl(cfg))

    print(f"\n{'='*60}")
    print("NTU-60 FL Summary (FedPure Augmentation)")
    print(f"{'='*60}")
    print(f"{'Algo':<12} {'Best':>8} {'Avg':>8} {'Last':>8}")
    for r in results:
        print(f"{r['algo']:<12} {r['best']:7.2f}% {r['avg']:7.2f}% {r['last']:7.2f}%")

    out_dir = os.path.join(TASL_DIR, "results", "exp2_ntu60_aug")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir}/results.json")


if __name__ == '__main__':
    main()
