"""
Quick test: TASL + FedPure fusion on 1/10 MNIST data.
5 clients, 10 rounds, 1 Byzantine (label_flip), 1 local epoch.
"""
import sys, os, random
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

TASL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASL_DIR)
sys.path.insert(0, os.path.join(TASL_DIR, "core"))
sys.path.insert(0, os.path.join(TASL_DIR, "scripts", "exp2"))

# ── Import components ──
from core.model import FedAvgCNN
from fedpure_tasl_fusion import compute_fused_trust_weights
from exp2_robustness_matrix import (
    compute_tasl_trust_weights, fedavg_aggregate, train_honest,
    state_sub, state_add, flatten_update, set_seed,
)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load 1/10 MNIST ──
from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
full_train = datasets.MNIST(root="./data/temp", train=True, download=True, transform=transform)
full_test = datasets.MNIST(root="./data/temp", train=False, download=True, transform=transform)

# Take only 1/10 of training data
n_samples = len(full_train) // 10
indices = np.random.choice(len(full_train), n_samples, replace=False)
subset = torch.utils.data.Subset(full_train, indices)
print(f"Training on {n_samples} / {len(full_train)} samples")

# Split into 5 clients IID
n_clients = 5
n_byz = 1
client_data = [[] for _ in range(n_clients)]
for i, idx in enumerate(indices):
    client_data[i % n_clients].append(idx)

client_loaders = []
for cid in range(n_clients):
    ds = torch.utils.data.Subset(full_train, client_data[cid])
    client_loaders.append(torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True))

test_loader = torch.utils.data.DataLoader(full_test, batch_size=256, shuffle=False)

# ── Test 3 configurations ──
configs = [
    {"name": "TASL only (no fusion)", "use_fusion": False},
    {"name": "TASL + FedPure fusion", "use_fusion": True, "fedpure_weight": 0.3},
    {"name": "TASL + FedPure fusion (strong)", "use_fusion": True, "fedpure_weight": 0.6},
]

def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

def train_label_flip(state, loader, lr=0.02, local_epochs=1):
    """Label flip attack: y -> 9 - y"""
    model = FedAvgCNN().to(device)
    model.load_state_dict(state)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_flipped = 9 - y  # Label flip
            opt.zero_grad()
            loss = criterion(model(x), y_flipped)
            loss.backward()
            opt.step()
    return {k: v.cpu() for k, v in model.state_dict().items()}

def train_honest_fn(state, loader, lr=0.02, local_epochs=1):
    model = FedAvgCNN().to(device)
    model.load_state_dict(state)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
    return {k: v.cpu() for k, v in model.state_dict().items()}

def get_flat(state_dict, ref_dict):
    delta = OrderedDict()
    for k in state_dict:
        if state_dict[k].is_floating_point():
            delta[k] = state_dict[k] - ref_dict[k]
    return np.concatenate([v.numpy().ravel() for v in delta.values()])

results = {}
for cfg in configs:
    print(f"\n{'='*60}")
    print(f"Config: {cfg['name']}")
    print(f"{'='*60}")
    
    # Init model
    global_state = FedAvgCNN().state_dict()
    global_model = FedAvgCNN().to(device)
    
    # Byzantine client index
    byz_idx = 0
    
    best_acc = 0.0
    ema_weights = None
    cos_history = []
    prev_anchor = None
    
    for r in range(1, 11):
        flat_updates = {}
        state_updates = {}
        
        for cid in range(n_clients):
            if cid == byz_idx:
                local = train_label_flip(global_state, client_loaders[cid])
            else:
                local = train_honest_fn(global_state, client_loaders[cid])
            
            state_updates[cid] = local
            flat_updates[cid] = get_flat(local, global_state)
        
        # Aggregate
        if cfg['use_fusion']:
            trust_weights, cos_sims, anchor, info = compute_fused_trust_weights(
                flat_updates,
                compute_tasl_fn=compute_tasl_trust_weights,
                fedpure_weight=cfg['fedpure_weight'],
                ema_weights=ema_weights,
                cos_history=cos_history,
                prev_anchor=prev_anchor,
            )
            ema_weights = dict(trust_weights)
            cos_history.append(dict(cos_sims))
            prev_anchor = anchor
            
            byz_weight = trust_weights.get(byz_idx, 0.0)
            detected = info.get('detected_malicious', [])
        else:
            trust_weights, cos_sims, anchor = compute_tasl_trust_weights(
                flat_updates,
                ema_weights=ema_weights,
                cos_history=cos_history,
                prev_anchor=prev_anchor,
            )
            ema_weights = dict(trust_weights)
            cos_history.append(dict(cos_sims))
            prev_anchor = anchor
            byz_weight = trust_weights.get(byz_idx, 0.0)
            detected = []
        
        agg_update = fedavg_aggregate(state_updates, trust_weights)
        for k in global_state:
            if k in agg_update and global_state[k].is_floating_point():
                global_state[k] = global_state[k] + agg_update[k]
        
        # Evaluate
        global_model.load_state_dict(global_state)
        acc = evaluate(global_model)
        best_acc = max(best_acc, acc)
        
        # Print every round
        det_str = f" detected={detected}" if detected else ""
        print(f"  R{r:02d} acc={acc:.2f}% best={best_acc:.2f}% byz_w={byz_weight:.4f}{det_str}")
    
    results[cfg['name']] = best_acc

# ── Summary ──
print(f"\n{'='*60}")
print(f"Fusion Test Summary (1/10 MNIST, 5 clients, 1 Byzantine)")
print(f"{'='*60}")
for name, acc in results.items():
    print(f"  {name:<40} best={acc:.2f}%")
