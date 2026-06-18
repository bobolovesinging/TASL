"""
Quick TASL hyperparameter sweep for Table 4 improvements.
Only runs TASL. Run from ~/TLF/TASL/ with:
  cd ~/TLF/TASL && python3 scripts/exp2/tasl_tune.py
"""
import sys, os, csv
sys.path.insert(0, "/home/njit516/TLF/TASL")
sys.path.insert(0, "/home/njit516/TLF/TASL/core")
sys.path.insert(0, "/home/njit516/TLF/TASL/scripts/exp2")

import torch, numpy as np
from model import FedAvgCNN
from exp2_run import (
    load_cifar10_data, load_fashionmnist_data,
    set_seed, evaluate, compute_tasl_trust_weights,
    fedavg_aggregate, flatten_update, state_sub, state_add,
    attack_label_flip, attack_sign_flip, attack_scaling, train_honest,
    create_model,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: %s" % DEVICE)

CONFIGS = [
    ('default',       5.0, 1.5, 0.7, 0.8),
    ('pow3_cap2',     3.0, 2.0, 0.7, 0.8),
    ('pow7',          7.0, 1.5, 0.7, 0.8),
    ('cap2',          5.0, 2.0, 0.7, 0.8),
    ('ema5',          5.0, 1.5, 0.5, 0.8),
]

ATTACK_FNS = {
    'label_flip': attack_label_flip,
    'sign_flip': attack_sign_flip,
    'scaling': attack_scaling,
}

def make_model_fn(cfg):
    return lambda: create_model(cfg)

def train_byz(gs, loader, fn, attack, n_classes=10):
    kwargs = {'num_classes': n_classes}
    if attack == 'scaling':
        kwargs['scaling_factor'] = 5.0
    return ATTACK_FNS[attack](gs, loader, DEVICE,
        model_fn=fn, local_epochs=1, lr=0.01,
        momentum=0.9, weight_decay=1e-4, **kwargs)

def train_clean(gs, loader, fn):
    return train_honest(gs, loader, DEVICE,
        model_fn=fn, local_epochs=1, lr=0.01,
        momentum=0.9, weight_decay=1e-4)

TESTS = [
    ('cifar10', 'sign_flip', 0.3, 100, 'resnet18_cifar'),
    ('cifar10', 'scaling', 0.3, 100, 'resnet18_cifar'),
    ('fashionmnist', 'label_flip', 0.4, 50, 'fedavgcnn'),
]

results = []

for ds, attack, ratio, rounds, model_name in TESTS:
    print("\n" + "=" * 60)
    print("Testing %s/%s @ %d%%, %dr" % (ds, attack, int(ratio*100), rounds))

    set_seed(42)
    cfg = {
        'data': {'dataset': ds, 'n_clients': 10, 'batch_size': 64,
                 'alpha': 0.3, 'split': 'non_iid'},
        'experiment': {'seed': 42},
        'model': {'name': model_name, 'num_classes': 10},
    }

    if ds == 'fashionmnist':
        loaders, test_ldr = load_fashionmnist_data(cfg)
    else:
        loaders, test_ldr = load_cifar10_data(cfg)

    n_clients = len(loaders)
    n_byz = max(1, int(n_clients * ratio))
    fn = make_model_fn(cfg)

    for cfg_name, trust_power, max_weight_ratio, ema_alpha, norm_pen in CONFIGS:
        set_seed(42)
        model = create_model(cfg).to(DEVICE)
        gs = model.state_dict()
        acc_hist, best_acc = [], 0.0
        ema_w = None; cos_h = []; prev_a = None; temp_a = None

        for r in range(1, rounds + 1):
            lup = {}; fup = {}
            for cid in range(n_clients):
                if cid < n_byz:
                    ts = train_byz(gs, loaders[cid], fn, attack)
                else:
                    ts = train_clean(gs, loaders[cid], fn)
                lup[cid] = state_sub(ts, gs)
                fup[cid] = flatten_update(lup[cid])

            tw, cs, na = compute_tasl_trust_weights(
                fup, trust_power=trust_power,
                max_weight_ratio=max_weight_ratio,
                norm_penalty_strength=norm_pen,
                refine_anchor=True, ema_weights=ema_w,
                ema_alpha=ema_alpha, cos_history=cos_h,
                prev_anchor=prev_a, temporal_anchor=temp_a)

            byz_w = sum(tw.get(b, 0.0) for b in range(n_byz))
            ema_w = dict(tw); cos_h.append(dict(cs)); prev_a = na
            au = fedavg_aggregate(lup, tw)
            af = flatten_update(au)
            if np.linalg.norm(af) > 1e-12:
                temp_a = af
            gs = state_add(gs, au)
            model.load_state_dict(gs)
            acc = evaluate(model, test_ldr, DEVICE)
            acc_hist.append(acc)
            best_acc = max(best_acc, acc)
            if r == 1 or r % 20 == 0:
                print("  %-15s R%03d acc=%.2f%% B=%.4f" % (cfg_name, r, acc, byz_w))

        avg = float(np.mean(acc_hist))
        results.append({
            'ds': ds, 'atk': attack, 'r': int(ratio*100),
            'cfg': cfg_name,
            'best': round(best_acc, 2), 'avg': round(avg, 2),
        })
        print("  => best=%.2f avg=%.2f" % (best_acc, avg))

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
for r in sorted(results, key=lambda x: (x['ds'], x['atk'], -x['avg'])):
    print("%-15s %-12s r%-3d | %-16s | best=%.2f avg=%.2f" % (
        r['ds'], r['atk'], r['r'], r['cfg'], r['best'], r['avg']))

csvf = os.path.expanduser('~/TLF/TASL/results/tasl_tune_results.csv')
with open(csvf, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)
print("Saved: %s" % csvf)
