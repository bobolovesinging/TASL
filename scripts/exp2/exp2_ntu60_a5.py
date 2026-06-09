# -*- coding: utf-8 -*-
"""
Exp2 NTU-60: Byzantine Resilience Matrix with FedPure Augmentation
=============================================
5 algorithms x 3 attacks, ST-GCN, augmented data (random_rot + crop_resize).

Usage:
  Quick test:  python exp2_ntu60_aug.py --quick
  Full run:    python exp2_ntu60_aug.py --rounds 65 --clients 10
"""
import csv, os, random, sys, argparse, yaml
import numpy as np
import torch, torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCKCHAIN_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CORE_DIR = os.path.join(BLOCKCHAIN_DIR, "core")
sys.path.insert(0, BLOCKCHAIN_DIR)
sys.path.insert(0, CORE_DIR)
sys.path.insert(0, SCRIPT_DIR)  # for ntu60_aug_loader

from ntu60_aug_loader import load_ntu60_augmented
from gradient_aggregator import GradientAggregator
from unified_data_layer import TrainStatsCollector, compute_unified_data_trust, fuse_trust_with_gradient
from ntu60_skeleton_attacks import apply_skeleton_attack
from gradient_attacks import GRADIENT_ATTACKS, apply_gradient_attack

# ST-GCN
STGCN_DIR = "/media/njit5/39e16d05-05ba-46b4-bc9b-47b3c98f1d4f/tlf/st-gcn"
sys.path.insert(0, STGCN_DIR)
from net.st_gcn import Model as STGCN

DATA_DIR = "/media/njit5/39e16d05-05ba-46b4-bc9b-47b3c98f1d4f/tlf/datasets/dir0.5"
NUM_CLASSES = 60

# Core TASL trust scorer
_tasl = GradientAggregator.__new__(GradientAggregator)

# ═══ LR Schedule: matches run_ntu60_aug.py (70.15% Clean) ═══
DEFAULT_WARMUP = 5
DEFAULT_LR_STEPS = [35, 55]
DEFAULT_BASE_LR = 0.1

def get_lr(round_num, base_lr=None, warmup=None, lr_steps=None):
    """Warmup + step decay schedule."""
    if base_lr is None: base_lr = DEFAULT_BASE_LR
    if warmup is None: warmup = DEFAULT_WARMUP
    if lr_steps is None: lr_steps = DEFAULT_LR_STEPS
    if round_num <= warmup:
        return base_lr * round_num / warmup
    factor = 1.0
    for s in lr_steps:
        if round_num > s:
            factor *= 0.1
    return base_lr * factor

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def create_model(device):
    return STGCN(num_class=NUM_CLASSES, in_channels=3,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True).to(device)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); c, t = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        c += model(x).argmax(1).eq(y).sum().item(); t += y.size(0)
    return 100.0 * c / t

@torch.no_grad()
def evaluate_full(model, loader, device, attack="none", num_classes=60):
    """Return (accuracy, ASR). ASR for label_flip = fraction predicted as flipped label."""
    model.eval()
    correct = total = flip_correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += preds.eq(y).sum().item()
        if attack == "label_flip":
            flipped = num_classes - 1 - y
            flip_correct += preds.eq(flipped).sum().item()
        total += y.size(0)
    acc = 100.0 * correct / max(total, 1)
    asr = 100.0 * flip_correct / max(total, 1) if attack == "label_flip" else 0.0
    return acc, asr


def state_sub(a, b):
    return {k: v - b[k] for k, v in a.items() if v.is_floating_point()}

def state_add(a, upd):
    return {k: v + upd.get(k, 0) for k, v in a.items() if v.is_floating_point()}

def flatten_update(upd):
    return np.concatenate([v.cpu().numpy().flatten() for v in upd.values()])

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y); loss.backward()
        optimizer.step()

def train_one_epoch_fedprox(model, loader, optimizer, criterion, device, mu, global_state):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        prox = 0.0
        for name, param in model.named_parameters():
            if name in global_state:
                prox += ((param - global_state[name].to(device)) ** 2).sum()
        loss += (mu / 2) * prox
        loss.backward(); optimizer.step()

SKELETON_ATTACKS = {"fgsm", "pgd", "bone_len", "hard_nobox"}

def train_on_model(model, loader, device, local_epochs=1, lr=0.01, attack=None,
                   stats_collector=None, cid=None, sk_atk_kwargs=None):
    """Train a pre-created model (in-place, reuses GPU memory).

    Supports:
      - label_flip: deterministic label flipping
      - fgsm/pgd: adversarial perturbation (needs model gradients)
      - bone_len/hard_nobox: structural skeleton attacks
    """
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    crit = nn.CrossEntropyLoss()
    if sk_atk_kwargs is None:
        sk_atk_kwargs = {}
    for _ in range(local_epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # ── Label-level attack ──
            if attack == "label_flip":
                y = (NUM_CLASSES - 1 - y) % NUM_CLASSES

            # ── Skeleton data-level attack ──
            if attack in SKELETON_ATTACKS:
                x, y = apply_skeleton_attack(model, x, y, attack, **sk_atk_kwargs)

            opt.zero_grad()
            loss = crit(model(x), y)
            loss_item = loss.item()
            loss.backward()
            opt.step()
            if stats_collector is not None and cid is not None:
                stats_collector.record_batch(cid, loss_item, model.parameters())
    return {k: v.cpu() for k, v in model.state_dict().items()}

# ════════════════ Aggregation ════════════════

def fedavg_aggregate(updates, weights=None):
    keys = [k for k in next(iter(updates.values())) if isinstance(next(iter(updates.values()))[k], torch.Tensor)]
    result = {}
    for k in keys:
        stacked = torch.stack([u[k].float() for u in updates.values()])
        if weights:
            w = torch.tensor([weights.get(c, 1.0/len(updates)) for c in updates], dtype=torch.float32)
            result[k] = (stacked * w.view(-1, *([1]*(stacked.dim()-1)))).sum(0)
        else:
            result[k] = stacked.mean(0)
    return result

def multi_krum_aggregate(flat_updates, updates, f):
    ids = list(flat_updates.keys())
    n = len(ids)
    scores = np.zeros(n)
    for i in range(n):
        dists = []
        for j in range(n):
            if i != j:
                dists.append(np.linalg.norm(flat_updates[ids[i]] - flat_updates[ids[j]]))
        dists.sort()
        scores[i] = sum(dists[:n - f - 1])
    selected = np.argsort(scores)[:n - f]
    sel_updates = {ids[s]: updates[ids[s]] for s in selected}
    return fedavg_aggregate(sel_updates)

def trimmed_mean_aggregate(updates, beta=0.2):
    keys = [k for k in next(iter(updates.values())) if isinstance(next(iter(updates.values()))[k], torch.Tensor)]
    result = {}
    n = len(updates)
    trim = int(n * beta)
    for k in keys:
        stacked = torch.stack([u[k].float() for u in updates.values()])
        sorted_t, _ = torch.sort(stacked, dim=0)
        result[k] = sorted_t[trim:n-trim].mean(0)
    return result

def fltrust_aggregate(flat_updates, updates, device):
    ids = list(updates.keys())
    n = len(ids)
    # Use honest update #0 as proxy root
    root_flat = flat_updates[ids[0]]
    root_norm = np.linalg.norm(root_flat) + 1e-12

    scores = {}
    for cid in ids:
        v = flat_updates[cid]
        v_norm = np.linalg.norm(v) + 1e-12
        cs = max(0, float(np.dot(v, root_flat) / (v_norm * root_norm)))
        scores[cid] = cs

    total = sum(scores.values())
    if total > 1e-12:
        weights = {cid: s/total for cid, s in scores.items()}
    else:
        weights = {cid: 1.0/n for cid in ids}
    return fedavg_aggregate(updates, weights)

# ════════════════ TASL Trust Scoring (delegated to core) ════════════════

def compute_tasl_trust_weights(flat_updates, **kwargs):
    """Thin wrapper → core GradientAggregator.compute_tasl_trust_weights."""
    torch_updates = {}
    for cid, v in flat_updates.items():
        if isinstance(v, np.ndarray):
            torch_updates[cid] = torch.from_numpy(v).float()
        else:
            torch_updates[cid] = v.float()
    if 'prev_anchor' in kwargs and kwargs['prev_anchor'] is not None:
        if isinstance(kwargs['prev_anchor'], np.ndarray):
            kwargs['prev_anchor'] = torch.from_numpy(kwargs['prev_anchor']).float()
    if 'temporal_anchor' in kwargs and kwargs['temporal_anchor'] is not None:
        if isinstance(kwargs['temporal_anchor'], np.ndarray):
            kwargs['temporal_anchor'] = torch.from_numpy(kwargs['temporal_anchor']).float()
    tw, cs, na = _tasl.compute_tasl_trust_weights(torch_updates, **kwargs)
    na_np = na.numpy() if isinstance(na, torch.Tensor) else na
    return tw, cs, na_np

# ════════════════ run_single ════════════════

def run_single(algo, attack, client_loaders, test_loader, n_clients, n_byz, rounds, device,
               local_epochs=1, seed=42, base_lr=None, warmup=None, lr_steps=None,
               attack_params=None):
    set_seed(seed)
    byz_set = set(range(n_byz))
    global_model = create_model(device)
    global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}
    if attack_params is None:
        attack_params = {}

    # Pre-create client models to avoid per-round OOM from model creation
    client_models = [create_model(device) for _ in range(n_clients)]

    acc_hist, asr_hist, ema_w, cos_h, prev_a, temp_a = [], [], None, [], None, None
    best = 0.0

    for r in range(1, rounds+1):
        cur_lr = get_lr(r, base_lr=base_lr, warmup=warmup, lr_steps=lr_steps)
        stats_collector = TrainStatsCollector()

        local_updates, flat_updates = {}, {}
        for cid in range(n_clients):
            # Load global weights into pre-created model
            client_models[cid].load_state_dict(
                {k: v.to(device) for k, v in global_state.items()})

            atk = attack if (cid in byz_set and attack != "none") else None

            # ── 梯度层后处理攻击（训练完后施加）──
            if atk in GRADIENT_ATTACKS:
                # 先正常训练
                train_on_model(client_models[cid], client_loaders[cid], device,
                              local_epochs=local_epochs, lr=cur_lr, attack=None,
                              stats_collector=stats_collector, cid=cid)
                state = {k: v.cpu() for k, v in client_models[cid].state_dict().items()}
                # 获取上一轮诚实平均更新（用于 min_max 攻击）
                honest_avg = None
                if atk == "min_max" and temp_a is not None:
                    honest_avg = torch.from_numpy(temp_a).float()
                local_state = apply_gradient_attack(
                    atk, state, global_state,
                    honest_avg_flat=honest_avg,
                    n_byz=n_byz, n_clients=n_clients,
                    attack_kwargs=attack_params.get(atk, {}),
                )

            # sign_flip: train normally first, then flip delta sign
            elif atk == "sign_flip":
                train_on_model(client_models[cid], client_loaders[cid], device,
                              local_epochs=local_epochs, lr=cur_lr, attack=None,
                              stats_collector=stats_collector, cid=cid)
                state = {k: v.cpu() for k, v in client_models[cid].state_dict().items()}
                local_state = {k: global_state[k] - (v - global_state[k])
                               for k, v in state.items() if k in global_state and v.is_floating_point()}
            elif atk == "gaussian_noise":
                train_on_model(client_models[cid], client_loaders[cid], device,
                              local_epochs=local_epochs, lr=cur_lr, attack=None,
                              stats_collector=stats_collector, cid=cid)
                state = client_models[cid].state_dict()
                local_state = {}
                for k, v in state.items():
                    if v.is_floating_point():
                        local_state[k] = v.cpu() + torch.randn_like(v.cpu()) * 0.1 * v.cpu().std()
                    else:
                        local_state[k] = v.cpu()
            # Skeleton data-layer attacks (fgsm, pgd, bone_len, hard_nobox)
            elif atk in SKELETON_ATTACKS:
                local_state = train_on_model(client_models[cid], client_loaders[cid], device,
                                             local_epochs=local_epochs, lr=cur_lr, attack=atk,
                                             stats_collector=stats_collector, cid=cid)
            # label_flip or no attack
            else:
                local_state = train_on_model(client_models[cid], client_loaders[cid], device,
                                             local_epochs=local_epochs, lr=cur_lr, attack=atk,
                                             stats_collector=stats_collector, cid=cid)

            local_updates[cid] = state_sub(local_state, global_state)
            flat_updates[cid] = flatten_update(local_updates[cid])

        # ── Data-layer trust ──
        round_stats = stats_collector.get_stats()
        data_trust, _ = compute_unified_data_trust(round_stats, n_clients)

        # Aggregation
        if algo == "fedavg":
            agg = fedavg_aggregate(local_updates)
        elif algo == "multi_krum":
            agg = multi_krum_aggregate(flat_updates, local_updates, f=n_byz)
        elif algo == "trimmed_mean":
            agg = trimmed_mean_aggregate(local_updates)
        elif algo == "fltrust":
            agg = fltrust_aggregate(flat_updates, local_updates, device)
        elif algo == "tasl":
            prev_anchor_t = torch.from_numpy(prev_a).float() if prev_a is not None else None
            tw, cs, na = compute_tasl_trust_weights(
                flat_updates, ema_weights=ema_w, cos_history=cos_h,
                prev_anchor=prev_anchor_t, data_trust=data_trust)
            ema_w = dict(tw); cos_h.append(dict(cs)); prev_a = na
            agg = fedavg_aggregate(local_updates, tw)
        else:
            raise ValueError(f"Unknown algo: {algo}")

        # 保存本轮聚合更新（供 min_max 攻击使用）
        af = flatten_update(agg)
        if np.linalg.norm(af) > 1e-12:
            temp_a = af

        global_state = state_add(global_state, agg)
        global_model.load_state_dict({k: v.to(device) for k, v in global_state.items()})
        acc, asr = evaluate_full(global_model, test_loader, device, attack=attack)
        acc_hist.append(acc); asr_hist.append(asr); best = max(best, acc)

        if r % 5 == 0 or r == 1:
            parts = [f"  [{algo}/{attack}] R{r:02d} acc={acc:.2f}% best={best:.2f}% lr={cur_lr:.5f}"]
            if algo=="tasl" and ema_w:
                bw = np.mean([ema_w.get(c,0.0) for c in byz_set])
                parts.append(f" byz_w={bw:.4f}")
            if attack == "label_flip":
                parts.append(f" asr={asr:.2f}%")
            if data_trust:
                bt = np.mean([data_trust.get(c,0.0) for c in byz_set]) if n_byz > 0 else 0
                parts.append(f" data_t={bt:.2f}")
            print("".join(parts), flush=True)

        torch.cuda.empty_cache()

    # Clean up
    for m in client_models: del m
    del global_model; torch.cuda.empty_cache()

    best_asr = float(np.max(asr_hist)) if asr_hist else 0.0
    avg_asr = float(np.mean(asr_hist)) if asr_hist else 0.0
    return {"best_acc": best, "avg_acc": float(np.mean(acc_hist)), "acc_history": acc_hist, "best_asr": best_asr, "avg_asr": avg_asr}

# ════════════════ main ════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--rounds", type=int, default=65)
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--byzantine-ratio", type=float, default=0.4)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip-clean", action="store_true", help="Skip Clean baseline (FedAvg no attack)")
    p.add_argument("--algorithms", type=str, default="fedavg,multi_krum,trimmed_mean,fltrust,tasl")
    p.add_argument("--attacks", type=str,
                    default="label_flip,sign_flip,gaussian_noise,fgsm,pgd,bone_len,hard_nobox,scaling,min_max")
    args = p.parse_args()

    # ── YAML config loading ──
    cfg = {}
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        data_cfg = cfg.get('data', {})
        train_cfg = cfg.get('training', {})
        byz_cfg = cfg.get('byzantine', {})
        alg_cfg = cfg.get('algorithms', None)
        atk_cfg = cfg.get('byzantine', {}).get('attacks', None)

        # YAML overwrites defaults, but CLI args take precedence (only set if still default)
        args.rounds = train_cfg.get('rounds', args.rounds)
        args.clients = data_cfg.get('n_clients', args.clients)
        args.local_epochs = train_cfg.get('local_epochs', args.local_epochs)
        args.byzantine_ratio = byz_cfg.get('ratio', args.byzantine_ratio)
        if alg_cfg:
            args.algorithms = ",".join(alg_cfg)
        if atk_cfg:
            args.attacks = ",".join(atk_cfg)

        # LR schedule from config
        lr_schedule = train_cfg.get('lr_schedule', {})
        base_lr = lr_schedule.get('base_lr', DEFAULT_BASE_LR)
        warmup = lr_schedule.get('warmup', DEFAULT_WARMUP)
        lr_steps = lr_schedule.get('steps', DEFAULT_LR_STEPS)
        weight_decay = train_cfg.get('weight_decay', 1e-4)
        # Load attack_params from YAML
        attack_params = cfg.get('attack_params', {})
    else:
        base_lr = DEFAULT_BASE_LR
        warmup = DEFAULT_WARMUP
        lr_steps = DEFAULT_LR_STEPS
        weight_decay = 1e-4
        attack_params = {}

    if args.quick:
        args.clients = 5; args.rounds = 10; args.byzantine_ratio = 0.4

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_clients = int(args.clients)
    n_byz = max(1, int(n_clients * args.byzantine_ratio))
    rounds = int(args.rounds)
    algos = args.algorithms.split(",")
    attacks_list = args.attacks.split(",")

    # ── Print LR schedule ──
    print(f"LR schedule: base={base_lr}, warmup={warmup}, steps={lr_steps}")
    check_rounds = [1, warmup, warmup+1] + list(lr_steps) + [s+1 for s in lr_steps]
    for r in sorted(set(check_rounds)):
        if r <= rounds:
            print(f"  R{r:03d} lr={get_lr(r, base_lr, warmup, lr_steps):.6f}")

    # ── Load augmented data ──
    print(f"Loading NTU-60 augmented data (random_rot + crop_resize)...")
    client_loaders, test_loader = load_ntu60_augmented(DATA_DIR, n_clients=n_clients, batch_size=64)
    print(f"  Data loaded: {n_clients} clients")

    # ── Print config ──
    print("=" * 70)
    print(f"[Exp2 NTU-60 Aug] Byzantine Resilience Matrix — ST-GCN")
    print(f"  {n_clients} clients, {n_byz} Byzantine ({args.byzantine_ratio:.0%})")
    print(f"  IID, FedPure augmentation (random_rot + valid_crop_resize)")
    print(f"  {rounds} rounds, local_epochs={args.local_epochs}, wd={weight_decay}")
    print(f"  LR: base={base_lr}, warmup={warmup}, steps={lr_steps}")
    print(f"  Algorithms: {algos}")
    print(f"  Attacks: {attacks_list}")
    print(f"  TASL: trust_power=5, weight_cap=1.5/n, ema_alpha=0.7, dual_anchor+norm_gate")
    print("=" * 70)

    # ── Clean baseline ──
    if args.skip_clean:
        print("\n>>> Clean Baseline SKIPPED (--skip-clean) <<<")
        clean_best, clean_avg = 0.0, 0.0
    else:
        print("\n>>> Clean Baseline <<<")
        clean = run_single("fedavg", "none", client_loaders, test_loader,
                           n_clients, 0, rounds, device, args.local_epochs, args.seed,
                           base_lr=base_lr, warmup=warmup, lr_steps=lr_steps,
                           attack_params=attack_params)
        clean_best, clean_avg = clean["best_acc"], clean["avg_acc"]
        print(f"  Clean: best={clean_best:.2f}%, avg={clean_avg:.2f}%")

    # ── Robustness matrix ──
    results = {}
    total = len(algos) * len(attacks_list)
    idx = 0

    for algo in algos:
        results[algo] = {}
        for attack in attacks_list:
            idx += 1
            print(f"\n>>> [{idx}/{total}] {algo} x {attack} <<<")
            res = run_single(algo, attack, client_loaders, test_loader,
                           n_clients, n_byz, rounds, device, args.local_epochs, args.seed,
                           base_lr=base_lr, warmup=warmup, lr_steps=lr_steps,
                           attack_params=attack_params)
            results[algo][attack] = res
            asr_str = f", best_asr={res['best_asr']:.2f}%" if attack == "label_flip" else ""
            print(f"  => {algo}/{attack}: best={res['best_acc']:.2f}%, avg={res['avg_acc']:.2f}%{asr_str}")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("SUMMARY: Best Accuracy")
    print("=" * 70)
    header = f"  {'Attack':<16}"
    for a in algos: header += f" {a:<16}"
    print(header)
    print("  " + "-" * (16 + 16*len(algos)))
    for attack in ["none"] + attacks_list:
        row = f"  {attack:<16}"
        for algo in algos:
            if attack == "none":
                val = f"{clean_best:.2f}%"
            else:
                val = f"{results[algo][attack]['best_acc']:.2f}%"
            row += f" {val:<16}"
        print(row)

    # ── Save CSV ──
    results_dir = os.path.join(BLOCKCHAIN_DIR, "results", "exp2_ntu60_aug")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "exp2_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo","attack","best_acc","avg_acc","best_asr","avg_asr"])
        for algo in algos:
            for attack in attacks_list:
                r = results[algo][attack]
                w.writerow([algo, attack, f"{r['best_acc']:.2f}", f"{r['avg_acc']:.2f}", f"{r.get('best_asr',0):.2f}", f"{r.get('avg_asr',0):.2f}"])
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    main()
