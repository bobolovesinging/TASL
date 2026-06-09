# -*- coding: utf-8 -*-
"""
Exp2 NTU-60 v3: Unified Data-Layer (loss + grad_norm) + Gradient-Layer Fusion + LR Schedule
============================================================================================
Improvements over v2:
  1. Unified data-layer trust (TrainStatsCollector + compute_unified_data_trust)
     — no prototype pre-extraction; computes trust per-round from loss + grad_norm
  2. New attack: bone_length (attack on skeleton bone lengths)
  3. Same LR warmup + step decay, FedPure augmentation, ST-GCN model

Usage:
  Quick test:  python exp2_ntu60_v3.py --quick
  Full run:    python exp2_ntu60_v3.py --rounds 65 --clients 10
"""
import csv, os, random, sys, argparse
import numpy as np
import torch, torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCKCHAIN_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CORE_DIR = os.path.join(BLOCKCHAIN_DIR, "core")
sys.path.insert(0, BLOCKCHAIN_DIR)
sys.path.insert(0, CORE_DIR)
sys.path.insert(0, SCRIPT_DIR)

from ntu60_aug_loader import load_ntu60_augmented
from unified_data_layer import TrainStatsCollector, compute_unified_data_trust
from attacks_advanced import bone_length_attack_np, apply_bone_length_to_batch

# ST-GCN
STGCN_DIR = "/media/njit5/39e16d05-05ba-46b4-bc9b-47b3c98f1d4f/tlf/st-gcn"
sys.path.insert(0, STGCN_DIR)
from net.st_gcn import Model as STGCN

DATA_DIR = "/media/njit5/39e16d05-05ba-46b4-bc9b-47b3c98f1d4f/tlf/datasets/dir0.1"
NUM_CLASSES = 60

# ── LR Schedule ───────────────────────────────────────
def get_lr_schedule(rounds, base_lr=0.1, warmup=5,
                    step_milestones=[35, 55], decay=0.1):
    """
    Warmup + step decay for NTU-60 training.

    Matches run_ntu60_aug.py schedule:
      R1-5:   linear warmup 0 → 0.1
      R6-20:  lr = 0.1
      R21-40: lr = 0.01
      R41-50: lr = 0.001
      R51-58: lr = 0.0001
      R59-65: lr = 0.00001
    """
    schedule = []
    for r in range(1, rounds + 1):
        if r <= warmup:
            lr = base_lr * r / warmup
        else:
            factor = 1.0
            for s in step_milestones:
                if r > s:
                    factor *= decay
            lr = base_lr * factor
        schedule.append(lr)
    return schedule

# ════════════════ Utils ════════════════

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

def state_sub(a, b):
    return {k: v - b[k] for k, v in a.items() if v.is_floating_point()}

def state_add(a, upd):
    return {k: v + upd.get(k, 0) for k, v in a.items() if v.is_floating_point()}

def flatten_update(upd):
    return np.concatenate([v.cpu().numpy().flatten() for v in upd.values()])

# ════════════════ Local Training ════════════════

def train_on_model_v2(model, loader, device, local_epochs=1, lr=0.01,
                       attack=None, data_noise_std=0.05,
                       stats_collector=None, cid=None):
    """
    Train a pre-created model (in-place, reuses GPU memory).
    v3: Supports bone_length attack + unified data-layer stats collection.
    """
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=1e-4, nesterov=True)
    crit = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # ── Data-layer attacks ──
            if attack == "data_noise":
                x = x + torch.randn_like(x) * data_noise_std
            elif attack == "bone_length":
                x = apply_bone_length_to_batch(x, scale=0.2)
            if attack == "label_flip":
                y = (NUM_CLASSES - 1 - y) % NUM_CLASSES

            opt.zero_grad()
            output = model(x)
            loss = crit(output, y)
            loss_item = loss.item()  # capture before backward
            loss.backward()
            opt.step()
            if stats_collector is not None and cid is not None:
                stats_collector.record_batch(cid, loss_item, model.parameters())
    return {k: v.cpu() for k, v in model.state_dict().items()}

# ════════════════ Aggregation ════════════════

def fedavg_aggregate(updates, weights=None):
    keys = [k for k in next(iter(updates.values()))
            if isinstance(next(iter(updates.values()))[k], torch.Tensor)]
    result = {}
    for k in keys:
        stacked = torch.stack([u[k].float() for u in updates.values()])
        if weights:
            w = torch.tensor([weights.get(c, 1.0/len(updates)) for c in updates],
                           dtype=torch.float32)
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
                dists.append(np.linalg.norm(
                    flat_updates[ids[i]] - flat_updates[ids[j]]))
        dists.sort()
        scores[i] = sum(dists[:n - f - 1])
    selected = np.argsort(scores)[:n - f]
    sel_updates = {ids[s]: updates[ids[s]] for s in selected}
    return fedavg_aggregate(sel_updates)

def trimmed_mean_aggregate(updates, beta=0.2):
    keys = [k for k in next(iter(updates.values()))
            if isinstance(next(iter(updates.values()))[k], torch.Tensor)]
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
    root_flat = flat_updates[ids[0]]
    root_norm = np.linalg.norm(root_flat) + 1e-12
    scores = {}
    for cid in ids:
        v = flat_updates[cid]
        v_norm = np.linalg.norm(v) + 1e-12
        cs = max(0, float(np.dot(v, root_flat) / (v_norm * root_norm)))
        scores[cid] = cs
    total = sum(scores.values())
    weights = {cid: s/total for cid, s in scores.items()} if total > 1e-12 \
              else {cid: 1.0/n for cid in ids}
    return fedavg_aggregate(updates, weights)

# ════════════════ TASL Trust Scoring (v3: unified data-layer) ════════════════

def compute_tasl_trust_weights(
    flat_updates, trust_power=5.0, max_weight_ratio=1.5, min_cos_threshold=0.2,
    norm_penalty_strength=0.8, refine_anchor=True,
    ema_weights=None, ema_alpha=0.7, cos_history=None, prev_anchor=None,
    temporal_anchor=None, data_trust=None,
):
    """
    TASL trust scoring v3: gradient-layer + unified data-layer multiplicative fusion.
    data_trust[c] x gradient_trust[c] -> one-vote veto from either layer.
    """
    n = len(flat_updates); ids = list(flat_updates.keys())
    if n == 0: return {}, {}, None

    # ── Norm stats ──
    norms = {cid: np.linalg.norm(v) for cid, v in flat_updates.items()}
    median_norm = np.median(list(norms.values()))
    norm_gate = median_norm * 2.5

    # ── Dual anchor ──
    stacked = np.stack(list(flat_updates.values()), axis=0)
    anchor = np.median(stacked, axis=0)
    anchor_norm = np.linalg.norm(anchor) + 1e-12

    # Pairwise consensus
    pairwise_medians = {}
    for i in ids:
        vi = flat_updates[i]; vi_norm = np.linalg.norm(vi) + 1e-12
        sims = [float(np.dot(vi, flat_updates[j]) /
                     (vi_norm*(np.linalg.norm(flat_updates[j])+1e-12)))
                for j in ids if j != i]
        pairwise_medians[i] = float(np.median(sims)) if sims else 0.5

    med_c = np.median(list(pairwise_medians.values()))
    high_ids = [cid for cid in ids if pairwise_medians[cid] >= med_c]
    if len(high_ids) >= max(2, n//2):
        cons_stacked = np.stack([flat_updates[cid] for cid in high_ids])
        anchor = 0.6 * anchor + 0.4 * np.median(cons_stacked, axis=0)
        anchor_norm = np.linalg.norm(anchor) + 1e-12

    # Refine anchor with high-cos subset
    if refine_anchor:
        fps = {cid: float(np.dot(flat_updates[cid], anchor) /
                         (np.linalg.norm(flat_updates[cid])+1e-12)/anchor_norm)
               for cid in ids}
        tids = [cid for cid in ids if fps[cid] > 0]
        if len(tids) >= max(2, n//2):
            ref = np.median(np.stack([flat_updates[c] for c in tids]), axis=0)
            anchor = 0.6 * ref + 0.4 * (prev_anchor if prev_anchor is not None else ref)
            anchor_norm = np.linalg.norm(anchor) + 1e-12

    # ── Cosine scores + adaptive threshold ──
    cos_sims = {cid: float(np.dot(v, anchor) /
                          (np.linalg.norm(v)+1e-12)/anchor_norm)
                for cid, v in flat_updates.items()}
    cv = sorted(cos_sims.values())
    if len(cv) >= 4:
        q1, q3 = np.percentile(cv, 25), np.percentile(cv, 75)
        iqr = q3 - q1
        adaptive_thr = min(0.4, max(min_cos_threshold, q1 - 1.0*iqr))
    else:
        adaptive_thr = min_cos_threshold

    # ── Raw trust scores ──
    rs = {}
    for cid in ids:
        cos = cos_sims[cid]
        if norms[cid] > norm_gate or cos < adaptive_thr:
            rs[cid] = 0.0; continue
        nr = norms[cid]/(median_norm+1e-12)
        np_ = np.exp(-norm_penalty_strength*abs(nr-1))
        con = max(0.01, pairwise_medians.get(cid, 0.5))
        rs[cid] = (cos**trust_power) * np_ * con

    # ── Persistent low-cos penalty ──
    if cos_history and len(cos_history) >= 2:
        for cid in ids:
            if all(ch.get(cid, 0.5) < 0.3 for ch in cos_history[-2:]):
                rs[cid] *= 0.1

    # ── v3: Unified data-layer multiplicative fusion ──
    if data_trust:
        for cid in ids:
            rs[cid] *= data_trust.get(cid, 1.0)

    # ── Normalize + cap + EMA ──
    sw = sum(rs.values())
    if sw > 1e-12:
        tw = {cid: w/sw for cid, w in rs.items()}
    else:
        tw = {cid: 1.0/n for cid in ids}

    mw = max_weight_ratio/n
    capped = {c: min(w, mw) for c, w in tw.items()}
    cs2 = sum(capped.values())
    if cs2 > 1e-12: tw = {c: w/cs2 for c, w in capped.items()}

    if ema_weights:
        sm = {cid: ema_alpha*tw.get(cid,0)+(1-ema_alpha)*ema_weights.get(cid,1.0/n)
              for cid in ids}
        ss = sum(sm.values())
        if ss > 1e-12: tw = {c: w/ss for c, w in sm.items()}

    return tw, cos_sims, anchor

# ════════════════ run_single (v3) ════════════════

def run_single(algo, attack, client_loaders, test_loader, n_clients, n_byz,
               rounds, device, local_epochs=1, lr=0.1, seed=42,
               lr_schedule=None, data_noise_std=0.05):
    set_seed(seed)
    byz_set = set(range(n_byz))
    global_model = create_model(device)
    global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}

    client_models = [create_model(device) for _ in range(n_clients)]

    acc_hist, ema_w, cos_h, prev_a, temp_a = [], None, [], None, None
    best = 0.0

    stats_collector = TrainStatsCollector()

    for r in range(1, rounds+1):
        # ── Get per-round LR ──
        cur_lr = lr_schedule[r-1] if lr_schedule else lr

        stats_collector.reset()

        local_updates, flat_updates = {}, {}
        for cid in range(n_clients):
            client_models[cid].load_state_dict(
                {k: v.to(device) for k, v in global_state.items()})

            atk = attack if (cid in byz_set and attack != "none") else None

            if atk == "sign_flip":
                train_on_model_v2(client_models[cid], client_loaders[cid], device,
                                 local_epochs=local_epochs, lr=cur_lr, attack=None,
                                 stats_collector=stats_collector, cid=cid)
                state = {k: v.cpu() for k, v in client_models[cid].state_dict().items()}
                local_state = {k: global_state[k] - (v - global_state[k])
                               for k, v in state.items()
                               if k in global_state and v.is_floating_point()}
            elif atk == "gaussian_noise":
                train_on_model_v2(client_models[cid], client_loaders[cid], device,
                                 local_epochs=local_epochs, lr=cur_lr, attack=None,
                                 stats_collector=stats_collector, cid=cid)
                state = client_models[cid].state_dict()
                local_state = {}
                for k, v in state.items():
                    if v.is_floating_point():
                        local_state[k] = v.cpu() + torch.randn_like(v.cpu()) * 0.1 * v.cpu().std()
                    else:
                        local_state[k] = v.cpu()
            else:
                local_state = train_on_model_v2(
                    client_models[cid], client_loaders[cid], device,
                    local_epochs=local_epochs, lr=cur_lr, attack=atk,
                    data_noise_std=data_noise_std,
                    stats_collector=stats_collector, cid=cid)

            local_updates[cid] = state_sub(local_state, global_state)
            flat_updates[cid] = flatten_update(local_updates[cid])

        # ── Compute unified data trust per round ──
        round_stats = stats_collector.get_stats()
        data_trust, diag = compute_unified_data_trust(round_stats, n_clients)

        # ── Aggregation ──
        if algo == "fedavg":
            agg = fedavg_aggregate(local_updates)
        elif algo == "multi_krum":
            agg = multi_krum_aggregate(flat_updates, local_updates, f=n_byz)
        elif algo == "trimmed_mean":
            agg = trimmed_mean_aggregate(local_updates)
        elif algo == "fltrust":
            agg = fltrust_aggregate(flat_updates, local_updates, device)
        elif algo == "tasl":
            tw, cs, na = compute_tasl_trust_weights(
                flat_updates, ema_weights=ema_w, cos_history=cos_h,
                prev_anchor=prev_a, temporal_anchor=temp_a,
                data_trust=data_trust)
            ema_w = dict(tw); cos_h.append(dict(cs)); prev_a = na
            agg = fedavg_aggregate(local_updates, tw)
            af = flatten_update(agg)
            if np.linalg.norm(af) > 1e-12: temp_a = af
        else:
            raise ValueError(f"Unknown algo: {algo}")

        global_state = state_add(global_state, agg)
        global_model.load_state_dict({k: v.to(device) for k, v in global_state.items()})
        acc = evaluate(global_model, test_loader, device)
        acc_hist.append(acc); best = max(best, acc)

        if r % 10 == 0 or r == 1:
            parts = [f"  [{algo}/{attack}] R{r:02d} acc={acc:.2f}% best={best:.2f}%"]
            if algo == "tasl" and ema_w:
                bw = np.mean([ema_w.get(c, 0.0) for c in byz_set])
                parts.append(f"byz_w={bw:.4f}")
            if data_trust:
                dt_mean = np.mean([data_trust.get(c, 0.0) for c in byz_set])
                parts.append(f"data_t={dt_mean:.2f}")
            print("".join(parts))

        torch.cuda.empty_cache()

    for m in client_models: del m
    del global_model; torch.cuda.empty_cache()

    return {"best_acc": best, "avg_acc": float(np.mean(acc_hist)),
            "acc_history": acc_hist}

# ════════════════ main ════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=65)
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--byzantine-ratio", type=float, default=0.4)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--algorithms", type=str,
                   default="fedavg,multi_krum,trimmed_mean,fltrust,tasl")
    p.add_argument("--attacks", type=str,
                   default="label_flip,sign_flip,gaussian_noise,bone_length")
    p.add_argument("--data-noise-std", type=float, default=0.05,
                   help="Noise std for data_noise attack")
    p.add_argument("--lr-warmup", type=int, default=5)
    p.add_argument("--lr-steps", type=str, default="35,55")
    args = p.parse_args()

    if args.quick:
        args.clients = 5; args.rounds = 10; args.local_epochs = 1
        args.byzantine_ratio = 0.4

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_clients = int(args.clients)
    n_byz = max(1, int(n_clients * args.byzantine_ratio))
    rounds = int(args.rounds)
    algos = args.algorithms.split(",")
    attacks_list = args.attacks.split(",")

    # ── LR schedule ──
    lr_steps = [int(x) for x in args.lr_steps.split(",")]
    lr_schedule = get_lr_schedule(rounds, base_lr=args.lr,
                                  warmup=args.lr_warmup,
                                  step_milestones=lr_steps)
    print(f"LR schedule: warmup={args.lr_warmup}, "
          f"steps={lr_steps}, base_lr={args.lr}")
    print(f"  R001 lr={lr_schedule[0]:.5f} -> "
          f"R{rounds:03d} lr={lr_schedule[-1]:.5f}")

    # ── Load augmented data ──
    print(f"Loading NTU-60 augmented data (random_rot + crop_resize)...")
    client_loaders, test_loader = load_ntu60_augmented(
        DATA_DIR, n_clients=n_clients, batch_size=64)
    print(f"  Data loaded: {n_clients} clients")

    # ── Print config ──
    print("=" * 70)
    print(f"[Exp2 NTU-60 v3] Unified data-layer (loss + grad_norm) + TASL — ST-GCN")
    print(f"  {n_clients} clients, {n_byz} Byzantine ({args.byzantine_ratio:.0%})")
    print(f"  Non-IID alpha=0.1, FedPure augmentation (random_rot + valid_crop_resize)")
    print(f"  {rounds} rounds, local_epochs={args.local_epochs}, lr schedule")
    print(f"  Algorithms: {algos}")
    print(f"  Attacks: {attacks_list}")
    print(f"  Data-layer: Unified (loss + grad_norm, per-round)")
    print(f"  TASL: trust_power=5, weight_cap=1.5/n, ema_alpha=0.7, "
          f"dual_anchor+norm_gate+data_fusion")
    print("=" * 70)

    # ── Clean baseline ──
    print("\n>>> Clean Baseline <<<")
    clean = run_single("fedavg", "none", client_loaders, test_loader,
                       n_clients, 0, rounds, device,
                       args.local_epochs, args.lr, args.seed,
                       lr_schedule=lr_schedule)
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
                           n_clients, n_byz, rounds, device,
                           args.local_epochs, args.lr, args.seed,
                           lr_schedule=lr_schedule,
                           data_noise_std=args.data_noise_std)
            results[algo][attack] = res
            print(f"  => {algo}/{attack}: best={res['best_acc']:.2f}%, "
                  f"avg={res['avg_acc']:.2f}%")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("SUMMARY: Best Accuracy")
    print("=" * 70)
    header = f"  {'Attack':<18}"
    for a in algos: header += f" {a:<18}"
    print(header)
    print("  " + "-" * (18 + 18*len(algos)))
    for attack in ["none"] + attacks_list:
        row = f"  {attack:<18}"
        for algo in algos:
            val = f"{clean_best:.2f}%" if attack == "none" \
                  else f"{results[algo][attack]['best_acc']:.2f}%"
            row += f" {val:<18}"
        print(row)

    # ── Save CSV ──
    results_dir = os.path.join(BLOCKCHAIN_DIR, "results", "exp2_ntu60_v3")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "exp2_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "attack", "best_acc", "avg_acc"])
        for algo in algos:
            for attack in attacks_list:
                r = results[algo][attack]
                w.writerow([algo, attack, f"{r['best_acc']:.2f}", f"{r['avg_acc']:.2f}"])
    print(f"\nResults saved to {csv_path}")

    # ── Also save config ──
    import json
    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "rounds": rounds, "n_clients": n_clients, "n_byz": n_byz,
            "alpha": 0.1, "lr_schedule": [float(x) for x in lr_schedule],
            "data_layer": "unified",
            "data_noise_std": args.data_noise_std,
            "clean_best": clean_best, "clean_avg": clean_avg,
        }, f, indent=2)
    print(f"Config saved to {config_path}")

if __name__ == "__main__":
    main()
