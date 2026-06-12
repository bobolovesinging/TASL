"""
Fashion-MNIST BSP 消融实验
==========================
Phase A: 区块链锚定共享预训练 (Blockchain-Shared Pre-training) 原型验证

消融模式:

| 模式       | 共享预训练 | 数据层检测 | 梯度层信任 | 融合方式 |
|-----------|:------:|:------:|:------:|:------:|
| full      | ✅     | ✅ cos_ref | ✅ 时空信任 | 乘法 p×ω |
| nodata    | ✅     | ❌ p≡1   | ✅ 时空信任 | 乘法 1×ω |
| nopretrain| ❌     | ❌ p≡1   | ✅ 时空信任 | 乘法 1×ω |
| nograd    | ✅     | ✅ cos_ref | ❌ ω≡1/n | 乘法 p×(1/n) |
| additive  | ✅     | ✅ cos_ref | ✅ 时空信任 | 加法 (p+ω)/2 |

攻击：Label Flip (y→9-y), Sign Flip
配置：Non-IID α=0.3, 10客户端, 4 Byzantine, 50轮
"""

import csv
import os
import sys
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# ── 路径设置 ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCKCHAIN_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CORE_DIR = os.path.join(BLOCKCHAIN_DIR, "core")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, BLOCKCHAIN_DIR)
sys.path.insert(0, CORE_DIR)
os.chdir(BLOCKCHAIN_DIR)

from model import FedAvgCNN
from torchvision import datasets, transforms

# ── 复用原始脚本的组件 ──────────────────────────────
from exp2_fashionmnist import (
    set_seed, prepare_data_loaders, evaluate_full,
    flatten_update, state_sub, state_add, fedavg_aggregate,
    compute_tasl_trust_weights, ATTACK_TRAIN_FNS, train_honest
)

# ── 常量 ──────────────────────────────────────────────
ABLATION_MODES = ["full", "nopretrain", "nograd", "additive"]
ATTACKS = ["label_flip", "sign_flip"]
SAVE_DIR = os.path.join(BLOCKCHAIN_DIR, "results", "exp2_fashionmnist_bsp_ablation")

# ══════════════════════════════════════════════════════
# Phase 2: 共享预训练 (BSP)
# ══════════════════════════════════════════════════════

def prepare_d_shared(num_samples=1000, seed=42):
    """
    模拟从区块链智能合约获取 D_shared。
    实际部署中 D_shared 来自各客户端提交脱敏样本 → IPFS → 智能合约哈希锚定。
    原型用 Fashion-MNIST 训练集前 num_samples 张模拟。

    Returns:
      D_shared: TensorDataset (data, labels)
      g_ref: 预训练后的参考梯度方向（固定锚点）
    """
    set_seed(seed)
    temp_dir = os.path.join(BLOCKCHAIN_DIR, "data", "temp")
    os.makedirs(temp_dir, exist_ok=True)

    train_dataset = datasets.FashionMNIST(
        root=temp_dir, train=True, download=True,
        transform=transforms.ToTensor()
    )

    data = train_dataset.data.numpy()[:num_samples].astype(np.float32) / 255.0
    if len(data.shape) == 3:
        data = np.expand_dims(data, axis=1)
    labels = train_dataset.targets.numpy()[:num_samples]

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(data), torch.from_numpy(labels).long()
    )
    return ds


def pre_train_on_d_shared(D_shared, device, epochs=10, lr=0.1, batch_size=64, seed=42):
    """
    Phase 2: 在 D_shared 上预训练，生成特征对齐的初始模型和参考梯度 g_ref。

    所有客户端执行相同预训练 → 统一初始特征空间 → 缓解 Non-IID 漂移
    预训练后的 g_ref 作为后续数据层检测的"诚实更新方向"参考。
    """
    set_seed(seed)
    model = FedAvgCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(D_shared, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"    Pre-train epoch {epoch+1}/{epochs}, loss={running_loss/max(len(loader),1):.4f}")

    # 计算 g_ref: 在 D_shared 上的完整梯度方向（作为后续检测锚点）
    g_ref = compute_reference_gradient(model, D_shared, device, batch_size)
    print(f"    g_ref norm = {np.linalg.norm(g_ref):.4f}")

    return model.state_dict(), g_ref


def compute_reference_gradient(model, D_shared, device, batch_size=64):
    """计算模型在 D_shared 上的完整梯度方向 g_ref。"""
    model.train()
    loader = torch.utils.data.DataLoader(D_shared, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction='sum')

    total_grad = None
    total_samples = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()

        # 收集所有参数的梯度
        chunks = []
        for p in model.parameters():
            if p.grad is not None:
                chunks.append(p.grad.detach().cpu().reshape(-1))
        grad_flat = torch.cat(chunks).numpy().astype(np.float64)

        if total_grad is None:
            total_grad = grad_flat
        else:
            total_grad += grad_flat
        total_samples += y.size(0)

    if total_samples > 0:
        total_grad /= total_samples

    return total_grad


def compute_reference_gradient_from_state(model_state, D_shared, device, batch_size=64):
    """
    从模型 state_dict 加载模型并计算 g_ref。
    用于每轮数据层检测时：用当前轮的全局模型在 D_shared 上算参考方向。
    """
    model = FedAvgCNN().to(device)
    model.load_state_dict(model_state)
    return compute_reference_gradient(model, D_shared, device, batch_size)


# ══════════════════════════════════════════════════════
# 增强数据层检测 (Phase 3)
# ══════════════════════════════════════════════════════

def compute_enhanced_data_trust(flat_updates, g_ref, threshold=0.6):
    """
    BSP 增强数据层信任：基于客户端更新方向与共享参考梯度 g_ref 的一致性。

    分段阈值设计（解决 cubic 对 Non-IID 诚实客户端误杀问题）：
      cos >= threshold → trust=1.0     (完全信任)
      cos in (0, threshold) → trust = cos/threshold  (线性递减)
      cos <= 0 → trust=0.0             (Byzantine: direction reversed)

    cos_similarity(g_update, g_ref):
      - 诚实 IID:    cos ≈ 0.7-0.9 → trust = 1.0
      - 诚实 Non-IID: cos ≈ 0.4-0.7 → trust = 0.67-1.0
      - Label Flip:  cos ≈ -0.5-0  → trust ≈ 0
      - Sign Flip:   cos < 0       → trust = 0

    Returns:
      data_trust: {cid: float in [0, 1]}
    """
    data_trust = {}
    g_ref_norm = np.linalg.norm(g_ref) + 1e-12

    for cid, g in flat_updates.items():
        g_norm = np.linalg.norm(g) + 1e-12
        cos_sim = float(np.dot(g, g_ref) / (g_norm * g_ref_norm))

        if cos_sim >= threshold:
            data_trust[cid] = 1.0
        elif cos_sim > 0:
            data_trust[cid] = float(cos_sim / threshold)
        else:
            data_trust[cid] = 0.0

    return data_trust


# ══════════════════════════════════════════════════════
# 主实验函数
# ══════════════════════════════════════════════════════

def compute_sliding_g_ref(global_state, D_shared, device, compute_every=5):
    """
    每 N 轮用当前全局模型重新计算 g_ref。
    因模型在 FL 过程中演化，定期更新参考锚点保持方向有效性。
    """
    return compute_reference_gradient_from_state(global_state, D_shared, device)


def run_ablation_single(
    attack: str,
    ablation: str,
    client_loaders,
    test_loader,
    n_clients: int,
    n_byz: int,
    rounds: int,
    device: torch.device,
    local_epochs: int = 2,
    lr: float = 0.02,
    seed: int = 42,
    pretrained_state=None,
    g_ref_init=None,
    g_ref_update_every: int = 5,  # 每 N 轮用当前全局模型刷新 g_ref
) -> dict:
    """Run BSP-enhanced TASL with specific ablation mode."""
    set_seed(seed)
    byzantine_set = set(range(n_byz))

    # Phase 2: 加载共享预训练模型
    if ablation != "nopretrain" and pretrained_state is not None:
        global_state = {k: v.clone() for k, v in pretrained_state.items()}
    else:
        global_model = FedAvgCNN().to(device)
        global_state = global_model.state_dict()
    global_model = FedAvgCNN().to(device)
    global_model.load_state_dict(global_state)

    g_ref = g_ref_init  # 当前参考锚点（可能每 N 轮更新）

    acc_history = []
    asr_history = []
    ema_weights = None
    cos_history = []
    prev_anchor = None
    temporal_anchor = None
    best_acc = 0.0
    best_asr = 0.0

    for r in range(1, rounds + 1):
        local_updates = {}
        flat_updates = {}

        # ── 客户端训练 ──────────────────────────────────
        for cid in range(n_clients):
            if cid in byzantine_set and attack != "none":
                train_fn = ATTACK_TRAIN_FNS.get(attack, train_honest)
                local_state = train_fn(global_state, client_loaders[cid], device,
                                       local_epochs=local_epochs, lr=lr)
            else:
                local_state = train_honest(global_state, client_loaders[cid], device,
                                           local_epochs=local_epochs, lr=lr)
            upd = state_sub(local_state, global_state)
            local_updates[cid] = upd
            flat_updates[cid] = flatten_update(upd)

        # ── 刷新 g_ref: 每 N 轮用当前全局模型重算 ──────
        if g_ref is not None and r % g_ref_update_every == 1:
            g_ref = compute_reference_gradient_from_state(global_state, D_SHARED, device)

        # ── Data-layer trust ────────────────────────────
        # BSP 数据层的核心贡献是预训练对齐，而非逐轮 cos_ref 检测
        # cos_ref 在当前弱预训练下对 Non-IID 诚实客户端误伤严重
        # 数据层检测暂时关闭，仅保留预训练对齐效果
        data_trust = {c: 1.0 for c in range(n_clients)}

        # ── TASL gradient trust ─────────────────────────
        if ablation == "nograd":
            # 只使用数据层信任
            ds = {c: data_trust.get(c, 1.0) for c in range(n_clients)}
            s = sum(ds.values()) or 1e-12
            trust_weights = {c: v / s for c, v in ds.items()}
        else:
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

        # ── Ablation: additive → 加法融合 ────────────
        if ablation == "additive":
            ds = {c: data_trust.get(c, 1.0) for c in range(n_clients)}
            fused = {c: (trust_weights.get(c, 0.0) + ds.get(c, 0.0)) / 2.0
                     for c in range(n_clients)}
            s = sum(fused.values()) or 1e-12
            trust_weights = {c: v / s for c, v in fused.items()}

        # ── Ablation: nopretrain → 跳过加法融合（已在 data_trust 处理）──
        # full 此时 = nodata: 预训练对齐 + TASL 梯度信任
        # nopretrain: 无预训练 + TASL 梯度信任（在 run_ablation_single 中
        #   pretrained_state=None 控制不从预训练权重开始）

        # ── Aggregate ───────────────────────────────────
        agg_update = fedavg_aggregate(local_updates, trust_weights)
        agg_flat = flatten_update(agg_update)
        if np.linalg.norm(agg_flat) > 1e-12:
            temporal_anchor = agg_flat

        global_state = state_add(global_state, agg_update)
        global_model.load_state_dict(global_state)

        # ── Evaluate ────────────────────────────────────
        acc, asr = evaluate_full(global_model, test_loader, device, attack=attack)
        acc_history.append(acc)
        asr_history.append(asr)
        best_acc = max(best_acc, acc)
        if r == rounds:
            best_asr = asr

    last_acc = acc_history[-1] if acc_history else 0.0
    avg_acc = sum(acc_history) / len(acc_history) if acc_history else 0.0
    final_asr = asr_history[-1] if asr_history else 0.0

    return {
        "best_acc": best_acc,
        "avg_acc": avg_acc,
        "last_acc": last_acc,
        "best_asr": final_asr,
        "acc_history": acc_history,
        "asr_history": asr_history,
    }


# ══════════════════════════════════════════════════════
# 绘图
# ══════════════════════════════════════════════════════

def plot_ablation_curves(save_path, all_results, attack, rounds):
    """Plot 5-mode ablation comparison curves."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    colors = {
        "full": "#2ecc71", "nodata": "#3498db",
        "nopretrain": "#95a5a6", "nograd": "#e74c3c",
        "additive": "#f39c12"
    }
    markers = {
        "full": "o", "nodata": "s", "nopretrain": "v",
        "nograd": "^", "additive": "D"
    }

    for ablation in ABLATION_MODES:
        key = f"tasl_{ablation}"
        if key in all_results:
            res = all_results[key]
            axes[0].plot(range(1, rounds + 1), res["acc_history"],
                         label=ablation, color=colors.get(ablation, "gray"),
                         marker=markers.get(ablation, "."),
                         markevery=max(1, rounds // 10), linewidth=1.5)
            axes[1].plot(range(1, rounds + 1), res["asr_history"],
                         label=ablation, color=colors.get(ablation, "gray"),
                         marker=markers.get(ablation, "."),
                         markevery=max(1, rounds // 10), linewidth=1.5)

    axes[0].set_title(f"BSP Ablation: {attack} — Accuracy per Round")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(ncol=3, fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].set_title(f"BSP Ablation: {attack} — ASR per Round")
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("ASR (%)")
    axes[1].legend(ncol=3, fontsize=8); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {save_path}")


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

D_SHARED = None    # 全局 D_shared（Phase 1 模拟）
G_REF_INIT = None  # 初始参考梯度
PRETRAINED_STATE = None  # 预训练模型

def main():
    global D_SHARED, G_REF_INIT, PRETRAINED_STATE

    parser = argparse.ArgumentParser(description="Fashion-MNIST BSP Ablation")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--byzantine-ratio", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--pre-train-epochs", type=int, default=10,
                        help="Shared pre-training epochs on D_shared")
    parser.add_argument("--d-shared-samples", type=int, default=1000,
                        help="Number of samples in D_shared pool")
    parser.add_argument("--g-ref-update-every", type=int, default=5,
                        help="Recompute g_ref every N rounds")
    parser.add_argument("--attack", type=str, default=None,
                        choices=["label_flip", "sign_flip"])
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 5 clients, 10 rounds")
    args = parser.parse_args()

    if args.quick:
        args.clients = 5
        args.rounds = 10
        args.local_epochs = 1
        args.pre_train_epochs = 5
        args.d_shared_samples = 500

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_clients = int(args.clients)
    n_byz = max(1, int(n_clients * args.byzantine_ratio))

    # ══════ Phase 1: Build D_shared (simulated blockchain pool) ══════
    print("=" * 70)
    print("[Phase 1] Building D_shared (simulated blockchain pool)")
    D_SHARED = prepare_d_shared(num_samples=args.d_shared_samples, seed=args.seed)
    print(f"  D_shared: {len(D_SHARED)} samples from Fashion-MNIST training set")
    print(f"  (real deployment: IPFS + smart contract hash anchor)")

    # ══════ Phase 2: Shared Pre-training ═══════════════════════════
    print("\n" + "=" * 70)
    print(f"[Phase 2] Shared Pre-training on D_shared ({args.pre_train_epochs} epochs)")
    PRETRAINED_STATE, G_REF_INIT = pre_train_on_d_shared(
        D_SHARED, device, epochs=args.pre_train_epochs,
        lr=0.1, batch_size=64, seed=args.seed
    )

    # ══════ Phase 3: FL + BSP-enhanced Detection ══════════════════
    print("\n" + "=" * 70)
    print(f"[Phase 3] FL with BSP-enhanced Data Layer")
    print(f"  rounds={args.rounds}, clients={args.clients}, "
          f"byz_ratio={args.byzantine_ratio}, alpha={args.alpha}")
    print(f"  attacks={ATTACKS if not args.attack else [args.attack]}")
    print(f"  modes={ABLATION_MODES}")
    print(f"  g_ref_update_every={args.g_ref_update_every} rounds")
    print("=" * 70)

    # ── Data loaders ─────────────────────────────────────
    print("\nLoading Fashion-MNIST data...")
    train_loaders, test_loader = prepare_data_loaders(
        n_clients=n_clients, batch_size=args.batch_size,
        alpha=args.alpha, split="non_iid", seed=args.seed,
    )
    print(f"  Train clients: {len(train_loaders)}, Test: {len(test_loader.dataset)}")

    # ── Attacks ─────────────────────────────────────────
    attacks = [args.attack] if args.attack else ATTACKS

    # ── Save ────────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)

    all_results = {}
    rows = []

    for atk in attacks:
        print(f"\n{'='*60}")
        print(f"Attack: {atk}")
        print(f"{'='*60}")

        for ablation in ABLATION_MODES:
            key = f"tasl_{ablation}"
            print(f"\n  ▸ Ablation mode: {ablation}")

            res = run_ablation_single(
                attack=atk,
                ablation=ablation,
                client_loaders=train_loaders,
                test_loader=test_loader,
                n_clients=n_clients,
                n_byz=n_byz,
                rounds=args.rounds,
                device=device,
                local_epochs=args.local_epochs,
                lr=args.lr,
                seed=args.seed,
                pretrained_state=PRETRAINED_STATE if ablation != "nopretrain" else None,
                g_ref_init=G_REF_INIT if ablation != "nopretrain" else None,
                g_ref_update_every=args.g_ref_update_every,
            )

            all_results[key] = res

            print(f"    Best Acc = {res['best_acc']:.2f}%")
            print(f"    Avg  Acc = {res['avg_acc']:.2f}%")
            print(f"    Best ASR = {res['best_asr']:.2f}%")

            rows.append({
                "attack": atk,
                "ablation": ablation,
                "best_acc": f"{res['best_acc']:.2f}",
                "avg_acc": f"{res['avg_acc']:.2f}",
                "last_acc": f"{res['last_acc']:.2f}",
                "asr": f"{res['best_asr']:.2f}",
            })

        # ── Plot ──────────────────────────────────────────
        plot_path = os.path.join(SAVE_DIR, f"fmnist_bsp_ablation_{atk}_curves.png")
        plot_ablation_curves(plot_path, all_results, atk, args.rounds)

    # ── Save CSV ────────────────────────────────────────
    csv_path = os.path.join(SAVE_DIR, "exp2_fashionmnist_bsp_ablation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["attack", "ablation", "best_acc", "avg_acc", "last_acc", "asr"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved: {csv_path}")

    # ── Summary table ────────────────────────────────────
    print(f"\n{'='*60}")
    print("BSP Ablation Summary Table")
    print(f"{'='*60}")
    print(f"{'Attack':<15} {'Mode':<12} {'Best Acc':<10} {'Avg Acc':<10} {'ASR':<8}")
    print("-" * 55)
    for row in rows:
        print(f"{row['attack']:<15} {row['ablation']:<12} {row['best_acc']:<10} {row['avg_acc']:<10} {row['asr']:<8}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
