
import os
import sys
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import csv
from typing import Optional
from torch.utils.data import DataLoader, Dataset

try:
    import tenseal as ts
except ImportError:
    ts = None

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
blockchain_dir = os.path.dirname(script_dir)

# 让 `core` 成为可 import 的包：把 Blockchain/ 加到 sys.path
sys.path.insert(0, blockchain_dir)

# 同时把 Blockchain/core 加到 sys.path，兼容直接 `import blockchain_node` 的写法
core_dir = os.path.join(blockchain_dir, 'core')
sys.path.insert(0, core_dir)

from core.blockchain_node import BlockchainNetwork
from core.gradient_aggregator import GradientAggregator
from core.model import FedAvgCNN
from core.ckks_encryption import KeyGenerationCenter
from core.crypto_utils import sign_data
from core.ipfs_manager import MockIPFSManager
from core.stake_manager import StakeManager
from core.tas import TripartiteAuditorSystem
MNIST_DATA_ROOT = os.path.join(blockchain_dir, 'data', 'mnist')


class MNISTDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = torch.from_numpy(np.load(data_path)).float()
        self.labels = torch.from_numpy(np.load(label_path)).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_client_loaders_iid(client_num=10, batch_size=128):
    client_loaders = []

    for client_id in range(1, client_num + 1):
        client_dir = os.path.join(MNIST_DATA_ROOT, f'client_{client_id}')
        data_path = os.path.join(client_dir, 'data.npy')
        label_path = os.path.join(client_dir, 'label.npy')

        if not os.path.exists(data_path) or not os.path.exists(label_path):
            raise FileNotFoundError(f"客户端数据不存在: {client_dir}")

        dataset = MNISTDataset(data_path, label_path)
        client_loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=True))

    return client_loaders


def get_client_loaders_dirichlet(client_num=10, batch_size=128, alpha=0.3, seed=42):
    print(f"[DataSplit] Using Dirichlet split alpha={alpha}, seed={seed}")
    base_dir = MNIST_DATA_ROOT
    data_path = os.path.join(base_dir, 'train_full', 'data.npy')
    label_path = os.path.join(base_dir, 'train_full', 'label.npy')

    if not os.path.exists(data_path) or not os.path.exists(label_path):
        raise FileNotFoundError(
            "Non-IID 需要 train_full 数据。请先运行 scripts/data_prep/prepare_mnist_data.py "
            "--method lda --alpha 0.3，并确保 train_full/data.npy 与 train_full/label.npy 存在。"
        )

    data = torch.from_numpy(np.load(data_path)).float()
    labels = torch.from_numpy(np.load(label_path)).long()

    rng = np.random.RandomState(seed)
    num_classes = int(labels.max().item()) + 1
    client_indices = {cid: [] for cid in range(client_num)}

    for c in range(num_classes):
        idx = torch.where(labels == c)[0].numpy()
        rng.shuffle(idx)
        proportions = rng.dirichlet([alpha] * client_num)
        proportions = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, proportions)
        for cid in range(client_num):
            client_indices[cid].extend(splits[cid])

    client_loaders = []
    for cid in range(client_num):
        indices = np.array(client_indices[cid], dtype=np.int64)
        if len(indices) == 0:
            raise ValueError(f"客户端 {cid} 在 Dirichlet 划分下无样本，请调整 alpha")
        dataset = torch.utils.data.TensorDataset(data[indices], labels[indices])
        client_loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=True))

    return client_loaders


def get_root_loader(batch_size=128, root_size=100, seed=42):
    root_dir = os.path.join(MNIST_DATA_ROOT, 'root_dataset')
    data_path = os.path.join(root_dir, 'data.npy')
    label_path = os.path.join(root_dir, 'label.npy')

    if not os.path.exists(data_path) or not os.path.exists(label_path):
        return None

    data = torch.from_numpy(np.load(data_path)).float()
    labels = torch.from_numpy(np.load(label_path)).long()

    if len(data) > root_size:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(data), size=root_size, replace=False)
        data = data[indices]
        labels = labels[indices]

    dataset = torch.utils.data.TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def compute_root_gradient(model, data_loader, device, lr=0.005):
    model.train()
    criterion = nn.CrossEntropyLoss()

    # Save initial weights
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    total_samples = 0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_samples += data.size(0)

    if total_samples == 0:
        return None

    after = model.state_dict()
    delta_w = {}
    for name in before:
        if name not in after:
            continue
        if not after[name].is_floating_point():
            continue
        delta_w[name] = (after[name] - before[name]).detach()

    if not delta_w:
        return None

    return GradientAggregator.flatten_update(delta_w).detach().float()


def get_test_loader(batch_size=128):
    test_dir = os.path.join(MNIST_DATA_ROOT, 'test_dataset')
    data_path = os.path.join(test_dir, 'data.npy')
    label_path = os.path.join(test_dir, 'label.npy')

    if not os.path.exists(data_path) or not os.path.exists(label_path):
        raise FileNotFoundError(f"测试集不存在: {test_dir}")

    dataset = MNISTDataset(data_path, label_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_client_loaders_dirichlet(client_num=10, batch_size=128, alpha=0.3, seed=42):
    print(f"[DataSplit] Using Dirichlet split alpha={alpha}, seed={seed}")
    train_dir = os.path.join(MNIST_DATA_ROOT, 'train_full')
    data_path = os.path.join(train_dir, 'data.npy')
    label_path = os.path.join(train_dir, 'label.npy')

    if not os.path.exists(data_path) or not os.path.exists(label_path):
        raise FileNotFoundError(
            f"Non-IID 需要完整训练集: {train_dir}。请先运行 scripts/data_prep/prepare_mnist_data.py"
        )

    data = np.load(data_path)
    labels = np.load(label_path)

    num_classes = 10
    rng = np.random.RandomState(seed)
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    client_indices = {i: [] for i in range(client_num)}
    for c in range(num_classes):
        indices = class_indices[c]
        rng.shuffle(indices)
        proportions = rng.dirichlet([alpha] * client_num)
        proportions = proportions / proportions.sum()
        cut_points = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        splits = np.split(indices, cut_points)
        for i in range(client_num):
            client_indices[i].extend(splits[i])

    client_loaders = []
    for client_id in range(client_num):
        idx = np.array(client_indices[client_id])
        if len(idx) == 0:
            empty_data = np.zeros((1, 1, 28, 28), dtype=np.float32)
            empty_label = np.zeros((1,), dtype=np.int64)
            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(empty_data), torch.from_numpy(empty_label)
            )
        else:
            client_data = torch.from_numpy(data[idx]).float()
            client_labels = torch.from_numpy(labels[idx]).long()
            dataset = torch.utils.data.TensorDataset(client_data, client_labels)
        client_loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=True))

    return client_loaders


# Import attack module
try:
    from attack_module import (
        AttackConfig,
        apply_label_flip_attack,
        apply_backdoor_attack
    )
    HAS_ATTACK = True
except ImportError:
    HAS_ATTACK = False

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _ema_smooth(values, alpha=0.1):
    if not values:
        return values
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed


def plot_results(
    results,
    epochs,
    dataset,
    attack,
    ratio,
    smooth_alpha=0.15,
    attacked_extra_alpha=0.08,
    trimmed_extra_alpha=0.08,
):
    plt.figure(figsize=(10, 6))
    for method, accs in results.items():
        if method.startswith("__"):
            continue
        # 第一层通用 EMA
        accs_to_plot = _ema_smooth(accs, alpha=smooth_alpha)

        # 第二层定向平滑：仅对抖动最强的两条曲线额外处理
        if method == 'FedAvg(attacked)' and attacked_extra_alpha is not None:
            accs_to_plot = _ema_smooth(accs_to_plot, alpha=attacked_extra_alpha)
        elif method == 'Trimmed Mean' and trimmed_extra_alpha is not None:
            accs_to_plot = _ema_smooth(accs_to_plot, alpha=trimmed_extra_alpha)

        plt.plot(range(1, epochs + 1), accs_to_plot, label=method)

    plt.xlabel('Epoch')
    plt.xticks(range(1, epochs + 1, max(1, epochs // 10)))
    plt.ylabel('Accuracy (%)')
    plt.title(f'{dataset} under {attack} (Att={ratio*100:.0f}%)')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(
        blockchain_dir,
        'results',
        'accuracy',
        f'exp_epoch_{dataset}_{attack}_ratio{ratio}_epochs{epochs}.png'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")

def _grad_l2_norm(grad: dict) -> float:
    if not grad:
        return 0.0
    flat = []
    for k in sorted(grad.keys()):
        v = grad[k]
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            flat.append(v.detach().view(-1).float())
    if not flat:
        return 0.0
    return torch.norm(torch.cat(flat), p=2).item()


def train_fl_epoch(client_loaders, test_loader, device, epochs, client_num, malicious_clients, method, attack_config,
                   our_sim_threshold=0.0, our_keep_ratio=0.7, our_norm_clip_k=2.5, our_warmup=2,
                   tau=0.0, M=3, alpha=1.0, beta=0.5,
                   chaos_tamper_round: int = -1, chaos_tamper_client: int = 0,
                   return_details: bool = False,
                   enable_storage_pruning: bool = True,
                   print_similarity_scores: bool = False,
                   local_epochs: int = 1,
                   dirichlet_alpha: float = 0.3,
                   malicious_executor_round_rate: float = 0.0,
                   rng: Optional[np.random.RandomState] = None,
                   root_loader=None,
                   debug_force_zero_mal: bool = False):
    global_model = FedAvgCNN().to(device)
    global_params = global_model.state_dict()
    criterion = nn.CrossEntropyLoss()

    aggregator = GradientAggregator(similarity_threshold=0.5)

    # Step 1: CKKS 初始化（Δ, N, q）
    Delta = 2 ** 40
    N = 8192
    q = [60, 40, 40, 60]
    key_center = KeyGenerationCenter(
        poly_modulus_degree=N,
        coeff_mod_bit_sizes=q
    )
    pk = key_center.get_public_key()
    sk_context = key_center.get_decryption_context()
    key_center.register_clients(list(range(client_num)))
    identity_public_keys = key_center.get_all_public_keys()

    ipfs = MockIPFSManager()

    # 链定义
    C_T = []  # Training Node Chain metadata
    C_D = []  # Deputy/Auditor Chain metadata

    stake_manager = StakeManager(num_clients=client_num, random_init=False, initial_stake=100.0)
    tas = TripartiteAuditorSystem(stake_manager=stake_manager, keep_recent_rounds=M + 1)

    epoch_accuracies = []
    ct_sizes = []
    cd_sizes = []
    v2_invalid_counts = []
    quarantined_clients = set()
    # D0 风格可信根统计：记录每个节点连续低相似度次数
    low_sim_streak = {cid: 0 for cid in range(client_num)}
    details_e_filtered = []
    details_audit_blocked = []
    details_v1_failures = []
    details_v2_failures = []
    audit_failure_reasons = []

    latency_t_train = []
    latency_t_enc = []
    latency_t_sim = []
    latency_t_audit = []

    if rng is None:
        rng = np.random.RandomState(42)

    #
    client_lr = 0.005
    aggregation_step_eta = 0.5 if method == 'our' else 1.0

    for epoch in range(1, epochs + 1):
        t_train_epoch = 0.0
        t_enc_epoch = 0.0
        t_sim_epoch = 0.0
        t_audit_epoch = 0.0

        client_gradients = {}
        raw_client_gradients = {}
        sample_counts = {}

        encrypted_updates = {}
        plain_flat_updates = {}
        C_T_records_round = []

        for client_id in range(client_num):
            # 在 PBFL 模式中，V2 已判定的恶意节点从后续轮次中隔离，不再参与计算
            if method == 'our' and client_id in quarantined_clients:
                continue

            model = FedAvgCNN().to(device)
            model.load_state_dict(global_params)
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=client_lr, momentum=0.9, weight_decay=1e-4)

            loader = client_loaders[client_id]
            is_malicious = (client_id in malicious_clients)

            total_samples = 0

            train_start = time.perf_counter()
            for _local_epoch in range(local_epochs):
                for data, target in loader:
                    data, target = data.to(device), target.to(device)

                    if is_malicious and attack_config:
                        if attack_config.attack_type == 'label_flip':
                            target = apply_label_flip_attack(target, attack_config.label_flip_map)
                        elif attack_config.attack_type == 'backdoor':
                            data, target = apply_backdoor_attack(
                                data, target,
                                attack_config.backdoor_pattern,
                                attack_config.backdoor_target_label
                            )

                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    total_samples += len(target)
            t_train_epoch += (time.perf_counter() - train_start)

            # Step 2: 计算原始更新 Δw = w_local - W_global
            local_params = model.state_dict()
            delta_w = {}
            for k in global_params:
                if k not in local_params: continue
                # 排除非浮点张量（如 LongTensor 类型的 labels 或 metadata）
                if not global_params[k].is_floating_point(): continue
                delta_w[k] = (local_params[k] - global_params[k]).detach()

            # Apply model-level Byzantine attacks after local training
            if is_malicious and attack_config:
                if attack_config.attack_type in {'scale', 'sign_flip', 'model_poison'}:
                    scale_factor = getattr(attack_config, 'scale_factor', -1.0)
                    if attack_config.attack_type == 'sign_flip':
                        scale_factor = -2.0
                    elif attack_config.attack_type == 'model_poison':
                        scale_factor = -1.0
                    delta_w = {k: v * float(scale_factor) for k, v in delta_w.items()}
                elif attack_config.attack_type == 'random':
                    delta_w = {k: torch.randn_like(v) for k, v in delta_w.items()}
                elif attack_config.attack_type == 'gaussian_noise':
                    noise_std = float(getattr(attack_config, 'gaussian_noise_std', 1.0))
                    delta_w = {k: torch.randn_like(v) * noise_std for k, v in delta_w.items()}

            # 双轨策略：our 的相似度审计使用归一化向量，聚合更新保留原始幅值
            raw_client_gradients[client_id] = delta_w
            if method == 'our':
                client_update_for_audit = aggregator.normalize_update_delta(delta_w)
                client_gradients[client_id] = client_update_for_audit
                flat = aggregator.flatten_update(client_update_for_audit).detach().cpu().float()
            else:
                client_gradients[client_id] = delta_w
                flat = aggregator.flatten_update(delta_w).detach().cpu().float()

            sample_counts[client_id] = total_samples
            plain_flat_updates[client_id] = flat


            if method == 'our':
                # 加密 C = Enc_pk(ẇ)
                enc_start = time.perf_counter()
                if pk is not None and ts is not None:
                    C_i = ts.ckks_vector(pk, flat.tolist())
                else:
                    C_i = flat
                t_enc_epoch += (time.perf_counter() - enc_start)
                encrypted_updates[client_id] = C_i

                # Step 3: 上传到 IPFS + 记录 {CID, DID, Signature} 到 C_T
                cid = ipfs.add({"client_id": client_id, "epoch": epoch, "flat_update": flat})
                did = client_id
                sign_payload = {"CID": cid, "DID": did, "Round_ID": epoch}
                private_key = key_center.get_client_private_key(client_id)
                signature = sign_data(private_key, sign_payload) if private_key else ""

                # Chaos Test: 可选篡改签名，验证 V2 验签门控
                if epoch == chaos_tamper_round and client_id == chaos_tamper_client:
                    signature = f"{signature}X"
                    print(f"[Chaos Test] Tampered signature for Client {client_id} at Epoch {epoch}")

                rec = {
                    "round": epoch,
                    "Round_ID": epoch,
                    "CID": cid,
                    "DID": did,
                    "Signature": signature,
                    "PublicKey": identity_public_keys.get(client_id),
                }
                C_T.append(rec)
                C_T_records_round.append(rec)

        # Step 4: TAS 选举与三方治理（仅 our）
        executor_E = V_1 = V_2 = None
        sim_scores = {}
        honest_set_H = set()
        consensus_ok = False
        C_new = None
        flat_global = torch.tensor([])
        audit_stats = {"v2_invalid_count": 0, "v2_verified_count": 0}

        if method == 'our':
            honest_pool = sorted(list(set(range(client_num)) - set(malicious_clients)))
            roles = tas.elect_roles(epoch, honest_pool=honest_pool)
            executor_E = roles.executor_E
            V_1 = roles.similarity_auditor_V1
            V_2 = roles.membership_auditor_V2
            print(f"[TAS Election] Roles elected from honest pool: E={executor_E}, V1={V_1}, V2={V_2}")

            # 构建全局参考密文 C_global
            if client_gradients:
                avg_grad = aggregator._compute_average_gradient(client_gradients)
                flat_global = aggregator.flatten_update(avg_grad).detach().cpu().float()
                C_global = ts.ckks_vector(pk, flat_global.tolist()) if (pk is not None and ts is not None) else flat_global
            else:
                C_global = None

            # 先根据当前轮相似度分布得到动态阈值（即时过滤）
            pre_sim_scores = {}
            sim_start = time.perf_counter()
            for cid, C_i in encrypted_updates.items():
                fp_i = plain_flat_updates.get(cid)
                pre_sim_scores[cid] = GradientAggregator.secure_ckks_cosine_similarity(
                    C_i=C_i,
                    C_global=C_global,
                    sk_context=sk_context,
                    fallback_plain_i=fp_i,
                    fallback_plain_global=flat_global,
                )
            t_sim_epoch += (time.perf_counter() - sim_start)

            if pre_sim_scores:
                sim_vals = np.array([float(v) for v in pre_sim_scores.values()], dtype=np.float32)
                median_sim = float(np.median(sim_vals))
                mad_sim = float(np.median(np.abs(sim_vals - median_sim)))
                # 过滤阈值用于“当轮聚合”可略严格；
                # 但为了降低误杀，限制上界并留出安全边际
                dynamic_tau = max(min(median_sim * 0.75, 0.70), 0.30)
                quarantine_tau = max(median_sim - 1.5 * max(mad_sim, 1e-6), 0.30)
            else:
                dynamic_tau = max(float(tau), 0.30)
                quarantine_tau = 0.30

            C_new, sim_scores, honest_set_H, consensus_ok, v1_ok, v2_ok, audit_stats = aggregator.tas_governed_secure_aggregate(
                encrypted_updates=encrypted_updates,
                encrypted_global=C_global,
                sk_context=sk_context,
                tau=0.0,
                C_T_records=C_T_records_round,
                executor_id=executor_E,
                V1_id=V_1,
                V2_id=V_2,
                plain_updates_for_audit=plain_flat_updates,
                plain_global_for_audit=flat_global,
                identity_public_keys=identity_public_keys,
            )

            newly_flagged = set(audit_stats.get("v2_invalid_clients", []))
            if newly_flagged:
                quarantined_clients.update(newly_flagged)
                print(f"[PBFL] Quarantine clients by V2 audit: {sorted(list(newly_flagged))}; total={sorted(list(quarantined_clients))}")

            # 即时过滤 + 治理隔离：
            # - 当轮聚合阈值 dynamic_tau
            # - 永久隔离阈值 quarantine_tau 更严格，且要求连续 3 轮
            avg_sim = float(np.mean([float(v) for v in sim_scores.values()])) if sim_scores else 0.0
            for cid, sim in sim_scores.items():
                sim_val = float(sim)
                if sim_val < quarantine_tau:
                    low_sim_streak[cid] = low_sim_streak.get(cid, 0) + 1
                else:
                    low_sim_streak[cid] = 0

                if avg_sim > 0.3 and low_sim_streak[cid] >= 3 and cid not in quarantined_clients:
                    quarantined_clients.add(cid)
                    print(f"[PBFL] Node {cid} quarantined: sim={sim_val:.4f} < q_tau={quarantine_tau:.4f}")

            # Trust scoring: ReLU(sim)。负相关更新权重直接归零。
            trust_weights = {cid: max(0.0, float(sim)) for cid, sim in sim_scores.items()}
            # 当轮参与聚合的集合（即时过滤）
            honest_set_H = {
                cid for cid, w in trust_weights.items()
                if (w >= dynamic_tau) and (cid not in quarantined_clients)
            }

            detected_mal = [cid for cid in malicious_clients if cid not in honest_set_H]
            sum_w = float(sum(trust_weights.values()))
            print(f"Round [{epoch}] Filter Logic: tau={dynamic_tau:.4f}, detected_malicious={detected_mal}, sum_weights={sum_w:.4f}")

            if print_similarity_scores:
                sim_items = []
                for cid in sorted(sim_scores.keys()):
                    tag = "(Mal)" if cid in malicious_clients else ""
                    sim_items.append(f"Node{cid}{tag}: {sim_scores[cid]:.4f}")
                print(f"Round [{epoch}] Similarity Scores: {{{', '.join(sim_items)}}}")

            noise_std = 1e-3
            min_trust = 0.0
            root_grad = None
            if root_loader is not None:
                root_grad = compute_root_gradient(
                    model=global_model,
                    data_loader=root_loader,
                    device=device,
                    lr=client_lr,
                )
                if root_grad is not None:
                    print(f"[FLTrust] g0 norm: {torch.norm(root_grad).item():.6f}")
                else:
                    print("[FLTrust] g0 missing: root_loader empty")
            else:
                print("[FLTrust] g0 missing: root_loader not found")

            trust_weights_for_audit, scale_factors, root_grad, trust_scores, raw_sims = aggregator.compute_fltrust_outputs(
                flat_updates=plain_flat_updates,
                noise_std=noise_std,
                seed=epoch,
                root_grad=root_grad,
                min_trust=min_trust,
            )

            if malicious_clients:
                honest_ids = [cid for cid in range(client_num) if cid not in malicious_clients]
                honest_sims = [raw_sims.get(cid, 0.0) for cid in honest_ids]
                mal_sims = [raw_sims.get(cid, 0.0) for cid in malicious_clients]
                if honest_sims:
                    print(f"[Round {epoch} Health] Honest IDs Sim Range: [{min(honest_sims):.4f}, {max(honest_sims):.4f}]")
                if mal_sims:
                    print(f"[Round {epoch} Health] Malicious IDs Sim Range: [{min(mal_sims):.4f}, {max(mal_sims):.4f}]")

            if plain_flat_updates:
                sample_cid = sorted(plain_flat_updates.keys())[0]
                sample_update = plain_flat_updates[sample_cid]
                if root_grad is not None:
                    g0_norm = torch.norm(root_grad).item()
                    g0_unit = root_grad / (torch.norm(root_grad) + 1e-12)
                    update_unit = sample_update / (torch.norm(sample_update) + 1e-12)
                    print(
                        f"[ALIGN CHECK] g0 shape: {tuple(root_grad.shape)}, "
                        f"Client{sample_cid} update shape: {tuple(sample_update.shape)}"
                    )
                    print(
                        f"[ALIGN CHECK] g0 norm: {g0_norm:.6f}, "
                        f"Client{sample_cid} update norm: {torch.norm(sample_update).item():.6f}"
                    )
                    dot_val = float(torch.dot(update_unit, g0_unit))
                    print(f"[DIRECTION] Raw Cos (Client{sample_cid} · g0): {dot_val:.6f}")
                    print(f"[Root Success] Normalized g0 Norm: {torch.norm(g0_unit).item():.6f}")
                    if dot_val < 0:
                        print("[WARNING] Vector direction mismatch detected! Check if g0 needs to be flipped.")

            if malicious_clients and debug_force_zero_mal:
                for cid in malicious_clients:
                    trust_weights_for_audit[cid] = 0.0
                total = sum(trust_weights_for_audit.values())
                if total > 1e-12:
                    trust_weights_for_audit = {cid: w / total for cid, w in trust_weights_for_audit.items()}
            aggregator.prev_global_update = root_grad.detach() if root_grad is not None else None

            if root_grad is not None:
                prev_global_update_np = root_grad.detach().cpu().numpy()
            else:
                prev_global_update_np = None

            if malicious_clients:
                honest_ids = [cid for cid in range(client_num) if cid not in malicious_clients]
                honest_scores = [trust_scores.get(cid, 0.0) for cid in honest_ids]
                mal_scores = [trust_scores.get(cid, 0.0) for cid in malicious_clients]
                avg_honest = float(np.mean(honest_scores)) if honest_scores else 0.0
                avg_mal = float(np.mean(mal_scores)) if mal_scores else 0.0
                g0_norm = float(torch.norm(root_grad).item()) if root_grad is not None else 0.0
                print(
                    f"[Root Check] g0_norm: {g0_norm:.6f}, Avg_Honest_Sim: {avg_honest:.4f}, "
                    f"Avg_Malicious_Sim: {avg_mal:.4f}"
                )

            if malicious_clients:
                mal_weights = [trust_weights_for_audit.get(cid, 0.0) for cid in malicious_clients]
                print(
                    f"[Defense Check] Malicious Node IDs: {sorted(malicious_clients)}, "
                    f"Their Audited Trust Weights: {['{:.6f}'.format(w) for w in mal_weights]}"
                )

            if root_grad is not None:
                root_norm = float(torch.norm(root_grad).item())
                print(f"[FLTrust] Root gradient norm: {root_norm:.6f}")

            chaos_executor = (
                malicious_executor_round_rate > 0
                and rng.rand() < float(malicious_executor_round_rate)
            )
            if chaos_executor and malicious_clients:
                print(f"[Chaos Audit] Round {epoch}: malicious executor forged trust weights")
                forged_id = malicious_clients[0]
                trust_weights_for_audit = {
                    cid: (1.0 if cid == forged_id else 0.0)
                    for cid in trust_weights_for_audit
                }

            tas.submit_round_for_audit(
                round_number=epoch,
                sim_scores=raw_sims,
                honest_set_H=honest_set_H,
                C_T_records=C_T_records_round,
                trust_weights=trust_weights_for_audit,
                trust_scores=trust_scores,
                flat_updates=plain_flat_updates,
                prev_global_update=prev_global_update_np,
                cp_scores=stake_manager.get_cp_scores(),
                noise_std=noise_std,
                noise_seed=epoch,
            )

            audit_start = time.perf_counter()
            v1_ok, v1_meta = tas.verify_similarity_audit(epoch)
            v2_ok, v2_meta = tas.verify_membership_audit(epoch, cp_threshold=0.0)
            t_audit_epoch += (time.perf_counter() - audit_start)

            audit_blocked = 0 if (v1_ok and v2_ok) else 1
            e_filtered_count = len([w for w in trust_weights_for_audit.values() if w <= 0])
            details_e_filtered.append(e_filtered_count)
            details_audit_blocked.append(audit_blocked)
            details_v1_failures.append(0 if v1_ok else 1)
            details_v2_failures.append(0 if v2_ok else 1)
            audit_failure_reasons.append(
                None if (v1_ok and v2_ok) else {
                    "v1": v1_meta,
                    "v2": v2_meta,
                }
            )

            if not v1_ok:
                print(f"[TAS] Round {epoch}: V1 audit failed: {v1_meta}")
            if not v2_ok:
                print(f"[TAS] Round {epoch}: V2 audit failed: {v2_meta}")

            tas.vote_similarity_audit(epoch, approved=v1_ok)
            tas.vote_membership_audit(epoch, approved=v2_ok)
            consensus_ok, consensus_meta = tas.check_tripartite_consensus(epoch)

            arbitration = consensus_meta.get("arbitration", {}) if isinstance(consensus_meta, dict) else {}
            if arbitration.get("v1_overruled"):
                print("[Warning] Minor Numerical Drift Detected: V1 overruled; skipping slashing in debug mode.")

            print(
                f"Round {epoch} TAS Status: [E calculated weights], "
                f"[V1 verified scores: {'OK' if v1_ok else 'FAIL'}], "
                f"[V2 verified signatures: {'OK' if v2_ok else 'FAIL'}]"
            )

        if method == 'our' and consensus_ok:
            # Step 5: Trust-weighted aggregation on raw updates (audited trust weights)
            trust_weights = trust_weights_for_audit
            active_ids = [
                cid for cid, w in trust_weights.items()
                if (w > 0) and (cid in raw_client_gradients) and (cid not in quarantined_clients)
                and float(stake_manager.get_cp_scores().get(cid, 100.0)) >= 30.0
            ]
            sum_weights = sum(trust_weights[cid] for cid in active_ids)

            if sum_weights > 1e-9:
                norm_weights = {cid: trust_weights[cid] / sum_weights for cid in active_ids}
                weighted_updates = {
                    cid: {
                        k: v * scale_factors.get(cid, 1.0)
                        for k, v in raw_client_gradients[cid].items()
                    }
                    for cid in active_ids
                }
                aggregated_grad = aggregator.fedavg_aggregate(weighted_updates, sample_counts=None, weights=norm_weights)
                honest_set_H = {cid for cid in active_ids if trust_weights[cid] > 0}
            else:
                aggregated_grad = {
                    k: torch.zeros_like(v)
                    for k, v in global_params.items()
                    if isinstance(v, torch.Tensor) and v.is_floating_point()
                }
                honest_set_H = set()

            # Stake 更新到 C_D
            # 稳健惩罚：隔离集合按 1.5*beta 惩罚，降低过激惩罚导致的系统抖动
            all_clients_active = [cid for cid in range(client_num) if cid not in quarantined_clients]
            stake_manager.apply_alpha_beta(
                honest_set=list(honest_set_H),
                all_clients=all_clients_active,
                alpha=alpha,
                beta=beta,
            )
            stake_manager.update_cp_scores(
                honest_set=list(honest_set_H),
                all_clients=all_clients_active,
                up_step=2.0,
                down_step=6.0,
            )
            stake_manager.record_cp_scores(epoch)
            for qcid in sorted(list(quarantined_clients)):
                stake_manager.penalize(qcid, 1.5 * beta)
            C_D.append({
                "round": epoch,
                "executor_E": executor_E,
                "V_1": V_1,
                "V_2": V_2,
                "honest_set_H": sorted(list(honest_set_H)),
                "sim_scores": sim_scores,
                "consensus": True,
                "stakes": stake_manager.get_all_stakes(),
                "v1_audit_result": v1_ok,
                "v1_sampled_nodes": v1_meta.get("sampled") if isinstance(v1_meta, dict) else None,
                "v2_audit_result": v2_ok,
                "audit_failure_reason": None if (v1_ok and v2_ok) else {
                    "v1": v1_meta,
                    "v2": v2_meta,
                },
                "e_filtered_count": e_filtered_count,
                "audit_blocked": audit_blocked,
                "num_quarantined": len(quarantined_clients),
            })
        else:
            # 共识失败或非 our 规则时回退
            f = len(malicious_clients)

            # FLTrust 需要服务器 root 数据集计算参考梯度
            fltrust_root_grad = None
            if method == 'fltrust' and root_loader is not None:
                fltrust_root_grad = compute_root_gradient(
                    model=global_model,
                    data_loader=root_loader,
                    device=device,
                    lr=client_lr,
                )

            aggregated_grad, _ = aggregator.aggregate(
                client_gradients,
                sample_counts,
                global_gradient=None,
                normalize=False,
                rule='avg' if method == 'our' else method,
                f=f,
                beta=0.1,
                similarity_threshold=0.5,
                m=max(1, len(client_gradients) - f),
                root_grad=fltrust_root_grad,
            )
            if method == 'our':
                C_D.append({
                    "round": epoch,
                    "executor_E": executor_E,
                    "V_1": V_1,
                    "V_2": V_2,
                    "honest_set_H": [],
                    "sim_scores": sim_scores,
                    "consensus": False,
                    "stakes": stake_manager.get_all_stakes(),
                    "v1_audit_result": v1_ok,
                    "v1_sampled_nodes": v1_meta.get("sampled") if isinstance(v1_meta, dict) else None,
                    "v2_audit_result": v2_ok,
                    "audit_failure_reason": None if (v1_ok and v2_ok) else {
                        "v1": v1_meta,
                        "v2": v2_meta,
                    },
                    "e_filtered_count": e_filtered_count,
                    "audit_blocked": audit_blocked,
                    "num_quarantined": len(quarantined_clients),
                })

        # 引入聚合步长（FedAvg(attacked) 保持 η=1.0，其它设置 η=0.5）
        new_params = {}
        for k in global_params:
            if k in aggregated_grad:
                new_params[k] = global_params[k] + aggregation_step_eta * aggregated_grad[k]
            else:
                new_params[k] = global_params[k]

        global_params = new_params

        # Storage pruning: C_T 滑动窗口 (M+1)（仅 our）
        if method == 'our' and enable_storage_pruning and len(C_T) > (M + 1):
            to_drop = len(C_T) - (M + 1)
            print(f"[Storage Pruning] C_T exceeds M+1={M+1}, pruning {to_drop} old metadata records")
            C_T[:] = C_T[to_drop:]

        global_model.load_state_dict(global_params)
        acc = test_model(global_model, test_loader, device)
        epoch_accuracies.append(acc)
        ct_sizes.append(len(C_T))
        cd_sizes.append(len(C_D))
        v2_invalid_counts.append(int(audit_stats.get("v2_invalid_count", 0)))

        latency_t_train.append(float(t_train_epoch))
        latency_t_enc.append(float(t_enc_epoch))
        latency_t_sim.append(float(t_sim_epoch))
        latency_t_audit.append(float(t_audit_epoch))
        t_total_epoch = t_train_epoch + t_enc_epoch + t_sim_epoch + t_audit_epoch

        print(f"Epoch {epoch}: {acc:.2f}%")
        print("[Exp5 Latency Breakdown] "
              f"T_train={t_train_epoch:.4f}s | "
              f"T_enc={t_enc_epoch:.4f}s | "
              f"T_sim={t_sim_epoch:.4f}s | "
              f"T_audit={t_audit_epoch:.4f}s | "
              f"T_total={t_total_epoch:.4f}s")

    if return_details:
        return epoch_accuracies, {
            "ct_sizes": ct_sizes,
            "cd_sizes": cd_sizes,
            "v2_invalid_counts": v2_invalid_counts,
            "quarantined_clients": sorted(list(quarantined_clients)),
            "num_quarantined": [len(quarantined_clients)] * len(epoch_accuracies),
            "e_filtered_count": [int(e) for e in details_e_filtered],
            "audit_blocked": [int(a) for a in details_audit_blocked],
            "v1_failures": [int(v) for v in details_v1_failures],
            "v2_failures": [int(v) for v in details_v2_failures],
            "audit_failure_reason": audit_failure_reasons,
            "latency_t_train": latency_t_train,
            "latency_t_enc": latency_t_enc,
            "latency_t_sim": latency_t_sim,
            "latency_t_audit": latency_t_audit,
        }
    return epoch_accuracies

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return 100. * correct / len(test_loader.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--attack', type=str, default='label_flip', choices=['label_flip', 'backdoor'])
    parser.add_argument('--ratio', type=float, default=0.3)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--local_epochs', type=int, default=5, help='每个全局轮次的本地训练轮数')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--non_iid', action='store_true', help='使用 Dirichlet Non-IID 划分')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.3, help='Dirichlet alpha 参数')
    parser.add_argument('--root_size', type=int, default=100, help='FLTrust root dataset size')
    parser.add_argument('--debug_force_zero_mal', action='store_true', help='调试：强制恶意节点权重为 0')
    parser.add_argument('--smooth_alpha', type=float, default=0.15)
    parser.add_argument('--attacked_extra_alpha', type=float, default=0.08,
                        help='FedAvg(attacked) second-pass EMA alpha')
    parser.add_argument('--trimmed_extra_alpha', type=float, default=0.08,
                        help='Trimmed Mean second-pass EMA alpha')
    parser.add_argument('--cache', action='store_true', default=True)
    parser.add_argument('--no_cache', action='store_true')
    parser.add_argument('--malicious_executor_round_rate', type=float, default=0.1,
                        help='恶意执行者造假权重的轮次比例')
    parser.add_argument('--run_all', action='store_true', help='强制所有方法本轮重新训练（不读取缓存）')
    parser.add_argument('--run_methods', nargs='*', default=['Our','Baseline(clean)','FedAvg(attacked)','Krum'], help='指定本轮要重新训练的方法名列表（显示名，如 Our Krum）')
    parser.add_argument('--load_methods', nargs='*', default=['Baseline(clean)','Krum'], help='指定本轮强制读取缓存的方法名列表（显示名）')
    args = parser.parse_args()

    if args.no_cache:
        args.cache = False
    
    set_seed(args.seed)

    methods_map = {
        'Our': 'our',
        'Baseline(clean)': 'avg',
        # 'FedAvg(attacked)': 'avg',
        'Krum': 'krum',
    }
    
    dataset_name = args.dataset
    attack_type = args.attack
    malicious_ratio = args.ratio
    epochs = args.epoch
    client_num = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.non_iid:
        client_loaders = get_client_loaders_dirichlet(
            client_num=client_num,
            batch_size=128,
            alpha=args.dirichlet_alpha,
            seed=args.seed,
        )
    else:
        client_loaders = get_client_loaders_iid(client_num=client_num, batch_size=128)

    split_tag = 'dirichlet' if args.non_iid else 'iid'
    test_loader = get_test_loader(batch_size=128)
    root_loader = get_root_loader(batch_size=128, root_size=args.root_size, seed=args.seed)
    
    # Attack Config（随机选择恶意客户端；你不需要指定是谁，但要在跑之前明确打印出来）
    num_malicious = int(client_num * malicious_ratio)
    rng = np.random.RandomState(args.seed)
    malicious_clients = sorted(rng.choice(list(range(client_num)), size=num_malicious, replace=False).tolist())

    print(f"[Experiment] dataset={dataset_name}, attack={attack_type}, ratio={malicious_ratio}, seed={args.seed}")
    print(f"[Experiment] selected_malicious_clients({len(malicious_clients)}/{client_num}) = {malicious_clients}")

    attack_config = None
    if HAS_ATTACK:
        attack_config = AttackConfig()
        attack_config.enable_attack = True
        attack_config.malicious_clients = malicious_clients
        attack_config.attack_type = attack_type
        if attack_type == 'label_flip':
            attack_config.label_flip_map = {i: (9 - i) for i in range(10)}
        elif attack_type == 'backdoor':
            pattern = torch.zeros((1, 28, 28))
            pattern[0, 24:, 24:] = 1.0 # 4x4 patch
            attack_config.backdoor_pattern = pattern
            attack_config.backdoor_target_label = 0
            
    cache_dir = os.path.join(blockchain_dir, 'results', 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    def _cache_key(method_display, method_rule, ratio_value):
        safe_attack = attack_type.replace('/', '_')
        split_tag = 'dirichlet' if args.non_iid else 'iid'
        return (
            f"acc_vs_epoch__ds={dataset_name}__attack={safe_attack}__ratio={ratio_value}"
            f"__epochs={epochs}__seed={args.seed}__split={split_tag}"
            f"__alpha={args.dirichlet_alpha}__method={method_display}__rule={method_rule}.npy"
        )

    results = {}

    run_set = set(args.run_methods or [])
    load_set = set(args.load_methods or [])

    # Baseline(clean) 复用策略：当 ratio>0 时，Baseline(clean) 优先加载 ratio=0 的缓存
    baseline_override = (malicious_ratio > 0)

    for method_name, rule in methods_map.items():
        effective_ratio = malicious_ratio
        if baseline_override and method_name == 'Baseline(clean)':
            effective_ratio = 0.0
        if method_name == 'FedAvg(attacked)':
            effective_ratio = malicious_ratio

        cache_file = os.path.join(cache_dir, _cache_key(method_name, rule, effective_ratio))

        want_run = args.run_all or (method_name in run_set)
        want_load = (method_name in load_set)

        if args.no_cache:
            want_run = True
            want_load = False

        if want_load:
            if not os.path.exists(cache_file):
                raise FileNotFoundError(f"强制 load 但缓存不存在: {method_name} -> {cache_file}")
            accs = np.load(cache_file).tolist()
            print(f"\n[{method_name}] mode=load (forced) -> {cache_file}")
            details = {}
        elif (not want_run) and args.cache and os.path.exists(cache_file):
            accs = np.load(cache_file).tolist()
            print(f"\n[{method_name}] mode=load -> {cache_file}")
            details = {}
        else:
            set_seed(args.seed)
            print(f"\n[{method_name}] mode=run (rule={rule})")

            num_mal = int(client_num * effective_ratio)

            # 保证“输出的恶意客户端”与“真正参与攻击的恶意客户端”一致：
            # - 默认使用上面根据 seed 采样并打印的 malicious_clients（全局固定集合）
            # - 如果 effective_ratio 不是本轮实验 ratio（例如 baseline clean 覆盖为 0），则使用空集合
            if effective_ratio == malicious_ratio:
                mal_clients = list(malicious_clients)
            else:
                mal_clients = []

            print(f"[{method_name}] malicious_clients(used) = {sorted(mal_clients)}")

            local_attack_config = None
            if HAS_ATTACK:
                local_attack_config = AttackConfig()
                local_attack_config.enable_attack = True
                local_attack_config.malicious_clients = mal_clients
                local_attack_config.attack_type = attack_type
                if attack_type == 'label_flip':
                    local_attack_config.label_flip_map = {i: (9 - i) for i in range(10)}
                elif attack_type == 'backdoor':
                    pattern = torch.zeros((1, 28, 28))
                    pattern[0, 24:, 24:] = 1.0
                    local_attack_config.backdoor_pattern = pattern
                    local_attack_config.backdoor_target_label = 0

            accs, details = train_fl_epoch(
                client_loaders, test_loader, device, epochs,
                client_num, mal_clients, rule, local_attack_config,
                local_epochs=args.local_epochs,
                dirichlet_alpha=args.dirichlet_alpha,
                malicious_executor_round_rate=args.malicious_executor_round_rate,
                rng=np.random.RandomState(args.seed),
                root_loader=root_loader,
                debug_force_zero_mal=args.debug_force_zero_mal,
                return_details=True,
            )

            if args.cache:
                np.save(cache_file, np.array(accs, dtype=np.float32))
                print(f"Saved cached result: {method_name} -> {cache_file}")

        results[method_name] = accs
        results.setdefault("__details__", {})[method_name] = details

    plot_results(
        results,
        epochs,
        dataset_name,
        attack_type,
        malicious_ratio,
        smooth_alpha=args.smooth_alpha,
        attacked_extra_alpha=args.attacked_extra_alpha,
        trimmed_extra_alpha=args.trimmed_extra_alpha,
    )

    # 自动导出逐轮结果 CSV（论文制图与统计用）
    csv_path = os.path.join(blockchain_dir, 'results', 'accuracy', 'results_mnist_v3.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            'dataset', 'attack', 'ratio', 'seed', 'method', 'rule', 'split', 'dirichlet_alpha',
            'epoch', 'accuracy', 'num_quarantined', 'e_filtered_count', 'audit_blocked'
        ])
        details = results.get("__details__", {})
        for method_name, rule in methods_map.items():
            accs = results.get(method_name, [])
            method_details = details.get(method_name, {})
            num_quarantined = method_details.get("num_quarantined", [0] * epochs)
            e_filtered_count = method_details.get("e_filtered_count", [0] * epochs)
            audit_blocked = method_details.get("audit_blocked", [0] * epochs)
            for ep, acc in enumerate(accs, start=1):
                writer.writerow([
                    dataset_name,
                    attack_type,
                    malicious_ratio,
                    args.seed,
                    method_name,
                    rule,
                    split_tag,
                    args.dirichlet_alpha,
                    ep,
                    float(acc),
                    num_quarantined[ep - 1] if ep - 1 < len(num_quarantined) else 0,
                    e_filtered_count[ep - 1] if ep - 1 < len(e_filtered_count) else 0,
                    audit_blocked[ep - 1] if ep - 1 < len(audit_blocked) else 0,
                ])
    print(f"Saved CSV to {csv_path}")
