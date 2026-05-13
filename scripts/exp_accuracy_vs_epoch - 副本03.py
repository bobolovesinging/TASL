
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import csv
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


def get_test_loader(batch_size=128):
    test_dir = os.path.join(MNIST_DATA_ROOT, 'test_dataset')
    data_path = os.path.join(test_dir, 'data.npy')
    label_path = os.path.join(test_dir, 'label.npy')

    if not os.path.exists(data_path) or not os.path.exists(label_path):
        raise FileNotFoundError(f"测试集不存在: {test_dir}")

    dataset = MNISTDataset(data_path, label_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


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


def plot_results(results, epochs, dataset, attack, ratio, smooth_alpha=0.15):
    plt.figure(figsize=(10, 6))
    for method, accs in results.items():
        accs_to_plot = _ema_smooth(accs, alpha=smooth_alpha)
        plt.plot(range(1, epochs + 1), accs_to_plot, label=method)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{dataset} under {attack} (Att={ratio*100:.0f}%)')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(blockchain_dir, 'results', f'exp_epoch_{dataset}_{attack}_ratio{ratio}.png')
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
                   print_similarity_scores: bool = False):
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

    #
    client_lr = 0.005
    aggregation_step_eta = 1

    for epoch in range(1, epochs + 1):
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

            # Step 2: 计算原始更新 Δw = w_local - W_global
            local_params = model.state_dict()
            delta_w = {}
            for k in global_params:
                if k not in local_params: continue
                # 排除非浮点张量（如 LongTensor 类型的 labels 或 metadata）
                if not global_params[k].is_floating_point(): continue
                delta_w[k] = (local_params[k] - global_params[k]).detach()

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
                if pk is not None and ts is not None:
                    C_i = ts.ckks_vector(pk, flat.tolist())
                else:
                    C_i = flat
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
            roles = tas.elect_roles(epoch)
            executor_E = roles.executor_E
            V_1 = roles.similarity_auditor_V1
            V_2 = roles.membership_auditor_V2

            # 构建全局参考密文 C_global
            if client_gradients:
                avg_grad = aggregator._compute_average_gradient(client_gradients)
                flat_global = aggregator.flatten_update(avg_grad).detach().cpu().float()
                C_global = ts.ckks_vector(pk, flat_global.tolist()) if (pk is not None and ts is not None) else flat_global
            else:
                C_global = None

            # 先根据当前轮相似度分布得到动态阈值（即时过滤）
            pre_sim_scores = {}
            for cid, C_i in encrypted_updates.items():
                fp_i = plain_flat_updates.get(cid)
                pre_sim_scores[cid] = GradientAggregator.secure_ckks_cosine_similarity(
                    C_i=C_i,
                    C_global=C_global,
                    sk_context=sk_context,
                    fallback_plain_i=fp_i,
                    fallback_plain_global=flat_global,
                )

            if pre_sim_scores:
                sim_vals = np.array([float(v) for v in pre_sim_scores.values()], dtype=np.float32)
                median_sim = float(np.median(sim_vals))
                mad_sim = float(np.median(np.abs(sim_vals - median_sim)))
                # 过滤阈值用于“当轮聚合”可略严格；
                # 但为了降低误杀，限制上界并留出安全边际
                dynamic_tau = max(min(median_sim * 0.75, 0.70), 0.45)
                quarantine_tau = max(median_sim - 1.5 * max(mad_sim, 1e-6), 0.30)
            else:
                dynamic_tau = max(float(tau), 0.45)
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
            for cid, sim in sim_scores.items():
                sim_val = float(sim)
                if sim_val < quarantine_tau:
                    low_sim_streak[cid] = low_sim_streak.get(cid, 0) + 1
                else:
                    low_sim_streak[cid] = 0

                if low_sim_streak[cid] >= 3 and cid not in quarantined_clients:
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

            tas.submit_round_for_audit(
                round_number=epoch,
                sim_scores=sim_scores,
                honest_set_H=honest_set_H,
                C_T_records=C_T_records_round,
            )
            tas.vote_similarity_audit(epoch, approved=v1_ok)
            tas.vote_membership_audit(epoch, approved=v2_ok)
            consensus_ok, _ = tas.check_tripartite_consensus(epoch)

        if method == 'our' and consensus_ok:
            # Step 5: Trust-weighted aggregation on raw updates（ReLU(sim) 加权）
            trust_weights = {cid: max(0.0, float(sim)) for cid, sim in sim_scores.items()}
            active_ids = [cid for cid in trust_weights if cid in raw_client_gradients and cid not in quarantined_clients]
            sum_weights = sum(trust_weights[cid] for cid in active_ids)

            if sum_weights > 1e-9:
                norm_weights = {cid: trust_weights[cid] / sum_weights for cid in active_ids}
                weighted_updates = {cid: raw_client_gradients[cid] for cid in active_ids}
                aggregated_grad = aggregator.fedavg_aggregate(weighted_updates, sample_counts=None, weights=norm_weights)
                honest_set_H = {cid for cid in active_ids if trust_weights[cid] >= max(dynamic_tau, 0.5)}
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
            })
        else:
            # 共识失败或非 our 规则时回退
            f = len(malicious_clients)
            aggregated_grad, _ = aggregator.aggregate(
                client_gradients,
                sample_counts,
                global_gradient=None,
                normalize=False,
                rule='avg' if method == 'our' else method,
                f=f,
                beta=0.1,
                similarity_threshold=0.5,
                m=max(1, len(client_gradients) - f)
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
        print(f"Epoch {epoch}: {acc:.2f}%")

    if return_details:
        return {
            "accuracies": epoch_accuracies,
            "ct_sizes": ct_sizes,
            "cd_sizes": cd_sizes,
            "v2_invalid_counts": v2_invalid_counts,
            "quarantined_clients": sorted(list(quarantined_clients)),
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
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--smooth_alpha', type=float, default=0.15)
    parser.add_argument('--cache', action='store_true', default=True)
    parser.add_argument('--no_cache', action='store_true')
    parser.add_argument('--run_all', action='store_true', help='强制所有方法本轮重新训练（不读取缓存）')
    parser.add_argument('--run_methods', nargs='*', default=['Our','Baseline(clean)','FedAvg(attacked)','Bulyan','Krum','Trimmed Mean'], help='指定本轮要重新训练的方法名列表（显示名，如 Our Krum）')
    parser.add_argument('--load_methods', nargs='*', default=['Baseline(clean)','FedAvg(attacked)','Krum','Trimmed Mean'], help='指定本轮强制读取缓存的方法名列表（显示名）')
    args = parser.parse_args()

    if args.no_cache:
        args.cache = False
    
    set_seed(args.seed)

    methods_map = {
        'Our': 'our',
        'Baseline(clean)': 'avg',
        'FedAvg(attacked)': 'avg',
        'Krum': 'krum',
        'Trimmed Mean': 'trimmed_mean'
    }
    
    dataset_name = args.dataset
    attack_type = args.attack
    malicious_ratio = args.ratio
    epochs = args.epoch
    client_num = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client_loaders = get_client_loaders_iid(client_num=client_num, batch_size=128)
    test_loader = get_test_loader(batch_size=128)
    
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
        return f"acc_vs_epoch__ds={dataset_name}__attack={safe_attack}__ratio={ratio_value}__epochs={epochs}__seed={args.seed}__method={method_display}__rule={method_rule}.npy"

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
        elif (not want_run) and args.cache and os.path.exists(cache_file):
            accs = np.load(cache_file).tolist()
            print(f"\n[{method_name}] mode=load -> {cache_file}")
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

            accs = train_fl_epoch(
                client_loaders, test_loader, device, epochs,
                client_num, mal_clients, rule, local_attack_config
            )

            if args.cache:
                np.save(cache_file, np.array(accs, dtype=np.float32))
                print(f"Saved cached result: {method_name} -> {cache_file}")

        results[method_name] = accs

    plot_results(results, epochs, dataset_name, attack_type, malicious_ratio, smooth_alpha=args.smooth_alpha)

    # 自动导出逐轮结果 CSV（论文制图与统计用）
    csv_path = os.path.join(blockchain_dir, 'results', 'results_mnist_v2.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'attack', 'ratio', 'seed', 'method', 'rule', 'epoch', 'accuracy'])
        for method_name, rule in methods_map.items():
            accs = results.get(method_name, [])
            for ep, acc in enumerate(accs, start=1):
                writer.writerow([
                    dataset_name,
                    attack_type,
                    malicious_ratio,
                    args.seed,
                    method_name,
                    rule,
                    ep,
                    float(acc),
                ])
    print(f"Saved CSV to {csv_path}")
