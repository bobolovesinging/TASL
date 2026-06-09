"""
梯度聚合模块（FedSGD）
实现归一化、余弦相似度剔除恶意梯度、梯度聚合
用于FedSGD算法：客户端计算梯度，服务器聚合梯度
"""
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple
import copy

try:
    from .crypto_utils import verify_signature
except ImportError:
    from crypto_utils import verify_signature

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    ts = None
    TENSEAL_AVAILABLE = False


class GradientAggregator:
    """梯度聚合器：实现归一化、相似度过滤、梯度聚合（用于FedSGD）"""
    
    def __init__(self, similarity_threshold: float = 0.3):
        """
        初始化梯度聚合器
        :param similarity_threshold: 余弦相似度阈值，低于此值的梯度将被剔除
        """
        self.similarity_threshold = similarity_threshold
        self.prev_global_update: Optional[torch.Tensor] = None
        self.last_anchor_conflict_sim: Optional[float] = None
        self.last_anchor_conflict_sim: Optional[float] = None
    
    @staticmethod
    def normalize_gradient(gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        归一化梯度（L2归一化）
        :param gradient: 梯度字典
        :return: 归一化后的梯度字典
        """
        normalized = {}
        for name, param in gradient.items():
            if isinstance(param, torch.Tensor):
                # L2归一化
                norm = torch.norm(param)
                if norm > 0:
                    normalized[name] = param / norm
                else:
                    normalized[name] = param
            else:
                normalized[name] = param
        return normalized
    
    @staticmethod
    def normalize_gradients(gradients: Dict[int, Dict[str, torch.Tensor]]) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        批量归一化多个梯度
        :param gradients: {client_id: gradient_dict}
        :return: 归一化后的梯度字典
        """
        normalized = {}
        for client_id, grad in gradients.items():
            normalized[client_id] = GradientAggregator.normalize_gradient(grad)
        return normalized
    
    @staticmethod
    def calculate_cosine_similarity(grad1: Dict[str, torch.Tensor], 
                                   grad2: Dict[str, torch.Tensor]) -> float:
        """
        计算两个梯度的余弦相似度
        :param grad1: 第一个梯度
        :param grad2: 第二个梯度
        :return: 余弦相似度（-1到1之间）
        """
        # 展平梯度
        flat1 = GradientAggregator._flatten_gradient(grad1)
        flat2 = GradientAggregator._flatten_gradient(grad2)
        
        if flat1.numel() == 0 or flat2.numel() == 0:
            return 0.0
        
        # 确保形状一致
        min_len = min(flat1.numel(), flat2.numel())
        flat1 = flat1[:min_len].float()
        flat2 = flat2[:min_len].float()
        
        # 计算余弦相似度
        dot_product = torch.dot(flat1, flat2)
        norm1 = torch.norm(flat1)
        norm2 = torch.norm(flat2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = (dot_product / (norm1 * norm2)).item()
        return max(-1.0, min(1.0, similarity))
    
    @staticmethod
    def _flatten_gradient(gradient: Dict[str, torch.Tensor]) -> torch.Tensor:
        """展平梯度字典为一维张量"""
        flat_list = []
        for name in sorted(gradient.keys()):
            param = gradient[name]
            if isinstance(param, torch.Tensor) and param.is_floating_point():
                flat_list.append(param.detach().view(-1).float())
            elif isinstance(param, torch.Tensor):
                flat_list.append(param.detach().view(-1).float())
            else:
                flat_list.append(torch.tensor([float(param)], dtype=torch.float32))
        
        if not flat_list:
            return torch.tensor([], dtype=torch.float32)
        
        return torch.cat(flat_list)
    
    def filter_by_similarity(self, 
                            client_gradients: Dict[int, Dict[str, torch.Tensor]],
                            global_gradient: Optional[Dict[str, torch.Tensor]] = None,
                            similarity_threshold: Optional[float] = None) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, float]]:
        """
        根据余弦相似度过滤恶意梯度
        :param client_gradients: {client_id: gradient_dict}
        :param global_gradient: 全局梯度（如果为None，计算平均梯度）
        :param similarity_threshold: 相似度阈值（如果为None，使用self.similarity_threshold）
        :return: (过滤后的梯度字典, {client_id: similarity})
        """
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
        
        if not client_gradients:
            return {}, {}
        
        # 如果没有提供全局梯度，计算平均梯度
        if global_gradient is None:
            global_gradient = self._compute_average_gradient(client_gradients)
        
        # 计算每个客户端梯度与全局梯度的相似度
        similarities = {}
        for client_id, client_grad in client_gradients.items():
            similarity = self.calculate_cosine_similarity(client_grad, global_gradient)
            similarities[client_id] = similarity
        
        # 过滤：只保留相似度 >= threshold 的梯度
        filtered_gradients = {
            client_id: grad 
            for client_id, grad in client_gradients.items()
            if similarities[client_id] >= similarity_threshold
        }
        
        print(f"[GradientAggregator] 相似度过滤: {len(client_gradients)} -> {len(filtered_gradients)} 个梯度")
        print(f"[GradientAggregator] 被过滤的客户端: {[cid for cid in client_gradients.keys() if cid not in filtered_gradients]}")
        
        return filtered_gradients, similarities
    
    @staticmethod
    def _compute_average_gradient(gradients: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """计算平均梯度"""
        if not gradients:
            return {}
        
        avg_gradient = {}
        first_client_id = list(gradients.keys())[0]
        first_grad = gradients[first_client_id]
        
        for name in first_grad.keys():
            if isinstance(first_grad[name], torch.Tensor):
                # 堆叠所有客户端的该参数
                stacked = torch.stack([
                    gradients[cid][name] 
                    for cid in gradients 
                    if name in gradients[cid]
                ])
                avg_gradient[name] = torch.mean(stacked, dim=0)
            else:
                # 非tensor类型，计算平均值
                values = [gradients[cid][name] for cid in gradients if name in gradients[cid]]
                if values:
                    avg_gradient[name] = sum(values) / len(values)
        
        return avg_gradient
    
    @staticmethod
    def fedavg_aggregate(gradients: Dict[int, Dict[str, torch.Tensor]],
                        sample_counts: Optional[Dict[int, int]] = None,
                        weights: Optional[Dict[int, float]] = None) -> Dict[str, torch.Tensor]:
        """
        梯度聚合算法（加权平均）
        用于FedSGD：聚合客户端梯度，返回加权平均梯度
        :param gradients: {client_id: gradient_dict}
        :param sample_counts: {client_id: sample_count} 每个客户端的样本数量（用于加权）
        :param weights: {client_id: weight} 自定义权重（如果提供，优先使用）
        :return: 聚合后的梯度字典
        """
        if not gradients:
            raise ValueError("没有梯度可聚合")
        
        # 计算权重
        if weights is None:
            if sample_counts is None:
                # 如果没有样本数量，使用平均权重
                weights = {cid: 1.0 / len(gradients) for cid in gradients.keys()}
            else:
                # 根据样本数量计算权重
                total_samples = sum(sample_counts.get(cid, 0) for cid in gradients.keys())
                if total_samples > 0:
                    weights = {cid: sample_counts.get(cid, 0) / total_samples for cid in gradients.keys()}
                else:
                    weights = {cid: 1.0 / len(gradients) for cid in gradients.keys()}
        
        # 初始化聚合梯度
        aggregated = {}
        first_client_id = list(gradients.keys())[0]
        first_grad = gradients[first_client_id]
        
        # 需要特殊处理的参数名（通常是整数类型，不进行加权平均）
        integer_params = ['num_batches_tracked']  # BatchNorm 的批次计数
        
        for name in first_grad.keys():
            if isinstance(first_grad[name], torch.Tensor):
                # 检查是否是整数类型参数
                if name in integer_params or first_grad[name].dtype in [torch.int64, torch.int32, torch.long]:
                    # 对于整数类型参数，直接使用第一个客户端的值（不进行聚合）
                    aggregated[name] = first_grad[name].clone()
                else:
                    # 对于浮点类型参数，初始化为零并准备聚合
                    aggregated[name] = torch.zeros_like(first_grad[name], dtype=torch.float32)
            else:
                # 非 Tensor 类型，检查是否是整数
                if isinstance(first_grad[name], (int, np.integer)):
                    aggregated[name] = first_grad[name]  # 直接使用第一个客户端的值
                else:
                    aggregated[name] = 0.0
        
        # 加权聚合
        for client_id, grad in gradients.items():
            weight = weights.get(client_id, 0.0)
            for name in aggregated.keys():
                if name in grad:
                    if isinstance(grad[name], torch.Tensor):
                        # 跳过整数类型参数的聚合
                        if name in integer_params or grad[name].dtype in [torch.int64, torch.int32, torch.long]:
                            continue  # 已经使用第一个客户端的值
                        # 确保类型匹配
                        if aggregated[name].dtype != grad[name].dtype:
                            grad_value = grad[name].float()
                        else:
                            grad_value = grad[name]
                        aggregated[name] += weight * grad_value
                    else:
                        # 非 Tensor 类型
                        if isinstance(grad[name], (int, np.integer)):
                            continue  # 跳过整数类型
                        aggregated[name] += weight * grad[name]
        
        return aggregated
    
    @staticmethod
    def _compute_pairwise_distances(gradients: Dict[int, torch.Tensor]) -> Dict[int, Dict[int, float]]:
        """计算梯度间的成对欧氏距离"""
        ids = list(gradients.keys())
        distances = {i: {} for i in ids}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_i, id_j = ids[i], ids[j]
                # dist = torch.norm(gradients[id_i] - gradients[id_j]).item()
                # 优化: 避免重复计算
                dist = torch.norm(gradients[id_i] - gradients[id_j]).item()
                distances[id_i][id_j] = dist
                distances[id_j][id_i] = dist
            distances[ids[i]][ids[i]] = 0.0
        return distances

    @staticmethod
    def krum(gradients: Dict[int, Dict[str, torch.Tensor]], f: int) -> Dict[str, torch.Tensor]:
        """
        Krum 聚合算法
        选择一个梯度，使得它与其他 n-f-2 个最近邻居的距离之和最小
        :param gradients: 客户端梯度字典
        :param f: 恶意客户端数量的估计上限
        """
        if not gradients:
            return {}
        
        # 1. 展平所有梯度
        flat_grads = {cid: GradientAggregator._flatten_gradient(grad) for cid, grad in gradients.items()}
        ids = list(flat_grads.keys())
        n = len(ids)
        
        if n <= 2 * f + 2:
            # Krum 要求 n >= 2f + 3, 如果不满足，回退到平均
            # print(f"[Krum] Warning: n={n} is not enough for f={f} (req n >= 2f+3). Fallback to Average.")
            # 为了实验能跑通，这里我们可以稍微放宽，或者直接取平均
            # 但严格来说 Krum 需要 n - f - 2 > 0
            k = max(1, n - f - 2)
        else:
            k = n - f - 2

        # 2. 计算成对距离
        distances = GradientAggregator._compute_pairwise_distances(flat_grads)
        
        # 3. 计算每个梯度的得分 (距离最近的 k 个邻居的距离和)
        scores = {}
        for i in ids:
            dists = sorted([distances[i][j] for j in ids if i != j])
            # 取最近的 k 个
            scores[i] = sum(dists[:k])
            
        # 4. 选择得分最小的梯度
        best_id = min(scores, key=scores.get)
        print(f"[Krum] Selected client {best_id} with score {scores[best_id]:.6f}")
        
        return gradients[best_id]

    @staticmethod
    def multi_krum(gradients: Dict[int, Dict[str, torch.Tensor]], f: int, m: int) -> Dict[str, torch.Tensor]:
        """
        Multi-Krum 聚合算法
        选择 m 个最佳梯度，然后取平均
        """
        if not gradients:
            return {}
            
        flat_grads = {cid: GradientAggregator._flatten_gradient(grad) for cid, grad in gradients.items()}
        ids = list(flat_grads.keys())
        n = len(ids)
        k = max(1, n - f - 2)
        
        distances = GradientAggregator._compute_pairwise_distances(flat_grads)
        
        scores = {}
        for i in ids:
            dists = sorted([distances[i][j] for j in ids if i != j])
            scores[i] = sum(dists[:k])
            
        # 选择得分最小的 m 个
        best_ids = sorted(scores, key=scores.get)[:m]
        # print(f"[Multi-Krum] Selected clients {best_ids}")
        
        # 聚合这 m 个梯度
        selected_grads = {cid: gradients[cid] for cid in best_ids}
        return GradientAggregator._compute_average_gradient(selected_grads)

    @staticmethod
    def bulyan(gradients: Dict[int, Dict[str, torch.Tensor]], f: int) -> Dict[str, torch.Tensor]:
        """
        Bulyan 聚合算法
        先用 Multi-Krum 选择 n - 2f 个梯度，再对这些梯度用 Trimmed Mean
        要求 n >= 4f + 3
        """
        n = len(gradients)
        if n <= 4 * f + 3:
            # Fallback
            # print(f"[Bulyan] Warning: n={n} too small for f={f}. Fallback to Krum.")
            return GradientAggregator.krum(gradients, f)
            
        # 1. Multi-Krum selection
        # Select n - 2f gradients
        m = n - 2 * f
        flat_grads = {cid: GradientAggregator._flatten_gradient(grad) for cid, grad in gradients.items()}
        ids = list(flat_grads.keys())
        k = n - f - 2
        
        distances = GradientAggregator._compute_pairwise_distances(flat_grads)
        scores = {}
        for i in ids:
            dists = sorted([distances[i][j] for j in ids if i != j])
            scores[i] = sum(dists[:k])
            
        best_ids = sorted(scores, key=scores.get)[:m]
        selected_grads = {cid: gradients[cid] for cid in best_ids}
        
        # 2. Trimmed Mean on selected gradients
        # Use beta equivalent to remove f more from the selected m
        # In Bulyan, usually we use Trimmed Mean on the selected set
        # Bulyan paper says: apply dimension-wise trimmed mean to the selected set
        # removing 2*f values is not possible if m = n - 2f. 
        # Actually Bulyan usually removes theta neighbors?
        # Simplified Bulyan: Apply Trimmed Mean with beta = f/m (approx) or just average if m is small
        
        return GradientAggregator.trimmed_mean(selected_grads, beta= 1.0 / m ) # Weak trim if m is large

    @staticmethod
    def trimmed_mean(gradients: Dict[int, Dict[str, torch.Tensor]], beta: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Trimmed Mean 聚合
        对每个维度，去掉最大和最小的 beta * n 个值，然后求平均
        """
        if not gradients:
            return {}
            
        first_grad = list(gradients.values())[0]
        n_clients = len(gradients)
        # trim_count = int(n_clients * beta)
        # 确保至少保留一个
        trim_count = max(0, min(int(n_clients * beta), int((n_clients - 1) / 2)))
        
        aggregated = {}
        
        for name in first_grad.keys():
            if isinstance(first_grad[name], torch.Tensor):
                # Stack all
                stacked = torch.stack([g[name] for g in gradients.values()])
                # Sort along client dimension (dim=0)
                sorted_vals, _ = torch.sort(stacked, dim=0)
                
                if trim_count > 0:
                    # Trim
                    trimmed = sorted_vals[trim_count : n_clients - trim_count]
                else:
                    trimmed = sorted_vals
                    
                aggregated[name] = torch.mean(trimmed, dim=0)
            else:
                 # fallback for non-tensor
                 aggregated[name] = first_grad[name]
                 
        return aggregated

    @staticmethod
    def median(gradients: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Coordinate-wise Median 聚合
        """
        if not gradients:
            return {}
            
        first_grad = list(gradients.values())[0]
        aggregated = {}
        
        for name in first_grad.keys():
            if isinstance(first_grad[name], torch.Tensor):
                stacked = torch.stack([g[name] for g in gradients.values()])
                # Median
                aggregated[name] = torch.median(stacked, dim=0).values
            else:
                aggregated[name] = first_grad[name]
                
        return aggregated

    def compute_v_space(self, flat_updates: Dict[int, torch.Tensor]) -> Optional[torch.Tensor]:
        if not flat_updates:
            return None
        stacked = torch.stack([u.float() for u in flat_updates.values()])
        return torch.median(stacked, dim=0).values

    def build_anchor_vector(
        self,
        prev_global_update: Optional[torch.Tensor],
        v_space: Optional[torch.Tensor],
        momentum_weight: float = 0.5,
        space_weight: float = 0.5,
        conflict_threshold: float = 0.0,
    ) -> Optional[torch.Tensor]:
        if prev_global_update is None and v_space is None:
            self.last_anchor_conflict_sim = None
            return None
        if prev_global_update is None:
            self.last_anchor_conflict_sim = None
            return v_space
        if v_space is None:
            self.last_anchor_conflict_sim = None
            return prev_global_update

        eps = 1e-12
        sim = float(
            torch.dot(prev_global_update, v_space)
            / ((torch.norm(prev_global_update) + eps) * (torch.norm(v_space) + eps))
        )
        self.last_anchor_conflict_sim = sim

        if sim < conflict_threshold:
            return v_space
        return momentum_weight * prev_global_update + space_weight * v_space

    def apply_foolsgold_penalty(
        self,
        flat_updates: Dict[int, torch.Tensor],
        trust_weights: Dict[int, float],
        anchor: torch.Tensor,
        eps: float = 1e-12,
        collusion_threshold: float = 0.98,
    ) -> Dict[int, float]:
        if not flat_updates or not trust_weights:
            return trust_weights

        ids = list(flat_updates.keys())
        updates = {cid: (flat_updates[cid] / (torch.norm(flat_updates[cid]) + eps)) for cid in ids}
        max_pair_sim = {cid: 0.0 for cid in ids}

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sim = float(torch.dot(updates[ids[i]], updates[ids[j]]))
                max_pair_sim[ids[i]] = max(max_pair_sim[ids[i]], sim)
                max_pair_sim[ids[j]] = max(max_pair_sim[ids[j]], sim)

        anchor_norm = anchor / (torch.norm(anchor) + eps)
        for cid in ids:
            anchor_sim = float(torch.dot(updates[cid], anchor_norm))
            if max_pair_sim[cid] >= collusion_threshold and anchor_sim < 0.5:
                trust_weights[cid] *= 0.1

        return trust_weights

    @staticmethod
    def compute_fltrust_outputs(
        flat_updates: Dict[int, torch.Tensor],
        noise_std: float = 1e-3,
        seed: Optional[int] = None,
        root_grad: Optional[torch.Tensor] = None,
        min_trust: float = 0.2,
    ) -> Tuple[Dict[int, float], Dict[int, float], Optional[torch.Tensor], Dict[int, float], Dict[int, float]]:
        if not flat_updates:
            return {}, {}, None, {}

        device = next(iter(flat_updates.values())).device

        if root_grad is None:
            stacked = torch.stack([u.to(device).float() for u in flat_updates.values()])
            root_grad = torch.mean(stacked, dim=0)

            if noise_std > 0:
                if seed is not None:
                    torch.manual_seed(seed)
                root_grad = root_grad + torch.randn_like(root_grad) * noise_std
        else:
            root_grad = root_grad.to(device)

        raw_sims = {}
        trust_scores = {}
        scale_factors = {}

        update_norms = [torch.norm(u).item() for u in flat_updates.values()]
        mean_update_norm = float(np.mean(update_norms)) if update_norms else 0.0

        for cid, update in flat_updates.items():
            root_on_device = root_grad.to(update.device)
            root_norm = torch.norm(root_on_device).item()
            update_norm = torch.norm(update) + 1e-12

            # keep cosine similarity for trust score
            root_unit = root_on_device / (root_norm + 1e-12)
            update_unit = update / update_norm
            sim = torch.dot(update_unit, root_unit)
            raw_sims[cid] = float(sim)
            score = max(0.0, float(sim))
            trust_scores[cid] = 0.0 if score < min_trust else score

            # scale by effective norm to avoid vanishing g0
            effective_root_norm = max(root_norm, mean_update_norm, 1e-12)
            scale_factors[cid] = float(effective_root_norm / update_norm)

            if cid == sorted(flat_updates.keys())[0]:
                print(
                    f"[FLTrust] root_norm={root_norm:.6f}, mean_update_norm={mean_update_norm:.6f}, "
                    f"effective_root_norm={effective_root_norm:.6f}"
                )

        sum_w = sum(trust_scores.values())
        if sum_w > 1e-12:
            trust_weights = {cid: w / sum_w for cid, w in trust_scores.items()}
        else:
            trust_weights = {cid: 0.0 for cid in trust_scores}

        return trust_weights, scale_factors, root_grad, trust_scores, raw_sims

    def compute_trust_weights(
        self,
        flat_updates: Dict[int, torch.Tensor],
        prev_global_update: Optional[torch.Tensor],
        cp_scores: Optional[Dict[int, float]] = None,
        trust_power: float = 3.0,
        momentum_weight: float = 0.5,
        space_weight: float = 0.5,
        conflict_threshold: float = 0.0,
        hard_threshold: float = 0.0,
    ) -> Dict[int, float]:
        """
        Spatial-Temporal Dual-Anchor + Cubic Trust Scoring.

        流程:
          1. 空间校准: 计算本轮所有更新的 coordinate-wise median → v_space
          2. 混合锚点: anchor = 0.5 * prev_global + 0.5 * v_space
             （如二者方向冲突，则以 v_space 为主）
          3. 以 anchor 为参考，对每个更新计算余弦相似度
          4. Cubic Trust Score: ω_j = max(0, cos_j)^3
          5. 归一化权重
        """
        eps = 1e-12
        ids = list(flat_updates.keys())
        if not ids:
            return {}

        # ── Step 1: 空间中位数锚点 ──────────────────────────────────────────
        v_space = self.compute_v_space(flat_updates)

        # ── Step 2: 混合锚点 ─────────────────────────────────────────────────
        anchor = self.build_anchor_vector(
            prev_global_update=prev_global_update,
            v_space=v_space,
            momentum_weight=momentum_weight,
            space_weight=space_weight,
            conflict_threshold=conflict_threshold,
        )

        # 若无锚点（第一轮且 v_space 也为 None），退化为均等权重
        if anchor is None:
            n = len(ids)
            return {cid: 1.0 / n for cid in ids}

        anchor_norm = anchor / (torch.norm(anchor) + eps)

        # ── Step 3 & 4: Cubic Trust Score ────────────────────────────────────
        raw_scores: Dict[int, float] = {}
        for cid, update in flat_updates.items():
            u = update.float()
            u_norm = u / (torch.norm(u) + eps)
            cos_sim = float(torch.dot(u_norm, anchor_norm))
            # Cubic gating: 正方向才有权重，三次幂产生非线性门控
            raw_scores[cid] = max(0.0, cos_sim) ** trust_power

        # ── Step 5: 归一化 ───────────────────────────────────────────────────
        sum_w = sum(raw_scores.values())
        if sum_w > eps:
            trust_weights = {cid: w / sum_w for cid, w in raw_scores.items()}
        else:
            # 全部得分为 0（极端攻击），退化为均等权重（保证训练不崩溃）
            n = len(ids)
            trust_weights = {cid: 1.0 / n for cid in ids}

        # ── 更新历史动量 ──────────────────────────────────────────────────────
        aggregated_flat = sum(
            trust_weights[cid] * flat_updates[cid].float() for cid in ids
        )
        self.prev_global_update = aggregated_flat.detach()
        self.last_anchor_conflict_sim = getattr(self, 'last_anchor_conflict_sim', None)

        return trust_weights

    @staticmethod
    def compute_tasl_trust_weights(
        flat_updates: Dict[int, torch.Tensor],
        trust_power: float = 5.0,
        max_weight_ratio: float = 1.5,
        min_cos_threshold: float = 0.2,
        norm_penalty_strength: float = 0.8,
        ema_weights: Optional[Dict[int, float]] = None,
        ema_alpha: float = 0.7,
        cos_history: Optional[List[Dict[int, float]]] = None,
        prev_anchor: Optional[torch.Tensor] = None,
        temporal_anchor: Optional[torch.Tensor] = None,
        data_trust: Optional[Dict[int, float]] = None,
    ) -> Tuple[Dict[int, float], Dict[int, float], torch.Tensor]:
        """
        TASL Full Gradient-Layer Trust Scoring.

        Pipeline:
          1. Spatial Median Anchor → element-wise median
          2. Norm Gate → zero weights for |L2| > 2.5×median_norm
          3. Pairwise Consensus → median inter-client similarity, filter low-consensus
          4. Anchor Blending → 0.6×refined + 0.4×prev_anchor
          4.5 Temporal Anchor Blending → 0.6×anchor + 0.4×temporal_anchor (if dir_agree > 0.3)
          5. Cubic Trust → max(0, cos)^power × norm_penalty × consensus
          6. IQR Adaptive cos Threshold → dynamic via Q1 - IQR
          7. Temporal Penalty → 2 consecutive rounds cos<0.3 → ×0.1
          8. Optional: data_trust multiplicative fusion
          9. Weight Cap → 1.5/N
          10. EMA Smoothing → α=0.7

        Returns:
          trust_weights, cos_sims, new_anchor
        """
        eps = 1e-12
        ids = list(flat_updates.keys())
        n = len(ids)
        if n == 0:
            return {}, {}, None

        # Convert torch tensors to numpy for flexible ops
        np_updates = {cid: v.detach().cpu().numpy() for cid, v in flat_updates.items()}

        # ── 1. Norm Gate ──
        norms = {cid: float(np.linalg.norm(v)) for cid, v in np_updates.items()}
        median_norm = float(np.median(list(norms.values())))
        norm_gate = median_norm * 2.5

        # ── 2. Spatial Median Anchor ──
        stacked = np.stack(list(np_updates.values()), axis=0)
        anchor = np.median(stacked, axis=0)
        anchor_norm = float(np.linalg.norm(anchor)) + eps

        # ── 3. Pairwise Consensus ──
        pairwise_medians = {}
        for i in ids:
            vi = np_updates[i]; vin = float(np.linalg.norm(vi)) + eps
            sims = [float(np.dot(vi, np_updates[j]) / (vin * (float(np.linalg.norm(np_updates[j])) + eps)))
                    for j in ids if j != i]
            pairwise_medians[i] = float(np.median(sims)) if sims else 0.5

        med_c = float(np.median(list(pairwise_medians.values())))
        high_ids = [cid for cid in ids if pairwise_medians[cid] >= med_c]
        if len(high_ids) >= max(2, n // 2):
            cons_stacked = np.stack([np_updates[cid] for cid in high_ids])
            anchor = 0.6 * anchor + 0.4 * np.median(cons_stacked, axis=0)
            anchor_norm = float(np.linalg.norm(anchor)) + eps

        # ── 4. Refined Anchor (blending) ──
        fps = {cid: float(np.dot(np_updates[cid], anchor) /
                         (float(np.linalg.norm(np_updates[cid])) + eps) / anchor_norm)
               for cid in ids}
        tids = [cid for cid in ids if fps[cid] > 0]
        if len(tids) >= max(2, n // 2):
            ref = np.median(np.stack([np_updates[c] for c in tids]), axis=0)
            prev = prev_anchor.detach().cpu().numpy() if prev_anchor is not None else ref
            anchor = 0.6 * ref + 0.4 * prev
            anchor_norm = float(np.linalg.norm(anchor)) + eps

        # ── 4.5 Temporal Anchor Blending (v3) ──
        if temporal_anchor is not None:
            temp_np = temporal_anchor.detach().cpu().numpy()
            temp_norm = float(np.linalg.norm(temp_np)) + eps
            dir_agree = float(np.dot(anchor, temp_np) / (anchor_norm * temp_norm))
            if dir_agree > 0.3:
                anchor = 0.6 * anchor + 0.4 * temp_np
                anchor_norm = float(np.linalg.norm(anchor)) + eps

        # ── 5. Cosine Similarities & IQR Threshold ──
        cos_sims = {cid: float(np.dot(v, anchor) / (float(np.linalg.norm(v)) + eps) / anchor_norm)
                    for cid, v in np_updates.items()}
        cv = sorted(cos_sims.values())
        if len(cv) >= 4:
            q1, q3 = float(np.percentile(cv, 25)), float(np.percentile(cv, 75))
            iqr = q3 - q1
            adaptive_thr = min(0.4, max(min_cos_threshold, q1 - 1.0 * iqr))
        else:
            adaptive_thr = min_cos_threshold

        # ── 6. Raw Trust Scores ──
        rs = {}
        for cid in ids:
            cos = cos_sims[cid]
            if norms[cid] > norm_gate or cos < adaptive_thr:
                rs[cid] = 0.0
                continue
            nr = norms[cid] / (median_norm + eps)
            np_factor = float(np.exp(-norm_penalty_strength * abs(nr - 1)))
            con = max(0.01, pairwise_medians.get(cid, 0.5))
            rs[cid] = (cos ** trust_power) * np_factor * con

        # ── 7. Temporal Penalty ──
        if cos_history and len(cos_history) >= 2:
            for cid in ids:
                if all(ch.get(cid, 0.5) < 0.3 for ch in cos_history[-2:]):
                    rs[cid] *= 0.1

        # ── 8. Data Trust Fusion (multiplicative) ──
        if data_trust:
            for cid in ids:
                rs[cid] *= data_trust.get(cid, 1.0)

        # ── 9. Normalize ──
        sw = sum(rs.values())
        if sw > eps:
            tw = {cid: w / sw for cid, w in rs.items()}
        else:
            tw = {cid: 1.0 / n for cid in ids}

        # ── 10. Weight Cap ──
        mw = max_weight_ratio / n
        capped = {c: min(w, mw) for c, w in tw.items()}
        cs_sum = sum(capped.values())
        if cs_sum > eps:
            tw = {c: w / cs_sum for c, w in capped.items()}

        # ── 11. EMA Smoothing ──
        if ema_weights:
            sm = {cid: ema_alpha * tw.get(cid, 0) + (1 - ema_alpha) * ema_weights.get(cid, 1.0 / n)
                  for cid in ids}
            ss = sum(sm.values())
            if ss > eps:
                tw = {c: w / ss for c, w in sm.items()}

        # Return anchor as torch.Tensor
        new_anchor = torch.from_numpy(anchor).float()
        return tw, cos_sims, new_anchor

    def aggregate(self,
                 client_gradients: Dict[int, Dict[str, torch.Tensor]],
                 sample_counts: Optional[Dict[int, int]] = None,
                 global_gradient: Optional[Dict[str, torch.Tensor]] = None,
                 normalize: bool = True,
                 rule: str = 'avg', # 'avg', 'psdl', 'krum', 'bulyan', 'trimmed_mean', 'median'
                 **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[int, float]]:
        """
        通用聚合入口
        """
        similarities = {}
        
        # 0. 预处理: 归一化 (Krum等通常也建议归一化，或者基于梯度的方向)
        # PSDL 需要归一化来计算相似度
        if normalize or rule == 'our':
            # print("[GradientAggregator] Normalizing gradients...")
            client_gradients = self.normalize_gradients(client_gradients)
        
        # 1. 计算相似度 (用于记录/审计，即使不用于过滤)
        if global_gradient is None and rule in ['psdl']:
             # PSDL 依赖 global_gradient，如果没有，用平均值代替作为基准
             global_gradient = self._compute_average_gradient(client_gradients)

        if global_gradient is not None:
            for cid, grad in client_gradients.items():
                similarities[cid] = self.calculate_cosine_similarity(grad, global_gradient)
        
        # 2. 根据规则聚合
        aggregated_grad = {}

        if rule == 'psdl':
            # Strict Governance (Similarity Filtering)
            threshold = kwargs.get('similarity_threshold', self.similarity_threshold)
            filtered_grads, _ = self.filter_by_similarity(client_gradients, global_gradient, threshold)
            if not filtered_grads:
                 print("[Warning] All gradients filtered out by PSDL! Fallback to Average.")
                 aggregated_grad = self.fedavg_aggregate(client_gradients, sample_counts)
            else:
                 aggregated_grad = self.fedavg_aggregate(filtered_grads, sample_counts)

        elif rule == 'our':
            # Our TASL: Spatial-Temporal Dual-Anchor + Cubic Trust Scoring
            # ── 取历史动量 ──────────────────────────────────────────────────
            v_ref = self.prev_global_update  # 可为 None（第一轮）

            # ── 展平梯度 ────────────────────────────────────────────────────
            flat_updates = {
                cid: self.flatten_update(g).detach().float()
                for cid, g in client_gradients.items()
            }

            # ── 计算混合锚点信任权重（内部已更新 self.prev_global_update）──
            trust_weights = self.compute_trust_weights(
                flat_updates=flat_updates,
                prev_global_update=v_ref,
                trust_power=3.0,          # Cubic Trust Scoring
                momentum_weight=0.5,      # 历史动量权重
                space_weight=0.5,         # 空间中位数权重
                conflict_threshold=0.0,   # 方向冲突时以空间中位数为主
            )

            # ── 加权聚合 ────────────────────────────────────────────────────
            aggregated_grad = self.fedavg_aggregate(
                client_gradients,
                sample_counts=None,
                weights=trust_weights,
            )

            # 记录相似度（用于审计）
            similarities = {cid: trust_weights.get(cid, 0.0) for cid in trust_weights}

            # Note: prev_global_update 已在 compute_trust_weights 内部更新

        elif rule == 'fltrust':
            # FLTrust: root-dataset based trust scores (original FLTrust paper)
            flat_updates = {
                cid: self.flatten_update(g).detach().float()
                for cid, g in client_gradients.items()
            }
            fltrust_root = kwargs.get('root_grad', None)
            trust_weights, scale_factors, root_grad, trust_scores, raw_sims = \
                self.compute_fltrust_outputs(flat_updates=flat_updates, noise_std=1e-3, root_grad=fltrust_root)
            if root_grad is not None:
                self.prev_global_update = root_grad.detach()
            aggregated_grad = self.fedavg_aggregate(
                client_gradients,
                sample_counts=None,
                weights=trust_weights,
            )
            similarities = {cid: trust_weights.get(cid, 0.0) for cid in trust_weights}

        elif rule == 'krum':
            f = kwargs.get('f', 0)
            aggregated_grad = self.krum(client_gradients, f)
            
        elif rule == 'multi_krum':
            f = kwargs.get('f', 0)
            m = kwargs.get('m', len(client_gradients) - f)
            aggregated_grad = self.multi_krum(client_gradients, f, m)
            
        elif rule == 'bulyan':
            f = kwargs.get('f', 0)
            aggregated_grad = self.bulyan(client_gradients, f)
            
        elif rule == 'trimmed_mean':
            beta = kwargs.get('beta', 0.1)
            aggregated_grad = self.trimmed_mean(client_gradients, beta)
            
        elif rule == 'median':
            aggregated_grad = self.median(client_gradients)
            
        else:
            # Default FedAvg
            aggregated_grad = self.fedavg_aggregate(client_gradients, sample_counts)
            
        return aggregated_grad, similarities

    # ============================
    # PBFL + TAS helpers
    # ============================
    @staticmethod
    def flatten_update(update: Dict[str, torch.Tensor]) -> torch.Tensor:
        return GradientAggregator._flatten_gradient(update).float()

    @staticmethod
    def normalize_update_delta(delta_w: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        flat = GradientAggregator._flatten_gradient(delta_w).float()
        norm = torch.norm(flat, p=2)
        if norm <= 0:
            return {k: v.clone() for k, v in delta_w.items()}
        return {
            k: (v / norm).clone() if isinstance(v, torch.Tensor) else v
            for k, v in delta_w.items()
        }

    @staticmethod
    def _recursive_rotation_sum(enc_vec: Any, vector_len: int) -> Any:
        """递归旋转求和（Recursive Rotation-Summation）。"""
        if enc_vec is None:
            return None

        # 优先使用 rotate + add 的递归求和
        if hasattr(enc_vec, "rotate"):
            result = enc_vec
            step = 1
            while step < max(1, int(vector_len)):
                result = result + result.rotate(step)
                step <<= 1
            return result

        # 兼容 TenSEAL 的 sum API
        if hasattr(enc_vec, "sum"):
            return enc_vec.sum()

        return enc_vec

    @staticmethod
    def secure_ckks_cosine_similarity(
        C_i: Any,
        C_global: Any,
        sk_context: Any,
        fallback_plain_i: Optional[torch.Tensor] = None,
        fallback_plain_global: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Secure cosine similarity with CKKS dot-product pathway:
        Sim = Dec_sk(<C_i, C_global>)

        CKKS inner-product uses multiply + recursive rotation-summation.
        """
        if TENSEAL_AVAILABLE and C_i is not None and C_global is not None and sk_context is not None:
            try:
                if isinstance(C_i, (bytes, bytearray)):
                    c_i = ts.ckks_vector_from(sk_context, C_i)
                else:
                    c_i = C_i

                if isinstance(C_global, (bytes, bytearray)):
                    c_g = ts.ckks_vector_from(sk_context, C_global)
                else:
                    c_g = C_global

                mul_enc = c_i * c_g
                vec_len = 0
                if fallback_plain_i is not None:
                    vec_len = int(fallback_plain_i.numel())
                summed_enc = GradientAggregator._recursive_rotation_sum(mul_enc, vector_len=max(1, vec_len))

                sim_dec = summed_enc.decrypt()
                if isinstance(sim_dec, list):
                    return float(sim_dec[0]) if sim_dec else 0.0
                return float(sim_dec)
            except Exception:
                pass

        if fallback_plain_i is None or fallback_plain_global is None:
            return 0.0
        dot = torch.dot(fallback_plain_i.float(), fallback_plain_global.float())
        return float(dot.item())

    @staticmethod
    def tas_governed_secure_aggregate(
        encrypted_updates: Dict[int, Any],
        encrypted_global: Any,
        sk_context: Any,
        tau: float,
        C_T_records: List[Dict[str, Any]],
        executor_id: int,
        V1_id: int,
        V2_id: int,
        plain_updates_for_audit: Optional[Dict[int, torch.Tensor]] = None,
        plain_global_for_audit: Optional[torch.Tensor] = None,
        identity_public_keys: Optional[Dict[int, str]] = None,
    ) -> Tuple[Optional[Any], Dict[int, float], Set[int], bool, bool, bool, Dict[str, Any]]:
        """
        严格 Tripartite Governance：
        1) E(Executor) 计算相似度
        2) V_1(Similarity Auditor) 审计相似度逻辑
        3) V_2(Membership Auditor) 审计 H 与 C_T 成员一致性
        4) 仅在三方治理规则满足时执行同态加和
        """
        # 角色互斥检查：E / V_1 / V_2 必须是三个不同节点
        if len({executor_id, V1_id, V2_id}) != 3:
            print(
                f"[TAS] Governance rejected: roles are not mutually exclusive "
                f"(E={executor_id}, V_1={V1_id}, V_2={V2_id})"
            )
            return None, {}, set(), False, False, False, {"v2_invalid_count": 0, "v2_verified_count": 0}

        # E: 相似度计算
        sim_scores: Dict[int, float] = {}
        for cid, C_i in encrypted_updates.items():
            fp_i = plain_updates_for_audit.get(cid) if plain_updates_for_audit else None
            sim_scores[cid] = GradientAggregator.secure_ckks_cosine_similarity(
                C_i=C_i,
                C_global=encrypted_global,
                sk_context=sk_context,
                fallback_plain_i=fp_i,
                fallback_plain_global=plain_global_for_audit,
            )

        honest_set_H = {cid for cid, sim in sim_scores.items() if sim >= tau}

        # V_1: 相似度审计（重算）
        v1_ok = True
        if plain_updates_for_audit is not None and plain_global_for_audit is not None:
            for cid in honest_set_H:
                if cid not in plain_updates_for_audit:
                    v1_ok = False
                    break
                ref = float(torch.dot(plain_updates_for_audit[cid].float(), plain_global_for_audit.float()).item())
                if abs(ref - sim_scores[cid]) > 1e-2:
                    v1_ok = False
                    break

        # V_2: 成员审计（H ⊆ C_T 记录节点 + 逐条验签）
        c_t_member_ids = set()
        c_t_valid_records = 0
        v2_invalid_count = 0
        v2_verified_count = 0
        v2_invalid_clients: Set[int] = set()
        for rec in C_T_records:
            did = rec.get("DID")
            cid = rec.get("CID")
            sig = rec.get("Signature")
            round_id = rec.get("Round_ID", rec.get("round"))
            if did is None or cid is None or sig is None or round_id is None:
                print("[V2 Audit] FAILED: Invalid record fields")
                v2_invalid_count += 1
                if did is not None:
                    v2_invalid_clients.add(int(did))
                continue

            did_int = int(did)
            pub_key = None
            if identity_public_keys is not None:
                pub_key = identity_public_keys.get(did_int)
            if pub_key is None:
                pub_key = rec.get("PublicKey")

            payload = {"CID": cid, "DID": did_int, "Round_ID": int(round_id)}
            if pub_key is None or not verify_signature(pub_key, payload, sig):
                print(f"[V2 Audit] FAILED: Invalid signature for Client {did_int}")
                v2_invalid_count += 1
                v2_invalid_clients.add(did_int)
                continue

            v2_verified_count += 1
            c_t_member_ids.add(did_int)
            c_t_valid_records += 1

        v2_ok = honest_set_H.issubset(c_t_member_ids)

        # 严格治理：E必须有提交、V1/V2都通过
        executor_submitted = executor_id in c_t_member_ids
        consensus_ok = bool(executor_submitted and v1_ok and v2_ok)

        print(
            f"[TAS] Secure aggregation governance => consensus={consensus_ok}, "
            f"E_submitted={executor_submitted}, V_1={v1_ok}, V_2={v2_ok}, "
            f"|H|={len(honest_set_H)}, valid_C_T_records={c_t_valid_records}"
        )

        audit_stats = {
            "v2_invalid_count": int(v2_invalid_count),
            "v2_verified_count": int(v2_verified_count),
            "v2_invalid_clients": sorted(list(v2_invalid_clients)),
        }

        if not consensus_ok:
            return None, sim_scores, honest_set_H, False, v1_ok, v2_ok, audit_stats

        # 同态加和 C_new = Σ C_honest
        C_new = None
        for cid in sorted(honest_set_H):
            Ci = encrypted_updates[cid]
            if C_new is None:
                C_new = Ci
            else:
                C_new = C_new + Ci

        return C_new, sim_scores, honest_set_H, True, v1_ok, v2_ok, audit_stats


