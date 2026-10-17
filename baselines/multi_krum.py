"""
Multi-Krum: 基于欧氏距离的鲁棒聚合
选择与最近邻居距离之和最小的m个梯度进行平均
"""
from typing import Dict, Tuple, Any
import torch
from .base import BaseAggregator


class MultiKrumAggregator(BaseAggregator):
    """
    Multi-Krum 聚合器

    Args:
        num_clients: 客户端总数
        f: 恶意客户端数量上限
        m: 选择的梯度数量（默认: n - 2f）
        multi_krum: True使用Multi-Krum，False使用单Krum
    """

    def __init__(self, num_clients: int, f: int = 4, m: int = None, multi_krum: bool = True, **kwargs):
        super().__init__(num_clients, **kwargs)
        self.f = f
        self.m = m if m else num_clients - 2 * f
        self.multi_krum = multi_krum

    def _flatten_gradients(self, gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
        """将梯度字典展平为向量"""
        flat_list = []
        for name in sorted(gradients.keys()):
            param = gradients[name]
            if isinstance(param, torch.Tensor):
                flat_list.append(param.detach().view(-1).float())
        return torch.cat(flat_list) if flat_list else torch.tensor([])

    def _compute_pairwise_distances(self, flat_grads: Dict[int, torch.Tensor]) -> Dict[int, Dict[int, float]]:
        """计算所有客户端之间的欧氏距离"""
        ids = list(flat_grads.keys())
        distances = {i: {} for i in ids}

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                dist = torch.norm(flat_grads[ids[i]] - flat_grads[ids[j]]).item()
                distances[ids[i]][ids[j]] = dist
                distances[ids[j]][ids[i]] = dist
            distances[ids[i]][ids[i]] = 0.0

        return distances

    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        sample_counts: Dict[int, int],
        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        执行Multi-Krum聚合
        """
        if not client_updates:
            return {}, {}

        n = len(client_updates)

        # 如果客户端数不足，回退到FedAvg
        if n < 2 * self.f + 3:
            from .fedavg import fedavg_aggregate
            aggregated = fedavg_aggregate(client_updates, sample_counts)
            return aggregated, {'aggregator': 'MultiKrum-Fallback', 'reason': 'n too small'}

        # 展平梯度
        flat_grads = {cid: self._flatten_gradients(grad) for cid, grad in client_updates.items()}

        # 计算成对距离
        distances = self._compute_pairwise_distances(flat_grads)

        # 计算每个客户端的分数（与最近k个邻居的距离和）
        k = n - self.f - 2
        scores = {}
        for cid in flat_grads.keys():
            dists = sorted([distances[cid][j] for j in flat_grads.keys() if i != cid])
            scores[cid] = sum(dists[:k])

        # 选择分数最小的m个客户端
        selected_ids = sorted(scores, key=scores.get)[:self.m]

        # 对选中的梯度进行FedAvg
        selected_updates = {cid: client_updates[cid] for cid in selected_ids}
        selected_counts = {cid: sample_counts[cid] for cid in selected_ids}

        from .fedavg import fedavg_aggregate
        aggregated = fedavg_aggregate(selected_updates, selected_counts)

        info = {
            'aggregator': 'MultiKrum' if self.multi_krum else 'Krum',
            'f': self.f,
            'k': k,
            'selected_clients': selected_ids,
            'scores': {cid: f"{scores[cid]:.4f}" for cid in selected_ids}
        }

        return aggregated, info

    def get_name(self) -> str:
        return f"MultiKrum(f={self.f}, m={self.m})"
