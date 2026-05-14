"""
FLTrust: 基于服务器根信任的聚合
使用服务器端干净数据生成的基准梯度来评估客户端信任度
"""
from typing import Dict, Tuple, Any
import torch
import torch.nn as nn
from .base import BaseAggregator, fedavg_aggregate


class FLTrustAggregator(BaseAggregator):
    """
    FLTrust 聚合器

    原理:
    1. 服务器使用根数据集(干净数据)计算基准梯度 g0
    2. 对每个客户端更新计算与 g0 的余弦相似度作为信任分数
    3. 只使用信任分数大于阈值的更新进行加权平均

    Args:
        num_clients: 客户端总数
        root_model: 服务器端基准模型
        root_loader: 根数据集的DataLoader
        min_trust: 最小信任阈值
    """

    def __init__(
        self,
        num_clients: int,
        root_model: nn.Module = None,
        root_loader = None,
        min_trust: float = 0.2,
        device: torch.device = None,
        **kwargs
    ):
        super().__init__(num_clients, **kwargs)
        self.root_model = root_model
        self.root_loader = root_loader
        self.min_trust = min_trust
        self.device = device or torch.device("cpu")

    def _compute_root_gradient(self, global_model: nn.Module, criterion = None) -> Dict[str, torch.Tensor]:
        """
        使用根数据集计算基准梯度

        Args:
            global_model: 全局模型

        Returns:
            基准梯度字典
        """
        if self.root_model is None or self.root_loader is None:
            # 如果没有根数据，返回None
            return None

        # 克隆模型参数作为基准
        root_model_state = {k: v.clone() for k, v in global_model.state_dict().items()}

        # 复制全局模型到root_model
        self.root_model.load_state_dict(root_model_state)
        self.root_model.to(self.device)
        self.root_model.eval()

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # 计算梯度
        self.root_model.zero_grad()
        for data, target in self.root_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.root_model(data)
            loss = criterion(output, target)
            loss.backward()

        # 收集梯度
        root_gradients = {}
        for name, param in self.root_model.named_parameters():
            if param.grad is not None:
                root_gradients[name] = param.grad.clone()

        return root_gradients

    def _flatten_gradients(self, gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
        """将梯度字典展平为向量"""
        flat_list = []
        for name in sorted(gradients.keys()):
            param = gradients[name]
            if isinstance(param, torch.Tensor):
                flat_list.append(param.detach().view(-1).float())
        return torch.cat(flat_list) if flat_list else torch.tensor([])

    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        sample_counts: Dict[int, int],
        global_model: nn.Module = None,
        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        执行FLTrust聚合
        """
        if not client_updates:
            return {}, {}

        # 计算根梯度
        root_gradients = self._compute_root_gradient(global_model) if global_model else None

        if root_gradients is None:
            # 没有根数据，回退到FedAvg
            aggregated = fedavg_aggregate(client_updates, sample_counts)
            return aggregated, {'aggregator': 'FLTrust-Fallback', 'reason': 'no root data'}

        # 展平根梯度
        flat_root = self._flatten_gradients(root_gradients)
        root_norm = torch.norm(flat_root) + 1e-12

        # 计算每个客户端的信任分数
        trust_scores = {}
        cos_sims = {}

        for cid, gradients in client_updates.items():
            flat_grad = self._flatten_gradients(gradients)
            grad_norm = torch.norm(flat_grad) + 1e-12

            # 余弦相似度
            cos_sim = torch.dot(flat_grad, flat_root) / (grad_norm * root_norm)
            cos_sim_val = max(0.0, cos_sim.item())  # 只取正值

            trust_scores[cid] = cos_sim_val if cos_sim_val >= self.min_trust else 0.0
            cos_sims[cid] = cos_sim.item()

        # 归一化信任权重
        total_score = sum(trust_scores.values())
        if total_score > 1e-12:
            weights = {cid: score / total_score for cid, score in trust_scores.items()}
        else:
            # 所有分数都很低，回退到FedAvg
            aggregated = fedavg_aggregate(client_updates, sample_counts)
            return aggregated, {'aggregator': 'FLTrust-Fallback', 'reason': 'all scores too low'}

        # 加权聚合
        aggregated = fedavg_aggregate(client_updates, sample_counts, weights)

        info = {
            'aggregator': 'FLTrust',
            'min_trust': self.min_trust,
            'cos_sims': {cid: f"{s:.4f}" for cid, s in cos_sims.items()},
            'trust_weights': {cid: f"{w:.4f}" for cid, w in weights.items()}
        }

        return aggregated, info

    def get_name(self) -> str:
        return f"FLTrust(min_trust={self.min_trust})"
