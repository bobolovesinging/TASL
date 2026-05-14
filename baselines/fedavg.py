"""
FedAvg: Federated Averaging
标准联邦平均聚合算法
"""
from typing import Dict, Tuple, Any
import torch
from .base import BaseAggregator, fedavg_aggregate


class FedAvgAggregator(BaseAggregator):
    """
    FedAvg 聚合器
    根据样本数量对客户端更新进行加权平均
    """

    def __init__(self, num_clients: int, **kwargs):
        super().__init__(num_clients, **kwargs)

    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        sample_counts: Dict[int, int],
        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        执行FedAvg聚合

        Returns:
            (aggregated_gradient, info_dict)
        """
        if not client_updates:
            return {}, {}

        # 计算权重
        total_samples = sum(sample_counts.values())
        weights = {cid: sample_counts[cid] / total_samples for cid in client_updates.keys()}

        # 聚合
        aggregated = fedavg_aggregate(client_updates, sample_counts, weights)

        info = {
            'aggregator': 'FedAvg',
            'num_clients': len(client_updates),
            'weights': weights
        }

        return aggregated, info

    def get_name(self) -> str:
        return "FedAvg"
