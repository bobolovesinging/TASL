"""
Trimmed Mean: 坐标级修剪聚合
对每个维度去掉最大和最小的若干值后求平均
"""
from typing import Dict, Tuple, Any
import torch
from .base import BaseAggregator


class TrimmedMeanAggregator(BaseAggregator):
    """
    Trimmed Mean 聚合器

    Args:
        num_clients: 客户端总数
        beta: 修剪比例 (0.0 ~ 0.5)，每端去掉 beta*n 个值
    """

    def __init__(self, num_clients: int, beta: float = 0.1, **kwargs):
        super().__init__(num_clients, **kwargs)
        self.beta = beta

    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        sample_counts: Dict[int, int],
        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        执行Trimmed Mean聚合
        """
        if not client_updates:
            return {}, {}

        n = len(client_updates)
        trim_count = max(0, min(int(n * self.beta), int((n - 1) / 2)))

        # 获取第一个客户端的参数列表作为模板
        first_cid = list(client_updates.keys())[0]
        param_names = list(client_updates[first_cid].keys())

        aggregated = {}

        # 对每个参数进行Trimmed Mean
        for param_name in param_names:
            # 堆叠所有客户端的参数
            tensors = []
            for cid in client_updates.keys():
                param = client_updates[cid][param_name]
                if isinstance(param, torch.Tensor):
                    tensors.append(param.float())

            if not tensors:
                continue

            stacked = torch.stack(tensors, dim=0)  # [n, *param_shape]

            # 排序
            sorted_vals, _ = torch.sort(stacked, dim=0)

            # 修剪
            if trim_count > 0:
                trimmed = sorted_vals[trim_count:n - trim_count]
            else:
                trimmed = sorted_vals

            # 求平均
            aggregated[param_name] = torch.mean(trimmed, dim=0)

        info = {
            'aggregator': 'TrimmedMean',
            'beta': self.beta,
            'trim_count': trim_count,
            'n_clients': n
        }

        return aggregated, info

    def get_name(self) -> str:
        return f"TrimmedMean(beta={self.beta})"
