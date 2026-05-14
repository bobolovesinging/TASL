"""
聚合策略基类
所有聚合算法需继承 BaseAggregator
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import torch
import numpy as np


class BaseAggregator(ABC):
    """
    聚合器基类
    所有聚合算法必须实现 aggregate() 方法
    """

    def __init__(self, num_clients: int, **kwargs):
        self.num_clients = num_clients
        self.config = kwargs

    @abstractmethod
    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        sample_counts: Dict[int, int],
        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        聚合客户端更新

        Args:
            client_updates: {client_id: {param_name: gradient}}
            sample_counts: {client_id: num_samples}
            **kwargs: 其他参数

        Returns:
            (aggregated_update, extra_info)
            extra_info 包含信任权重、过滤信息等
        """
        pass

    def get_name(self) -> str:
        """获取聚合器名称"""
        return self.__class__.__name__


def fedavg_aggregate(
    client_updates: Dict[int, Dict[str, torch.Tensor]],
    sample_counts: Dict[int, int],
    weights: Dict[int, float] = None
) -> Dict[str, torch.Tensor]:
    """
    FedAvg 加权平均聚合

    Args:
        client_updates: 客户端梯度字典
        sample_counts: 样本数量
        weights: 自定义权重（可选）

    Returns:
        聚合后的梯度
    """
    if not client_updates:
        return {}

    # 计算权重
    if weights is None:
        total_samples = sum(sample_counts.values())
        if total_samples > 0:
            weights = {cid: sample_counts[cid] / total_samples for cid in client_updates.keys()}
        else:
            weights = {cid: 1.0 / len(client_updates) for cid in client_updates.keys()}

    # 初始化聚合梯度
    first_cid = list(client_updates.keys())[0]
    aggregated = {}

    for param_name in client_updates[first_cid].keys():
        # 获取第一个客户端的参数作为模板
        template = client_updates[first_cid][param_name]

        # 跳过整数类型参数（BatchNorm的num_batches_tracked）
        if isinstance(template, torch.Tensor) and template.dtype in [torch.int64, torch.int32, torch.long]:
            aggregated[param_name] = template.clone()
            continue

        # 初始化为零张量
        aggregated[param_name] = torch.zeros_like(template.float())

    # 加权累加
    for client_id, updates in client_updates.items():
        weight = weights.get(client_id, 0.0)
        for param_name in aggregated.keys():
            if param_name in updates:
                param = updates[param_name]
                if isinstance(param, torch.Tensor):
                    # 跳过整数类型参数（BatchNorm的num_batches_tracked等）
                    if param.dtype in [torch.int64, torch.int32, torch.long]:
                        continue
                    if aggregated[param_name].dtype != param.dtype:
                        param = param.float()
                    aggregated[param_name] += weight * param

    return aggregated
