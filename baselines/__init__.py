"""
聚合策略模块
支持: FedAvg, Multi-Krum, Trimmed Mean, FLTrust
"""
from .base import BaseAggregator, fedavg_aggregate
from .fedavg import FedAvgAggregator
from .multi_krum import MultiKrumAggregator
from .trimmed_mean import TrimmedMeanAggregator
from .fltrust import FLTrustAggregator


def get_aggregator(
    name: str,
    num_clients: int,
    config: dict = None,
    **kwargs
) -> BaseAggregator:
    """
    工厂函数：根据配置获取聚合器

    Args:
        name: 聚合器名称
        num_clients: 客户端数量
        config: 聚合器配置参数字典
        **kwargs: 其他参数（如FLTrust的root_model）

    Returns:
        聚合器实例
    """
    config = config or {}

    if name.lower() == 'fedavg':
        return FedAvgAggregator(num_clients)
    elif name.lower() == 'multi_krum':
        return MultiKrumAggregator(
            num_clients,
            f=config.get('f', num_clients // 4),
            m=config.get('m', num_clients - 2 * config.get('f', num_clients // 4))
        )
    elif name.lower() == 'trimmed_mean':
        return TrimmedMeanAggregator(
            num_clients,
            beta=config.get('beta', 0.1)
        )
    elif name.lower() == 'fltrust':
        return FLTrustAggregator(
            num_clients,
            root_model=kwargs.get('root_model'),
            root_loader=kwargs.get('root_loader'),
            min_trust=config.get('min_trust', 0.2),
            device=kwargs.get('device')
        )
    elif name.lower() == 'median':
        # 使用TrimmedMean with beta=0.5 作为Median近似
        return TrimmedMeanAggregator(num_clients, beta=0.5)
    else:
        # 默认返回FedAvg
        print(f"[Warning] Unknown aggregator '{name}', using FedAvg")
        return FedAvgAggregator(num_clients)


__all__ = [
    'BaseAggregator',
    'FedAvgAggregator',
    'MultiKrumAggregator',
    'TrimmedMeanAggregator',
    'FLTrustAggregator',
    'get_aggregator',
    'fedavg_aggregate'
]
