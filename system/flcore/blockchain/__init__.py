"""
区块链联邦学习模块
"""
from .blockchain import Blockchain, Block, ModelUpdate, aggregate_model_updates
from .blockchain_node import BlockchainNode, BlockchainNetwork
from .data_distribution import DataDistributionStats, aggregate_distributions, compute_distribution_similarity
from .distribution_update import DistributionUpdate, DistributionBlock

__all__ = [
    'Blockchain',
    'Block',
    'ModelUpdate',
    'aggregate_model_updates',
    'BlockchainNode',
    'BlockchainNetwork',
    'DataDistributionStats',
    'aggregate_distributions',
    'compute_distribution_similarity',
    'DistributionUpdate',
    'DistributionBlock'
]

