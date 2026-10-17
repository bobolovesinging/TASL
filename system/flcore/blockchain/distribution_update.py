"""
数据分布更新交易模块
扩展区块链以支持数据分布信息的共享
"""
import hashlib
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from .data_distribution import DataDistributionStats


@dataclass
class DistributionUpdate:
    """数据分布更新交易"""
    client_id: int
    round_number: int
    distribution_stats: Dict[str, Any]  # 数据分布统计信息
    timestamp: float
    previous_hash: str
    hash: str = None
    
    def __post_init__(self):
        """计算交易的哈希值"""
        if self.hash is None:
            self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """计算交易的哈希值"""
        # 将分布统计转换为可哈希的字符串
        stats_str = json.dumps(self.distribution_stats, sort_keys=True, default=str)
        data_string = f"{self.client_id}{self.round_number}{stats_str}{self.timestamp}{self.previous_hash}"
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DistributionUpdate':
        """从字典恢复"""
        return cls(**data)


@dataclass
class DistributionBlock:
    """数据分布区块"""
    index: int
    timestamp: float
    distribution_updates: list  # DistributionUpdate列表
    previous_hash: str
    aggregated_distribution: Optional[Dict[str, Any]] = None  # 聚合后的全局分布
    validator: int = 0
    hash: str = None
    
    def __post_init__(self):
        """计算区块的哈希值"""
        if self.hash is None:
            self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """计算区块的哈希值"""
        updates_str = json.dumps([u.to_dict() for u in self.distribution_updates], sort_keys=True, default=str)
        agg_dist_str = json.dumps(self.aggregated_distribution, sort_keys=True, default=str) if self.aggregated_distribution else ""
        data_string = f"{self.index}{self.timestamp}{updates_str}{agg_dist_str}{self.previous_hash}{self.validator}"
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def seal_block(self, validator_id: int):
        """PoA验证者签名区块"""
        self.validator = validator_id
        self.hash = self.calculate_hash()
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'distribution_updates': [u.to_dict() for u in self.distribution_updates],
            'previous_hash': self.previous_hash,
            'hash': self.hash,
            'validator': self.validator,
            'aggregated_distribution': self.aggregated_distribution
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DistributionBlock':
        """从字典恢复"""
        updates = [DistributionUpdate.from_dict(u) for u in data['distribution_updates']]
        return cls(
            index=data['index'],
            timestamp=data['timestamp'],
            distribution_updates=updates,
            previous_hash=data['previous_hash'],
            hash=data['hash'],
            validator=data.get('validator', 0),
            aggregated_distribution=data.get('aggregated_distribution')
        )

