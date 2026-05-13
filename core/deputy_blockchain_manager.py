"""
TAS 链管理器
将三方治理角色节点映射到各自客户端区块链实例。
命名约定：
- C_T: Training Node Chain（训练节点链）
- C_D: Deputy/Auditor Chain（治理链）
"""
import copy
from typing import Dict, List, Optional

try:
    from .blockchain_node import BlockchainNode, BlockchainNetwork
except ImportError:
    from blockchain_node import BlockchainNode, BlockchainNetwork


class TASBlockchainManager:
    """TAS 节点区块链管理器（沿用各节点自己的链存储）"""

    def __init__(self, tas_node_ids: List[int], blockchain_network: BlockchainNetwork):
        if not tas_node_ids:
            raise ValueError("tas_node_ids 不能为空")
        self.tas_node_ids = tas_node_ids.copy()
        self.blockchain_network = blockchain_network
        self.tas_nodes: Dict[int, BlockchainNode] = {}
        self._init_tas_blockchains()

    def _init_tas_blockchains(self):
        for node_id in self.tas_node_ids:
            self.tas_nodes[node_id] = self.blockchain_network.get_node(node_id)
        print(f"[TASBlockchainManager] 已初始化 {len(self.tas_nodes)} 个 TAS 节点")

    def update_tas_nodes(self, new_tas_node_ids: List[int]):
        old_ids = set(self.tas_node_ids)
        new_ids = set(new_tas_node_ids)

        for removed in (old_ids - new_ids):
            self.tas_nodes.pop(removed, None)
            print(f"[TASBlockchainManager] 节点 {removed} 不再是 TAS 节点")

        for added in (new_ids - old_ids):
            self.tas_nodes[added] = self.blockchain_network.get_node(added)
            print(f"[TASBlockchainManager] 节点 {added} 成为 TAS 节点")

        self.tas_node_ids = new_tas_node_ids.copy()
        print(f"[TASBlockchainManager] TAS 节点已更新: {self.tas_node_ids}")

    def is_tas_node(self, node_id: int) -> bool:
        return node_id in self.tas_node_ids

    def get_tas_node(self, node_id: int) -> Optional[BlockchainNode]:
        return self.tas_nodes.get(node_id)

    def get_all_tas_nodes(self) -> Dict[int, BlockchainNode]:
        return self.tas_nodes.copy()

    def sync_tas_blockchains(self):
        if not self.tas_nodes:
            return

        best_chain = None
        best_value = 0
        best_node_id = None

        for node_id, node in self.tas_nodes.items():
            chain_value = node.blockchain.get_chain_length()
            if chain_value > best_value and node.blockchain.validate_chain():
                best_chain = node.blockchain
                best_value = chain_value
                best_node_id = node_id

        if best_chain:
            for node_id, node in self.tas_nodes.items():
                if node_id != best_node_id:
                    node.sync_blockchain(best_chain)
            print(f"[TASBlockchainManager] 已同步 TAS 链，来源节点 {best_node_id}，长度指标={best_value}")

    def get_storage_info(self) -> Dict:
        info = {
            'num_tas_nodes': len(self.tas_node_ids),
            'tas_node_ids': self.tas_node_ids.copy(),
            'tas_storage': {}
        }
        for node_id, node in self.tas_nodes.items():
            memory_info = node.blockchain.get_memory_info()
            info['tas_storage'][node_id] = {
                'chain_length': node.blockchain.get_chain_length(),
                'in_memory_blocks': memory_info.get('in_memory_blocks', 0),
                'disk_blocks': memory_info.get('disk_blocks', 0),
                'memory_bytes': memory_info.get('estimated_memory_bytes', 0),
                'disk_bytes': memory_info.get('disk_size_bytes', 0),
            }
        return info


# 兼容旧导入
class DeputyBlockchainManager(TASBlockchainManager):
    pass
