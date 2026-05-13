"""
区块链节点模块（定义区块链节点和区块链网络
每个客户端作为区块链网络中的一个节点
"""
import copy
import json
import os
from typing import List, Dict, Optional
try:
    from .blockchain import Blockchain, ModelUpdate, aggregate_model_updates
except ImportError:
    # 如果相对导入失败，使用绝对导入
    from blockchain import Blockchain, ModelUpdate, aggregate_model_updates
import torch
import numpy as np

class BlockchainNode:
    """区块链节点，代表一个客户端"""
    
    def __init__(self, node_id: int, blockchain: Optional[Blockchain] = None, 
                 validators: Optional[List[int]] = None,
                 lightweight_mode: bool = False,
                 max_blocks: Optional[int] = None,
                 keep_recent_blocks: int = 10,
                 disk_storage: bool = True,
                 storage_dir: Optional[str] = None,
                 in_memory_blocks: int = 5,
                 circular_mode: bool = False,
                 circular_length: Optional[int] = None):
        """
        初始化节点
        :param node_id: 节点ID（对应客户端ID）
        :param blockchain: 初始区块链（如果为None则创建新的）
        :param validators: 授权验证者列表
        :param lightweight_mode: 是否启用轻量级模式
        :param max_blocks: 最大区块数（超过此数量会自动修剪）
        :param keep_recent_blocks: 修剪时保留的最近区块数量
        :param disk_storage: 是否启用磁盘存储
        :param storage_dir: 存储目录（None则使用默认目录）
        :param in_memory_blocks: 内存中保留的区块数量
        :param circular_mode: 是否启用循环模式（固定长度M+1）
        :param circular_length: 循环链长度（M+1）
        """
        self.node_id = node_id
        if storage_dir is None:
            # Ensure storage_dir is a string if None is passed
            storage_dir = os.path.join("blockchain_storage", f"node_{node_id}")
        
        self.blockchain = blockchain if blockchain else Blockchain(
            validators=validators,
            lightweight_mode=lightweight_mode,
            max_blocks=max_blocks,
            keep_recent_blocks=keep_recent_blocks,
            disk_storage=disk_storage,
            storage_dir=storage_dir,
            in_memory_blocks=in_memory_blocks,
            circular_mode=circular_mode,
            circular_length=circular_length
        )
        self.peers: List[int] = []  # 其他节点的ID列表
        
    def add_peer(self, peer_id: int):
        """添加对等节点"""
        if peer_id != self.node_id and peer_id not in self.peers:
            self.peers.append(peer_id)
    
    def submit_model_update(self, round_number: int, model_params: Dict[str, torch.Tensor], ipfs_manager=None) -> ModelUpdate:
        """
        提交模型更新到区块链
        :param round_number: 轮次编号
        :param model_params: 模型参数字典（PyTorch tensor）
        :param ipfs_manager: IPFS管理器实例 (必须提供)
        :return: 创建的ModelUpdate对象
        """
        # 1. 上传到 IPFS (优先)
        cid = None
        if ipfs_manager:
            # 直接上传原始 PyTorch Tensor 字典 (Bit-Exact)
            cid = ipfs_manager.add(model_params)
            print(f"[Node {self.node_id}] Uploaded model update to IPFS (Binary), CID: {cid[:8]}...")
        else:
             raise ValueError("Must provide ipfs_manager to submit update")
        
        # 2. 提交 CID 到区块链
        update = self.blockchain.add_model_update(
            client_id=self.node_id,
            round_number=round_number,
            model_params=None # 链上不再存储参数
        )
        update.model_cid = cid # 设置 CID
        update.hash = update.calculate_hash()
        
        return update
        
        # 2. 提交 CID 到区块链 (不再提交 full params)
        update = self.blockchain.add_model_update(
            client_id=self.node_id,
            round_number=round_number,
            model_params=None # 链上不再存储
        )
        update.model_cid = cid # 设置 CID
        # 重新计算哈希以包含CID
        update.hash = update.calculate_hash()
        
        return update
    
    def validate_and_seal_block(self, aggregated_model: Optional[Dict[str, torch.Tensor]] = None, 
                                validator_id: Optional[int] = None, round_number: Optional[int] = None,
                                rewards: Optional[Dict[int, float]] = None,
                                similarities: Optional[Dict[int, float]] = None,
                                scores: Optional[Dict[int, float]] = None,
                                ipfs_manager=None) -> bool:
        """
        验证并封装新区块（PoA共识）
        :param aggregated_model: 聚合后的模型参数
        :param ipfs_manager: IPFS管理器实例 (必须提供，用于上传聚合模型)
        :return: 是否成功创建区块
        """
        if not self.blockchain.pending_updates:
            return False
        
        # 将聚合模型上传 IPFS
        aggregated_cid = None
        
        if aggregated_model:
            if ipfs_manager:
                # 直接上传原始字典 (Binary)
                aggregated_cid = ipfs_manager.add(aggregated_model)
                print(f"[Node {self.node_id}] Uploaded aggregated model to IPFS (Binary), CID: {aggregated_cid[:8]}...")
            else:
                print(f"[WARNING] No IPFS manager provided, aggregated model will NOT be stored on IPFS.")
        
        # 注意：这里我们不再需要构造 aggregated_dict (JSON List)，因为不再存入区块体
        # 如果需要兼容旧代码逻辑，aggregated_dict 可以设为 None 或者空字典
        aggregated_dict = None

        try:
            # 创建区块 (传入 CID)
            block = self.blockchain.create_block(
                aggregated_model=None, # 不再存入区块体
                validator_id=validator_id,
                round_number=round_number,
                rewards=rewards,
                similarities=similarities,
                scores=scores
            )
            block.aggregated_model_cid = aggregated_cid
            # 重新计算哈希
            block.hash = block.calculate_hash()
            
            return True
        except Exception as e:
            print(f"节点 {self.node_id} 区块验证失败: {e}")
            return False
    
    def get_latest_model(self, ipfs_manager=None) -> Optional[Dict[str, torch.Tensor]]:
        """
        获取最新的聚合模型
        :param ipfs_manager: IPFS管理器实例 (必须提供)
        :return: 模型参数字典（PyTorch tensor格式）
        """
        latest_block = self.blockchain.get_latest_block()
        cid = latest_block.aggregated_model_cid
        
        if not cid:
             # 尝试兼容旧模式
             aggregated_numpy = self.blockchain.get_aggregated_model()
             if aggregated_numpy is None:
                 return None
        else:
             if ipfs_manager is None:
                 print("[Error] No IPFS manager provided to download model.")
                 return None
             print(f"[Node {self.node_id}] Downloading global model from IPFS, CID: {cid[:8]}...")
             aggregated_numpy = ipfs_manager.get(cid)
             if aggregated_numpy is None:
                 print(f"[Error] Failed to retrieve model from IPFS CID: {cid}")
                 return None
        
        # 转换为PyTorch tensor
        model_params = {}
        for key, value in aggregated_numpy.items():
            model_params[key] = torch.tensor(value, dtype=torch.float32)
        
        return model_params
    
    def sync_blockchain(self, other_blockchain: Blockchain):
        """
        同步其他节点的区块链（选择更新的有效链）
        :param other_blockchain: 其他节点的区块链
        """
        # 判断是否应该同步：循环模式比较total_blocks，普通模式比较链长度
        should_sync = False
        if self.blockchain.circular_mode or other_blockchain.circular_mode:
            # 循环模式：比较总轮次数（total_blocks）
            if other_blockchain.total_blocks > self.blockchain.total_blocks:
                should_sync = True
        else:
            # 普通模式：比较链长度
            if len(other_blockchain.chain) > len(self.blockchain.chain):
                should_sync = True
        
        if should_sync:
            if other_blockchain.validate_chain():
                self.blockchain = copy.deepcopy(other_blockchain)
                # 重要：同步后也要同步pending_updates（应该已经被清空）
                self.blockchain.pending_updates = other_blockchain.pending_updates.copy()
                if self.blockchain.circular_mode:
                    print(f"[DEBUG] 节点 {self.node_id} 同步了更新的区块链，总轮次数: {self.blockchain.total_blocks}, 链长度: {len(self.blockchain.chain)}")
                else:
                    print(f"[DEBUG] 节点 {self.node_id} 同步了更长的区块链，链长度: {len(self.blockchain.chain)}")
    
    def get_pending_updates_count(self) -> int:
        """获取待处理的更新数量"""
        return len(self.blockchain.pending_updates)
    
    def save_blockchain(self, filepath: str):
        """保存区块链到文件"""
        with open(filepath, 'w') as f:
            json.dump(self.blockchain.to_dict(), f, indent=2, default=str)
    
    def load_blockchain(self, filepath: str):
        """从文件加载区块链"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.blockchain = Blockchain.from_dict(data)
        else:
            print(f"文件 {filepath} 不存在，使用新的区块链")


class BlockchainNetwork:
    """区块链网络，管理所有节点"""
    
    def __init__(self, num_nodes: int, validators: Optional[List[int]] = None,
                 lightweight_mode: bool = False,
                 max_blocks: Optional[int] = None,
                 keep_recent_blocks: int = 10,
                 disk_storage: bool = True,
                 storage_dir: str = "blockchain_storage",
                 in_memory_blocks: int = 5,
                 circular_mode: bool = False,
                 circular_length: Optional[int] = None):
        """
        初始化网络
        :param num_nodes: 节点数量
        :param validators: 授权验证者ID列表（如果为None则所有节点都是验证者）
        :param lightweight_mode: 是否启用轻量级模式
        :param max_blocks: 最大区块数（超过此数量会自动修剪）
        :param keep_recent_blocks: 修剪时保留的最近区块数量
        :param disk_storage: 是否启用磁盘存储（旧区块保存到磁盘）
        :param storage_dir: 存储目录（每个节点会有子目录）
        :param in_memory_blocks: 内存中保留的区块数量
        :param circular_mode: 是否启用循环模式（固定长度M+1）
        :param circular_length: 循环链长度（M+1）
        """
        self.num_nodes = num_nodes
        self.nodes: Dict[int, BlockchainNode] = {}
        # 默认所有节点都是验证者
        self.validators = validators if validators else list(range(num_nodes))
        
        # 创建所有节点，第一个节点初始化区块链
        main_storage_dir = os.path.join(storage_dir, "node_0") if disk_storage else None
        
        for i in range(num_nodes):
            if i == 0:
                blockchain = Blockchain(
                    validators=self.validators,
                    lightweight_mode=lightweight_mode,
                    max_blocks=max_blocks,
                    keep_recent_blocks=keep_recent_blocks,
                    disk_storage=disk_storage,
                    storage_dir=main_storage_dir if disk_storage else "blockchain_storage",
                    in_memory_blocks=in_memory_blocks,
                    circular_mode=circular_mode,
                    circular_length=circular_length
                )
                node = BlockchainNode(
                    i, blockchain, validators=self.validators,
                    lightweight_mode=lightweight_mode,
                    max_blocks=max_blocks,
                    keep_recent_blocks=keep_recent_blocks,
                    disk_storage=disk_storage,
                    storage_dir=main_storage_dir,
                    in_memory_blocks=in_memory_blocks,
                    circular_mode=circular_mode,
                    circular_length=circular_length
                )
            else:
                # 其他节点复制第一个节点的区块链
                blockchain = copy.deepcopy(self.nodes[0].blockchain)
                # 为每个节点设置独立的存储目录
                node_storage_dir = os.path.join(storage_dir, f"node_{i}") if disk_storage else None
                blockchain.storage_dir = node_storage_dir if disk_storage else blockchain.storage_dir
                if disk_storage and node_storage_dir:
                    os.makedirs(node_storage_dir, exist_ok=True)
                
                node = BlockchainNode(
                    i, blockchain, validators=self.validators,
                    lightweight_mode=lightweight_mode,
                    max_blocks=max_blocks,
                    keep_recent_blocks=keep_recent_blocks,
                    disk_storage=disk_storage,
                    storage_dir=node_storage_dir,
                    in_memory_blocks=in_memory_blocks,
                    circular_mode=circular_mode,
                    circular_length=circular_length
                )
            
            # 添加所有其他节点作为对等节点
            for j in range(num_nodes):
                if i != j:
                    node.add_peer(j)
            
            self.nodes[i] = node
    
    def get_node(self, node_id: int) -> BlockchainNode:
        """获取指定节点"""
        return self.nodes[node_id]
    
    def sync_all_nodes(self):
        """同步所有节点的区块链（循环模式比较total_blocks，普通模式使用最长链原则）"""
        # 检查是否启用循环模式（假设所有节点使用相同的模式）
        circular_mode = False
        if self.nodes:
            circular_mode = next(iter(self.nodes.values())).blockchain.circular_mode
        
        # 找到最新的有效链
        best_chain = None
        best_value = 0
        
        for node in self.nodes.values():
            if circular_mode:
                # 循环模式：比较总轮次数（total_blocks）
                if node.blockchain.total_blocks > best_value:
                    if node.blockchain.validate_chain():
                        best_chain = node.blockchain
                        best_value = node.blockchain.total_blocks
            else:
                # 普通模式：比较链长度
                if len(node.blockchain.chain) > best_value:
                    if node.blockchain.validate_chain():
                        best_chain = node.blockchain
                        best_value = len(node.blockchain.chain)
        
        # 同步所有节点
        if best_chain:
            for node in self.nodes.values():
                node.sync_blockchain(best_chain)
    
    def aggregate_and_seal(self, round_number: int, selected_clients: List[int]) -> Dict[str, torch.Tensor]:
        """
        聚合选中的客户端更新并由验证者封装区块
        :param round_number: 轮次编号
        :param selected_clients: 选中的客户端ID列表
        :return: 聚合后的模型参数
        """
        print(f"[DEBUG] 开始聚合 {len(selected_clients)} 个客户端的更新...")
        
        # 收集所有待处理的更新
        updates = []
        for client_id in selected_clients:
            node = self.nodes[client_id]
            client_updates = node.blockchain.pending_updates
            updates.extend(client_updates)
            print(f"[DEBUG] 客户端 {client_id} 的更新数: {len(client_updates)}")
        
        if not updates:
            raise ValueError("没有待处理的模型更新")
        
        print(f"[DEBUG] 总共收集到 {len(updates)} 个模型更新")
        
        # 聚合模型更新
        print("[DEBUG] 执行FedAvg聚合算法...")
        aggregated_numpy = aggregate_model_updates(updates, len(selected_clients))
        print(f"[DEBUG] 聚合完成，参数层数: {len(aggregated_numpy)}")
        
        # 转换为PyTorch tensor
        aggregated_model = {}
        for key, value in aggregated_numpy.items():
            if isinstance(value, list):
                aggregated_model[key] = torch.tensor(value, dtype=torch.float32)
            else:
                aggregated_model[key] = torch.tensor(np.array(value), dtype=torch.float32)
        
        # 同步所有节点的待处理更新（确保所有节点都有相同的待处理更新）
        print("[DEBUG] 同步所有节点的待处理更新...")
        # 收集所有节点的待处理更新（去重）
        all_updates_dict = {}  # 使用(client_id, round_number)作为键去重
        for node in self.nodes.values():
            for update in node.blockchain.pending_updates:
                key = (update.client_id, update.round_number)
                if key not in all_updates_dict:
                    all_updates_dict[key] = update
        
        all_updates_list = list(all_updates_dict.values())
        print(f"[DEBUG] 去重后的总更新数: {len(all_updates_list)}")
        
        # 将所有更新同步到所有节点（创建区块前）
        for node in self.nodes.values():
            node.blockchain.pending_updates = all_updates_list.copy()
        
        # 选择一个验证者节点创建区块（轮询方式由区块链内部管理）
        validator_id = self.validators[0]
        validator_node = self.nodes[validator_id]
        print(f"[DEBUG] 验证者节点 {validator_id} 开始封装区块...")
        print(f"[DEBUG] 验证者节点待处理更新数: {len(validator_node.blockchain.pending_updates)}")
        
        # 由验证者封装区块（create_block会自动清空pending_updates）
        # 注意: BlockchainNetwork需要持有ipfs_manager才能传给validate_and_seal_block
        # 但目前设计上BlockchainNetwork没有ipfs_manager，我们假设调用方会修改或者我们给BlockchainNetwork添加一个默认的
        # 为了更简单，我们暂时创建一个临时的 IPFS Manager (如果还没传入) - 或者更好，修改BlockchainNetwork支持它
        
        # [Hack] 动态引入 MockIPFSManager
        try:
            from .ipfs_manager import MockIPFSManager
            ipfs = MockIPFSManager() # 使用默认路径
        except ImportError:
            from ipfs_manager import MockIPFSManager
            ipfs = MockIPFSManager()

        result = validator_node.validate_and_seal_block(aggregated_model, ipfs_manager=ipfs)
        if result:
            print(f"[DEBUG] 区块封装成功")
            # 创建区块后，所有节点的pending_updates应该都被清空了
            print(f"[DEBUG] 区块创建后，验证者节点待处理更新数: {len(validator_node.blockchain.pending_updates)}")
        else:
            print(f"[WARNING] 区块封装失败！")
        
        # 重要：创建区块后，验证者节点的pending_updates已被清空
        # 需要手动清空所有节点的pending_updates，然后同步
        print("[DEBUG] 清空所有节点的待处理更新...")
        for node in self.nodes.values():
            node.blockchain.pending_updates = []
        
        # 同步所有节点的区块链
        print("[DEBUG] 同步所有节点的区块链...")
        self.sync_all_nodes()
        
        # 验证所有节点的pending_updates都已清空
        all_cleared = True
        for node_id, node in self.nodes.items():
            pending_count = len(node.blockchain.pending_updates)
            if pending_count > 0:
                print(f"[WARNING] 节点 {node_id} 仍有 {pending_count} 个待处理更新！")
                all_cleared = False
            else:
                print(f"[DEBUG] 节点 {node_id} 待处理更新已清空")
        
        if all_cleared:
            print("[DEBUG] ✓ 所有节点的待处理更新已成功清空")
        
        print("[DEBUG] 所有节点同步完成")
        
        return aggregated_model

