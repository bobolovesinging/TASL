"""
区块链节点模块
每个客户端作为区块链网络中的一个节点
"""
import copy
import json
import os
from typing import List, Dict, Optional
from .blockchain import Blockchain, ModelUpdate, aggregate_model_updates
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
            storage_dir = f"blockchain_storage_node_{node_id}"
        
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
    
    def submit_model_update(self, round_number: int, model_params: Dict[str, torch.Tensor],
                            gradient_params: Optional[Dict[str, torch.Tensor]] = None,
                            sample_count: int = 0,
                            train_loss: float = 0.0) -> ModelUpdate:
        """
        提交模型更新到区块链
        :param round_number: 轮次编号
        :param model_params: 模型参数字典（PyTorch tensor）
        :return: 创建的ModelUpdate对象
        """
        # 将PyTorch tensor转换为可序列化的格式
        params_dict = {}
        for key, value in model_params.items():
            if isinstance(value, torch.Tensor):
                params_dict[key] = value.cpu().detach().numpy().tolist()
            else:
                params_dict[key] = value
        
        gradient_dict = None
        if gradient_params:
            gradient_dict = {}
            for key, value in gradient_params.items():
                if isinstance(value, torch.Tensor):
                    gradient_dict[key] = value.cpu().detach().numpy().tolist()
                else:
                    gradient_dict[key] = value
        
        update = self.blockchain.add_model_update(
            client_id=self.node_id,
            round_number=round_number,
            model_params=params_dict,
            gradient_params=gradient_dict,
            sample_count=sample_count,
            train_loss=train_loss
        )
        
        return update
    
    def validate_and_seal_block(self, aggregated_model: Optional[Dict[str, torch.Tensor]] = None, validator_id: Optional[int] = None) -> bool:
        """
        验证并封装新区块（PoA共识）
        :param aggregated_model: 聚合后的模型参数
        :param validator_id: 验证者ID（如果为None则由区块链轮询选择）
        :return: 是否成功创建区块
        """
        if not self.blockchain.pending_updates:
            return False
        
        # 将聚合模型转换为可序列化格式
        aggregated_dict = None
        if aggregated_model:
            aggregated_dict = {}
            for key, value in aggregated_model.items():
                if isinstance(value, torch.Tensor):
                    aggregated_dict[key] = value.cpu().detach().numpy().tolist()
                else:
                    aggregated_dict[key] = value
        
        try:
            self.blockchain.create_block(aggregated_model=aggregated_dict, validator_id=validator_id)
            return True
        except Exception as e:
            print(f"节点 {self.node_id} 区块验证失败: {e}")
            return False
    
    def get_latest_model(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        获取最新的聚合模型
        :return: 模型参数字典（PyTorch tensor格式）
        """
        aggregated_numpy = self.blockchain.get_aggregated_model()
        if aggregated_numpy is None:
            return None
        
        # 转换为PyTorch tensor
        model_params = {}
        for key, value in aggregated_numpy.items():
            model_params[key] = torch.tensor(value, dtype=torch.float32)
        
        return model_params
    
    def sync_blockchain(self, other_blockchain: Blockchain):
        """
        同步其他节点的区块链（选择最长的有效链）
        优化：避免深拷贝整个区块链，只同步链结构
        :param other_blockchain: 其他节点的区块链
        """
        if len(other_blockchain.chain) > len(self.blockchain.chain):
            if other_blockchain.validate_chain():
                # 优化：只复制链结构，不深拷贝整个区块链对象
                # 这样可以避免复制大量模型参数数据
                self.blockchain.chain = []
                for block in other_blockchain.chain:
                    # 只复制区块的元数据，不复制完整的model_updates（因为lightweight_mode已经处理了）
                    if block.lightweight:
                        # 轻量级区块：直接复制（不包含完整模型参数）
                        # 注意：这里仍然需要创建新对象，避免引用共享
                        block_copy = Block(
                            index=block.index,
                            timestamp=block.timestamp,
                            model_updates=[],  # 轻量级模式不包含完整更新
                            previous_hash=block.previous_hash,
                            validator=block.validator,
                            hash=block.hash,
                            aggregated_model=block.aggregated_model,
                            lightweight=True,
                            update_hashes=getattr(block, 'update_hashes', None),
                            client_ids=getattr(block, 'client_ids', None)
                        )
                        self.blockchain.chain.append(block_copy)
                    else:
                        # 完整区块：转换为轻量级后再复制（节省内存）
                        block_copy = Block(
                            index=block.index,
                            timestamp=block.timestamp,
                            model_updates=[],  # 不复制完整更新
                            previous_hash=block.previous_hash,
                            validator=block.validator,
                            hash=block.hash,
                            aggregated_model=block.aggregated_model,
                            lightweight=True,  # 标记为轻量级
                            update_hashes=getattr(block, 'update_hashes', None),
                            client_ids=getattr(block, 'client_ids', None)
                        )
                        self.blockchain.chain.append(block_copy)
                
                # 同步其他必要属性
                self.blockchain.validators = other_blockchain.validators.copy()
                self.blockchain.current_validator_index = other_blockchain.current_validator_index
                self.blockchain.total_blocks = other_blockchain.total_blocks
                
                # 重要：同步后也要同步pending_updates（应该已经被清空）
                # 使用浅拷贝即可，因为pending_updates应该已经被清空
                self.blockchain.pending_updates = []
                
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
                # 其他节点复制第一个节点的区块链（优化：只复制必要的数据，避免深拷贝整个区块链）
                # 创建新的区块链实例，但共享相同的配置
                blockchain = Blockchain(
                    validators=self.validators,
                    lightweight_mode=lightweight_mode,
                    max_blocks=max_blocks,
                    keep_recent_blocks=keep_recent_blocks,
                    disk_storage=disk_storage,
                    storage_dir=os.path.join(storage_dir, f"node_{i}") if disk_storage else "blockchain_storage",
                    in_memory_blocks=in_memory_blocks,
                    circular_mode=circular_mode,
                    circular_length=circular_length
                )
                # 只同步链结构（不包括pending_updates，因为每个节点独立）
                # 注意：这里不复制pending_updates，因为每个节点的pending_updates应该是独立的
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
        """同步所有节点的区块链（使用最长链原则）"""
        # 找到最长的有效链
        longest_chain = None
        longest_length = 0
        
        for node in self.nodes.values():
            if len(node.blockchain.chain) > longest_length:
                if node.blockchain.validate_chain():
                    longest_chain = node.blockchain
                    longest_length = len(node.blockchain.chain)
        
        # 同步所有节点
        if longest_chain:
            for node in self.nodes.values():
                node.sync_blockchain(longest_chain)
    
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
        # 注意：这里直接使用updates列表，因为updates已经包含了所有选中客户端的更新
        # 不需要再次遍历所有节点，避免重复复制
        all_updates_list = updates.copy()  # 直接使用已经收集的updates
        print(f"[DEBUG] 待同步的更新数: {len(all_updates_list)}")
        
        # 将所有更新同步到所有节点（创建区块前）
        # 优化：使用浅拷贝即可，因为pending_updates会在创建区块后立即清空
        for node in self.nodes.values():
            node.blockchain.pending_updates = all_updates_list.copy()
        
        # 选择一个验证者节点创建区块（轮询方式由区块链内部管理）
        validator_id = self.validators[0]
        validator_node = self.nodes[validator_id]
        print(f"[DEBUG] 验证者节点 {validator_id} 开始封装区块...")
        print(f"[DEBUG] 验证者节点待处理更新数: {len(validator_node.blockchain.pending_updates)}")
        
        # 由验证者封装区块（create_block会自动清空pending_updates）
        result = validator_node.validate_and_seal_block(aggregated_model)
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

