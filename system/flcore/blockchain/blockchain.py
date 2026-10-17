"""
区块链核心模块
用于FedAvg的去中心化模型更新管理
"""
import hashlib
import json
import time
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import torch


@dataclass
class ModelUpdate:
    """模型更新交易"""
    client_id: int
    round_number: int
    model_params: Dict[str, Any]  # 模型参数字典
    timestamp: float
    previous_hash: str
    gradient_params: Optional[Dict[str, Any]] = None
    sample_count: int = 0
    train_loss: float = 0.0
    hash: str = None
    
    def __post_init__(self):
        """计算交易的哈希值"""
        if self.hash is None:
            self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """计算交易的哈希值"""
        # 将模型参数转换为可哈希的字符串
        params_str = json.dumps(self.model_params, sort_keys=True, default=str)
        grad_str = json.dumps(self.gradient_params, sort_keys=True, default=str) if self.gradient_params else ""
        data_string = f"{self.client_id}{self.round_number}{params_str}{grad_str}{self.sample_count}{self.train_loss}{self.timestamp}{self.previous_hash}"
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


@dataclass
class Block:
    """区块链中的区块"""
    index: int
    timestamp: float
    model_updates: List[ModelUpdate]  # 该区块包含的模型更新
    previous_hash: str
    validator: int = 0  # 验证者ID（用于PoA）
    hash: str = None
    aggregated_model: Optional[Dict[str, Any]] = None  # 聚合后的模型
    # 轻量级模式：只保存元数据，不保存完整更新
    lightweight: bool = False  # 是否为轻量级区块
    update_hashes: Optional[List[str]] = None  # 模型更新的哈希值列表（轻量级模式）
    client_ids: Optional[List[int]] = None  # 参与的客户端ID列表（轻量级模式）
    
    def __post_init__(self):
        """计算区块的哈希值"""
        if self.hash is None:
            self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """计算区块的哈希值"""
        if self.lightweight and self.update_hashes:
            # 轻量级模式：只使用哈希值计算
            updates_str = json.dumps(self.update_hashes, sort_keys=True)
        else:
            # 完整模式：使用完整更新计算
            updates_str = json.dumps([u.to_dict() for u in self.model_updates], sort_keys=True, default=str)
        data_string = f"{self.index}{self.timestamp}{updates_str}{self.previous_hash}{self.validator}"
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def seal_block(self, validator_id: int):
        """PoA验证者签名区块（替代PoW挖矿）"""
        self.validator = validator_id
        self.hash = self.calculate_hash()
    
    def to_lightweight(self):
        """转换为轻量级区块（只保留元数据，释放模型参数）"""
        if not self.lightweight:
            self.update_hashes = [u.hash for u in self.model_updates]
            self.client_ids = [u.client_id for u in self.model_updates]
            self.model_updates = []  # 清空完整更新
            self.lightweight = True
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        result = {
            'index': self.index,
            'timestamp': self.timestamp,
            'previous_hash': self.previous_hash,
            'hash': self.hash,
            'validator': self.validator,
            'aggregated_model': self.aggregated_model,
            'lightweight': self.lightweight
        }
        
        if self.lightweight:
            result['update_hashes'] = self.update_hashes
            result['client_ids'] = self.client_ids
            result['model_updates'] = []  # 轻量级模式不保存
        else:
            result['model_updates'] = [u.to_dict() for u in self.model_updates]
            result['update_hashes'] = None
            result['client_ids'] = None
        
        return result


class Blockchain:
    """区块链类，用于管理FedAvg的模型更新"""
    
    def __init__(self, validators: Optional[List[int]] = None, 
                 lightweight_mode: bool = False,
                 max_blocks: Optional[int] = None,
                 keep_recent_blocks: int = 10,
                 disk_storage: bool = True,
                 storage_dir: str = "blockchain_storage",
                 in_memory_blocks: int = 5,
                 circular_mode: bool = False,
                 circular_length: Optional[int] = None):
        """
        初始化区块链（使用PoA共识）
        :param validators: 授权验证者ID列表
        :param lightweight_mode: 是否启用轻量级模式（创建区块后自动清理详细更新）
        :param max_blocks: 最大区块数（超过此数量会自动修剪旧区块，None表示不限制）
        :param keep_recent_blocks: 修剪时保留的最近区块数量
        :param disk_storage: 是否启用磁盘存储（旧区块保存到磁盘）
        :param storage_dir: 磁盘存储目录
        :param in_memory_blocks: 内存中保留的最近区块数量
        :param circular_mode: 是否启用循环模式（固定长度M+1，新轮次覆盖旧轮次）
        :param circular_length: 循环链长度（M+1），如果为None且circular_mode=True，则使用keep_recent_blocks+1
        """
        self.chain: List[Block] = [self.create_genesis_block()]
        self.validators = validators if validators else [0]  # 默认验证者
        self.pending_updates: List[ModelUpdate] = []  # 待处理的模型更新
        self.current_validator_index = 0  # 轮询验证者索引
        self.lightweight_mode = lightweight_mode  # 轻量级模式
        self.max_blocks = max_blocks  # 最大区块数
        self.keep_recent_blocks = keep_recent_blocks  # 保留的最近区块数
        self.disk_storage = disk_storage  # 磁盘存储模式
        self.storage_dir = storage_dir  # 存储目录
        self.in_memory_blocks = in_memory_blocks  # 内存中保留的区块数
        self.total_blocks = 1  # 总区块数（包括磁盘中的）
        
        # 循环模式配置
        self.circular_mode = circular_mode  # 是否启用循环模式
        if circular_mode:
            # 循环链长度 = M + 1（M个数据区块 + 1个创世区块）
            self.circular_length = circular_length if circular_length is not None else (keep_recent_blocks + 1)
            # 初始化循环链：创世区块 + M个空槽位
            # 注意：第0个是创世区块，第1到M个是数据区块
            while len(self.chain) < self.circular_length:
                # 创建占位区块（稍后会被覆盖）
                placeholder = Block(
                    index=len(self.chain),
                    timestamp=0.0,
                    model_updates=[],
                    previous_hash=self.chain[-1].hash if self.chain else "0",
                    aggregated_model=None
                )
                self.chain.append(placeholder)
            print(f"[Blockchain] 启用循环模式，链长度固定为 {self.circular_length} (1个创世区块 + {self.circular_length-1}个数据区块)")
        else:
            self.circular_length = None
        
        # 创建存储目录
        if self.disk_storage:
            os.makedirs(self.storage_dir, exist_ok=True)
        
    def create_genesis_block(self) -> Block:
        """创建创世区块"""
        genesis = Block(
            index=0,
            timestamp=time.time(),
            model_updates=[],
            previous_hash="0",
            aggregated_model=None
        )
        genesis.hash = genesis.calculate_hash()
        return genesis
    
    def _save_block_to_disk(self, block: Block):
        """将区块保存到磁盘"""
        if not self.disk_storage:
            return
        
        block_file = os.path.join(self.storage_dir, f"block_{block.index}.json")
        with open(block_file, 'w') as f:
            json.dump(block.to_dict(), f, indent=2, default=str)
    
    def _load_block_from_disk(self, block_index: int) -> Optional[Block]:
        """从磁盘加载指定索引的区块"""
        if not self.disk_storage:
            return None
        
        block_file = os.path.join(self.storage_dir, f"block_{block_index}.json")
        if not os.path.exists(block_file):
            return None
        
        try:
            with open(block_file, 'r') as f:
                block_data = json.load(f)
            
            lightweight = block_data.get('lightweight', False)
            model_updates = []
            if not lightweight and block_data.get('model_updates'):
                model_updates = [ModelUpdate(**u) for u in block_data['model_updates']]
            
            block = Block(
                index=block_data['index'],
                timestamp=block_data['timestamp'],
                model_updates=model_updates,
                previous_hash=block_data['previous_hash'],
                hash=block_data['hash'],
                validator=block_data.get('validator', 0),
                aggregated_model=block_data.get('aggregated_model'),
                lightweight=lightweight,
                update_hashes=block_data.get('update_hashes'),
                client_ids=block_data.get('client_ids')
            )
            return block
        except Exception as e:
            print(f"加载区块 {block_index} 失败: {e}")
            return None
    
    def _move_old_blocks_to_disk(self):
        """将旧的区块移动到磁盘，只保留最近的N个在内存中"""
        if not self.disk_storage:
            return
        
        # 如果内存中的区块数超过限制，将最旧的保存到磁盘
        moved_count = 0
        while len(self.chain) > self.in_memory_blocks:
            old_block = self.chain.pop(0)  # 移除最旧的区块
            self._save_block_to_disk(old_block)
            moved_count += 1
        
        if moved_count > 0:
            print(f"[DEBUG] 已将 {moved_count} 个旧区块保存到磁盘 (blockchain_storage/)")
            print(f"[DEBUG] 当前内存中区块数: {len(self.chain)}, 磁盘中区块数: {self.total_blocks - len(self.chain)}")
    
    def get_latest_block(self) -> Block:
        """获取最新的区块"""
        if self.circular_mode:
            # 循环模式：返回最新的数据区块（不是创世区块）
            # 找到最新的非空数据区块
            for i in range(len(self.chain) - 1, 0, -1):  # 从后往前查找，跳过创世区块
                if self.chain[i].timestamp > 0:  # 时间戳大于0表示是有效数据区块
                    return self.chain[i]
            # 如果没有找到有效数据区块，返回创世区块
            return self.chain[0]
        else:
            return self.chain[-1]
    
    def get_block(self, index: int) -> Optional[Block]:
        """获取指定索引的区块（可能从磁盘加载）"""
        # 检查是否在内存中
        for block in self.chain:
            if block.index == index:
                return block
        
        # 尝试从磁盘加载
        if self.disk_storage:
            return self._load_block_from_disk(index)
        
        return None
    
    def add_model_update(self, client_id: int, round_number: int,
                         model_params: Dict[str, Any],
                         gradient_params: Optional[Dict[str, Any]] = None,
                         sample_count: int = 0,
                         train_loss: float = 0.0) -> ModelUpdate:
        """
        添加模型更新交易
        :param client_id: 客户端ID
        :param round_number: 轮次编号
        :param model_params: 模型参数字典
        :return: 创建的ModelUpdate对象
        """
        previous_hash = self.get_latest_block().hash
        update = ModelUpdate(
            client_id=client_id,
            round_number=round_number,
            model_params=model_params,
            gradient_params=gradient_params,
            sample_count=sample_count,
            train_loss=train_loss,
            timestamp=time.time(),
            previous_hash=previous_hash
        )
        self.pending_updates.append(update)
        return update
    
    def create_block(self, aggregated_model: Optional[Dict[str, Any]] = None, validator_id: Optional[int] = None) -> Block:
        """
        创建新区块并添加到链中（使用PoA共识）
        如果启用循环模式，新轮次会覆盖旧轮次（固定长度M+1）
        :param aggregated_model: 聚合后的模型参数
        :param validator_id: 验证者ID（如果为None则轮询选择）
        :return: 新创建的区块
        """
        if not self.pending_updates:
            raise ValueError("没有待处理的模型更新")
        
        # 选择验证者（轮询机制）
        if validator_id is None:
            validator_id = self.validators[self.current_validator_index]
            self.current_validator_index = (self.current_validator_index + 1) % len(self.validators)
        elif validator_id not in self.validators:
            raise ValueError(f"验证者 {validator_id} 未授权")
        
        # 计算实际区块索引和位置
        if self.circular_mode:
            # 循环模式：计算应该覆盖的位置
            # total_blocks 记录总轮次数（包括被覆盖的）
            actual_round = self.total_blocks  # 当前是第几轮（从1开始，0是创世区块）
            # 计算在循环链中的位置：1 + (actual_round - 1) % (circular_length - 1)
            # 第0个位置是创世区块，第1到M个位置是数据区块
            if actual_round == 0:
                # 创世区块，不应该调用create_block
                raise ValueError("创世区块已存在，不应再次创建")
            
            # 计算要覆盖的位置（1到M之间）
            position_in_chain = 1 + ((actual_round - 1) % (self.circular_length - 1))
            
            # 获取前一个区块的哈希（可能是上一个数据区块，或者创世区块）
            if position_in_chain == 1:
                # 覆盖第1个数据区块，前一个区块是创世区块
                previous_hash = self.chain[0].hash
            else:
                # 覆盖其他位置，前一个区块是上一个数据区块
                previous_hash = self.chain[position_in_chain - 1].hash
            
            # 释放旧区块的内存（如果存在）
            old_block = self.chain[position_in_chain]
            if old_block.model_updates:
                for update in old_block.model_updates:
                    if update.model_params:
                        update.model_params.clear()
                    if update.gradient_params:
                        update.gradient_params.clear()
                old_block.model_updates.clear()
            
            # 创建新区块，覆盖旧位置
            block = Block(
                index=actual_round,  # 使用实际轮次作为索引
                timestamp=time.time(),
                model_updates=self.pending_updates.copy(),
                previous_hash=previous_hash,
                validator=validator_id,
                aggregated_model=aggregated_model
            )
            
            # PoA验证者签名
            block.seal_block(validator_id)
            
            # 如果启用轻量级模式，创建后立即转换为轻量级
            if self.lightweight_mode:
                block.to_lightweight()
            
            # 覆盖旧区块
            self.chain[position_in_chain] = block
            self.total_blocks += 1  # 总轮次数+1
            
            print(f"[Blockchain] 循环模式: 第{actual_round}轮覆盖位置{position_in_chain}，链长度保持{self.circular_length}")
            
        else:
            # 普通模式：追加到链尾
            block = Block(
                index=len(self.chain),
                timestamp=time.time(),
                model_updates=self.pending_updates.copy(),
                previous_hash=self.get_latest_block().hash,
                validator=validator_id,
                aggregated_model=aggregated_model
            )
            
            # PoA验证者签名（替代PoW挖矿）
            block.seal_block(validator_id)
            
            # 如果启用轻量级模式，创建后立即转换为轻量级
            if self.lightweight_mode:
                block.to_lightweight()
            
            # 添加到链中
            self.chain.append(block)
            self.total_blocks = len(self.chain)
            
            # 如果启用磁盘存储，将旧区块移动到磁盘
            if self.disk_storage:
                print(f"[DEBUG] 创建区块 {block.index} 完成，开始管理内存...")
                self._move_old_blocks_to_disk()
                # 更新总区块数（包括磁盘中的）
                disk_blocks = len([f for f in os.listdir(self.storage_dir) if f.startswith('block_') and f.endswith('.json')])
                self.total_blocks = len(self.chain) + disk_blocks
                print(f"[DEBUG] 区块管理完成: 内存中={len(self.chain)}, 磁盘中={disk_blocks}, 总计={self.total_blocks}")
        
        # 清空待处理更新并释放内存
        # 先释放每个更新中的模型参数内存
        for update in self.pending_updates:
            if update.model_params:
                # 注意：model_params是字典，值可能是numpy数组或列表
                # 这里主要是确保Python对象引用被释放
                update.model_params.clear()
            if update.gradient_params:
                update.gradient_params.clear()
        
        self.pending_updates = []
        
        # 清理CUDA缓存（如果有GPU）
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        # 如果设置了最大区块数且不是循环模式，检查是否需要修剪
        if not self.circular_mode and self.max_blocks is not None and self.total_blocks > self.max_blocks:
            self.prune_blocks()
        
        return block
    
    def get_aggregated_model(self) -> Optional[Dict[str, Any]]:
        """获取最新的聚合模型"""
        latest_block = self.get_latest_block()
        return latest_block.aggregated_model
    
    def validate_chain(self) -> bool:
        """验证区块链的有效性"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # 验证当前区块的哈希
            if current_block.hash != current_block.calculate_hash():
                return False
            
            # 验证与前一个区块的连接
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True
    
    def get_chain_length(self) -> int:
        """获取区块链长度（包括磁盘中的区块）"""
        if self.circular_mode:
            # 循环模式：返回总轮次数（包括被覆盖的）
            return self.total_blocks
        elif self.disk_storage:
            return self.total_blocks
        else:
            return len(self.chain)
    
    def prune_blocks(self, keep_recent: Optional[int] = None):
        """
        修剪旧区块，只保留最近的N个区块
        :param keep_recent: 保留的最近区块数量（默认使用self.keep_recent_blocks）
        """
        if keep_recent is None:
            keep_recent = self.keep_recent_blocks
        
        # 删除磁盘中的旧区块文件
        if self.disk_storage:
            files_to_delete = []
            for filename in os.listdir(self.storage_dir):
                if filename.startswith('block_') and filename.endswith('.json'):
                    try:
                        block_index = int(filename.replace('block_', '').replace('.json', ''))
                        if block_index > 0 and block_index < (self.total_blocks - keep_recent):
                            files_to_delete.append(os.path.join(self.storage_dir, filename))
                    except ValueError:
                        continue
            
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"删除文件 {file_path} 失败: {e}")
        
        # 如果内存中的区块数超过限制，只保留最近的
        if len(self.chain) > keep_recent:
            blocks_to_keep = max(1, keep_recent - 1)  # 至少保留1个（除了创世区块）
            
            # 先转换旧区块为轻量级模式（如果还没有）
            for block in self.chain[1:-blocks_to_keep]:
                if not block.lightweight:
                    block.to_lightweight()
                # 保存到磁盘（如果启用）
                if self.disk_storage:
                    self._save_block_to_disk(block)
            
            # 重新构建链：保留创世区块和最近的区块
            new_chain = [self.chain[0]]  # 保留创世区块
            new_chain.extend(self.chain[-blocks_to_keep:])  # 保留最近的区块
            
            # 更新索引
            for i, block in enumerate(new_chain):
                block.index = i
            
            # 更新previous_hash
            for i in range(1, len(new_chain)):
                new_chain[i].previous_hash = new_chain[i-1].hash
            
            removed_count = len(self.chain) - len(new_chain)
            self.chain = new_chain
            self.total_blocks = len(new_chain)
            
            print(f"已修剪 {removed_count} 个旧区块，保留 {len(new_chain)} 个区块在内存中")
    
    def convert_to_lightweight(self, block_indices: Optional[List[int]] = None):
        """
        将指定区块转换为轻量级模式
        :param block_indices: 要转换的区块索引列表（None表示转换所有非轻量级区块）
        """
        if block_indices is None:
            # 转换所有非轻量级区块（除了创世区块和最新区块）
            for i in range(1, len(self.chain) - 1):
                if not self.chain[i].lightweight:
                    self.chain[i].to_lightweight()
        else:
            for idx in block_indices:
                if 0 < idx < len(self.chain) and not self.chain[idx].lightweight:
                    self.chain[idx].to_lightweight()
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        获取区块链的内存使用信息
        :return: 内存信息字典
        """
        import sys
        
        total_size = sys.getsizeof(self.chain)
        lightweight_count = sum(1 for b in self.chain if b.lightweight)
        full_count = len(self.chain) - lightweight_count
        
        # 估算每个区块的大小
        block_sizes = []
        for block in self.chain:
            size = sys.getsizeof(block)
            if not block.lightweight:
                size += sum(sys.getsizeof(u) for u in block.model_updates)
            block_sizes.append(size)
        
        # 计算磁盘中的区块数
        disk_blocks = 0
        disk_size = 0
        if self.disk_storage and os.path.exists(self.storage_dir):
            disk_files = [f for f in os.listdir(self.storage_dir) if f.startswith('block_') and f.endswith('.json')]
            disk_blocks = len(disk_files)
            for filename in disk_files:
                file_path = os.path.join(self.storage_dir, filename)
                try:
                    disk_size += os.path.getsize(file_path)
                except:
                    pass
        
        return {
            'total_blocks': self.total_blocks if self.disk_storage else len(self.chain),
            'in_memory_blocks': len(self.chain),
            'disk_blocks': disk_blocks,
            'lightweight_blocks': lightweight_count,
            'full_blocks': full_count,
            'estimated_memory_bytes': sum(block_sizes),
            'disk_size_bytes': disk_size,
            'average_block_size_bytes': sum(block_sizes) / len(block_sizes) if block_sizes else 0
        }
    
    def get_round_updates(self, round_number: int) -> List[ModelUpdate]:
        """获取指定轮次的所有模型更新"""
        updates = []
        for block in self.chain:
            for update in block.model_updates:
                if update.round_number == round_number:
                    updates.append(update)
        return updates
    
    def to_dict(self) -> Dict:
        """将整个区块链转换为字典"""
        return {
            'chain': [block.to_dict() for block in self.chain],
            'pending_updates': [u.to_dict() for u in self.pending_updates],
            'validators': self.validators,
            'current_validator_index': self.current_validator_index,
            'lightweight_mode': self.lightweight_mode,
            'max_blocks': self.max_blocks,
            'keep_recent_blocks': self.keep_recent_blocks
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Blockchain':
        """从字典恢复区块链"""
        # 兼容旧版本（使用difficulty）和新版本（使用validators）
        validators = data.get('validators', [0])
        lightweight_mode = data.get('lightweight_mode', False)
        max_blocks = data.get('max_blocks', None)
        keep_recent_blocks = data.get('keep_recent_blocks', 10)
        
        blockchain = cls(
            validators=validators,
            lightweight_mode=lightweight_mode,
            max_blocks=max_blocks,
            keep_recent_blocks=keep_recent_blocks
        )
        blockchain.chain = []
        
        for block_data in data['chain']:
            lightweight = block_data.get('lightweight', False)
            model_updates = []
            if not lightweight and block_data.get('model_updates'):
                model_updates = [ModelUpdate(**u) for u in block_data['model_updates']]
            
            block = Block(
                index=block_data['index'],
                timestamp=block_data['timestamp'],
                model_updates=model_updates,
                previous_hash=block_data['previous_hash'],
                hash=block_data['hash'],
                validator=block_data.get('validator', 0),
                aggregated_model=block_data.get('aggregated_model'),
                lightweight=lightweight,
                update_hashes=block_data.get('update_hashes'),
                client_ids=block_data.get('client_ids')
            )
            blockchain.chain.append(block)
        
        blockchain.pending_updates = [
            ModelUpdate(**u) for u in data.get('pending_updates', [])
        ]
        blockchain.current_validator_index = data.get('current_validator_index', 0)
        
        return blockchain


def aggregate_model_updates(updates: List[ModelUpdate], client_num: int) -> Dict[str, Any]:
    """
    聚合多个客户端的模型更新（FedAvg算法）
    :param updates: 模型更新列表
    :param client_num: 参与聚合的客户端数量（未使用，保留用于兼容性）
    :return: 聚合后的模型参数字典（numpy数组格式，便于序列化）
    """
    if not updates:
        raise ValueError("没有模型更新可聚合")
    
    # 初始化聚合模型
    first_params = updates[0].model_params
    aggregated = {}
    for key in first_params:
        # 确保参数是numpy数组格式
        if isinstance(first_params[key], list):
            param_array = np.array(first_params[key], dtype=np.float32)
        else:
            param_array = np.array(first_params[key], dtype=np.float32)
        aggregated[key] = np.zeros_like(param_array, dtype=np.float32)
    
    # 加权平均
    weight = 1.0 / len(updates)
    for update in updates:
        for key in aggregated:
            if isinstance(update.model_params[key], list):
                param_array = np.array(update.model_params[key], dtype=np.float32)
            else:
                param_array = np.array(update.model_params[key], dtype=np.float32)
            aggregated[key] += param_array * weight
    
    # 转换为列表以便序列化
    aggregated_list = {}
    for key, value in aggregated.items():
        aggregated_list[key] = value.tolist()
    
    return aggregated_list

