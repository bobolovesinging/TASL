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
    model_params: Optional[Dict[str, Any]] = None # Deprecated: 只在内存中使用，不序列化到链上
    timestamp: float = 0.0
    previous_hash: str = ""
    model_cid: str = None  # New: IPFS CID
    hash: str = None
    
    def __post_init__(self):
        """计算交易的哈希值"""
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.hash is None:
            self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """计算交易的哈希值 (仅基于CID)"""
        if self.model_cid:
            # 轻量级模式：只使用CID计算哈希
            data_string = f"{self.client_id}{self.round_number}{self.model_cid}{self.timestamp}{self.previous_hash}"
        else:
            # 兼容旧模式（但不推荐）
            params_str = json.dumps(self.model_params, sort_keys=True, default=str) if self.model_params else ""
            data_string = f"{self.client_id}{self.round_number}{params_str}{self.timestamp}{self.previous_hash}"
            
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """转换为字典 (链上存储格式)"""
        d = asdict(self)
        # 移除 model_params，只保留 CID
        if 'model_params' in d:
            del d['model_params']
        return d


@dataclass
class Block:
    """区块链中的区块"""
    index: int
    timestamp: float
    model_updates: List[ModelUpdate]  # 该区块包含的模型更新
    previous_hash: str
    validator: int = 0  # 验证者ID
    hash: str = None
    aggregated_model_cid: Optional[str] = None  # New: 聚合模型的 CID
    aggregated_model: Optional[Dict[str, Any]] = None  # Deprecated: 仅内存使用
    round_number: Optional[int] = None
    rewards: Optional[Dict[int, float]] = None
    similarities: Optional[Dict[int, float]] = None
    scores: Optional[Dict[int, float]] = None
    lightweight: bool = True  # 默认为 True
    
    def __post_init__(self):
        if self.hash is None:
            self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """计算区块哈希"""
        # 序列化 updates (利用 ModelUpdate.to_dict 自动忽略 model_params)
        updates_str = json.dumps([u.to_dict() for u in self.model_updates], sort_keys=True, default=str)
        
        rewards_str = json.dumps(self.rewards, sort_keys=True) if self.rewards else ""
        similarities_str = json.dumps(self.similarities, sort_keys=True) if self.similarities else ""
        scores_str = json.dumps(self.scores, sort_keys=True) if self.scores else ""
        round_str = str(self.round_number) if self.round_number is not None else ""
        agg_cid = self.aggregated_model_cid if self.aggregated_model_cid else ""
        
        data_string = f"{self.index}{self.timestamp}{updates_str}{self.previous_hash}{self.validator}{round_str}{rewards_str}{similarities_str}{scores_str}{agg_cid}"
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def seal_block(self, validator_id: int):
        self.validator = validator_id
        self.hash = self.calculate_hash()
    
    def to_lightweight(self):
        """已经是轻量级了，保留此方法兼容接口"""
        self.lightweight = True
        # 清理内存中的重型数据 (如果有)
        self.aggregated_model = None
        for update in self.model_updates:
            update.model_params = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        if 'aggregated_model' in result:
            del result['aggregated_model'] # 不存储具体的聚合模型参数
        # model_updates 里的 model_params 已经在 ModelUpdate.to_dict 中移除
        result['model_updates'] = [u.to_dict() for u in self.model_updates]
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
            # 检查磁盘上是否已存在创世块
            # 循环模式下，创世块使用position_in_chain=0
            position = 0 if self.circular_mode else None
            genesis_block = self._load_block_from_disk(0, position_in_chain=position)
            if genesis_block is not None:
                # 如果磁盘上存在创世块，使用它替换新创建的创世块
                self.chain[0] = genesis_block
                print(f"[Blockchain] 从磁盘加载创世块: {self.storage_dir}/block_0.json")
            else:
                # 如果磁盘上不存在，保存新创建的创世块到磁盘
                self._save_block_to_disk(self.chain[0], position_in_chain=0)
                print(f"[Blockchain] 创建新创世块并保存到磁盘")
        
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
    
    def _save_block_to_disk(self, block: Block, position_in_chain: Optional[int] = None):
        """
        将区块保存到磁盘
        :param block: 要保存的区块
        :param position_in_chain: 在链中的位置（循环模式下使用，None则使用block.index）
        """
        if not self.disk_storage:
            return
        
        # 循环模式下，使用位置索引保存（确保只有M个数据区块文件）
        # 普通模式下，使用block.index
        if self.circular_mode and position_in_chain is not None:
            # 创世块使用index 0，数据区块使用位置索引（1到M）
            if position_in_chain == 0:
                block_file = os.path.join(self.storage_dir, f"block_0.json")
            else:
                block_file = os.path.join(self.storage_dir, f"block_{position_in_chain}.json")
        else:
            block_file = os.path.join(self.storage_dir, f"block_{block.index}.json")
        
        with open(block_file, 'w') as f:
            json.dump(block.to_dict(), f, indent=2, default=str)
    
    def _load_block_from_disk(self, block_index: int, position_in_chain: Optional[int] = None) -> Optional[Block]:
        """
        从磁盘加载指定索引的区块
        :param block_index: 区块索引（用于普通模式）
        :param position_in_chain: 在链中的位置（用于循环模式，None则使用block_index）
        """
        if not self.disk_storage:
            return None
        
        # 循环模式下，使用位置索引加载
        if self.circular_mode and position_in_chain is not None:
            if position_in_chain == 0:
                block_file = os.path.join(self.storage_dir, f"block_0.json")
            else:
                block_file = os.path.join(self.storage_dir, f"block_{position_in_chain}.json")
        else:
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
                round_number=block_data.get('round_number'),
                rewards=block_data.get('rewards'),
                similarities=block_data.get('similarities'),
                scores=block_data.get('scores'),
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
            # 循环模式下，使用区块在链中的原始位置（index）
            position = old_block.index if self.circular_mode else None
            self._save_block_to_disk(old_block, position_in_chain=position)
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
            # 循环模式下，如果是创世块，使用position_in_chain=0
            position = 0 if (self.circular_mode and index == 0) else None
            loaded_block = self._load_block_from_disk(index, position_in_chain=position)
            # 如果是创世块（index=0），加载后将其放回内存
            if loaded_block is not None and index == 0:
                self.chain[0] = loaded_block
                print(f"[Blockchain] 从磁盘加载创世块到内存")
            return loaded_block
        
        return None
    
    def add_model_update(self, client_id: int, round_number: int, model_params: Dict[str, Any]) -> ModelUpdate:
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
            timestamp=time.time(),
            previous_hash=previous_hash
        )
        self.pending_updates.append(update)
        return update
    
    def create_block(self, aggregated_model: Optional[Dict[str, Any]] = None, validator_id: Optional[int] = None,
                     round_number: Optional[int] = None, rewards: Optional[Dict[int, float]] = None,
                     similarities: Optional[Dict[int, float]] = None, scores: Optional[Dict[int, float]] = None) -> Block:
        """
        创建新区块并添加到链中（使用PoA共识）
        如果启用循环模式，新轮次会覆盖旧轮次（固定长度M+1）
        :param aggregated_model: 聚合后的模型参数
        :param validator_id: 验证者ID（如果为None则轮询选择）
        :param round_number: 轮次编号
        :param rewards: 客户端奖赏信息 {client_id: reward_score}
        :param similarities: 客户端相似度 {client_id: similarity}
        :param scores: 客户端得分 {client_id: score}
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
            # 优先使用传入的round_number，如果没有则使用total_blocks
            if round_number is not None:
                actual_round = round_number  # 使用传入的轮次编号
            else:
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
            
            # 释放旧区块的内存（如果存在）- 清理所有大对象
            old_block = self.chain[position_in_chain]
            if old_block.model_updates:
                for update in old_block.model_updates:
                    if update.model_params:
                        # 清理模型参数字典中的所有值
                        for key, value in update.model_params.items():
                            if isinstance(value, (list, np.ndarray)):
                                del value
                        update.model_params.clear()
                old_block.model_updates.clear()
            # 清理aggregated_model等大对象（重要：防止内存泄漏）
            if old_block.aggregated_model:
                for key, value in old_block.aggregated_model.items():
                    if isinstance(value, (list, np.ndarray)):
                        del value
                old_block.aggregated_model = None
            # 清理其他大对象
            old_block.rewards = None
            old_block.similarities = None
            old_block.scores = None
            
            # 创建新区块，覆盖旧位置
            block = Block(
                index=actual_round,  # 使用实际轮次作为索引
                timestamp=time.time(),
                model_updates=self.pending_updates.copy(),
                previous_hash=previous_hash,
                validator=validator_id,
                aggregated_model=aggregated_model,
                round_number=round_number,
                rewards=rewards,
                similarities=similarities,
                scores=scores
            )
            
            # PoA验证者签名
            block.seal_block(validator_id)
            
            # 如果启用轻量级模式，创建后立即转换为轻量级
            if self.lightweight_mode:
                block.to_lightweight()
            
            # 覆盖旧区块
            self.chain[position_in_chain] = block
            # 更新total_blocks：如果传入round_number，使用round_number+1；否则递增
            if round_number is not None:
                self.total_blocks = max(self.total_blocks, round_number + 1)
            else:
                self.total_blocks += 1  # 总轮次数+1
            
            # 如果启用磁盘存储，保存区块到磁盘（使用位置索引）
            if self.disk_storage:
                self._save_block_to_disk(block, position_in_chain=position_in_chain)
                # 保存到磁盘后，如果启用轻量级模式，清理内存中的大对象（但保留aggregated_model用于读取）
                if self.lightweight_mode and block.aggregated_model:
                    # 注意：aggregated_model需要保留在内存中，因为可能被读取
                    # 但如果内存压力大，可以考虑只保留在磁盘上
                    pass
                print(f"[Blockchain] 循环模式: 第{actual_round}轮覆盖位置{position_in_chain}，已保存到磁盘 block_{position_in_chain}.json，链长度保持{self.circular_length}")
            else:
                print(f"[Blockchain] 循环模式: 第{actual_round}轮覆盖位置{position_in_chain}，链长度保持{self.circular_length}")
            
        else:
            # 普通模式：追加到链尾
            block = Block(
                index=len(self.chain),
                timestamp=time.time(),
                model_updates=self.pending_updates.copy(),
                previous_hash=self.get_latest_block().hash,
                validator=validator_id,
                aggregated_model=aggregated_model,
                round_number=round_number,
                rewards=rewards,
                similarities=similarities,
                scores=scores
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
                # 清理字典中的所有值
                for key, value in update.model_params.items():
                    if isinstance(value, (list, np.ndarray)):
                        del value
                update.model_params.clear()
        
        self.pending_updates = []
        
        # 如果启用轻量级模式，清理内存中旧区块的大对象（循环模式下已清理）
        if self.lightweight_mode and not self.circular_mode:
            # 对于普通模式，清理已保存到磁盘的旧区块的大对象
            for block in self.chain[:-self.in_memory_blocks]:  # 除了最近N个区块
                if block.aggregated_model and not block.lightweight:
                    # 如果区块已保存到磁盘，清理aggregated_model（可以从磁盘重新加载）
                    for key, value in block.aggregated_model.items():
                        if isinstance(value, (list, np.ndarray)):
                            del value
                    block.aggregated_model = None
        
        # 清理CUDA缓存（如果有GPU）
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        # 如果设置了最大区块数且不是循环模式，检查是否需要修剪
        if not self.circular_mode and self.max_blocks is not None and self.total_blocks > self.max_blocks:
            self.prune_blocks()
        
        return block
    
    def get_aggregated_model(self) -> Optional[Dict[str, Any]]:
        """获取最新的聚合模型（从创世块读取全局模型）"""
        # 根据用户需求，全局模型保存在创世块（block_0）中
        genesis_block = self.get_block(0)  # 获取创世块
        if genesis_block and genesis_block.aggregated_model is not None:
            return genesis_block.aggregated_model
        
        # 如果创世块中没有全局模型，尝试从最新区块获取（向后兼容）
        latest_block = self.get_latest_block()
        if latest_block and latest_block.aggregated_model is not None:
            return latest_block.aggregated_model
        
        return None
    
    def update_genesis_block_model(self, aggregated_model: Dict[str, Any]):
        """
        更新创世块的全局模型参数
        :param aggregated_model: 聚合后的模型参数字典
        """
        # 确保创世块在内存中（如果不在，从磁盘加载）
        self._ensure_genesis_block_in_memory()
        
        genesis_block = self.chain[0]  # 直接访问内存中的创世块
        if genesis_block is None or genesis_block.index != 0:
            raise ValueError("创世块不存在或索引错误")
        
        # 更新创世块的全局模型
        genesis_block.aggregated_model = aggregated_model
        
        # 重新计算哈希（因为aggregated_model改变了）
        genesis_block.hash = genesis_block.calculate_hash()
        
        # 如果启用磁盘存储，保存创世块到磁盘
        if self.disk_storage:
            self._save_block_to_disk(genesis_block, position_in_chain=0)
        
        print(f"[Blockchain] 创世块的全局模型已更新并保存到磁盘")
    
    def _ensure_genesis_block_in_memory(self):
        """
        确保创世块在内存中，如果不在则从磁盘加载
        """
        # 检查内存中的第一个区块是否是创世块
        if len(self.chain) > 0 and self.chain[0].index == 0:
            return  # 创世块已在内存中
        
        # 如果不在内存中，尝试从磁盘加载
        if self.disk_storage:
            # 循环模式下，创世块使用position_in_chain=0
            position = 0 if self.circular_mode else None
            genesis_block = self._load_block_from_disk(0, position_in_chain=position)
            if genesis_block is not None:
                # 如果链为空，直接添加；否则替换第一个区块
                if len(self.chain) == 0:
                    self.chain.append(genesis_block)
                else:
                    self.chain[0] = genesis_block
                print(f"[Blockchain] 从磁盘加载创世块到内存")
            else:
                # 如果磁盘上也没有，创建新的创世块
                genesis_block = self.create_genesis_block()
                if len(self.chain) == 0:
                    self.chain.append(genesis_block)
                else:
                    self.chain[0] = genesis_block
                if self.disk_storage:
                    self._save_block_to_disk(genesis_block, position_in_chain=0)
                    print(f"[Blockchain] 创建新创世块")
    
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
            for i, block in enumerate(self.chain[1:-blocks_to_keep], start=1):
                if not block.lightweight:
                    block.to_lightweight()
                # 保存到磁盘（如果启用）
                if self.disk_storage:
                    self._save_block_to_disk(block, position_in_chain=i if self.circular_mode else None)
            
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
                round_number=block_data.get('round_number'),
                rewards=block_data.get('rewards'),
                similarities=block_data.get('similarities'),
                scores=block_data.get('scores'),
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
        return {}
    
    # [Mod] 过滤掉 model_params 为 None 的更新 (IPFS 模式)
    valid_updates = [u for u in updates if u.model_params is not None]
    if not valid_updates:
        # 如果所有更新都没有参数(都是CID)，且没有被预先解析，则返回空
        #这会提示调用者必须先解析CID
        print("[Warning] aggregate_model_updates received updates without model_params. Ensure CIDs are resolved.")
        return {}
            
    # 初始化聚合模型
    first_params = valid_updates[0].model_params
    aggregated = {}
    for key in first_params:
        # 确保参数是numpy数组格式
        if isinstance(first_params[key], list):
            param_array = np.array(first_params[key], dtype=np.float32)
        else:
            param_array = np.array(first_params[key], dtype=np.float32)
        aggregated[key] = np.zeros_like(param_array, dtype=np.float32)
    
    # 加权平均
    weight = 1.0 / len(valid_updates)
    for update in valid_updates:
        for key in aggregated:
            if key not in update.model_params: continue # 安全检查
            
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

