"""
区块结构模块
实现DPOS区块链的区块和区块链结构
"""
import hashlib
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import torch


@dataclass
class BlockHeader:
    """区块头"""
    version: int = 1  # 区块版本
    previous_hash: str = ""  # 前一个区块的哈希
    merkle_root: str = ""  # Merkle树根哈希
    timestamp: float = 0.0  # 时间戳
    round_number: int = 0  # 轮次编号
    deputy_id: int = -1  # 出块的deputy节点ID
    nonce: int = 0  # 随机数（用于工作量证明，DPOS中可简化）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def to_string(self) -> str:
        """转换为字符串（用于哈希计算）"""
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass
class Block:
    """区块结构"""
    header: BlockHeader
    transactions: List[Dict[str, Any]]  # 交易列表（梯度提交）
    signature: Optional[str] = None  # 区块签名（deputy节点的签名）
    
    def __post_init__(self):
        """初始化后计算Merkle根和区块哈希"""
        if not self.header.merkle_root:
            self.header.merkle_root = self._calculate_merkle_root()
        self.hash = self._calculate_hash()
    
    def _calculate_merkle_root(self) -> str:
        """计算Merkle树根哈希"""
        if not self.transactions:
            return hashlib.sha256(b"").hexdigest()
        
        # 简化版Merkle树：将所有交易哈希后拼接
        tx_hashes = []
        for tx in self.transactions:
            tx_str = json.dumps(tx, sort_keys=True)
            tx_hash = hashlib.sha256(tx_str.encode()).hexdigest()
            tx_hashes.append(tx_hash)
        
        # 如果只有一个交易，直接返回其哈希
        if len(tx_hashes) == 1:
            return tx_hashes[0]
        
        # 递归计算Merkle根
        while len(tx_hashes) > 1:
            new_hashes = []
            for i in range(0, len(tx_hashes), 2):
                if i + 1 < len(tx_hashes):
                    combined = tx_hashes[i] + tx_hashes[i + 1]
                else:
                    combined = tx_hashes[i] + tx_hashes[i]  # 奇数个时复制最后一个
                new_hash = hashlib.sha256(combined.encode()).hexdigest()
                new_hashes.append(new_hash)
            tx_hashes = new_hashes
        
        return tx_hashes[0]
    
    def _calculate_hash(self) -> str:
        """计算区块哈希"""
        header_str = self.header.to_string()
        transactions_str = json.dumps(self.transactions, sort_keys=True)
        combined = header_str + transactions_str
        if self.signature:
            combined += self.signature
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "header": self.header.to_dict(),
            "transactions": self.transactions,
            "signature": self.signature,
            "hash": self.hash
        }
    
    def verify(self, previous_block: Optional['Block'] = None) -> bool:
        """
        验证区块的有效性
        :param previous_block: 前一个区块（用于验证previous_hash）
        :return: 是否有效
        """
        # 验证previous_hash
        if previous_block:
            if self.header.previous_hash != previous_block.hash:
                return False
        
        # 验证Merkle根
        calculated_merkle = self._calculate_merkle_root()
        if self.header.merkle_root != calculated_merkle:
            return False
        
        # 验证区块哈希
        calculated_hash = self._calculate_hash()
        if self.hash != calculated_hash:
            return False
        
        return True


class Blockchain:
    """区块链结构（DPOS）"""
    
    def __init__(self):
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict[str, Any]] = []  # 待处理的交易（梯度提交）
        self.difficulty: int = 1  # 难度（DPOS中可设为1，因为不需要挖矿）
    
    def create_genesis_block(self) -> Block:
        """创建创世区块"""
        header = BlockHeader(
            version=1,
            previous_hash="0" * 64,  # 创世区块没有前一个区块
            timestamp=time.time(),
            round_number=0,
            deputy_id=-1  # 创世区块没有deputy
        )
        genesis_block = Block(
            header=header,
            transactions=[],
            signature=None
        )
        return genesis_block
    
    def add_block(self, block: Block) -> bool:
        """
        添加区块到链上
        :param block: 要添加的区块
        :return: 是否成功添加
        """
        # 验证区块
        previous_block = self.get_latest_block()
        if not block.verify(previous_block):
            return False
        
        # 添加到链
        self.chain.append(block)
        
        # 清理旧区块，只保留最近两轮
        self._prune_old_blocks()
        
        return True
    
    def create_block(self, deputy_id: int, round_number: int, 
                    transactions: Optional[List[Dict[str, Any]]] = None) -> Block:
        """
        创建新区块（由deputy节点调用）
        :param deputy_id: 出块的deputy节点ID
        :param round_number: 轮次编号
        :param transactions: 交易列表（如果为None，使用pending_transactions）
        :return: 新创建的区块
        """
        # 获取前一个区块
        previous_block = self.get_latest_block()
        previous_hash = previous_block.hash if previous_block else "0" * 64
        
        # 使用提供的交易或pending交易
        if transactions is None:
            transactions = self.pending_transactions.copy()
            self.pending_transactions.clear()  # 清空pending交易
        
        # 创建区块头
        header = BlockHeader(
            version=1,
            previous_hash=previous_hash,
            timestamp=time.time(),
            round_number=round_number,
            deputy_id=deputy_id
        )
        
        # 创建区块
        block = Block(
            header=header,
            transactions=transactions,
            signature=None  # 签名由deputy节点添加
        )
        
        return block
    
    def add_transaction(self, transaction: Dict[str, Any]):
        """添加交易到pending列表"""
        self.pending_transactions.append(transaction)
    
    def get_latest_block(self) -> Optional[Block]:
        """获取最新的区块"""
        if not self.chain:
            return None
        return self.chain[-1]
    
    def get_block_by_hash(self, block_hash: str) -> Optional[Block]:
        """根据哈希获取区块"""
        for block in self.chain:
            if block.hash == block_hash:
                return block
        return None
    
    def get_block_by_round(self, round_number: int) -> Optional[Block]:
        """根据轮次获取区块"""
        for block in self.chain:
            if block.header.round_number == round_number:
                return block
        return None
    
    def verify_chain(self) -> bool:
        """验证整个链的有效性"""
        if not self.chain:
            return True
        
        # 验证创世区块
        if self.chain[0].header.previous_hash != "0" * 64:
            return False
        
        # 验证每个区块
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            if not current_block.verify(previous_block):
                return False
        
        return True
    
    def get_chain_length(self) -> int:
        """获取链的长度"""
        return len(self.chain)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "chain": [block.to_dict() for block in self.chain],
            "pending_transactions": self.pending_transactions,
            "difficulty": self.difficulty
        }
    
    def get_transactions_by_round(self, round_number: int) -> List[Dict[str, Any]]:
        """获取指定轮次的所有交易"""
        block = self.get_block_by_round(round_number)
        if block:
            return block.transactions
        return []
    
    def _prune_old_blocks(self, keep_recent_rounds: int = 2):
        """
        清理旧区块，只保留最近两轮的区块（保留创世区块）
        :param keep_recent_rounds: 保留的最近轮次数（默认2轮）
        """
        if len(self.chain) <= 1:
            # 只有创世区块或没有区块，不需要清理
            return
        
        # 获取当前最新区块的轮次
        latest_block = self.chain[-1]
        current_round = latest_block.header.round_number
        
        # 如果当前轮次 <= 1，不需要清理（只有创世区块和第一轮）
        if current_round <= 1:
            return
        
        # 计算要保留的最小轮次
        min_round_to_keep = max(1, current_round - keep_recent_rounds + 1)
        
        # 保留的区块：创世区块（round_number=0）和最近两轮的区块
        blocks_to_keep = []
        removed_count = 0
        
        for block in self.chain:
            round_num = block.header.round_number
            # 保留创世区块或最近两轮的区块
            if round_num == 0 or round_num >= min_round_to_keep:
                blocks_to_keep.append(block)
            else:
                removed_count += 1
        
        # 更新链
        if removed_count > 0:
            # 按轮次排序保留的区块
            blocks_to_keep.sort(key=lambda b: b.header.round_number)
            self.chain = blocks_to_keep
            
            # 重新连接链：修复previous_hash以确保链的连续性
            if len(self.chain) > 1:
                # 修复previous_hash：每个区块指向前一个区块
                for i in range(1, len(self.chain)):
                    current_block = self.chain[i]
                    previous_block = self.chain[i - 1]
                    
                    # 如果previous_hash不匹配，修复它
                    if current_block.header.previous_hash != previous_block.hash:
                        current_block.header.previous_hash = previous_block.hash
                        current_block.hash = current_block._calculate_hash()
            
            print(f"[Blockchain] 已清理旧区块：删除 {removed_count} 个区块，保留 {len(self.chain)} 个区块（创世区块 + 最近{keep_recent_rounds}轮）")

