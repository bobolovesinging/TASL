"""
DPOS共识机制模块
实现Delegated Proof of Stake共识算法
"""
import time
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from flcore.blockchain.block import Block, Blockchain, BlockHeader
from flcore.blockchain.stake_manager import StakeManager


class DPOSConsensus:
    """DPOS共识机制"""
    
    def __init__(self, blockchain: Blockchain, stake_manager: StakeManager, 
                 num_deputies: int, block_time: float = 5.0):
        """
        初始化DPOS共识
        :param blockchain: 区块链实例
        :param stake_manager: 权益管理器
        :param num_deputies: deputy节点数量
        :param block_time: 出块时间间隔（秒）
        """
        self.blockchain = blockchain
        self.stake_manager = stake_manager
        self.num_deputies = num_deputies
        self.block_time = block_time
        self.deputy_schedule: Dict[int, int] = {}  # {round: deputy_id} 出块调度
        self.deputy_performance: Dict[int, Dict[str, Any]] = {}  # {deputy_id: {stats}}
    
    def initialize_blockchain(self):
        """初始化区块链（创建创世区块）"""
        if self.blockchain.get_chain_length() == 0:
            genesis_block = self.blockchain.create_genesis_block()
            self.blockchain.add_block(genesis_block)
            print("[DPOS] 已创建创世区块")
    
    def schedule_deputy(self, round_number: int, deputies: List[int]) -> Optional[int]:
        """
        调度deputy出块（轮流出块）
        :param round_number: 轮次编号
        :param deputies: deputy节点列表
        :return: 应该出块的deputy ID，如果调度失败返回None
        """
        if not deputies:
            return None
        
        # 轮流出块：根据round_number选择deputy
        deputy_index = round_number % len(deputies)
        scheduled_deputy = deputies[deputy_index]
        self.deputy_schedule[round_number] = scheduled_deputy
        
        print(f"[DPOS] Round {round_number}: 调度deputy {scheduled_deputy} 出块")
        return scheduled_deputy
    
    def create_block_by_deputy(self, deputy_id: int, round_number: int,
                               transactions: List[Dict[str, Any]],
                               signature: Optional[str] = None) -> Optional[Block]:
        """
        Deputy节点创建区块
        :param deputy_id: deputy节点ID
        :param round_number: 轮次编号
        :param transactions: 交易列表（梯度提交）
        :param signature: 区块签名
        :return: 创建的区块，如果失败返回None
        """
        # 检查是否有出块权限
        scheduled_deputy = self.deputy_schedule.get(round_number)
        if scheduled_deputy != deputy_id:
            print(f"[DPOS] 警告：deputy {deputy_id} 没有Round {round_number}的出块权限（应该是deputy {scheduled_deputy}）")
            return None
        
        # 创建区块
        block = self.blockchain.create_block(
            deputy_id=deputy_id,
            round_number=round_number,
            transactions=transactions
        )
        
        # 添加签名
        if signature:
            block.signature = signature
            block.hash = block._calculate_hash()  # 重新计算哈希
        
        # 验证区块
        previous_block = self.blockchain.get_latest_block()
        if not block.verify(previous_block):
            print(f"[DPOS] 警告：deputy {deputy_id} 创建的区块验证失败")
            return None
        
        # 添加到链
        if self.blockchain.add_block(block):
            # 记录deputy表现
            self._record_deputy_performance(deputy_id, round_number, success=True)
            print(f"[DPOS] Deputy {deputy_id} 成功创建并添加区块（Round {round_number}）")
            return block
        else:
            self._record_deputy_performance(deputy_id, round_number, success=False)
            return None
    
    def verify_block(self, block: Block, previous_block: Optional[Block] = None) -> Tuple[bool, str]:
        """
        验证区块（可以由其他节点调用）
        :param block: 要验证的区块
        :param previous_block: 前一个区块（如果为None，从链中获取）
        :return: (是否有效, 错误信息)
        """
        if previous_block is None:
            previous_block = self.blockchain.get_latest_block()
        
        # 验证区块结构
        if not block.verify(previous_block):
            return False, "区块验证失败：结构或哈希不正确"
        
        # 验证deputy是否有出块权限
        scheduled_deputy = self.deputy_schedule.get(block.header.round_number)
        if scheduled_deputy is None:
            return False, f"Round {block.header.round_number} 没有调度deputy"
        
        if block.header.deputy_id != scheduled_deputy:
            return False, f"Deputy {block.header.deputy_id} 没有Round {block.header.round_number}的出块权限"
        
        # 验证时间戳（可选：检查是否在合理范围内）
        current_time = time.time()
        time_diff = abs(current_time - block.header.timestamp)
        if time_diff > 3600:  # 1小时内的区块才认为是有效的
            return False, f"区块时间戳异常：时间差 {time_diff:.2f} 秒"
        
        return True, ""
    
    def _record_deputy_performance(self, deputy_id: int, round_number: int, success: bool):
        """记录deputy的表现"""
        if deputy_id not in self.deputy_performance:
            self.deputy_performance[deputy_id] = {
                "total_blocks": 0,
                "successful_blocks": 0,
                "failed_blocks": 0,
                "last_round": -1
            }
        
        stats = self.deputy_performance[deputy_id]
        stats["total_blocks"] += 1
        if success:
            stats["successful_blocks"] += 1
        else:
            stats["failed_blocks"] += 1
        stats["last_round"] = round_number
    
    def get_deputy_performance(self, deputy_id: int) -> Dict[str, Any]:
        """获取deputy的表现统计"""
        return self.deputy_performance.get(deputy_id, {
            "total_blocks": 0,
            "successful_blocks": 0,
            "failed_blocks": 0,
            "last_round": -1
        })
    
    def get_deputy_success_rate(self, deputy_id: int) -> float:
        """获取deputy的成功率"""
        stats = self.get_deputy_performance(deputy_id)
        if stats["total_blocks"] == 0:
            return 0.0
        return stats["successful_blocks"] / stats["total_blocks"]
    
    def should_slash_deputy(self, deputy_id: int, failure_threshold: float = 0.5) -> bool:
        """
        判断是否应该对deputy进行Slashing惩罚
        :param deputy_id: deputy节点ID
        :param failure_threshold: 失败率阈值（超过此值应该惩罚）
        :return: 是否应该惩罚
        """
        success_rate = self.get_deputy_success_rate(deputy_id)
        failure_rate = 1.0 - success_rate
        return failure_rate > failure_threshold

