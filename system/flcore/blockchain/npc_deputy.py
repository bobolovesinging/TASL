"""
NPC Deputy管理模块
负责选举和管理NPC deputy节点，这些节点负责管理区块链账单和聚合
"""
import copy
import random
import math
from typing import List, Dict, Optional, Set, Tuple, Any
import torch


class NPCDeputyManager:
    """NPC Deputy管理器：选举和管理deputy节点"""
    
    def __init__(self, num_deputies: int, stake_manager, keep_recent_rounds: Optional[int] = None):
        """
        初始化NPC Deputy管理器（DPOS超级节点，采用FIFO轮换）
        :param num_deputies: deputy节点数量（M）
        :param stake_manager: 权益管理器实例
        :param keep_recent_rounds: 内存中保留的最近轮次数（默认M轮，即num_deputies）
        """
        self.num_deputies = num_deputies
        self.stake_manager = stake_manager
        self.current_deputies: List[int] = []
        self.deputy_history: Dict[int, List[int]] = {}  # {round: [deputy_ids]}
        # 默认只保留最近M轮数据（M = num_deputies）
        self.keep_recent_rounds = keep_recent_rounds if keep_recent_rounds is not None else num_deputies
        self.deputy_validation_failures: Dict[int, int] = {}  # {deputy_id: failure_count}
        self.transaction_votes: Dict[int, Dict[int, bool]] = {}  # {round: {deputy_id: vote_result}}
        self.pending_transactions: Dict[int, Dict[str, Any]] = {}  # {round: transaction_data}
        self.executor_deputy_history: List[int] = []  # 执行deputy历史记录（FIFO轮换）
    
    def elect_deputies(self, round_number: int) -> List[int]:
        """
        选举NPC deputy（基于权益）
        :param round_number: 当前轮次
        :return: deputy节点ID列表
        """
        # 选择权益最高的M个客户端作为deputy
        deputies = self.stake_manager.get_top_stake_clients(self.num_deputies)
        self.current_deputies = deputies
        self.deputy_history[round_number] = copy.deepcopy(deputies)
        
        print(f"[NPCDeputy] Round {round_number}: 选举出 {len(deputies)} 个NPC deputy: {deputies}")
        return deputies
    
    def is_deputy(self, client_id: int) -> bool:
        """判断客户端是否为当前deputy"""
        return client_id in self.current_deputies
    
    def get_deputies(self) -> List[int]:
        """获取当前deputy列表"""
        return copy.deepcopy(self.current_deputies)
    
    def update_deputies_from_scores(self, round_number: int, top_k: Optional[int] = None, 
                                   performance_weight: float = 0.6, stake_weight: float = 0.4):
        """
        根据此轮表现和权益综合更新下一轮的deputy（采用FIFO轮换模式）
        :param round_number: 当前轮次
        :param top_k: 选择前k个客户端（默认使用num_deputies）
        :param performance_weight: 表现权重（相似度/得分）
        :param stake_weight: 权益权重
        """
        if top_k is None:
            top_k = self.num_deputies
        
        # 获取当前轮次的得分
        round_scores = self.stake_manager.round_scores.get(round_number, {})
        if not round_scores:
            # 如果没有得分，回退到只根据权益
            top_clients = self.stake_manager.get_top_stake_clients(top_k)
            next_round = round_number + 1
            self.current_deputies = top_clients
            self.deputy_history[next_round] = copy.deepcopy(top_clients)
            print(f"[NPCDeputy] Round {round_number} -> {next_round}: 根据权益更新deputy: {top_clients}")
            return
        
        # 综合得分 = performance_weight * 归一化表现 + stake_weight * 归一化权益
        all_stakes = self.stake_manager.get_all_stakes()
        max_stake = max(all_stakes.values()) if all_stakes.values() else 1.0
        max_score = max(round_scores.values()) if round_scores.values() else 1.0
        
        combined_scores = {}
        for client_id in range(self.stake_manager.num_clients):
            # 归一化表现（得分）
            normalized_performance = round_scores.get(client_id, 0.0) / max_score if max_score > 0 else 0.0
            # 归一化权益
            normalized_stake = all_stakes.get(client_id, 0.0) / max_stake if max_stake > 0 else 0.0
            # 综合得分（表现权重更高）
            combined_score = performance_weight * normalized_performance + stake_weight * normalized_stake
            combined_scores[client_id] = combined_score
        
        # 根据综合得分排序
        sorted_by_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_clients = [cid for cid, _ in sorted_by_combined[:top_k * 2]]  # 选择前2k个作为候选
        
        # FIFO轮换：纯先进先出模式，不检查任期
        current_deputies = self.current_deputies.copy()
        
        if not current_deputies:
            # 如果没有当前deputy，直接选择得分最高的
            next_deputies = [cid for cid, _ in sorted_by_combined[:top_k]]
        else:
            # FIFO：保留至少一半的上一轮deputy（按顺序保留，先进先出）
            min_keep = math.ceil(len(current_deputies) / 2)  # 至少保留一半（向上取整）
            keep_count = min(min_keep, len(current_deputies))
            
            if keep_count > 0:
                # FIFO：按顺序保留前面的deputy（先进先出）
                keep_deputies = current_deputies[:keep_count]
            else:
                keep_deputies = []
            
            # 从候选列表中选择新的deputy（排除已保留的）
            excluded = set(keep_deputies)
            remaining_candidates = [cid for cid in candidate_clients if cid not in excluded]
            need_new = top_k - len(keep_deputies)
            
            if need_new > 0 and remaining_candidates:
                new_deputies = remaining_candidates[:need_new]
                next_deputies = keep_deputies + new_deputies
            else:
                # 如果候选不足，从所有客户端中选择
                all_clients = [cid for cid in range(self.stake_manager.num_clients) if cid not in excluded]
                new_deputies = [cid for cid, _ in sorted_by_combined if cid not in excluded][:need_new]
                next_deputies = keep_deputies + new_deputies
        
        # 确保数量正确
        next_deputies = next_deputies[:top_k]
        
        next_round = round_number + 1
        self.current_deputies = next_deputies
        self.deputy_history[next_round] = copy.deepcopy(next_deputies)
        
        print(f"[NPCDeputy] Round {round_number} -> {next_round}: FIFO轮换更新deputy:")
        print(f"  - 保留的deputy (FIFO): {[d for d in next_deputies if d in current_deputies]}")
        print(f"  - 新增的deputy: {[d for d in next_deputies if d not in current_deputies]}")
        print(f"  - 最终deputy列表: {next_deputies}")
    
    def get_deputy_for_round(self, round_number: int) -> List[int]:
        """获取指定轮次的deputy列表"""
        return copy.deepcopy(self.deputy_history.get(round_number, []))
    
    def can_accept_gradient(self, client_id: int, round_number: int) -> bool:
        """
        判断是否接受客户端的梯度上传
        只有deputy节点可以接受和管理梯度
        :param client_id: 客户端ID
        :param round_number: 轮次
        :return: 是否可以接受
        """
        # 检查是否为当前deputy
        return self.is_deputy(client_id)
    
    def get_gradient_storage_limit(self) -> int:
        """获取梯度存储限制（M份）"""
        return self.num_deputies
    
    def record_validation_failure(self, deputy_id: int):
        """记录deputy验证失败"""
        if deputy_id not in self.deputy_validation_failures:
            self.deputy_validation_failures[deputy_id] = 0
        self.deputy_validation_failures[deputy_id] += 1
    
    def get_validation_failure_count(self, deputy_id: int) -> int:
        """获取deputy验证失败次数"""
        return self.deputy_validation_failures.get(deputy_id, 0)
    
    def reset_validation_failures(self, deputy_id: int):
        """重置deputy验证失败计数（成功验证后调用）"""
        if deputy_id in self.deputy_validation_failures:
            self.deputy_validation_failures[deputy_id] = 0
    
    def should_penalize_deputy(self, deputy_id: int, failure_threshold: int = 3) -> bool:
        """
        判断是否应该惩罚deputy（验证失败次数过多）
        :param deputy_id: deputy节点ID
        :param failure_threshold: 失败次数阈值
        :return: 是否应该惩罚
        """
        failure_count = self.get_validation_failure_count(deputy_id)
        return failure_count >= failure_threshold
    
    def get_executor_deputy(self, round_number: int) -> Optional[int]:
        """
        获取执行deputy节点ID（第0轮选择权益最高的，之后FIFO轮换）
        :param round_number: 当前轮次
        :return: 执行deputy节点ID，如果没有deputy则返回None
        """
        if not self.current_deputies:
            return None
        
        if round_number == 0:
            # 第0轮：选择权益最高的deputy
            deputy_stakes = {deputy_id: self.stake_manager.get_stake(deputy_id) 
                            for deputy_id in self.current_deputies}
            executor_deputy = max(deputy_stakes.items(), key=lambda x: x[1])[0]
            self.executor_deputy_history = [executor_deputy]
            print(f"[NPCDeputy] Round {round_number}: 选择权益最高的deputy {executor_deputy} 作为执行deputy")
            return executor_deputy
        else:
            # 后续轮次：FIFO轮换
            # 从当前deputy列表中选择下一个（按FIFO顺序）
            if not self.executor_deputy_history:
                # 如果没有历史记录，选择第一个deputy
                executor_deputy = self.current_deputies[0]
                self.executor_deputy_history = [executor_deputy]
            else:
                # 找到上一个执行deputy在当前deputy列表中的位置
                last_executor = self.executor_deputy_history[-1]
                if last_executor in self.current_deputies:
                    last_index = self.current_deputies.index(last_executor)
                    # 选择下一个deputy（FIFO）
                    next_index = (last_index + 1) % len(self.current_deputies)
                    executor_deputy = self.current_deputies[next_index]
                else:
                    # 如果上一个执行deputy不在当前列表中，选择第一个
                    executor_deputy = self.current_deputies[0]
                
                self.executor_deputy_history.append(executor_deputy)
                # 只保留最近的历史记录（避免内存增长）
                if len(self.executor_deputy_history) > 100:
                    self.executor_deputy_history = self.executor_deputy_history[-100:]
            
            print(f"[NPCDeputy] Round {round_number}: FIFO轮换，选择deputy {executor_deputy} 作为执行deputy")
            return executor_deputy
    
    def get_highest_stake_deputy(self) -> Optional[int]:
        """
        获取权益最大的deputy节点ID（已废弃，使用get_executor_deputy代替）
        :return: deputy节点ID，如果没有deputy则返回None
        """
        # 为了向后兼容，保留此方法，但建议使用get_executor_deputy
        if not self.current_deputies:
            return None
        
        # 获取所有deputy的权益
        deputy_stakes = {deputy_id: self.stake_manager.get_stake(deputy_id) 
                        for deputy_id in self.current_deputies}
        
        # 返回权益最大的deputy
        highest_stake_deputy = max(deputy_stakes.items(), key=lambda x: x[1])[0]
        return highest_stake_deputy
    
    def submit_transaction_for_verification(self, round_number: int, transaction_data: Dict[str, Any]):
        """
        提交交易供其他deputy验证
        :param round_number: 轮次
        :param transaction_data: 交易数据（包含聚合结果、相似度信息等）
        """
        self.pending_transactions[round_number] = transaction_data
        self.transaction_votes[round_number] = {}
        print(f"[NPCDeputy] Round {round_number}: 交易已提交，等待deputy验证")
    
    def vote_on_transaction(self, round_number: int, deputy_id: int, is_valid: bool):
        """
        Deputy节点对交易进行投票（防止重复投票）
        :param round_number: 轮次
        :param deputy_id: deputy节点ID
        :param is_valid: 交易是否有效
        """
        if round_number not in self.transaction_votes:
            self.transaction_votes[round_number] = {}
        
        # 检查是否已经投票过
        if deputy_id in self.transaction_votes[round_number]:
            return
        
        self.transaction_votes[round_number][deputy_id] = is_valid
        print(f"[NPCDeputy] Deputy {deputy_id} 对Round {round_number}的交易投票: {'通过' if is_valid else '拒绝'}")
    
    def check_transaction_consensus(self, round_number: int, threshold_ratio: float = 2.0/3.0, 
                                   executor_deputy_id: Optional[int] = None) -> Tuple[bool, int, int]:
        """
        检查交易是否达到共识（大于等于M的2/3的deputy验证通过，M为总deputy数）
        :param round_number: 轮次
        :param threshold_ratio: 阈值比例（默认2/3）
        :param executor_deputy_id: 执行deputy节点ID（创建交易的deputy，不参与投票）
        :return: (是否达成共识, 通过票数, 总票数)
        """
        # M = 总deputy数
        total_deputies = len(self.current_deputies)
        
        # 共识阈值 = ceil(M * 2/3)，基于总deputy数M
        threshold = math.ceil(total_deputies * threshold_ratio)
        
        if round_number not in self.transaction_votes:
            # 排除执行deputy后的投票deputy数
            voting_deputies = [d for d in self.current_deputies if d != executor_deputy_id]
            return False, 0, len(voting_deputies)
        
        votes = self.transaction_votes[round_number]
        
        # 排除执行deputy的投票（执行deputy不参与投票）
        voting_deputies = [d for d in self.current_deputies if d != executor_deputy_id]
        total_voting_deputies = len(voting_deputies)
        
        # 只统计投票deputy的票数（排除执行deputy）
        approve_votes = sum(1 for deputy_id, vote in votes.items() 
                           if deputy_id != executor_deputy_id and vote)
        
        # 共识判断：通过票数 >= ceil(M * 2/3)
        consensus_reached = approve_votes >= threshold
        
        executor_info = f"（执行deputy: {executor_deputy_id}不参与投票）" if executor_deputy_id is not None else ""
        print(f"[NPCDeputy] Round {round_number} 交易投票统计{executor_info}:")
        print(f"  - 总deputy数(M): {total_deputies}, 投票deputy数: {total_voting_deputies}")
        print(f"  - 通过票数: {approve_votes}/{total_voting_deputies}, 阈值(ceil(M*2/3)): {threshold}")
        print(f"  - 共识: {'达成' if consensus_reached else '未达成'}")
        
        return consensus_reached, approve_votes, total_voting_deputies
    
    def clear_transaction_votes(self, round_number: int):
        """
        清除指定轮次的交易投票记录
        同时清理旧轮次的历史数据（不考虑溯源性，只保留最近数据）
        """
        # 清除指定轮次的投票和交易
        if round_number in self.transaction_votes:
            del self.transaction_votes[round_number]
        if round_number in self.pending_transactions:
            del self.pending_transactions[round_number]
        
        # 清理旧轮次的历史数据
        self._cleanup_old_history(round_number)
    
    def _cleanup_old_history(self, current_round: int):
        """
        清理旧轮次的历史数据（只保留最近keep_recent_rounds轮）
        :param current_round: 当前轮次
        """
        min_keep_round = max(0, current_round - self.keep_recent_rounds + 1)
        
        # 清理deputy_history
        rounds_to_remove = [r for r in self.deputy_history.keys() if r < min_keep_round]
        for round_num in rounds_to_remove:
            del self.deputy_history[round_num]
        
        # 清理transaction_votes
        rounds_to_remove = [r for r in self.transaction_votes.keys() if r < min_keep_round]
        for round_num in rounds_to_remove:
            del self.transaction_votes[round_num]
        
        # 清理pending_transactions
        rounds_to_remove = [r for r in self.pending_transactions.keys() if r < min_keep_round]
        for round_num in rounds_to_remove:
            del self.pending_transactions[round_num]
        
        if rounds_to_remove:
            print(f"[NPCDeputyManager] 已清理 {len(rounds_to_remove)} 个旧轮次的历史数据，保留最近 {self.keep_recent_rounds} 轮")

