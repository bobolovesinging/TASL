"""
权益管理系统
管理客户端的权益（stake），用于NPC deputy选举和奖励分配
"""
import copy
from typing import Dict, List, Optional
import numpy as np


class StakeManager:
    """权益管理器：管理每个客户端的权益和得分"""
    
    def __init__(self, num_clients: int, initial_stake: float = 100.0, random_init: bool = True, stake_range: tuple = (50.0, 150.0)):
        """
        初始化权益管理器
        :param num_clients: 客户端总数
        :param initial_stake: 初始权益值（如果random_init=False时使用）
        :param random_init: 是否随机初始化权益
        :param stake_range: 随机初始化时的权益范围 (min, max)
        """
        self.num_clients = num_clients
        if random_init:
            # 随机初始化每个客户端的权益
            import random
            min_stake, max_stake = stake_range
            self.stakes: Dict[int, float] = {
                i: random.uniform(min_stake, max_stake) 
                for i in range(num_clients)
            }
            print(f"[StakeManager] 已随机初始化权益: {self.stakes}")
        else:
            self.stakes: Dict[int, float] = {i: initial_stake for i in range(num_clients)}
        self.round_scores: Dict[int, Dict[int, float]] = {}  # {round: {client_id: score}}
        self.round_similarities: Dict[int, Dict[int, float]] = {}  # {round: {client_id: similarity}}
        
    def get_stake(self, client_id: int) -> float:
        """获取客户端权益"""
        return self.stakes.get(client_id, 0.0)
    
    def get_all_stakes(self) -> Dict[int, float]:
        """获取所有客户端权益"""
        return copy.deepcopy(self.stakes)
    
    def update_stake(self, client_id: int, delta: float):
        """更新客户端权益（delta可以是正数或负数）"""
        if client_id in self.stakes:
            self.stakes[client_id] = max(0.0, self.stakes[client_id] + delta)
    
    def set_stake(self, client_id: int, value: float):
        """设置客户端权益"""
        self.stakes[client_id] = max(0.0, value)
    
    def get_top_stake_clients(self, top_k: int) -> List[int]:
        """
        获取权益最高的top_k个客户端
        :param top_k: 返回前k个客户端
        :return: 客户端ID列表（按权益降序）
        """
        sorted_clients = sorted(
            self.stakes.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [client_id for client_id, _ in sorted_clients[:top_k]]
    
    def record_similarity(self, round_number: int, client_id: int, similarity: float):
        """记录客户端在某一轮的余弦相似度"""
        if round_number not in self.round_similarities:
            self.round_similarities[round_number] = {}
        self.round_similarities[round_number][client_id] = similarity
    
    def record_score(self, round_number: int, client_id: int, score: float):
        """记录客户端在某一轮的得分"""
        if round_number not in self.round_scores:
            self.round_scores[round_number] = {}
        self.round_scores[round_number][client_id] = score
    
    def calculate_score(self, client_id: int, similarity: float, round_number: int, 
                       similarity_weight: float = 0.7, stake_weight: float = 0.3) -> float:
        """
        计算客户端得分
        :param client_id: 客户端ID
        :param similarity: 余弦相似度
        :param round_number: 轮次
        :param similarity_weight: 相似度权重
        :param stake_weight: 权益权重
        :return: 得分
        """
        stake = self.get_stake(client_id)
        # 归一化权益（相对于最大权益）
        max_stake = max(self.stakes.values()) if self.stakes.values() else 1.0
        normalized_stake = stake / max_stake if max_stake > 0 else 0.0
        
        # 得分 = 相似度权重 * 相似度 + 权益权重 * 归一化权益
        score = similarity_weight * similarity + stake_weight * normalized_stake
        
        # 记录相似度和得分
        self.record_similarity(round_number, client_id, similarity)
        self.record_score(round_number, client_id, score)
        
        return score
    
    def update_stakes_from_scores(self, round_number: int, reward_factor: float = 0.1, 
                                  penalty_factor: float = 0.05):
        """
        根据得分更新权益
        :param round_number: 轮次
        :param reward_factor: 奖励因子（得分高的客户端增加权益）
        :param penalty_factor: 惩罚因子（得分低的客户端减少权益）
        """
        if round_number not in self.round_scores:
            return
        
        scores = self.round_scores[round_number]
        if not scores:
            return
        
        # 计算平均得分
        avg_score = np.mean(list(scores.values())) if scores.values() else 0.0
        
        # 根据得分与平均值的差异更新权益
        for client_id, score in scores.items():
            delta = score - avg_score
            if delta > 0:
                # 高于平均值，增加权益
                stake_delta = delta * reward_factor
            else:
                # 低于平均值，减少权益
                stake_delta = delta * penalty_factor
            
            self.update_stake(client_id, stake_delta)
    
    def get_top_score_clients(self, round_number: int, top_k: int) -> List[int]:
        """
        获取某一轮得分最高的top_k个客户端
        :param round_number: 轮次
        :param top_k: 返回前k个客户端
        :return: 客户端ID列表（按得分降序）
        """
        if round_number not in self.round_scores:
            return []
        
        scores = self.round_scores[round_number]
        sorted_clients = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [client_id for client_id, _ in sorted_clients[:top_k]]
    
    def get_round_stats(self, round_number: int) -> Dict:
        """获取某一轮的统计信息"""
        stats = {
            'round': round_number,
            'scores': self.round_scores.get(round_number, {}),
            'similarities': self.round_similarities.get(round_number, {}),
            'stakes': copy.deepcopy(self.stakes)
        }
        return stats
    
    def reward_deputy(self, deputy_id: int, reward_amount: float, reason: str = "出块奖励"):
        """
        奖励deputy节点（DPOS出块奖励）
        :param deputy_id: deputy节点ID
        :param reward_amount: 奖励金额
        :param reason: 奖励原因
        """
        if deputy_id in self.stakes:
            old_stake = self.stakes[deputy_id]
            self.stakes[deputy_id] += reward_amount
            print(f"[StakeManager] Deputy {deputy_id} 获得奖励: +{reward_amount:.2f} ({reason}), "
                  f"权益: {old_stake:.2f} -> {self.stakes[deputy_id]:.2f}")
    
    def slash_deputy(self, deputy_id: int, slash_amount: float, reason: str = "作恶惩罚"):
        """
        Slashing惩罚deputy节点（严重作恶）
        :param deputy_id: deputy节点ID
        :param slash_amount: 惩罚金额（可以是绝对值或百分比）
        :param reason: 惩罚原因
        """
        if deputy_id not in self.stakes:
            return
        
        old_stake = self.stakes[deputy_id]
        
        # 如果slash_amount在0-1之间，认为是百分比
        if 0 < slash_amount <= 1:
            actual_slash = old_stake * slash_amount
        else:
            actual_slash = slash_amount
        
        # 惩罚后权益不能为负
        self.stakes[deputy_id] = max(0.0, old_stake - actual_slash)
        
        print(f"[StakeManager] ⚠️ Deputy {deputy_id} 受到Slashing惩罚: -{actual_slash:.2f} ({reason}), "
              f"权益: {old_stake:.2f} -> {self.stakes[deputy_id]:.2f}")
    
    def penalize_deputy_failure(self, deputy_id: int, penalty_amount: float, reason: str = "失职惩罚"):
        """
        惩罚deputy节点失职（未及时出块、验证失败等）
        :param deputy_id: deputy节点ID
        :param penalty_amount: 惩罚金额
        :param reason: 惩罚原因
        """
        if deputy_id in self.stakes:
            old_stake = self.stakes[deputy_id]
            self.stakes[deputy_id] = max(0.0, old_stake - penalty_amount)
            print(f"[StakeManager] Deputy {deputy_id} 受到失职惩罚: -{penalty_amount:.2f} ({reason}), "
                  f"权益: {old_stake:.2f} -> {self.stakes[deputy_id]:.2f}")
    
    def get_deputy_tenure(self, deputy_id: int, deputy_history: Dict[int, List[int]]) -> int:
        """
        获取deputy的连续任期（连续担任deputy的轮次数）
        :param deputy_id: deputy节点ID
        :param deputy_history: deputy历史记录 {round: [deputy_ids]}
        :return: 连续任期（轮次数）
        """
        if not deputy_history:
            return 0
        
        rounds = sorted(deputy_history.keys(), reverse=True)
        tenure = 0
        
        for round_num in rounds:
            if deputy_id in deputy_history[round_num]:
                tenure += 1
            else:
                break
        
        return tenure

