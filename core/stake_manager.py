"""
权益管理系统
管理客户端的权益（stake），用于TAS治理选举和奖励分配
"""
import copy
from typing import Dict, List, Optional
import numpy as np


class StakeManager:
    """权益管理器：管理每个客户端的权益和得分"""

    def __init__(
        self,
        num_clients: int,
        initial_stake: float = 100.0,
        random_init: bool = True,
        stake_range: tuple = (50.0, 150.0),
    ):
        self.num_clients = num_clients
        if random_init:
            import random
            min_stake, max_stake = stake_range
            self.stakes: Dict[int, float] = {
                i: random.uniform(min_stake, max_stake)
                for i in range(num_clients)
            }
            print(f"[StakeManager] 已随机初始化权益: {self.stakes}")
        else:
            self.stakes: Dict[int, float] = {i: initial_stake for i in range(num_clients)}

        self.round_scores: Dict[int, Dict[int, float]] = {}
        self.round_similarities: Dict[int, Dict[int, float]] = {}
        self.cp_scores: Dict[int, float] = {i: 100.0 for i in range(num_clients)}
        self.cp_history: Dict[int, Dict[int, float]] = {}

    def get_stake(self, client_id: int) -> float:
        return self.stakes.get(client_id, 0.0)

    def get_all_stakes(self) -> Dict[int, float]:
        return copy.deepcopy(self.stakes)

    def get_cp_scores(self) -> Dict[int, float]:
        return copy.deepcopy(self.cp_scores)

    def record_cp_scores(self, round_number: int):
        self.cp_history[round_number] = copy.deepcopy(self.cp_scores)

    def update_cp_scores(self, honest_set: List[int], all_clients: List[int], up_step: float = 2.0, down_step: float = 6.0):
        honest = set(honest_set)
        for cid in all_clients:
            if cid in honest:
                self.cp_scores[cid] = min(100.0, self.cp_scores.get(cid, 100.0) + up_step)
            else:
                self.cp_scores[cid] = max(0.0, self.cp_scores.get(cid, 100.0) - down_step)

    def update_stake(self, client_id: int, delta: float):
        if client_id in self.stakes:
            self.stakes[client_id] = max(0.0, self.stakes[client_id] + delta)

    def set_stake(self, client_id: int, value: float):
        self.stakes[client_id] = max(0.0, value)

    def slash_stake(self, client_id: int, penalty: float):
        if client_id in self.stakes:
            old_stake = self.stakes[client_id]
            self.stakes[client_id] = max(0.0, self.stakes[client_id] - penalty)
            print(f"[StakeManager] Node {client_id} Slashed! Stake: {old_stake:.2f} -> {self.stakes[client_id]:.2f}")

    def reward(self, client_id: int, alpha: float):
        """按 α 奖励诚实节点。"""
        self.update_stake(client_id, float(alpha))
        print(
            f"[StakeManager] Reward node {client_id} by alpha={alpha:.4f}, "
            f"new_stake={self.get_stake(client_id):.4f}"
        )

    def penalize(self, client_id: int, beta: float):
        """按 β 惩罚异常节点。"""
        self.slash_stake(client_id, float(beta))

    def slash_lying_auditor(self, client_id: int, cp_penalty: float = 100.0):
        if client_id in self.stakes:
            old_stake = self.stakes[client_id]
            self.stakes[client_id] = max(0.0, self.stakes[client_id] * 0.2)
            print(
                f"[StakeManager] Lying auditor {client_id} slashed 80% stake: "
                f"{old_stake:.2f} -> {self.stakes[client_id]:.2f}"
            )
        self.cp_scores[client_id] = max(0.0, self.cp_scores.get(client_id, 0.0) - cp_penalty)

    def apply_alpha_beta(self, honest_set: List[int], all_clients: List[int], alpha: float, beta: float):
        """使用 α/β 进行奖励与惩罚。"""
        honest = set(honest_set)
        for cid in all_clients:
            if cid in honest:
                self.reward(cid, alpha)
            else:
                self.penalize(cid, beta)

    def audit_similarity_action(self, client_id: int, sim_value: float, alpha: float, beta: float):
        if sim_value < -0.1:
            self.penalize(client_id, beta)
            print(f"[Stake Audit] Node {client_id} Sim={sim_value:.4f}. Action: Slash")
        elif sim_value > 0:
            self.reward(client_id, alpha)
            print(f"[Stake Audit] Node {client_id} Sim={sim_value:.4f}. Action: Reward")
        else:
            print(f"[Stake Audit] Node {client_id} Sim={sim_value:.4f}. Action: Neutral")

    def get_top_stake_clients(self, top_k: int, allowlist: Optional[List[int]] = None) -> List[int]:
        if allowlist is not None:
            allow = set(allowlist)
            filtered = [(cid, stake) for cid, stake in self.stakes.items() if cid in allow]
        else:
            filtered = list(self.stakes.items())

        sorted_clients = sorted(filtered, key=lambda x: x[1], reverse=True)
        return [client_id for client_id, _ in sorted_clients[:top_k]]

    def record_similarity(self, round_number: int, client_id: int, similarity: float):
        if round_number not in self.round_similarities:
            self.round_similarities[round_number] = {}
        self.round_similarities[round_number][client_id] = similarity

    def record_score(self, round_number: int, client_id: int, score: float):
        if round_number not in self.round_scores:
            self.round_scores[round_number] = {}
        self.round_scores[round_number][client_id] = score

    def calculate_score(
        self,
        client_id: int,
        similarity: float,
        round_number: int,
        similarity_weight: float = 0.7,
        stake_weight: float = 0.3,
    ) -> float:
        stake = self.get_stake(client_id)
        max_stake = max(self.stakes.values()) if self.stakes.values() else 1.0
        normalized_stake = stake / max_stake if max_stake > 0 else 0.0
        score = similarity_weight * similarity + stake_weight * normalized_stake
        self.record_similarity(round_number, client_id, similarity)
        self.record_score(round_number, client_id, score)
        return score

    def update_stakes_from_scores(
        self,
        round_number: int,
        reward_factor: float = 0.1,
        penalty_factor: float = 0.05,
    ):
        if round_number not in self.round_scores:
            return

        scores = self.round_scores[round_number]
        if not scores:
            return

        avg_score = np.mean(list(scores.values())) if scores.values() else 0.0
        for client_id, score in scores.items():
            delta = score - avg_score
            if delta > 0:
                stake_delta = delta * reward_factor
            else:
                stake_delta = delta * penalty_factor
            self.update_stake(client_id, stake_delta)

    def compute_cp_tier(self, client_id: int) -> str:
        cp = self.cp_scores.get(client_id, 0.0)
        if cp >= 80:
            return "A"
        if cp >= 50:
            return "B"
        return "C"
