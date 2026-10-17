"""
TAS (Tripartite Auditor System) 管理模块
负责从节点池中选出三方治理角色：
- Executor (E): 执行同态相似度计算与同态聚合
- Similarity Auditor (V_1): 审计相似度计算正确性（抽样重算 + 跨轮统计检验）
- Membership Auditor (V_2): 审计诚实集合 H 与训练链 C_T 一致性
"""
import copy
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class TASRoles:
    executor_E: int
    similarity_auditor_V1: int
    membership_auditor_V2: int


class TripartiteAuditorSystem:
    """TAS 管理器：选举和管理 E / V_1 / V_2 三方角色"""

    def __init__(self, stake_manager, keep_recent_rounds: Optional[int] = None):
        self.stake_manager = stake_manager
        self.current_roles: Optional[TASRoles] = None
        self.role_history: Dict[int, TASRoles] = {}
        self.keep_recent_rounds = keep_recent_rounds if keep_recent_rounds is not None else 50

        self.pending_round_data: Dict[int, Dict[str, Any]] = {}
        self.audit_votes: Dict[int, Dict[str, bool]] = {}

        # ── 跨轮异常检测状态 ──────────────────────────────────────────────
        # 记录每轮每个客户端的 trust_weights，用于统计检验
        self._weight_history: deque = deque(maxlen=10)   # 最近 10 轮
        # 记录每轮的 honest_set_H 大小，用于突变检测
        self._honest_set_size_history: deque = deque(maxlen=10)

    def elect_roles(self, round_number: int, honest_pool: Optional[List[int]] = None) -> TASRoles:
        top3 = self.stake_manager.get_top_stake_clients(3, allowlist=honest_pool)
        if len(top3) < 3:
            raise ValueError("节点数量不足，无法选举 TAS 三方角色（至少需要 3 个节点）")

        roles = TASRoles(
            executor_E=top3[0],
            similarity_auditor_V1=top3[1],
            membership_auditor_V2=top3[2],
        )
        self.current_roles = roles
        self.role_history[round_number] = copy.deepcopy(roles)
        self._prune_history()

        print(
            f"[TAS] Round {round_number}: E={roles.executor_E}, "
            f"V_1={roles.similarity_auditor_V1}, V_2={roles.membership_auditor_V2}"
        )
        return roles

    def get_roles(self) -> Optional[TASRoles]:
        return copy.deepcopy(self.current_roles)

    def submit_round_for_audit(
        self,
        round_number: int,
        sim_scores: Dict[int, float],
        honest_set_H: Set[int],
        C_T_records: List[Dict[str, Any]],
        trust_weights: Optional[Dict[int, float]] = None,
        trust_scores: Optional[Dict[int, float]] = None,
        flat_updates: Optional[Dict[int, Any]] = None,
        prev_global_update: Optional[Any] = None,
        v_space: Optional[Any] = None,
        cp_scores: Optional[Dict[int, float]] = None,
        noise_std: Optional[float] = None,
        noise_seed: Optional[int] = None,
    ):
        self.pending_round_data[round_number] = {
            "sim_scores": copy.deepcopy(sim_scores),
            "honest_set_H": set(honest_set_H),
            "C_T_records": copy.deepcopy(C_T_records),
            "trust_weights": copy.deepcopy(trust_weights) if trust_weights is not None else None,
            "trust_scores": copy.deepcopy(trust_scores) if trust_scores is not None else None,
            "flat_updates": copy.deepcopy(flat_updates) if flat_updates is not None else None,
            "prev_global_update": copy.deepcopy(prev_global_update),
            "v_space": copy.deepcopy(v_space),
            "cp_scores": copy.deepcopy(cp_scores) if cp_scores is not None else None,
            "noise_std": noise_std,
            "noise_seed": noise_seed,
        }
        self.audit_votes[round_number] = {"V1": False, "V2": False}
        print(f"[TAS] Round {round_number}: 审计任务已提交")

    def vote_similarity_audit(self, round_number: int, approved: bool):
        if round_number not in self.audit_votes:
            self.audit_votes[round_number] = {"V1": False, "V2": False}
        self.audit_votes[round_number]["V1"] = bool(approved)
        print(f"[TAS] V_1 对 Round {round_number} 相似度审计: {'通过' if approved else '拒绝'}")

    def vote_membership_audit(self, round_number: int, approved: bool):
        if round_number not in self.audit_votes:
            self.audit_votes[round_number] = {"V1": False, "V2": False}
        self.audit_votes[round_number]["V2"] = bool(approved)
        print(f"[TAS] V_2 对 Round {round_number} 成员审计: {'通过' if approved else '拒绝'}")

    def verify_similarity_audit(
        self,
        round_number: int,
        sample_ratio: float = 0.2,
        tolerance: float = 2e-3,
    ) -> Tuple[bool, Dict[str, Any]]:
        payload = self.pending_round_data.get(round_number)
        if not payload:
            return False, {"reason": "missing_round_data"}

        trust_weights = payload.get("trust_weights") or {}
        trust_scores = payload.get("trust_scores") or {}
        flat_updates = payload.get("flat_updates") or {}
        noise_std = float(payload.get("noise_std") or 0.0)
        noise_seed = payload.get("noise_seed")
        prev_global_update = payload.get("prev_global_update")

        if not trust_weights or not flat_updates:
            return False, {"reason": "missing_inputs"}

        ids = sorted(list(flat_updates.keys()))
        if not ids:
            return False, {"reason": "empty_updates"}

        sample_size = max(1, int(len(ids) * sample_ratio))
        rng = np.random.RandomState(round_number)
        sampled = rng.choice(ids, size=sample_size, replace=False).tolist()

        if prev_global_update is not None:
            root_grad = np.array(prev_global_update, dtype=np.float64)
        else:
            updates = [np.array(flat_updates[cid], dtype=np.float64) for cid in ids]
            stacked = np.stack(updates, axis=0)
            root_grad = np.mean(stacked, axis=0)
            if noise_seed is not None and noise_std > 0:
                rng_noise = np.random.RandomState(int(noise_seed))
                root_grad = root_grad + rng_noise.normal(0.0, noise_std, size=root_grad.shape)

        root_norm = np.linalg.norm(root_grad) + 1e-12
        for cid in sampled:
            update = np.array(flat_updates[cid], dtype=np.float64)
            update_norm = np.linalg.norm(update) + 1e-12
            sim = float(np.dot(update, root_grad) / (update_norm * root_norm))
            recomputed = max(0.0, sim)

            expected = float(trust_scores.get(cid, trust_weights.get(cid, 0.0)))
            if abs(recomputed - expected) > tolerance:
                return False, {
                    "reason": "selective_weighting_attack",
                    "client_id": cid,
                    "expected": expected,
                    "recomputed": recomputed,
                }

        return True, {"sampled": sampled, "checked": len(sampled)}

    def verify_similarity_audit_enhanced(
        self,
        round_number: int,
        sample_ratio: float = 0.2,
        tolerance: float = 2e-3,
        enable_cross_round: bool = True,
        ks_alpha: float = 0.05,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        增强版 V1 审计：抽样重算 + 跨轮 KS 统计检验。

        1. 抽样重算（同 verify_similarity_audit）
        2. 跨轮统计检验：对比本轮 trust_weights 分布与历史分布，
           若 KS 统计量超过临界值（α=0.05），判定异常。
        """
        # ── Phase 1: 抽样重算 ────────────────────────────────────────────
        point_ok, point_detail = self.verify_similarity_audit(
            round_number, sample_ratio=sample_ratio, tolerance=tolerance
        )

        detail = {"point_check": point_detail}

        if not point_ok:
            detail["phase"] = "point_check"
            return False, detail

        # ── Phase 2: 跨轮 KS 检验 ──────────────────────────────────────
        if not enable_cross_round:
            detail["phase"] = "point_check_only"
            return True, detail

        payload = self.pending_round_data.get(round_number)
        if not payload:
            detail["phase"] = "point_check_only"
            return True, detail

        current_weights = payload.get("trust_weights") or {}
        if not current_weights:
            detail["phase"] = "point_check_only"
            return True, detail

        current_vals = sorted([float(w) for w in current_weights.values()])

        # 首次运行，无历史可对比
        if len(self._weight_history) < 3:
            self._weight_history.append(current_vals)
            detail["phase"] = "point_check_only_insufficient_history"
            return True, detail

        # 拼接历史权重值作为参考分布
        hist_vals = []
        for w_list in self._weight_history:
            hist_vals.extend(w_list)
        hist_vals = sorted(hist_vals)

        # KS 检验
        ks_stat, ks_pvalue = self._ks_test(current_vals, hist_vals)

        self._weight_history.append(current_vals)

        # 临界值表 (α=0.05, 双样本)
        n1, n2 = len(current_vals), len(hist_vals)
        if n1 > 0 and n2 > 0:
            critical = 1.36 * np.sqrt((n1 + n2) / (n1 * n2))
        else:
            critical = float("inf")

        if ks_stat > critical:
            detail["phase"] = "cross_round_ks"
            detail["ks_stat"] = float(ks_stat)
            detail["ks_critical"] = float(critical)
            detail["ks_pvalue"] = float(ks_pvalue)
            return False, detail

        detail["phase"] = "all_passed"
        detail["ks_stat"] = float(ks_stat)
        return True, detail

    @staticmethod
    def _ks_test(sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
        """双样本 Kolmogorov-Smirnov 检验（纯 numpy 实现）"""
        n1, n2 = len(sample1), len(sample2)
        if n1 == 0 or n2 == 0:
            return 0.0, 1.0

        data = sorted(sample1 + sample2)
        cdf1 = np.searchsorted(sample1, data, side="right") / n1
        cdf2 = np.searchsorted(sample2, data, side="right") / n2
        ks_stat = float(np.max(np.abs(cdf1 - cdf2)))

        # 近似 p 值
        en = np.sqrt(n1 * n2 / (n1 + n2))
        if en > 0:
            z = (ks_stat + 0.12 / en + 0.11 / en**2) * en
            pvalue = 2.0 * np.exp(-2.0 * z * z)
            pvalue = max(0.0, min(1.0, pvalue))
        else:
            pvalue = 1.0

        return ks_stat, pvalue

    def verify_membership_audit(self, round_number: int, cp_threshold: float = 30.0) -> Tuple[bool, Dict[str, Any]]:
        payload = self.pending_round_data.get(round_number)
        if not payload:
            return False, {"reason": "missing_round_data"}

        trust_weights = payload.get("trust_weights") or {}
        c_t_records = payload.get("C_T_records") or []
        cp_scores = payload.get("cp_scores") or {}

        valid_ids = {int(rec.get("DID")) for rec in c_t_records if rec.get("DID") is not None}
        included = {cid for cid, w in trust_weights.items() if float(w) > 0.0}

        missing = sorted([cid for cid in included if cid not in valid_ids])
        if missing:
            return False, {"reason": "unfair_inclusion", "missing": missing}

        low_cp = sorted([cid for cid in included if float(cp_scores.get(cid, 100.0)) < cp_threshold])
        if low_cp:
            return False, {"reason": "low_cp_inclusion", "clients": low_cp}

        return True, {"included": sorted(list(included))}

    def verify_membership_audit_enhanced(
        self,
        round_number: int,
        cp_threshold: float = 30.0,
        enable_size_anomaly: bool = True,
        size_drop_threshold: float = 0.4,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        增强版 V2 审计：CP 合规检查 + honest_set 大小突变检测。

        如果 honest_set_H 大小突然暴增（超过历史均值的 size_drop_threshold），
        可能意味着 E 非法纳入了隔离节点。
        """
        # ── Phase 1: 标准 CP 合规检查 ──────────────────────────────────
        base_ok, base_detail = self.verify_membership_audit(round_number, cp_threshold)
        detail = {"base_check": base_detail}

        if not base_ok:
            detail["phase"] = "cp_violation"
            return False, detail

        # ── Phase 2: honest_set 大小突变检测 ────────────────────────────
        if not enable_size_anomaly:
            detail["phase"] = "cp_check_only"
            return True, detail

        payload = self.pending_round_data.get(round_number)
        if not payload:
            detail["phase"] = "cp_check_only"
            return True, detail

        honest_set_H = payload.get("honest_set_H") or set()
        current_size = len(honest_set_H)
        self._honest_set_size_history.append(current_size)

        if len(self._honest_set_size_history) < 3:
            detail["phase"] = "cp_check_only_insufficient_history"
            return True, detail

        hist_sizes = list(self._honest_set_size_history)[:-1]  # 排除当前轮
        mean_size = float(np.mean(hist_sizes))
        std_size = float(np.std(hist_sizes)) if len(hist_sizes) > 1 else 0.0

        # 突变检测：当前集合大小超出均值 ± size_drop_threshold * 均值
        if mean_size > 0:
            relative_change = (current_size - mean_size) / mean_size
            if relative_change > size_drop_threshold:
                detail["phase"] = "honest_set_size_spike"
                detail["current_size"] = current_size
                detail["mean_size"] = float(mean_size)
                detail["relative_change"] = float(relative_change)
                return False, detail

        detail["phase"] = "all_passed"
        return True, detail

    def check_tripartite_consensus(self, round_number: int, require_executor_submission: bool = True) -> Tuple[bool, Dict[str, bool]]:
        votes = self.audit_votes.get(round_number, {"V1": False, "V2": False})
        has_executor_submission = round_number in self.pending_round_data

        ok = votes.get("V1", False) and votes.get("V2", False)
        if require_executor_submission:
            ok = ok and has_executor_submission

        arbitration = {
            "v1_overruled": False,
            "reason": None,
        }

        # 2-out-of-3 arbitration: if V1 rejects but V2 approves, treat V1 as malicious auditor
        if has_executor_submission and not votes.get("V1", False) and votes.get("V2", False):
            ok = True
            arbitration["v1_overruled"] = True
            arbitration["reason"] = "selective_auditing_attack"

        print(
            f"[TAS] Round {round_number}: consensus={'达成' if ok else '未达成'} "
            f"(E_submitted={has_executor_submission}, V_1={votes.get('V1', False)}, V_2={votes.get('V2', False)}, "
            f"V1_overruled={arbitration['v1_overruled']})"
        )
        votes_out = copy.deepcopy(votes)
        votes_out["arbitration"] = arbitration
        return ok, votes_out

    def clear_round(self, round_number: int):
        self.pending_round_data.pop(round_number, None)
        self.audit_votes.pop(round_number, None)

    def _prune_history(self):
        if len(self.role_history) <= self.keep_recent_rounds:
            return
        rounds = sorted(self.role_history.keys())
        overflow = len(rounds) - self.keep_recent_rounds
        for r in rounds[:overflow]:
            del self.role_history[r]


# backward compatibility
class NPCDeputyManager(TripartiteAuditorSystem):
    pass
