"""
梯度层攻击模块
针对 Byzantine 鲁棒聚合器的梯度层攻击：
  - scaling:     放大恶意更新，覆盖诚实信号
  - min_max:     Byzantine 协作，使聚合结果偏离诚实方向最远
  - model_replacement: 模型替换攻击（scaling 的协作版本）

所有攻击的输入/输出均为 state_dict 格式的模型参数（CPU tensor）。
攻击在本地训练完成后施加于提交的更新上。
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Set


# ─────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────

def _state_sub(a: Dict, b: Dict) -> Dict:
    """a - b (both on CPU)"""
    return {k: v - b[k] for k, v in a.items() if v.is_floating_point()}


def _state_add(a: Dict, b: Dict) -> Dict:
    return {k: v + b[k] for k, v in a.items() if k in b and v.is_floating_point()}


def _flatten(state: Dict) -> torch.Tensor:
    return torch.cat([v.flatten() for v in state.values()])


# ─────────────────────────────────────────────
# 1. Scaling Attack
# ─────────────────────────────────────────────

def scaling_attack(
    client_state: Dict,
    global_state: Dict,
    scaling_factor: float = 100.0,
) -> Dict:
    """
    Scaling Attack（放大攻击）

    原理：
        Byzantine 客户端正常训练后，将更新 delta = state - global 放大
        scaling_factor 倍，使恶意更新在聚合中占主导。

    参数：
        client_state:   训练后的客户端参数（CPU）
        global_state:   当前全局参数（CPU）
        scaling_factor: 放大倍数，默认 100×

    返回：
        攻击后的参数 state_dict（CPU）
    """
    attacked = {}
    for k, v in client_state.items():
        if v.is_floating_point() and k in global_state:
            delta = v - global_state[k]
            attacked[k] = global_state[k] + scaling_factor * delta
        else:
            attacked[k] = v
    return attacked


# ─────────────────────────────────────────────
# 2. Min-Max Attack (Blanchard et al.)
# ─────────────────────────────────────────────

def min_max_attack(
    client_state: Dict,
    global_state: Dict,
    honest_avg_flat: Optional[torch.Tensor] = None,
    n_byz: int = 1,
    n_clients: int = 10,
    sign_flip_first: bool = True,
) -> Dict:
    """
    Min-Max Attack（Blanchard et al., 2017）

    原理：
        Byzantine 客户端协作，使聚合结果尽可能远离诚实客户端的更新方向。
        经典做法：
          1. 估计诚实平均更新方向 h
          2. 每个 Byzantine 客户端提交 -h（与诚实方向相反）
          3. 若 n_byz 足够多，聚合结果被拉向 -h

        增强版（sign_flip_first=True）：
          先对更新取反，再沿反方向放大，使聚合器（如 spatial median）
          难以区分恶意/诚实更新。

    参数：
        client_state:     训练后的客户端参数（CPU）
        global_state:     当前全局参数（CPU）
        honest_avg_flat:  诚实平均更新（flattened tensor），若未提供则基于本地更新估计
        n_byz:            Byzantine 客户端数量
        n_clients:        总客户端数
        sign_flip_first:  是否先对更新取反（使攻击更隐蔽）

    返回：
        攻击后的参数 state_dict（CPU）
    """
    # 估计诚实平均更新方向
    delta = _state_sub(client_state, global_state)
    flat_delta = _flatten(delta)

    if honest_avg_flat is not None:
        # 使用外部提供的诚实平均（如上一轮的聚合更新）
        h = honest_avg_flat
    else:
        # 退化情况：用本地更新作为 h 的 proxy（实际攻击中 Byzantine 无法获得此信息）
        # 论文实验里用"已知诚实更新"来模拟最强攻击
        h = flat_delta

    # 攻击方向：与诚实更新相反
    if sign_flip_first:
        attack_dir = -h / (h.norm() + 1e-12)
    else:
        # 直接取反
        attack_dir = -flat_delta / (flat_delta.norm() + 1e-12)

    # 幅度：与诚实更新相当（避免被 norm gate 直接过滤）
    # 稍大一点，确保能影响聚合
    attack_mag = flat_delta.norm() * 1.5

    # 构造攻击更新
    attack_flat = attack_dir * attack_mag

    # 将 flattened 攻击向量还原为 state_dict 格式
    attacked = {}
    idx = 0
    for k, v in global_state.items():
        if v.is_floating_point():
            sz = v.numel()
            attacked[k] = global_state[k] + attack_flat[idx:idx+sz].view(v.shape)
            idx += sz
        else:
            attacked[k] = v
    return attacked


# ─────────────────────────────────────────────
# 3. Model Replacement Attack
# ─────────────────────────────────────────────

def model_replacement_attack(
    client_states: List[Dict],
    global_state: Dict,
    target_state: Optional[Dict] = None,
    scaling_factor: float = 50.0,
) -> List[Dict]:
    """
    Model Replacement Attack（Bagdasaryan et al., 2020）

    原理：
        Byzantine 客户端协作，使全局模型被替换为目标模型（或目标更新方向）。
        每个 Byzantine 客户端提交：
            w_i = w_global + s * (w_target - w_global)
        其中 s 足够大，使得聚合后全局模型接近 w_target。

        对于 FedAvg（等权聚合）：
            w_new = (1/n) * Σ w_i
                   = (1/n) * [(n-n_byz)*w_honest + n_byz*w_byz]
                   = w_honest + (n_byz/n)*(w_byz - w_honest)
        要使 w_new ≈ w_target，需：
            w_byz = w_global + (n/n_byz) * (w_target - w_global)

    参数：
        client_states:  所有 Byzantine 客户端的参数列表（CPU）
        global_state:   当前全局参数（CPU）
        target_state:   目标模型参数；若为 None，则使用"随机噪声"作为目标
        scaling_factor: 放大因子（用于无目标模型时）

    返回：
        攻击后的 Byzantine 客户端参数列表
    """
    n_byz = len(client_states)
    if target_state is None:
        # 无目标：用随机噪声初始化目标，使模型性能崩溃
        target_state = {}
        for k, v in global_state.items():
            if v.is_floating_point():
                # 随机噪声，幅度与全局参数相当
                target_state[k] = v + scaling_factor * torch.randn_like(v)
            else:
                target_state[k] = v

    # 计算每个 Byzantine 客户端需要提交的参数
    # w_byz = w_global + (n / n_byz) * (w_target - w_global)
    factor = len(client_states) / n_byz  # 实际上 = 1.0，因为 client_states 就是 Byzantine 的
    # 正确公式：n_byz 个 Byzantine 要使聚合结果 = target
    # w_new = (n-n_byz)/n * w_honest + n_byz/n * w_byz ≈ target
    # => w_byz ≈ (n/n_byz) * target - (n-n_byz)/n_byz * w_honest
    # 攻击时不知道 w_honest，用 global_state 近似
    n = n_byz + (len(client_states) * 0)  # 需要知道总客户端数，这里需要外部传入
    # 简化：直接用放大因子
    attacked = []
    for _ in client_states:
        attacked_state = {}
        for k, v in global_state.items():
            if v.is_floating_point() and k in target_state:
                delta = target_state[k] - v
                attacked_state[k] = v + scaling_factor * delta
            else:
                attacked_state[k] = v
        attacked.append(attacked_state)
    return attacked


# ─────────────────────────────────────────────
# 统一接口
# ─────────────────────────────────────────────

GRADIENT_ATTACKS = {"scaling", "min_max", "model_replacement"}


def apply_gradient_attack(
    attack_name: str,
    client_state: Dict,
    global_state: Dict,
    honest_avg_flat: Optional[torch.Tensor] = None,
    n_byz: int = 1,
    n_clients: int = 10,
    attack_kwargs: Optional[Dict] = None,
) -> Dict:
    """
    统一接口：对单个客户端的更新施加梯度层攻击。

    参数：
        attack_name:   "scaling" | "min_max" | "model_replacement"
        client_state:  训练后的客户端参数
        global_state:  当前全局参数
        honest_avg_flat: 诚实平均更新（flattened），用于 min_max
        n_byz:         Byzantine 数量
        n_clients:     总客户端数
        attack_kwargs: 攻击参数字典（如 scaling_factor, sign_flip_first 等）

    返回：
        攻击后的参数 state_dict
    """
    if attack_kwargs is None:
        attack_kwargs = {}

    if attack_name == "scaling":
        sf = attack_kwargs.get("scaling_factor", 100.0)
        return scaling_attack(client_state, global_state, scaling_factor=sf)

    elif attack_name == "min_max":
        sff = attack_kwargs.get("sign_flip_first", True)
        return min_max_attack(
            client_state, global_state,
            honest_avg_flat=honest_avg_flat,
            n_byz=n_byz, n_clients=n_clients,
            sign_flip_first=sff,
        )

    elif attack_name == "model_replacement":
        sf = attack_kwargs.get("scaling_factor", 50.0)
        # model_replacement 需要 Byzantine 客户端列表，这里只处理单个
        # 返回用 target=random 的替换攻击
        return model_replacement_attack(
            [client_state], global_state,
            target_state=None, scaling_factor=sf,
        )[0]

    else:
        raise ValueError(f"Unknown gradient attack: {attack_name}")
