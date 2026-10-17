"""
攻击策略模块
支持: Label Flipping, Sign Flipping, Gaussian Noise
"""
from .base import (
    BaseAttack,
    NoAttack,
    LabelFlippingAttack,
    SignFlippingAttack,
    GaussianNoiseAttack
)


def get_attack(
    name: str,
    malicious_ratio: float = 0.0,
    config: dict = None,
    **kwargs
) -> BaseAttack:
    """
    工厂函数：根据配置获取攻击策略

    Args:
        name: 攻击名称
        malicious_ratio: 恶意客户端占比
        config: 攻击参数字典

    Returns:
        攻击实例
    """
    config = config or {}

    if name.lower() == 'none' or malicious_ratio == 0:
        # 返回一个不做任何事的攻击
        return NoAttack(malicious_ratio=0)
    elif name.lower() == 'label_flipping':
        return LabelFlippingAttack(
            malicious_ratio=malicious_ratio,
            num_classes=config.get('num_classes', 10),
            flip_mapping=config.get('flip_mapping', '9-y')
        )
    elif name.lower() == 'sign_flipping':
        return SignFlippingAttack(
            malicious_ratio=malicious_ratio,
            sign_scale=config.get('sign_scale', -1.0)
        )
    elif name.lower() == 'gaussian_noise':
        return GaussianNoiseAttack(
            malicious_ratio=malicious_ratio,
            noise_scale=config.get('noise_scale', 2.0)
        )
    else:
        # 默认返回无攻击
        print(f"[Warning] Unknown attack '{name}', using no attack")
        return NoAttack(malicious_ratio=0)


__all__ = [
    'BaseAttack',
    'NoAttack',
    'LabelFlippingAttack',
    'SignFlippingAttack',
    'GaussianNoiseAttack',
    'get_attack'
]
