"""
攻击策略基类
所有攻击必须继承 BaseAttack
"""
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
import torch
import numpy as np


class BaseAttack(ABC):
    """
    攻击基类
    """

    def __init__(self, malicious_ratio: float = 0.0, **kwargs):
        self.malicious_ratio = malicious_ratio
        self.config = kwargs

    @abstractmethod
    def select_malicious_clients(
        self,
        total_clients: int,
        available_clients: List[int],
        round_idx: int
    ) -> List[int]:
        """
        选择恶意客户端

        Args:
            total_clients: 客户端总数
            available_clients: 本轮可用的客户端
            round_idx: 当前轮次

        Returns:
            恶意客户端ID列表
        """
        pass

    def get_name(self) -> str:
        """获取攻击名称"""
        return self.__class__.__name__

    def is_malicious(self, client_id: int, malicious_clients: List[int]) -> bool:
        """判断客户端是否为恶意"""
        return client_id in malicious_clients


class NoAttack(BaseAttack):
    """无攻击 - 不修改任何数据"""

    def __init__(self, malicious_ratio: float = 0.0, **kwargs):
        super().__init__(malicious_ratio, **kwargs)

    def select_malicious_clients(
        self,
        total_clients: int,
        available_clients: List[int],
        round_idx: int
    ) -> List[int]:
        """无恶意客户端"""
        return []


class LabelFlippingAttack(BaseAttack):
    """
    标签翻转攻击
    在数据加载阶段修改标签: y -> num_classes - 1 - y
    """

    def __init__(
        self,
        malicious_ratio: float = 0.0,
        num_classes: int = 10,
        flip_mapping: str = "9-y",  # 或 "custom"
        **kwargs
    ):
        super().__init__(malicious_ratio, **kwargs)
        self.num_classes = num_classes
        self.flip_mapping = flip_mapping

    def select_malicious_clients(
        self,
        total_clients: int,
        available_clients: List[int],
        round_idx: int
    ) -> List[int]:
        """基于随机种子的确定性恶意客户端选择"""
        num_malicious = int(total_clients * self.malicious_ratio)
        if num_malicious == 0:
            return []

        # 使用轮次作为种子，确保每轮选择相同
        np.random.seed(round_idx)
        all_clients = list(range(total_clients))
        malicious = np.random.choice(all_clients, num_malicious, replace=False).tolist()
        return sorted(malicious)

    def apply_to_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        应用标签翻转

        Args:
            labels: 原始标签

        Returns:
            翻转后的标签
        """
        if self.flip_mapping == "9-y":
            # MNIST/CIFAR10: 0->9, 1->8, ..., 9->0
            return self.num_classes - 1 - labels
        else:
            # 默认翻转
            return self.num_classes - 1 - labels


class SignFlippingAttack(BaseAttack):
    """
    符号翻转攻击
    在梯度上传阶段翻转符号或乘以负数
    """

    def __init__(
        self,
        malicious_ratio: float = 0.0,
        sign_scale: float = -1.0,
        **kwargs
    ):
        super().__init__(malicious_ratio, **kwargs)
        self.sign_scale = sign_scale

    def select_malicious_clients(
        self,
        total_clients: int,
        available_clients: List[int],
        round_idx: int
    ) -> List[int]:
        num_malicious = int(total_clients * self.malicious_ratio)
        if num_malicious == 0:
            return []

        np.random.seed(round_idx)
        all_clients = list(range(total_clients))
        malicious = np.random.choice(all_clients, num_malicious, replace=False).tolist()
        return sorted(malicious)

    def apply_to_gradients(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        应用符号翻转

        Args:
            gradients: 原始梯度字典

        Returns:
            翻转后的梯度
        """
        poisoned = {}
        for name, grad in gradients.items():
            if isinstance(grad, torch.Tensor):
                poisoned[name] = grad * self.sign_scale
            else:
                poisoned[name] = grad
        return poisoned


class GaussianNoiseAttack(BaseAttack):
    """
    高斯噪声攻击
    向梯度添加大尺度高斯噪声
    """

    def __init__(
        self,
        malicious_ratio: float = 0.0,
        noise_scale: float = 2.0,
        **kwargs
    ):
        super().__init__(malicious_ratio, **kwargs)
        self.noise_scale = noise_scale

    def select_malicious_clients(
        self,
        total_clients: int,
        available_clients: List[int],
        round_idx: int
    ) -> List[int]:
        num_malicious = int(total_clients * self.malicious_ratio)
        if num_malicious == 0:
            return []

        np.random.seed(round_idx)
        all_clients = list(range(total_clients))
        malicious = np.random.choice(all_clients, num_malicious, replace=False).tolist()
        return sorted(malicious)

    def apply_to_gradients(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """添加高斯噪声"""
        poisoned = {}
        with torch.no_grad():
            for name, grad in gradients.items():
                if isinstance(grad, torch.Tensor):
                    noise = torch.randn_like(grad) * self.noise_scale
                    poisoned[name] = grad + noise
                else:
                    poisoned[name] = grad
        return poisoned
