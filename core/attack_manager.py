"""
Attack Manager Module
用于模拟联邦学习中的恶意攻击行为
核心功能：
1. Label Flipping: 篡改训练数据标签
2. Gradient Poisoning: 篡改模型参数/梯度
"""
import torch
import numpy as np
import copy
from typing import Dict, Any, List

class AttackManager:
    def __init__(self, attack_type: str = "None", malicious_ratio: float = 0.0):
        self.attack_type = attack_type
        self.malicious_ratio = malicious_ratio
        
    def select_malicious_nodes(self, total_nodes: int, round_seed: int = 0) -> List[int]:
        """
        每轮随机选择恶意节点
        """
        if self.malicious_ratio <= 0:
            return []
            
        num_malicious = int(total_nodes * self.malicious_ratio)
        np.random.seed(round_seed)
        malicious_nodes = np.random.choice(range(total_nodes), num_malicious, replace=False).tolist()
        malicious_nodes.sort()
        return malicious_nodes

    def apply_label_flip(self, labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        """
        应用标签翻转攻击 (Label Flipping)
        策略: y -> 9 - y
        """
        if self.attack_type != "LabelFlip":
            return labels
            
        return num_classes - 1 - labels

    def apply_gradient_poisoning(self, model_params: Dict[str, torch.Tensor], noise_scale: float = 2.0) -> Dict[str, torch.Tensor]:
        """
        应用梯度/模型毒化攻击 (Model Poisoning)
        策略: 向参数添加大幅度高斯噪声
        """
        if self.attack_type != "GradientPoisoning":
            return model_params
            
        poisoned_params = {}
        with torch.no_grad():
            for name, param in model_params.items():
                noise = torch.randn_like(param) * noise_scale
                poisoned_params[name] = param + noise
                
        return poisoned_params
