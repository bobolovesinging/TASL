"""
联邦学习攻击模块
提供多种常见的攻击方式，用于测试联邦学习系统的鲁棒性

支持的攻击类型：
1. 随机攻击 (Random Attack)
2. 标签翻转攻击 (Label Flipping Attack)
3. 模型投毒攻击 (Model Poisoning Attack)
4. 缩放攻击 (Scaling Attack)
5. 后门攻击 (Backdoor Attack)
"""
import torch


class AttackConfig:
    """
    攻击配置类，用于配置各种攻击方式
    支持多种攻击模式，方便测试联邦学习系统的鲁棒性
    """
    def __init__(self):
        # 是否启用攻击
        self.enable_attack = False
        
        # 攻击类型: 'none', 'random', 'label_flip', 'model_poison', 'scale', 'sign_flip', 'gaussian_noise', 'backdoor'
        self.attack_type = 'none'
        self.gaussian_noise_std = 1.0
        
        # 恶意客户端ID列表（这些客户端将执行攻击）
        self.malicious_clients = []
        
        # 攻击强度参数
        self.attack_strength = 1.0  # 攻击强度（0.0-1.0，1.0为最强）
        
        # 标签翻转攻击：翻转映射（将标签i映射到标签flip_map[i]）
        self.label_flip_map = {}
        
        # 缩放攻击：恶意客户端返回的模型参数缩放倍数
        self.scale_factor = -1.0  # 负值表示反向攻击
        
        # 后门攻击：后门触发模式（特定样本特征）
        self.backdoor_pattern = None
        self.backdoor_target_label = 0  # 后门目标标签
    
    def is_malicious(self, client_id):
        """
        检查客户端是否为恶意客户端
        :param client_id: 客户端ID
        :return: True如果客户端是恶意的，否则False
        """
        return self.enable_attack and client_id in self.malicious_clients
    
    def get_attack_type(self, client_id):
        """
        获取客户端的攻击类型
        :param client_id: 客户端ID
        :return: 攻击类型字符串
        """
        if self.is_malicious(client_id):
            return self.attack_type
        return 'none'


def apply_random_attack(model_params, device, attack_strength=1.0):
    """
    随机攻击：返回随机初始化的模型参数
    :param model_params: 原始模型参数
    :param device: 设备
    :param attack_strength: 攻击强度（0.0-1.0）
    :return: 攻击后的模型参数
    """
    attacked_params = {}
    for key, value in model_params.items():
        if isinstance(value, torch.Tensor):
            # 生成随机参数，形状与原始参数相同
            random_params = torch.randn_like(value) * attack_strength
            # 混合原始参数和随机参数
            attacked_params[key] = (1 - attack_strength) * value + random_params
        else:
            attacked_params[key] = value
    return attacked_params


def apply_label_flip_attack(labels, flip_map):
    """
    标签翻转攻击：翻转数据标签
    :param labels: 原始标签
    :param flip_map: 标签翻转映射 {原标签: 新标签}
    :return: 翻转后的标签
    """
    flipped_labels = labels.clone()
    for original_label, new_label in flip_map.items():
        mask = (labels == original_label)
        flipped_labels[mask] = new_label
    return flipped_labels


def apply_model_poison_attack(model_params, global_params, device, attack_strength=1.0):
    """
    模型投毒攻击：返回与全局模型相反的更新（反向梯度攻击）
    :param model_params: 客户端训练后的模型参数
    :param global_params: 全局模型参数
    :param device: 设备
    :param attack_strength: 攻击强度（0.0-1.0）
    :return: 攻击后的模型参数
    """
    attacked_params = {}
    for key in model_params:
        if key in global_params:
            # 计算更新量
            update = model_params[key] - global_params[key].to(device)
            # 反向更新（投毒攻击）
            poisoned_update = -attack_strength * update
            attacked_params[key] = global_params[key].to(device) + poisoned_update
        else:
            attacked_params[key] = model_params[key]
    return attacked_params


def apply_scale_attack(model_params, global_params, device, scale_factor=-1.0):
    """
    缩放攻击：放大模型参数更新（可以是正向或反向）
    :param model_params: 客户端训练后的模型参数
    :param global_params: 全局模型参数
    :param device: 设备
    :param scale_factor: 缩放因子（负值表示反向攻击）
    :return: 攻击后的模型参数
    """
    attacked_params = {}
    for key in model_params:
        if key in global_params:
            # 计算更新量
            update = model_params[key] - global_params[key].to(device)
            # 缩放更新
            scaled_update = scale_factor * update
            attacked_params[key] = global_params[key].to(device) + scaled_update
        else:
            attacked_params[key] = model_params[key]
    return attacked_params


def apply_backdoor_attack(data, labels, backdoor_pattern, target_label, attack_ratio=0.1):
    """
    后门攻击：在数据中植入后门模式，并修改标签
    :param data: 输入数据 (N, C, T, V, M)
    :param labels: 原始标签
    :param backdoor_pattern: 后门模式（与data形状相同的tensor）
    :param target_label: 后门目标标签
    :param attack_ratio: 攻击比例（多少比例的数据被植入后门）
    :return: (攻击后的数据, 攻击后的标签)
    """
    attacked_data = data.clone()
    attacked_labels = labels.clone()
    
    # 随机选择一部分数据进行后门攻击
    num_samples = data.size(0)
    num_attack = int(num_samples * attack_ratio)
    if num_attack > 0:
        attack_indices = torch.randperm(num_samples)[:num_attack]
        
        if backdoor_pattern is not None:
            # 植入后门模式
            for idx in attack_indices:
                attacked_data[idx] = attacked_data[idx] + backdoor_pattern.to(data.device)
                attacked_labels[idx] = target_label
    
    return attacked_data, attacked_labels

