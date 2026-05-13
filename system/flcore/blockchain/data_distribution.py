"""
数据分布统计模块
用于计算和共享客户端的数据分布信息，解决no-iid问题
"""
import numpy as np
import torch
from typing import Dict, List, Any, Optional
from collections import Counter


class DataDistributionStats:
    """数据分布统计信息"""
    
    def __init__(self, client_id: int, num_classes: int):
        """
        初始化数据分布统计
        :param client_id: 客户端ID
        :param num_classes: 类别数量
        """
        self.client_id = client_id
        self.num_classes = num_classes
        self.class_distribution: Dict[int, int] = {}  # 每个类别的样本数量
        self.total_samples: int = 0
        self.class_weights: Dict[int, float] = {}  # 每个类别的权重（归一化）
        self.feature_mean: Optional[np.ndarray] = None  # 特征均值
        self.feature_std: Optional[np.ndarray] = None  # 特征标准差
        
    def compute_from_data(self, data_loader, model=None, device='cuda'):
        """
        从数据加载器计算数据分布统计
        :param data_loader: 数据加载器
        :param model: 可选的模型，用于提取特征统计
        :param device: 设备
        """
        self.class_distribution = {}
        self.total_samples = 0
        all_features = []
        all_labels = []
        
        # 如果是eval模式，确保模型在eval模式
        if model is not None:
            model.eval()
        
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(data_loader):
                if isinstance(data, list):
                    data = data[0]
                
                data = data.to(device) if torch.is_tensor(data) else data
                labels = labels.to(device) if torch.is_tensor(labels) else labels
                
                # 统计类别分布
                labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
                unique, counts = np.unique(labels_np, return_counts=True)
                for cls, cnt in zip(unique, counts):
                    cls = int(cls)
                    self.class_distribution[cls] = self.class_distribution.get(cls, 0) + int(cnt)
                
                self.total_samples += len(labels_np)
                
                # 如果提供了模型，提取特征统计
                if model is not None:
                    try:
                        # 尝试提取特征
                        if hasattr(model, 'encoder_q'):
                            # SkeletonCLR模型
                            features, _, _ = model.encoder_q(data)
                        elif hasattr(model, 'forward'):
                            # 其他模型，使用forward获取特征
                            output = model(data)
                            if isinstance(output, tuple):
                                features = output[0]
                            else:
                                features = output
                        else:
                            features = None
                        
                        if features is not None:
                            features_np = features.cpu().numpy() if torch.is_tensor(features) else features
                            # 展平特征
                            if len(features_np.shape) > 2:
                                features_np = features_np.reshape(features_np.shape[0], -1)
                            all_features.append(features_np)
                            all_labels.append(labels_np)
                    except Exception as e:
                        print(f"Warning: Failed to extract features: {e}")
                        continue
        
        # 计算类别权重
        if self.total_samples > 0:
            for cls in range(self.num_classes):
                count = self.class_distribution.get(cls, 0)
                self.class_weights[cls] = count / self.total_samples if self.total_samples > 0 else 0.0
        
        # 计算特征统计
        if len(all_features) > 0:
            all_features = np.concatenate(all_features, axis=0)
            self.feature_mean = np.mean(all_features, axis=0)
            self.feature_std = np.std(all_features, axis=0)
        
        return self
    
    def compute_from_labels(self, labels: List[int] or np.ndarray):
        """
        仅从标签计算类别分布（更轻量级的方法）
        :param labels: 标签列表或数组
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        elif isinstance(labels, list):
            labels = np.array(labels)
        
        self.class_distribution = dict(Counter(labels))
        self.total_samples = len(labels)
        
        # 计算类别权重
        for cls in range(self.num_classes):
            count = self.class_distribution.get(cls, 0)
            self.class_weights[cls] = count / self.total_samples if self.total_samples > 0 else 0.0
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于序列化"""
        # 确保所有键都是Python原生类型（不是numpy类型）
        class_dist = {int(k): int(v) for k, v in self.class_distribution.items()}
        class_weights = {int(k): float(v) for k, v in self.class_weights.items()}
        
        result = {
            'client_id': int(self.client_id),
            'num_classes': int(self.num_classes),
            'class_distribution': class_dist,
            'total_samples': int(self.total_samples),
            'class_weights': class_weights
        }
        
        if self.feature_mean is not None:
            result['feature_mean'] = [float(x) for x in self.feature_mean.tolist()]
        if self.feature_std is not None:
            result['feature_std'] = [float(x) for x in self.feature_std.tolist()]
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataDistributionStats':
        """从字典恢复"""
        stats = cls(data['client_id'], data['num_classes'])
        stats.class_distribution = {int(k): int(v) for k, v in data['class_distribution'].items()}
        stats.total_samples = data['total_samples']
        stats.class_weights = {int(k): float(v) for k, v in data['class_weights'].items()}
        
        if 'feature_mean' in data and data['feature_mean'] is not None:
            stats.feature_mean = np.array(data['feature_mean'])
        if 'feature_std' in data and data['feature_std'] is not None:
            stats.feature_std = np.array(data['feature_std'])
        
        return stats
    
    def get_class_imbalance_ratio(self) -> float:
        """
        计算类别不平衡比例
        返回最大类别样本数 / 最小类别样本数（排除0）
        """
        non_zero_counts = [cnt for cnt in self.class_distribution.values() if cnt > 0]
        if len(non_zero_counts) < 2:
            return 1.0
        return max(non_zero_counts) / min(non_zero_counts)


def aggregate_distributions(distributions: List[DataDistributionStats]) -> Dict[str, Any]:
    """
    聚合多个客户端的数据分布
    :param distributions: 数据分布统计列表
    :return: 聚合后的全局分布信息
    """
    if not distributions:
        return {}
    
    num_classes = distributions[0].num_classes
    total_samples = sum(d.total_samples for d in distributions)
    
    # 聚合类别分布（确保使用Python原生int类型）
    global_class_distribution = {}
    global_class_weights = {}
    
    for cls in range(num_classes):
        total_count = sum(d.class_distribution.get(cls, 0) for d in distributions)
        global_class_distribution[int(cls)] = int(total_count)
        global_class_weights[int(cls)] = float(total_count / total_samples if total_samples > 0 else 0.0)
    
    # 聚合特征统计（如果有）
    global_feature_mean = None
    global_feature_std = None
    
    distributions_with_features = [d for d in distributions if d.feature_mean is not None]
    if distributions_with_features:
        # 加权平均特征均值
        weighted_means = []
        weights = []
        for d in distributions_with_features:
            weighted_means.append(d.feature_mean * d.total_samples)
            weights.append(d.total_samples)
        
        if weights:
            total_weight = sum(weights)
            global_feature_mean = sum(weighted_means) / total_weight if total_weight > 0 else None
        
        # 计算全局标准差（简化版本）
        if distributions_with_features[0].feature_std is not None:
            global_feature_std = np.mean([d.feature_std for d in distributions_with_features], axis=0)
    
    return {
        'global_class_distribution': global_class_distribution,
        'global_class_weights': global_class_weights,
        'total_samples': int(total_samples),
        'num_clients': int(len(distributions)),
        'feature_mean': [float(x) for x in global_feature_mean.tolist()] if global_feature_mean is not None else None,
        'feature_std': [float(x) for x in global_feature_std.tolist()] if global_feature_std is not None else None
    }


def compute_distribution_similarity(dist1: DataDistributionStats, 
                                   dist2: DataDistributionStats) -> float:
    """
    计算两个数据分布之间的相似度（使用KL散度）
    :param dist1: 第一个分布
    :param dist2: 第二个分布
    :return: 相似度分数（0-1，1表示完全相同）
    """
    import scipy.stats as stats
    
    # 构建概率分布向量
    p1 = np.array([dist1.class_weights.get(i, 0.0) for i in range(dist1.num_classes)])
    p2 = np.array([dist2.class_weights.get(i, 0.0) for i in range(dist2.num_classes)])
    
    # 添加小的epsilon避免log(0)
    epsilon = 1e-10
    p1 = p1 + epsilon
    p2 = p2 + epsilon
    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum()
    
    # 计算KL散度（对称版本）
    kl_div = 0.5 * (stats.entropy(p1, p2) + stats.entropy(p2, p1))
    
    # 转换为相似度（0-1）
    similarity = np.exp(-kl_div)
    
    return float(similarity)

