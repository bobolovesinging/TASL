"""
准备MNIST客户端数据的脚本
支持三种划分方式：IID、LDA (Non-IID)、Sharding (Non-IID)
会自动下载MNIST数据集
"""
import os
import sys
import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from typing import Dict, List, Tuple

# 添加 CrosSCLR 项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==================== 配置参数 ====================
# 设置客户端数量
client_num = 10
# 数据保存路径: data/mnist/client_X
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
base_data_dir = os.path.join(project_root, 'data', 'mnist')

# ==================== 划分方式函数 ====================

def iid_split(labels: np.ndarray, client_num: int) -> Dict[int, List[int]]:
    """
    IID划分：每个客户端都有所有类别，且每个类别的数据量和比例大致相等
    """
    print("\n[IID划分] 开始IID数据划分...")
    client_indices = {i: [] for i in range(client_num)}
    
    # 获取每个类别的索引
    num_classes = 10
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    for c in range(num_classes):
        indices = class_indices[c]
        np.random.shuffle(indices)
        
        # 将该类别的数据平均分给所有客户端
        split_indices = np.array_split(indices, client_num)
        
        for i in range(client_num):
            client_indices[i].extend(split_indices[i])
            
    return client_indices


def lda_split(labels: np.ndarray, client_num: int, alpha: float = 0.5) -> Dict[int, List[int]]:
    """
    LDA (Latent Dirichlet Allocation) 划分：使用Dirichlet分布创建Non-IID数据分布
    """
    print(f"\n[LDA划分] 开始LDA数据划分 (alpha={alpha})...")
    
    num_classes = 10
    client_indices = {i: [] for i in range(client_num)}
    
    # 获取每个类别的索引
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    # 为每个类别生成客户端分配比例
    np.random.seed(42)
    
    # 记录每个客户端已分配的样本数（可选，用于调试）
    
    for c in range(num_classes):
        indices = class_indices[c]
        np.random.shuffle(indices)
        num_samples = len(indices)
        
        # 使用Dirichlet分布生成比例
        proportions = np.random.dirichlet([alpha] * client_num)

        # 使用 cumulative sum 确保分配所有样本
        proportions = (np.cumsum(proportions) * num_samples).astype(int)[:-1]
        
        split_indices = np.split(indices, proportions)
        
        for i in range(client_num):
            client_indices[i].extend(split_indices[i])
            
    return client_indices


def sharding_split(labels: np.ndarray, client_num: int) -> Dict[int, List[int]]:
    """
    Sharding划分：每个客户端只拥有少数几个类别（高度Non-IID）
    例如：MNIST有10类，10个客户端，每个客户端分2个分片，每个分片包含一个类别的部分数据
    这里简化实现：将数据按标签排序，然后切分成 2 * client_num 个分片，每个客户端随机分2个分片
    """
    print("\n[Sharding划分] 开始Sharding数据划分...")
    
    num_classes = 10
    shards_per_client = 2
    total_shards = client_num * shards_per_client
    
    # 对索引按标签排序
    idxs = np.argsort(labels)
    # 标签也按相同顺序排序（用于校验）
    # sorted_labels = labels[idxs]
    
    # 获取总样本数
    num_samples = len(labels)
    shard_size = num_samples // total_shards
    
    # 创建分片索引列表
    shards = [idxs[i * shard_size : (i + 1) * shard_size] for i in range(total_shards)]
    
    if len(shards) > total_shards:
        # 如果有剩余（整除不了），由于numpy split特性可能不会发生，但手动切片可能会有剩余
        # 简单起见，忽略最后的一点点余数，或者把余数加到最后一个分片
        pass
        
    # 随机分配分片给客户端
    random_shard_idxs = np.random.permutation(total_shards)
    client_indices = {i: [] for i in range(client_num)}
    
    for i in range(client_num):
        # 每个客户端分配 shards_per_client 个分片
        for j in range(shards_per_client):
            shard_idx = random_shard_idxs[i * shards_per_client + j]
            client_indices[i].extend(shards[shard_idx])
            
    return client_indices


# ==================== 主函数 ====================

def prepare_data(split_method: str = 'iid', alpha: float = 0.5):
    """
    准备MNIST客户端数据
    """
    # 1. 下载并加载MNIST数据
    print("下载/加载 MNIST 数据...")
    # 临时目录用于下载
    temp_data_dir = os.path.join(base_data_dir, 'temp')
    os.makedirs(temp_data_dir, exist_ok=True)
    
    train_dataset = datasets.MNIST(root=temp_data_dir, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root=temp_data_dir, train=False, download=True, transform=transforms.ToTensor())
    
    # 获取训练数据和标签
    train_data = train_dataset.data.numpy() # (60000, 28, 28) uint8
    train_labels = train_dataset.targets.numpy() # (60000,) int64

    # 保存完整训练集（用于脚本内 Dirichlet 划分）
    full_train_dir = os.path.join(base_data_dir, 'train_full')
    os.makedirs(full_train_dir, exist_ok=True)
    full_train_data = train_data.astype(np.float32) / 255.0
    full_train_data = np.expand_dims(full_train_data, axis=1)
    np.save(os.path.join(full_train_dir, 'data.npy'), full_train_data)
    np.save(os.path.join(full_train_dir, 'label.npy'), train_labels)

    # 保存 root_dataset（每类固定数量样本，用于 FLTrust g0）
    root_dir = os.path.join(base_data_dir, 'root_dataset')
    os.makedirs(root_dir, exist_ok=True)
    root_per_class = 10
    root_indices = []
    for c in range(10):
        indices = np.where(train_labels == c)[0]
        np.random.shuffle(indices)
        root_indices.extend(indices[:root_per_class].tolist())
    root_indices = np.array(root_indices, dtype=np.int64)
    root_data = full_train_data[root_indices]
    root_labels = train_labels[root_indices]
    np.save(os.path.join(root_dir, 'data.npy'), root_data)
    np.save(os.path.join(root_dir, 'label.npy'), root_labels)

    root_counts = {int(c): int((root_labels == c).sum()) for c in np.unique(root_labels)}
    print(f"Root dataset class distribution: {root_counts}")

    print(f"训练集形状: {train_data.shape}, 标签形状: {train_labels.shape}")
    
    # 2. 根据划分方式分配索引
    if split_method.lower() == 'iid':
        client_indices = iid_split(train_labels, client_num)
    elif split_method.lower() == 'lda':
        client_indices = lda_split(train_labels, client_num, alpha=alpha)
    elif split_method.lower() == 'sharding':
        client_indices = sharding_split(train_labels, client_num)
    else:
        raise ValueError(f"不支持的划分方式: {split_method}")
        
    # 3. 保存客户端数据
    print("\n开始保存客户端数据...")
    
    # 归一化并添加通道维度 (N, 28, 28) -> (N, 1, 28, 28) float32
    # 注意：为了节省空间，我们也可以保存uint8，在训练时归一化。但这里为了方便直接使用float32
    # 为了保持与NTU60脚本的一致性，我们将数据保存为numpy数组
    
    for client_id in range(client_num):
        indices = client_indices[client_id]
        indices = np.array(indices)
        np.random.shuffle(indices) # 打乱顺序
        
        # 提取数据
        c_data = train_data[indices]
        c_labels = train_labels[indices]
        
        # 转换格式： (N, 28, 28) -> (N, 1, 28, 28) normalized float32
        c_data = c_data.astype(np.float32) / 255.0
        c_data = np.expand_dims(c_data, axis=1)
        
        # 保存路径
        client_path = os.path.join(base_data_dir, f'client_{client_id+1}')
        os.makedirs(client_path, exist_ok=True)
        
        np.save(os.path.join(client_path, 'data.npy'), c_data)
        np.save(os.path.join(client_path, 'label.npy'), c_labels)
        
        # 保存索引（可选）
        np.save(os.path.join(client_path, 'indices.npy'), indices)
        
        unique_classes = np.unique(c_labels)
        print(f"客户端 {client_id+1}: {len(c_data)} 个样本, 类别数: {len(unique_classes)}, 类别: {unique_classes}")
        
    # 保存测试集（作为统一的测试集）
    print("\n保存测试集...")
    test_path = os.path.join(base_data_dir, 'test_dataset')
    os.makedirs(test_path, exist_ok=True)
    
    test_data = test_dataset.data.numpy().astype(np.float32) / 255.0
    test_data = np.expand_dims(test_data, axis=1)
    test_labels = test_dataset.targets.numpy()
    
    np.save(os.path.join(test_path, 'data.npy'), test_data)
    np.save(os.path.join(test_path, 'label.npy'), test_labels)
    print(f"测试集: {len(test_data)} 个样本")
    
    print(f"\n[完成] 数据已保存在: {base_data_dir}")


def main():
    parser = argparse.ArgumentParser(description='准备MNIST客户端数据')
    parser.add_argument('--method', type=str, default='iid', 
                       choices=['iid', 'lda', 'sharding'],
                       help='数据划分方式')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='LDA参数')
    parser.add_argument('--client_num', type=int, default=10,
                       help='客户端数量')
    
    args = parser.parse_args()
    
    global client_num
    client_num = args.client_num
    
    print("=" * 70)
    print("MNIST 客户端数据准备")
    print("=" * 70)
    
    prepare_data(split_method=args.method, alpha=args.alpha)


# NOTE: moved to scripts/data_prep/prepare_mnist_data.py
if __name__ == '__main__':
    main()
