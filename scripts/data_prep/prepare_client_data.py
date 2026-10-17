"""
准备客户端数据的脚本
将NTU60数据集分配给多个客户端用于联邦学习
支持三种划分方式：IID、LDA (Non-IID)、Sharding (Non-IID)
"""
import os
import sys
import pickle
import argparse
import numpy as np
from typing import Dict, List, Tuple

# 添加 CrosSCLR 项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==================== 配置参数 ====================
# 设置客户端数量
client_num = 10
client_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Client_datasets_10')

# NTU60 数据路径（使用 xview 或 xsub，这里使用 xview）
# 可以根据需要修改为 xsub
data_path = '../data/NTU60_frame50/xview/train_position.npy'
label_path = '../data/NTU-RGB-D/xview/train_label.pkl'

# ==================== 划分方式函数 ====================

def iid_split(class_data: Dict[int, List[int]], client_num: int, labels: np.ndarray) -> Dict[int, Tuple[List[int], List[int]]]:
    """
    IID划分：每个客户端都有所有类别，且每个类别的数据量大致相等
    :param class_data: {class_id: [indices]}
    :param client_num: 客户端数量
    :param labels: 标签数组
    :return: {client_id: (indices, labels)}
    """
    print("\n[IID划分] 开始IID数据划分...")
    client_data = {i: ([], []) for i in range(client_num)}
    
    num_classes = len(class_data)
    samples_per_client_per_class = [len(class_data[i]) // client_num for i in range(num_classes)]
    remain_samples_per_class = [len(class_data[i]) % client_num for i in range(num_classes)]
    
    for client_id in range(client_num):
        client_indices = []
        client_labels = []
        
        # 为每个类别分配数据
        for class_id in range(num_classes):
            start_idx = client_id * samples_per_client_per_class[class_id]
            end_idx = (client_id + 1) * samples_per_client_per_class[class_id]
            selected_indices = class_data[class_id][start_idx:end_idx]
            client_indices.extend(selected_indices)
            client_labels.extend([class_id] * len(selected_indices))
        
        # 处理剩余数据
        for class_id in range(num_classes):
            if remain_samples_per_class[class_id] > 0:
                remain_start = client_id * remain_samples_per_class[class_id] // client_num
                remain_end = (client_id + 1) * remain_samples_per_class[class_id] // client_num
                if remain_end > remain_start:
                    selected_indices = class_data[class_id][-remain_samples_per_class[class_id]:][remain_start:remain_end]
                    client_indices.extend(selected_indices)
                    client_labels.extend([class_id] * len(selected_indices))
        
        client_data[client_id] = (client_indices, client_labels)
    
    return client_data


def lda_split(class_data: Dict[int, List[int]], client_num: int, labels: np.ndarray, alpha: float = 0.5) -> Dict[int, Tuple[List[int], List[int]]]:
    """
    LDA (Latent Dirichlet Allocation) 划分：使用Dirichlet分布创建Non-IID数据分布
    每个客户端的数据分布不同，但可能包含多个类别
    :param class_data: {class_id: [indices]}
    :param client_num: 客户端数量
    :param labels: 标签数组
    :param alpha: Dirichlet分布的参数，alpha越小，数据分布越不均匀（Non-IID程度越高）
    :return: {client_id: (indices, labels)}
    """
    print(f"\n[LDA划分] 开始LDA数据划分 (alpha={alpha})...")
    print("  - alpha越小，Non-IID程度越高")
    print("  - alpha=0.1: 高度Non-IID（每个客户端主要只有少数类别）")
    print("  - alpha=1.0: 接近IID")
    print("  - alpha=0.5: 中等Non-IID（推荐）")
    
    num_classes = len(class_data)
    client_data = {i: ([], []) for i in range(client_num)}
    
    # 为每个类别生成客户端分配比例（使用Dirichlet分布）
    np.random.seed(42)  # 固定随机种子以确保可重复性
    
    for class_id in range(num_classes):
        class_indices = class_data[class_id]
        num_samples = len(class_indices)
        
        if num_samples == 0:
            continue
        
        # 使用Dirichlet分布生成每个客户端应该分配的比例
        # alpha越小，分布越不均匀
        proportions = np.random.dirichlet([alpha] * client_num)
        
        # 根据比例分配样本
        assigned_samples = 0
        for client_id in range(client_num):
            # 计算这个客户端应该分配多少样本
            num_for_client = int(proportions[client_id] * num_samples)
            
            # 确保不超过剩余样本数
            remaining = num_samples - assigned_samples
            if client_id == client_num - 1:
                # 最后一个客户端分配所有剩余样本
                num_for_client = remaining
            else:
                num_for_client = min(num_for_client, remaining)
            
            if num_for_client > 0:
                start_idx = assigned_samples
                end_idx = assigned_samples + num_for_client
                selected_indices = class_indices[start_idx:end_idx]
                
                client_data[client_id][0].extend(selected_indices)
                client_data[client_id][1].extend([class_id] * num_for_client)
                
                assigned_samples += num_for_client
    
    return client_data


def sharding_split(class_data: Dict[int, List[int]], client_num: int, labels: np.ndarray) -> Dict[int, Tuple[List[int], List[int]]]:
    """
    Sharding划分：每个客户端只分配部分类别（高度Non-IID）
    例如：60个类别，10个客户端，每个客户端6个类别
    :param class_data: {class_id: [indices]}
    :param client_num: 客户端数量
    :param labels: 标签数组
    :return: {client_id: (indices, labels)}
    """
    print("\n[Sharding划分] 开始Sharding数据划分...")
    print("  - 每个客户端只分配部分类别（高度Non-IID）")
    
    num_classes = len(class_data)
    classes_per_client = num_classes // client_num
    remain_classes = num_classes % client_num
    
    print(f"  - 总类别数: {num_classes}")
    print(f"  - 每个客户端类别数: {classes_per_client}")
    print(f"  - 剩余类别: {remain_classes}")
    
    client_data = {i: ([], []) for i in range(client_num)}
    
    # 为每个客户端分配类别
    assigned_classes = 0
    for client_id in range(client_num):
        # 计算这个客户端应该分配多少类别
        num_classes_for_client = classes_per_client
        if client_id < remain_classes:
            num_classes_for_client += 1  # 剩余类别分配给前几个客户端
        
        # 分配类别
        client_classes = list(range(assigned_classes, assigned_classes + num_classes_for_client))
        assigned_classes += num_classes_for_client
        
        print(f"  - 客户端 {client_id+1}: 类别 {min(client_classes)}-{max(client_classes)} ({len(client_classes)}个类别)")
        
        # 为这个客户端分配这些类别的所有数据
        for class_id in client_classes:
            client_data[client_id][0].extend(class_data[class_id])
            client_data[client_id][1].extend([class_id] * len(class_data[class_id]))
    
    return client_data


# ==================== 主函数 ====================

def prepare_data(split_method: str = 'iid', alpha: float = 0.5):
    """
    准备客户端数据
    :param split_method: 划分方式 ('iid', 'lda', 'sharding')
    :param alpha: LDA划分的参数（仅用于lda方式）
    """
# 创建客户端文件夹
print(f"创建 {client_num} 个客户端文件夹...")
for i in range(1, client_num + 1):
    os.makedirs(os.path.join(client_folder, f'client_{i}'), exist_ok=True)
print(f"[OK] 已创建 {client_num} 个客户端文件夹\n")

    # 加载NTU60数据
    print("开始读取NTU60数据...")
    if not os.path.exists(data_path):
        print(f"错误: 数据文件 {data_path} 不存在！")
        print("请确保数据文件路径正确，或修改 data_path 变量")
        sys.exit(1)
    
    if not os.path.exists(label_path):
        print(f"错误: 标签文件 {label_path} 不存在！")
        print("请确保标签文件路径正确，或修改 label_path 变量")
        sys.exit(1)
    
    # 加载数据（使用内存映射模式以节省内存）
    print(f"加载数据文件: {data_path}")
    data = np.load(data_path, mmap_mode='r')
    print(f"数据形状: {data.shape}")  # 应该是 (N, C, T, V, M)
    
    # 加载标签
    print(f"加载标签文件: {label_path}")
    with open(label_path, 'rb') as f:
        sample_name, labels = pickle.load(f)
    labels = np.array(labels)
    
    print(f"标签形状: {labels.shape}")
    print(f"类别数量: {len(np.unique(labels))}")
    print(f"总样本数: {len(labels)}\n")

    # 按类别组织数据
    num_classes = 60  # NTU60有60个类别
    class_data = {i: [] for i in range(num_classes)}
    
    for idx, label in enumerate(labels):
        class_data[label].append(idx)
    
    print("数据统计:")
    for i in range(num_classes):
        print(f"类别 {i}: {len(class_data[i])} 个样本")
    print()
    
    # 根据划分方式分配数据
    if split_method.lower() == 'iid':
        client_data = iid_split(class_data, client_num, labels)
    elif split_method.lower() == 'lda':
        client_data = lda_split(class_data, client_num, labels, alpha=alpha)
    elif split_method.lower() == 'sharding':
        client_data = sharding_split(class_data, client_num, labels)
    else:
        raise ValueError(f"不支持的划分方式: {split_method}。支持的方式: 'iid', 'lda', 'sharding'")

    # 保存客户端数据
    print("\n开始保存客户端数据...")
    for client_id in range(client_num):
        client_indices, client_labels = client_data[client_id]
        
        # 打乱数据（保持索引和标签对应）
        if len(client_indices) > 0:
            shuffle_idx = np.random.permutation(len(client_indices))
            client_indices = [client_indices[i] for i in shuffle_idx]
            client_labels = [client_labels[i] for i in shuffle_idx]
        
        # 保存客户端数据（保存索引，实际数据在训练时加载）
        client_path = os.path.join(client_folder, f'client_{client_id+1}')
        
        # 保存数据索引和标签
        np.save(os.path.join(client_path, 'data_indices.npy'), np.array(client_indices))
        np.save(os.path.join(client_path, 'label.npy'), np.array(client_labels))
        
        # 保存原始数据路径（用于训练时加载）
        with open(os.path.join(client_path, 'data_path.txt'), 'w') as f:
            f.write(data_path)
        
        # 保存划分方式信息（用于后续分析）
        with open(os.path.join(client_path, 'split_info.txt'), 'w') as f:
            f.write(f"split_method={split_method}\n")
            if split_method.lower() == 'lda':
                f.write(f"alpha={alpha}\n")
        
        unique_classes = np.unique(client_labels)
        print(f"客户端 {client_id+1}: {len(client_indices)} 个样本, "
              f"标签范围: {min(client_labels)}-{max(client_labels)}, "
              f"类别数: {len(unique_classes)}, "
              f"类别列表: {sorted(unique_classes.tolist())}")

print(f"\n[完成] 已为 {client_num} 个客户端准备好数据")
print(f"数据保存在: {client_folder}/")
    print(f"划分方式: {split_method.upper()}")
    if split_method.lower() == 'lda':
        print(f"LDA参数 alpha: {alpha}")
    print(f"注意: 实际数据文件路径保存在每个客户端的 data_path.txt 中")


def main():
    """主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(description='准备客户端数据（支持IID、LDA、Sharding三种划分方式）')
    parser.add_argument('--method', type=str, default='iid', 
                       choices=['iid', 'lda', 'sharding'],
                       help='数据划分方式: iid (IID), lda (LDA Non-IID), sharding (Sharding Non-IID)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='LDA划分的参数（仅用于lda方式）。alpha越小，Non-IID程度越高。推荐值: 0.1-1.0')
    parser.add_argument('--client_num', type=int, default=10,
                       help='客户端数量')
    parser.add_argument('--data_path', type=str, default=None,
                       help='数据文件路径（可选，默认使用脚本中的路径）')
    parser.add_argument('--label_path', type=str, default=None,
                       help='标签文件路径（可选，默认使用脚本中的路径）')
    
    args = parser.parse_args()
    
    # 更新全局变量
    global client_num, client_folder, data_path, label_path
    client_num = args.client_num
    client_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', f'Client_datasets_{client_num}')
    
    if args.data_path:
        data_path = args.data_path
    if args.label_path:
        label_path = args.label_path
    
    print("=" * 70)
    print("NTU60 客户端数据准备")
    print("=" * 70)
    print(f"划分方式: {args.method.upper()}")
    print(f"客户端数量: {client_num}")
    if args.method.lower() == 'lda':
        print(f"LDA参数 alpha: {args.alpha}")
    print("=" * 70)
    
    prepare_data(split_method=args.method, alpha=args.alpha)


# NOTE: moved to scripts/data_prep/prepare_client_data.py
if __name__ == '__main__':
    main()
