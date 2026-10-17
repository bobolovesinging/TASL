"""
联邦学习仿真框架 - 主程序
配置驱动：所有实验参数通过 config.yaml 设置
"""
import os
import sys
import argparse
import numpy as np
import torch
from torchvision import datasets, transforms

# 导入项目模块
from utils import load_config, set_seed, iid_split, lda_split, sharding_split, setup_logger
from trainer import FedServer, FedClient
from models import get_model


def prepare_data(config: dict):
    """
    准备数据集

    Args:
        config: 配置字典

    Returns:
        (client_data, client_labels, test_data, test_labels)
    """
    data_cfg = config['data']
    global_cfg = config['global']

    dataset_name = data_cfg.get('name', 'mnist').lower()
    num_clients = global_cfg.get('num_clients', 10)
    split_method = data_cfg.get('split', 'iid')
    alpha = data_cfg.get('alpha', 0.3)

    # 数据目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_root, 'data', dataset_name)
    temp_dir = os.path.join(data_root, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # 下载数据集
    print(f"[Data] Loading {dataset_name} dataset...")

    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST(
            root=temp_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        test_dataset = datasets.MNIST(
            root=temp_dir,
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
    elif dataset_name == 'fashionmnist':
        train_dataset = datasets.FashionMNIST(
            root=temp_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        test_dataset = datasets.FashionMNIST(
            root=temp_dir,
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
    elif dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root=temp_dir,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )
        test_dataset = datasets.CIFAR10(
            root=temp_dir,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # 获取训练数据和标签
    if hasattr(train_dataset, 'data'):
        train_data = train_dataset.data.numpy()
    else:
        train_data = np.array([train_dataset[i][0].numpy() for i in range(len(train_dataset))])

    if hasattr(train_dataset, 'targets'):
        train_labels = train_dataset.targets.numpy()
    else:
        train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])

    # 预处理
    if dataset_name in ['mnist', 'fashionmnist']:
        train_data = train_data.astype(np.float32) / 255.0
        if len(train_data.shape) == 3:
            train_data = np.expand_dims(train_data, axis=1)
    elif dataset_name == 'cifar10':
        train_data = train_data.astype(np.float32) / 255.0
        train_data = train_data.transpose(0, 3, 1, 2)  # NHWC -> NCHW

    # 数据划分
    print(f"[Data] Splitting data ({split_method})...")

    if split_method.lower() == 'iid':
        client_indices = iid_split(train_labels, num_clients)
    elif split_method.lower() == 'lda':
        client_indices = lda_split(train_labels, num_clients, alpha)
    elif split_method.lower() == 'sharding':
        client_indices = sharding_split(train_labels, num_clients)
    else:
        print(f"[Warning] Unknown split method '{split_method}', using IID")
        client_indices = iid_split(train_labels, num_clients)

    # 分配客户端数据
    client_data = []
    client_labels = []

    for i in range(num_clients):
        indices = client_indices[i]
        c_data = train_data[indices]
        c_labels = train_labels[indices]
        client_data.append(c_data)
        client_labels.append(c_labels)
        print(f"  Client {i}: {len(c_data)} samples")

    # 准备测试集
    if hasattr(test_dataset, 'data'):
        test_data = test_dataset.data.numpy()
    else:
        test_data = np.array([test_dataset[i][0].numpy() for i in range(len(test_dataset))])

    if hasattr(test_dataset, 'targets'):
        test_labels = test_dataset.targets.numpy()
    else:
        test_labels = np.array([test_dataset[i][1] for i in range(len(test_dataset))])

    if dataset_name in ['mnist', 'fashionmnist']:
        test_data = test_data.astype(np.float32) / 255.0
        if len(test_data.shape) == 3:
            test_data = np.expand_dims(test_data, axis=1)
    elif dataset_name == 'cifar10':
        test_data = test_data.astype(np.float32) / 255.0
        test_data = test_data.transpose(0, 3, 1, 2)

    return client_data, client_labels, test_data, test_labels


def create_clients(config: dict, client_data, client_labels):
    """
    创建联邦学习客户端

    Args:
        config: 配置字典
        client_data: 客户端数据列表
        client_labels: 客户端标签列表

    Returns:
        客户端列表
    """
    from attacks import get_attack

    model_cfg = config.get('model', {'name': 'simple_cnn', 'num_classes': 10})
    data_cfg = config['data']
    attack_cfg = config['attack']

    clients = []

    for i in range(len(client_data)):
        # 创建模型
        model = get_model(
            model_name=model_cfg.get('name', 'simple_cnn'),
            num_classes=model_cfg.get('num_classes', 10),
            dataset=data_cfg.get('name', 'mnist')
        )

        # 创建攻击（每个客户端独立）
        attack = get_attack(
            name=attack_cfg.get('name', 'none'),
            malicious_ratio=0.0,  # 恶意比例由服务器统一控制
            config=attack_cfg.get('params', {}),
            num_classes=model_cfg.get('num_classes', 10)
        )

        # 创建客户端
        client = FedClient(
            client_id=i,
            data=client_data[i],
            labels=client_labels[i],
            model=model,
            attack=attack
        )
        clients.append(client)

    return clients


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Federated Learning Simulation')

    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--rounds', type=int, default=None,
                        help='覆盖配置中的轮数')
    parser.add_argument('--clients', type=int, default=None,
                        help='覆盖配置中的客户端数')
    parser.add_argument('--lr', type=float, default=None,
                        help='覆盖学习率')
    parser.add_argument('--attack', type=str, default=None,
                        help='覆盖攻击类型')
    parser.add_argument('--malicious_ratio', type=float, default=None,
                        help='覆盖恶意比例')
    parser.add_argument('--aggregator', type=str, default=None,
                        help='覆盖聚合算法')
    parser.add_argument('--output', type=str, default=None,
                        help='覆盖输出目录')
    parser.add_argument('--seed', type=int, default=None,
                        help='覆盖随机种子')

    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 加载配置
    print(f"[Config] Loading from {args.config}")
    config = load_config(args.config)

    # 命令行参数覆盖
    if args.rounds:
        config['global']['rounds'] = args.rounds
    if args.clients:
        config['global']['num_clients'] = args.clients
    if args.lr:
        config['local']['lr'] = args.lr
    if args.attack:
        config['attack']['name'] = args.attack
    if args.malicious_ratio is not None:
        config['attack']['malicious_ratio'] = args.malicious_ratio
    if args.aggregator:
        config['aggregator']['name'] = args.aggregator
    if args.output:
        config['output']['save_dir'] = args.output
    if args.seed:
        config['global']['seed'] = args.seed

    # 设置随机种子
    seed = config['global'].get('seed', 42)
    set_seed(seed)
    print(f"[Seed] Random seed: {seed}")

    # 准备数据
    client_data, client_labels, test_data, test_labels = prepare_data(config)

    # 创建客户端
    print("[Clients] Creating clients...")
    clients = create_clients(config, client_data, client_labels)

    # 创建服务器
    print("[Server] Initializing server...")
    server = FedServer(config, test_data, test_labels)

    # 训练
    results = server.train(clients)

    # 保存模型
    if config.get('output', {}).get('save_model', False):
        save_dir = config.get('output', {}).get('save_dir', './results')
        model_path = os.path.join(save_dir, 'global_model.pth')
        server.save_model(model_path)

    return results


if __name__ == '__main__':
    main()
