"""
工具模块
包含: YAML解析、随机种子、日志配置、数据划分、绘图工具
"""
import os
import yaml
import random
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from datetime import datetime


# ==================== 配置加载 ====================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def merge_args_with_config(config: Dict, args_dict: Dict) -> Dict:
    """
    命令行参数覆盖配置文件

    Args:
        config: YAML配置字典
        args_dict: 命令行参数字典

    Returns:
        合并后的配置
    """
    merged = config.copy()
    for key, value in args_dict.items():
        if value is not None:
            # 支持嵌套键: "aggregator.name"
            keys = key.split('.')
            d = merged
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value
    return merged


# ==================== 随机种子 ====================

def set_seed(seed: int):
    """
    设置随机种子确保可复现性

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==================== 日志配置 ====================

def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    配置日志记录器

    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        level: 日志级别

    Returns:
        配置好的Logger对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricsLogger:
    """
    指标记录器：记录训练过程中的accuracy和loss
    """

    def __init__(self, save_dir: str = "./results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.history = {
            'round': [],
            'train_acc': [],
            'test_acc': [],
            'train_loss': [],
            'test_loss': [],
            'malicious_clients': [],
            'trust_weights': {}
        }

        # 创建带时间戳的日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(save_dir, f"metrics_{timestamp}.txt")

    def log_round(self, round_idx: int, metrics: Dict):
        """记录单轮指标"""
        self.history['round'].append(round_idx)
        self.history['train_acc'].append(metrics.get('train_acc', 0))
        self.history['test_acc'].append(metrics.get('test_acc', 0))
        self.history['train_loss'].append(metrics.get('train_loss', 0))
        self.history['test_loss'].append(metrics.get('test_loss', 0))
        self.history['malicious_clients'].append(metrics.get('malicious_clients', []))

        if 'trust_weights' in metrics:
            self.history['trust_weights'][round_idx] = metrics['trust_weights']

    def print_summary(self, round_idx: int, metrics: Dict):
        """打印轮次摘要"""
        msg = f"Round {round_idx:3d} | " \
              f"Train Acc: {metrics.get('train_acc', 0):6.2f}% | " \
              f"Test Acc: {metrics.get('test_acc', 0):6.2f}% | " \
              f"Train Loss: {metrics.get('train_loss', 0):.4f} | " \
              f"Test Loss: {metrics.get('test_loss', 0):.4f}"

        if metrics.get('malicious_clients'):
            msg += f" | Malicious: {metrics['malicious_clients']}"

        print(msg)

        # 写入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    def save_history(self, filename: str = "history.pth"):
        """保存训练历史"""
        save_path = os.path.join(self.save_dir, filename)
        torch.save(self.history, save_path)
        return save_path


# ==================== 数据划分 ====================

def iid_split(labels: np.ndarray, num_clients: int) -> Dict[int, List[int]]:
    """
    IID划分：每个客户端获得相同比例的各类别数据

    Args:
        labels: 数据标签数组
        num_clients: 客户端数量

    Returns:
        {client_id: [sample_indices]}
    """
    client_indices = {i: [] for i in range(num_clients)}
    num_classes = len(np.unique(labels))

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        splits = np.array_split(class_indices, num_clients)
        for i in range(num_clients):
            client_indices[i].extend(splits[i].tolist())

    return client_indices


def lda_split(labels: np.ndarray, num_clients: int, alpha: float = 0.5) -> Dict[int, List[int]]:
    """
    LDA划分：使用Dirichlet分布创建Non-IID数据分布

    Args:
        labels: 数据标签数组
        num_clients: 客户端数量
        alpha: Dirichlet分布参数（越小越倾斜）

    Returns:
        {client_id: [sample_indices]}
    """
    num_classes = len(np.unique(labels))
    client_indices = {i: [] for i in range(num_clients)}

    np.random.seed(42)  # 可复现

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)

        # Dirichlet分布生成分配比例
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        splits = np.split(class_indices, proportions)

        for i in range(num_clients):
            client_indices[i].extend(splits[i].tolist())

    return client_indices


def sharding_split(labels: np.ndarray, num_clients: int, shards_per_client: int = 2) -> Dict[int, List[int]]:
    """
    Sharding划分：每个客户端拥有少数类别（高度Non-IID）

    Args:
        labels: 数据标签数组
        num_clients: 客户端数量
        shards_per_client: 每个客户端的分片数

    Returns:
        {client_id: [sample_indices]}
    """
    num_classes = len(np.unique(labels))
    total_shards = num_clients * shards_per_client

    # 按标签排序索引
    sorted_indices = np.argsort(labels)
    shard_size = len(labels) // total_shards

    shards = [sorted_indices[i * shard_size:(i + 1) * shard_size]
              for i in range(total_shards)]

    # 随机分配分片给客户端
    np.random.shuffle(shards)
    client_indices = {i: [] for i in range(num_clients)}

    for i in range(num_clients):
        for j in range(shards_per_client):
            client_indices[i].extend(shards[i * shards_per_client + j].tolist())

    return client_indices


# ==================== 工具函数 ====================

def get_device() -> torch.device:
    """获取可用的计算设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """计算模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_accuracy(model: torch.nn.Module, data_loader, device: torch.device) -> Tuple[float, float]:
    """
    计算模型准确率和损失

    Returns:
        (accuracy, loss)
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item() * len(target)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return 100.0 * correct / total, total_loss / total


def flatten_gradients(gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    将梯度字典展平为1D张量

    Args:
        gradients: {param_name: gradient_tensor}

    Returns:
        展平后的梯度向量
    """
    flat_list = []
    for name in sorted(gradients.keys()):
        param = gradients[name]
        if isinstance(param, torch.Tensor):
            flat_list.append(param.detach().view(-1).float())
    return torch.cat(flat_list) if flat_list else torch.tensor([])


def compute_cosine_similarity(grad1: torch.Tensor, grad2: torch.Tensor) -> float:
    """计算两个向量的余弦相似度"""
    if grad1.numel() == 0 or grad2.numel() == 0:
        return 0.0

    norm1 = torch.norm(grad1)
    norm2 = torch.norm(grad2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return torch.dot(grad1, grad2) / (norm1 * norm2)
