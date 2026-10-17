"""
联邦学习训练模块
包含: FedClient, FedServer 类
"""
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm

from models import get_model
from baselines import get_aggregator
from attacks import get_attack
from utils import MetricsLogger, get_device, compute_accuracy


class FederatedDataset(Dataset):
    """联邦学习数据集包装器"""

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class FedClient:
    """
    联邦学习客户端

    功能:
    1. 本地数据加载（含攻击注入）
    2. 本地模型训练
    3. 梯度计算与上传
    """

    def __init__(
        self,
        client_id: int,
        data: np.ndarray,
        labels: np.ndarray,
        model: nn.Module,
        attack=None,
        device: torch.device = None
    ):
        self.client_id = client_id
        self.data = data
        self.labels = labels
        self.model = model
        self.attack = attack
        self.device = device or get_device()
        self.sample_count = len(labels)

        # 创建数据集
        self.dataset = FederatedDataset(data, labels)

        # 初始化模型
        self.model.to(self.device)

    def apply_attack_to_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """应用标签翻转攻击"""
        if self.attack and hasattr(self.attack, 'apply_to_labels'):
            return self.attack.apply_to_labels(labels)
        return labels

    def train(
        self,
        local_epochs: int,
        batch_size: int,
        lr: float,
        optimizer_name: str = 'sgd',
        attack_labels: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        本地训练

        Args:
            local_epochs: 本地训练轮数
            batch_size: 批大小
            lr: 学习率
            optimizer_name: 优化器名称
            attack_labels: 是否在训练时攻击标签

        Returns:
            (param_update, sample_count) - 参数更新(delta) = 原始参数 - 训练后参数
        """
        self.model.train()

        # 保存训练前的参数
        old_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # 创建DataLoader
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # 选择优化器
        if optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=lr)

        criterion = nn.CrossEntropyLoss()

        # 训练
        for epoch in range(local_epochs):
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)

                # 标签翻转攻击
                if attack_labels:
                    target = self.apply_attack_to_labels(target)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # 计算参数更新(delta): 新参数 - 旧参数
        # 这样服务器直接 add_ 即可: w_global_new = w_global + delta
        param_update = {}
        new_state = self.model.state_dict()
        for name in old_state:
            param_update[name] = new_state[name] - old_state[name]

        return param_update, self.sample_count

    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """获取模型参数"""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_model_state(self, state_dict: Dict[str, torch.Tensor]):
        """设置模型参数"""
        self.model.load_state_dict(state_dict)


class FedServer:
    """
    联邦学习服务器

    功能:
    1. 客户端选择
    2. 全局模型分发
    3. 聚合更新
    4. 全局评估
    """

    def __init__(
        self,
        config: Dict[str, Any],
        test_data: np.ndarray,
        test_labels: np.ndarray
    ):
        self.config = config
        self.global_config = config['global']
        self.data_config = config['data']
        self.agg_config = config['aggregator']
        self.attack_config = config['attack']
        self.local_config = config['local']
        self.model_config = config.get('model', {'name': 'simple_cnn', 'num_classes': 10})

        # 设备
        self.device = get_device()
        print(f"[Server] Using device: {self.device}")

        # 创建全局模型
        self.global_model = get_model(
            model_name=self.model_config.get('name', 'simple_cnn'),
            num_classes=self.model_config.get('num_classes', 10),
            dataset=self.data_config.get('name', 'mnist')
        )
        self.global_model.to(self.device)

        # 创建聚合器
        self.aggregator = get_aggregator(
            name=self.agg_config.get('name', 'fedavg'),
            num_clients=self.global_config.get('num_clients', 10),
            config=self.agg_config.get('params', {}),
            device=self.device
        )

        # 创建攻击管理器
        self.attack = get_attack(
            name=self.attack_config.get('name', 'none'),
            malicious_ratio=self.attack_config.get('malicious_ratio', 0.0),
            config=self.attack_config.get('params', {})
        )

        # 测试集
        self.test_dataset = FederatedDataset(test_data, test_labels)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)

        # 日志
        self.logger = MetricsLogger(
            save_dir=config.get('output', {}).get('save_dir', './results')
        )

        # 训练历史
        self.best_acc = 0.0

    def select_clients(self, round_idx: int) -> List[int]:
        """选择参与本轮训练的客户端"""
        num_clients = self.global_config.get('num_clients', 10)
        frac = self.global_config.get('frac', 1.0)
        num_selected = max(1, int(num_clients * frac))

        # 简单随机选择
        np.random.seed(round_idx)
        return np.random.choice(num_clients, num_selected, replace=False).tolist()

    def aggregate_updates(
        self,
        client_gradients: Dict[int, Dict[str, torch.Tensor]],
        sample_counts: Dict[int, int]
    ) -> Dict[str, torch.Tensor]:
        """聚合客户端更新"""
        aggregated, info = self.aggregator.aggregate(
            client_gradients,
            sample_counts
        )
        if info:
            print(f"[Server] Aggregation info: {info}")
        return aggregated

    def apply_gradients(self, param_updates: Dict[str, torch.Tensor], lr: float = 1.0):
        """将聚合后的参数更新应用到全局模型
        w_global = w_global + lr * aggregated_delta
        其中 delta = new_param - old_param（参数向梯度反方向移动）
        """
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in param_updates:
                    delta = param_updates[name]
                    # 处理类型不匹配
                    if delta.dtype != param.dtype:
                        delta = delta.to(param.dtype)
                    # 缩放更新量：lr < 1 表示部分更新，避免参数震荡
                    param.add_(delta, alpha=lr)

    def evaluate(self) -> Tuple[float, float]:
        """评估全局模型"""
        self.global_model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                total_loss += criterion(output, target).item() * len(target)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100.0 * correct / total
        avg_loss = total_loss / total

        return accuracy, avg_loss

    def train_round(
        self,
        round_idx: int,
        clients: List[FedClient]
    ) -> Dict[str, Any]:
        """
        执行一轮联邦学习

        Args:
            round_idx: 当前轮次
            clients: 客户端列表

        Returns:
            轮次指标
        """
        # 选择客户端
        selected_ids = self.select_clients(round_idx)

        # 选择恶意客户端
        malicious = self.attack.select_malicious_clients(
            total_clients=len(clients),
            available_clients=selected_ids,
            round_idx=round_idx
        )

        # 分发全局模型
        global_state = self.global_model.state_dict()
        for client in clients:
            client.set_model_state(copy.deepcopy(global_state))

        # 本地训练
        client_updates = {}
        sample_counts = {}

        for client in clients:
            if client.client_id in selected_ids:
                attack_this_round = client.client_id in malicious

                updates, count = client.train(
                    local_epochs=self.local_config.get('epochs', 2),
                    batch_size=self.local_config.get('batch_size', 32),
                    lr=self.local_config.get('lr', 0.01),
                    optimizer_name=self.local_config.get('optimizer', 'sgd'),
                    attack_labels=attack_this_round and hasattr(client.attack, 'apply_to_labels')
                )

                # 恶意客户端攻击梯度
                if attack_this_round and hasattr(client.attack, 'apply_to_gradients'):
                    updates = client.attack.apply_to_gradients(updates)

                client_updates[client.client_id] = updates
                sample_counts[client.client_id] = count

        # 聚合
        aggregated = self.aggregate_updates(client_updates, sample_counts)

        # 应用梯度（使用局部学习率缩放）
        lr = self.local_config.get('lr', 0.01)
        self.apply_gradients(aggregated, lr)

        # 评估
        test_acc, test_loss = self.evaluate()

        # 记录
        metrics = {
            'round': round_idx,
            'train_acc': 0.0,  # 可扩展训练集评估
            'test_acc': test_acc,
            'train_loss': 0.0,
            'test_loss': test_loss,
            'malicious_clients': malicious,
            'selected_clients': selected_ids
        }

        self.logger.log_round(round_idx, metrics)

        if test_acc > self.best_acc:
            self.best_acc = test_acc

        return metrics

    def train(self, clients: List[FedClient]) -> Dict[str, Any]:
        """
        完整联邦学习训练

        Args:
            clients: 客户端列表

        Returns:
            最终结果
        """
        rounds = self.global_config.get('rounds', 50)
        log_interval = self.config.get('output', {}).get('log_interval', 1)

        print("=" * 70)
        print("Federated Learning Training Started")
        print("=" * 70)
        print(f"Algorithm: {self.aggregator.get_name()}")
        print(f"Attack: {self.attack.get_name()} (ratio: {self.attack.malicious_ratio})")
        print(f"Rounds: {rounds}")
        print(f"Clients: {len(clients)}")
        print("=" * 70)

        for round_idx in range(1, rounds + 1):
            metrics = self.train_round(round_idx, clients)

            # 打印日志
            if round_idx % log_interval == 0 or round_idx == 1:
                self.logger.print_summary(round_idx, metrics)

        # 保存历史
        save_path = self.logger.save_history()

        print("=" * 70)
        print("Training Completed!")
        print(f"Best Test Accuracy: {self.best_acc:.2f}%")
        print(f"History saved to: {save_path}")
        print("=" * 70)

        return {
            'best_acc': self.best_acc,
            'history': self.logger.history,
            'save_path': save_path
        }

    def save_model(self, path: str):
        """保存全局模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.global_model.state_dict(), path)
        print(f"[Server] Model saved to {path}")
