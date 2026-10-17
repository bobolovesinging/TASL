"""
模型定义模块
支持: SimpleCNN (MNIST), ResNet18 (CIFAR10)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    简单CNN模型（适用于MNIST/CIFAR10）
    结构: Conv -> Pool -> Conv -> Pool -> FC -> FC
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

        # 根据输入尺寸计算FC层大小
        # MNIST: 1x28x28 -> after 2 pools -> 7x7x64 = 3136
        # CIFAR10: 3x32x32 -> after 2 pools -> 8x8x64 = 4096
        self.fc1 = nn.Linear(64 * 7 * 7, 512) if in_channels == 1 else nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28x28 -> 14x14

        # Conv Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14x14 -> 7x7

        # Flatten
        x = x.view(x.size(0), -1)

        # FC
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class FedAvgCNN(nn.Module):
    """
    FedAvg论文中的CNN模型
    适用于MNIST和CIFAR10
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super(FedAvgCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)

        # MNIST: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model(model_name: str, num_classes: int = 10, dataset: str = 'mnist') -> nn.Module:
    """
    工厂函数：根据配置获取模型

    Args:
        model_name: 模型名称
        num_classes: 分类数
        dataset: 数据集名称

    Returns:
        模型实例
    """
    if dataset.lower() in ['mnist', 'fashionmnist']:
        in_channels = 1
        input_size = 28
    elif dataset.lower() == 'cifar10':
        in_channels = 3
        input_size = 32
    else:
        in_channels = 3
        input_size = 32

    if model_name.lower() == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes, in_channels=in_channels)
    elif model_name.lower() == 'fedavg_cnn':
        return FedAvgCNN(num_classes=num_classes, in_channels=in_channels)
    else:
        # 默认返回SimpleCNN
        return SimpleCNN(num_classes=num_classes, in_channels=in_channels)


def test_model():
    """测试模型输出"""
    model = get_model('simple_cnn', num_classes=10, dataset='mnist')
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")


if __name__ == '__main__':
    test_model()
