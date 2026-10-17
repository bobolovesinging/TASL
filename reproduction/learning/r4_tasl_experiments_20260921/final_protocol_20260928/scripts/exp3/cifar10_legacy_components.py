"""CIFAR-10 components from the submitted governance experiment.

The model, data augmentation and Dirichlet partition follow
principal_5seed/scripts/exp3/exp_governance_v9_cifar10_ablation.py.
"""

import os

import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(4, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(4, out_channels), out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(min(4, out_channels), out_channels),
            )

    def forward(self, x):
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class CifarCNN(nn.Module):
    """The original CIFAR-10 ResNet-18 with GroupNorm."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.GroupNorm(4, 64), nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self._init_weights()

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        return self.fc(x.view(x.size(0), -1))


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def dirichlet_split(labels, n_clients, alpha=0.3):
    n_classes = len(np.unique(labels))
    idx_per_class = {c: np.where(labels == c)[0] for c in range(n_classes)}
    client_indices = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx_c = idx_per_class[c]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        proportions = (proportions * len(idx_c)).astype(int)
        remainder = len(idx_c) - proportions.sum()
        for i in range(remainder):
            proportions[i % n_clients] += 1
        start = 0
        for client_id in range(n_clients):
            n_client = proportions[client_id]
            client_indices[client_id].extend(idx_c[start:start + n_client].tolist())
            start += n_client
    return client_indices


def get_cifar10_loaders(client_num=10, batch_size=64, alpha=0.3, data_root=None):
    if data_root is None:
        data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "data")
    os.makedirs(data_root, exist_ok=True)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test)
    train_labels = np.asarray(train_set.targets)
    client_indices = dirichlet_split(train_labels, client_num, alpha=alpha)
    client_loaders = [
        DataLoader(Subset(train_set, indices), batch_size=batch_size,
                   shuffle=True, num_workers=2, pin_memory=True)
        for indices in client_indices
    ]
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)
    return client_loaders, test_loader
