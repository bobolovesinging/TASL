"""
CNN模型定义
"""
import torch
import torch.nn as nn


class CNN(nn.Module):
    """用于MNIST分类的CNN模型"""
    
    def __init__(self, in_channels=1, classes=10):
        """
        初始化CNN模型
        :param in_channels: 输入通道数，默认1（MNIST灰度图）
        :param classes: 分类数量，默认10（MNIST有10个类别）
        """
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, classes),
            nn.ReLU()
        )                              
                                        
    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = nn.Flatten()(x)
        x = self.classifier(x)
        return x


