# 基于区块链的联邦学习 - 解决Non-IID问题使用指南

## 概述

本系统使用区块链技术来共享客户端的数据分布信息，帮助解决联邦学习中的non-IID（非独立同分布）问题。通过透明地共享全局数据分布，客户端可以更好地理解整体数据状况，从而改善本地训练效果。

## 核心功能

1. **数据分布统计**: 自动计算每个客户端的数据分布（类别分布、样本数量等）
2. **区块链存储**: 将数据分布信息存储在区块链上，确保透明性和可追溯性
3. **全局分布聚合**: 从所有客户端聚合数据分布，形成全局视图
4. **分布共享**: 服务器将全局分布发送给所有客户端，用于改善训练

## 使用方法

### 1. 运行基于区块链的FedAvg

在 `main.py` 中，使用以下算法名称：

```bash
python main.py -algo FedAvgBlockchain [其他参数...]
```

或者使用简写：

```bash
python main.py -algo Blockchain [其他参数...]
```

### 2. 完整命令示例

```bash
python main.py \
    -algo FedAvgBlockchain \
    -data ntu60-01 \
    -nb 60 \
    -m skeletonclr \
    -nc 10 \
    -gr 100 \
    -ls 2 \
    -lr 0.01 \
    -lbs 64
```

### 3. 工作原理

#### 阶段1: 数据分布计算
- 每个客户端计算自己的数据分布统计（类别分布、样本数量等）
- 分布信息包括：
  - `class_distribution`: 每个类别的样本数量
  - `class_weights`: 每个类别的权重（归一化）
  - `total_samples`: 总样本数

#### 阶段2: 区块链存储（可选）
- 客户端的数据分布信息可以存储在区块链上
- 分布信息被打包成 `DistributionUpdate` 交易
- 交易被验证并存储在区块链区块中

#### 阶段3: 全局聚合
- 服务器从所有客户端收集数据分布
- 聚合形成全局数据分布视图
- 全局分布包括：
  - `global_class_distribution`: 全局类别分布
  - `global_class_weights`: 全局类别权重
  - `total_samples`: 全局总样本数

#### 阶段4: 分布共享
- 服务器将全局数据分布发送给所有客户端
- 客户端可以通过 `self.global_distribution` 访问全局分布
- 客户端可以使用全局分布信息来：
  - 调整本地训练策略
  - 处理类别不平衡问题
  - 改善模型在no-iid数据上的性能

## 代码示例

### 服务器端

服务器会自动处理数据分布收集和共享：

```python
from flcore.servers.serverblockchain import FedAvgBlockchain

# 在main.py中
if args.algorithm == "FedAvgBlockchain":
    server = FedAvgBlockchain(args, i)
    server.train()  # 自动处理数据分布共享
```

### 客户端使用全局分布

客户端可以通过 `self.global_distribution` 访问全局分布：

```python
# 在客户端训练代码中
if self.global_distribution is not None:
    global_weights = self.global_distribution['global_class_weights']
    
    # 使用全局权重调整损失函数
    # 例如：加权交叉熵损失
    class_weights = torch.tensor([
        global_weights.get(i, 1.0) 
        for i in range(self.num_classes)
    ]).to(self.device)
    
    # 创建加权损失函数
    weighted_loss = nn.CrossEntropyLoss(weight=class_weights)
```

## 配置参数

区块链相关配置在 `FedAvgBlockchain.__init__` 中：

```python
self.blockchain_network = BlockchainNetwork(
    num_nodes=self.num_clients,          # 节点数（等于客户端数）
    lightweight_mode=True,                 # 轻量级模式（节省内存）
    max_blocks=100,                        # 最大区块数
    keep_recent_blocks=50,                 # 保留的最近区块数
    disk_storage=True,                     # 启用磁盘存储
    storage_dir="blockchain_storage_distribution",  # 存储目录
    in_memory_blocks=10                    # 内存中保留的区块数
)
```

## 数据分布统计详情

### DataDistributionStats 类

```python
from flcore.blockchain.data_distribution import DataDistributionStats

# 创建统计对象
stats = DataDistributionStats(client_id=0, num_classes=60)

# 从数据加载器计算分布
stats.compute_from_data(trainloader, model=None)

# 或仅从标签计算（更轻量级）
stats.compute_from_labels(labels)

# 获取统计信息
print(stats.class_distribution)  # {0: 100, 1: 50, ...}
print(stats.class_weights)       # {0: 0.2, 1: 0.1, ...}
print(stats.total_samples)       # 500
```

## 优势

1. **解决non-IID问题**: 通过共享全局数据分布，客户端可以了解整体数据状况
2. **透明性**: 所有数据分布信息存储在区块链上，可追溯、可验证
3. **去中心化**: 不依赖单一中心服务器，分布式存储和验证
4. **安全性**: 区块链的不可篡改特性保证数据分布信息的真实性
5. **灵活性**: 客户端可以根据全局分布调整训练策略

## 注意事项

1. **性能开销**: 区块链操作会增加一定的计算和存储开销
2. **网络同步**: 确保所有节点同步区块链状态
3. **存储管理**: 使用轻量级模式和磁盘存储来控制内存使用
4. **数据隐私**: 当前实现共享类别分布统计，不共享原始数据

## 未来改进

1. **智能合约**: 使用智能合约自动处理数据分布聚合
2. **隐私保护**: 添加差分隐私保护数据分布信息
3. **动态调整**: 根据全局分布动态调整客户端训练策略
4. **性能优化**: 优化区块链存储和查询性能
5. **加权训练**: 根据全局分布实现加权损失函数

## 参考

- 区块链实现基于PoA（Proof of Authority）共识机制
- 数据分布统计基于类别分布和样本统计
- 兼容现有的FedAvg训练流程

