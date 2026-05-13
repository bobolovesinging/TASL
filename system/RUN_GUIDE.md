# 运行指南

## 基本运行方式

### 1. 使用默认配置文件运行（推荐）

配置文件默认路径：`config/attack_config.json`

```bash
# 从项目根目录运行
cd /home/njit516/TLF/Python_code/NoCGAcom_blockchain
python system/main.py

# 或者从system目录运行
cd system
python main.py
```

### 2. 指定自定义配置文件

```bash
python system/main.py --config_path config/attack_config.json
```

### 3. 命令行参数覆盖配置文件

配置文件中的参数可以通过命令行参数覆盖：

```bash
# 示例：使用配置文件，但覆盖数据集和全局轮数
python system/main.py \
    --config_path config/attack_config.json \
    --dataset ntu60 \
    --global_rounds 20 \
    --algorithm FedAvg
```

## 配置文件说明

当前配置文件：`config/attack_config.json`

### 主要配置项：

1. **federated（联邦学习基础配置）**
   - `num_clients`: 客户端总数（默认10）
   - `join_ratio`: 每轮参与训练的客户端比例（默认0.5，即5个客户端）
   - `dataset`: 数据集名称（当前为"small"）
   - `global_rounds`: 全局训练轮数（当前为10）

2. **smart_contract（智能合约配置）**
   - `enabled`: 是否启用智能合约（true/false）
   - `aggregation_method`: 聚合方法（fedavg/weighted_avg/median/krum）

3. **decentralized（去中心化配置）**
   - `p2p_enabled`: 是否启用P2P通信（true/false）
   - `p2p_mode`: P2P通信模式（local/grpc/websocket）

4. **blockchain_storage（区块链存储配置）**
   - `enabled`: 是否启用区块链存储
   - `lightweight`: 是否启用轻量级模式
   - `disk_storage`: 是否启用磁盘存储
   - `keep_recent_blocks`: 内存中保留的最近区块数

5. **npc_deputy（NPC Deputy机制配置）**
   - `enabled`: 是否启用NPC Deputy机制（true/false）
   - `num_deputies`: Deputy节点数量（默认3）
   - `keep_recent_rounds`: 内存中保留的最近轮次数（默认5轮）
   - `gradient_storage_dir`: 梯度存储目录

6. **attack（攻击配置）**
   - `enabled`: 是否启用攻击（true/false）
   - `malicious_ids`: 恶意客户端ID列表

## 常用运行命令示例

### 示例1：基础运行（使用配置文件）
```bash
python system/main.py
```

### 示例2：启用NPC Deputy机制
确保配置文件中 `npc_deputy.enabled = true`，然后运行：
```bash
python system/main.py
```

### 示例3：启用攻击模拟
修改配置文件中的 `attack.enabled = true`，然后运行：
```bash
python system/main.py
```

### 示例4：使用NTU60数据集
修改配置文件中的 `federated.dataset = "ntu60"`，或使用命令行：
```bash
python system/main.py --dataset ntu60
```

### 示例5：指定GPU设备
```bash
python system/main.py --device_id 0
```

### 示例6：运行多次实验
```bash
python system/main.py --times 5  # 运行5次实验
```

## 重要参数说明

### 必需参数（有默认值，但建议检查）：
- `--dataset` / `-data`: 数据集名称（默认"small"）
- `--model` / `-m`: 模型类型（默认"skeletonclr"）
- `--algorithm` / `-algo`: 算法名称（默认"FedAvg"）
- `--num_clients` / `-nc`: 客户端总数（默认10）
- `--global_rounds` / `-gr`: 全局训练轮数（默认10）

### 设备相关：
- `--device` / `-dev`: 设备类型（cuda/cpu，默认cuda）
- `--device_id` / `-did`: GPU设备ID（默认"0"）

### 训练相关：
- `--batch_size` / `-lbs`: 批次大小（默认128）
- `--local_epochs` / `-ls`: 本地训练轮数（默认2）
- `--local_learning_rate` / `-lr`: 本地学习率（默认0.01）

## 输出文件位置

运行后会生成以下文件/目录：

1. **TensorBoard日志**：`runs/FedAvg_normal_0/` 或 `runs/FedAvg_attack_0/`
   ```bash
   tensorboard --logdir runs
   ```

2. **区块链存储**：
   - 区块链数据：`blockchain_storage/`
   - 梯度存储：`blockchain_gradient_storage/`

3. **结果文件**：`results/` 目录下

## 故障排查

### 1. 数据集不存在
如果遇到数据集错误，检查：
- 数据集路径是否正确
- 数据集名称是否在支持列表中（如 "small", "ntu60"）

### 2. 内存不足
如果遇到内存不足：
- 减少 `batch_size`
- 减少 `num_clients`
- 启用 `blockchain_storage.lightweight = true`
- 启用 `blockchain_storage.disk_storage = true`
- 减少 `npc_deputy.keep_recent_rounds`（默认5）

### 3. 配置文件加载失败
- 检查配置文件路径是否正确
- 检查JSON格式是否正确（可以使用JSON验证器）

## 当前配置快速参考

根据 `config/attack_config.json`：
- ✅ NPC Deputy机制已启用
- ✅ 智能合约已启用
- ✅ P2P通信已启用
- ✅ 区块链存储已启用
- ❌ 攻击模拟已禁用
- 📊 数据集：small
- 🔄 全局轮数：10
- 👥 客户端数：10
- 🎯 每轮参与：5个客户端（50%）

## 运行示例

```bash
# 1. 进入项目目录
cd /home/njit516/TLF/Python_code/NoCGAcom_blockchain

# 2. 运行代码（使用默认配置文件）
python system/main.py

# 3. 查看TensorBoard（可选，另开终端）
tensorboard --logdir runs
```

运行成功后，你会看到：
- 配置加载信息
- NPC Deputy初始化信息
- 每轮训练进度
- 最终结果统计


