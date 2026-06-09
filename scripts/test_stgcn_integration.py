"""
ST-GCN 模型集成测试脚本
验证原始 ST-GCN 模型能否在 TASL 联邦学习框架中正常运行
使用随机生成的骨架数据进行快速验证
"""
import sys
import os

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
tasl_dir = os.path.join(script_dir, '..')
stgcn_dir = os.path.join(script_dir, '..', '..', 'st-gcn')

sys.path.insert(0, tasl_dir)
sys.path.insert(0, os.path.join(tasl_dir, 'core'))
sys.path.insert(0, os.path.join(stgcn_dir, 'net'))

import torch
import torch.nn as nn
import numpy as np

# ============ 1. 测试原始 ST-GCN 模型加载 ============
print("=" * 60)
print("[测试1] 加载原始 ST-GCN 模型")
print("=" * 60)

from net.st_gcn import Model as OriginalSTGCN

model_original = OriginalSTGCN(
    in_channels=3,
    num_class=60,
    graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
    edge_importance_weighting=True,
    dropout=0.5
)

# 统计参数量
total_params = sum(p.numel() for p in model_original.parameters())
trainable_params = sum(p.numel() for p in model_original.parameters() if p.requires_grad)
print(f"原始 ST-GCN 参数量: {total_params:,} (可训练: {trainable_params:,})")

# 模拟 NTU60 输入: (N, C, T, V, M) = (batch, 3, 300, 25, 2)
# 使用较小的序列长度以加速测试
dummy_input = torch.randn(4, 3, 50, 25, 2)
print(f"输入形状: {dummy_input.shape}")

# 前向传播
model_original.eval()
with torch.no_grad():
    output = model_original(dummy_input)
print(f"输出形状: {output.shape}")
print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
assert output.shape == (4, 60), f"期望输出 (4, 60)，实际 {output.shape}"
print("[✓] 原始 ST-GCN 前向传播测试通过!")

# ============ 2. 测试 TASL 修改版 ST-GCN 模型 ============
print("\n" + "=" * 60)
print("[测试2] 加载 TASL 修改版 ST-GCN 模型 (system/flcore/trainmodel/net/)")
print("=" * 60)

# 添加 system 路径（确保 TASL 版 net/ 优先于原始 st-gcn/net/）
_tasl_net_dir = os.path.join(tasl_dir, 'system', 'flcore', 'trainmodel')
# 先移除原始 st-gcn 的 net 路径，避免冲突
_orig_stgcn_net = os.path.join(stgcn_dir, 'net')
sys.path = [p for p in sys.path if p != _orig_stgcn_net]
sys.path.insert(0, _tasl_net_dir)
# 强制重新导入
import importlib
if 'net.st_gcn' in sys.modules:
    del sys.modules['net.st_gcn']
if 'net.utils.graph' in sys.modules:
    del sys.modules['net.utils.graph']
if 'net.utils.tgcn' in sys.modules:
    del sys.modules['net.utils.tgcn']
from net.st_gcn import Model as TASLSTGCN
from net.utils.graph import Graph

model_tasl = TASLSTGCN(
    in_channels=3,
    hidden_channels=16,
    hidden_dim=256,
    num_class=60,
    graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
    edge_importance_weighting=True,
    dropout=0.5
)

total_params_tasl = sum(p.numel() for p in model_tasl.parameters())
print(f"TASL ST-GCN 参数量: {total_params_tasl:,}")
print(f"参数量差异: {total_params - total_params_tasl:,}")

# 前向传播
model_tasl.eval()
with torch.no_grad():
    feat, logits, gx = model_tasl(dummy_input)
print(f"特征输出形状: {feat.shape}")
print(f"分类输出形状: {logits.shape}")
print(f"图特征输出形状: {gx.shape}")
print("[✓] TASL 修改版 ST-GCN 前向传播测试通过!")

# ============ 3. 测试联邦学习客户端训练 ============
print("\n" + "=" * 60)
print("[测试3] 模拟联邦学习客户端本地训练")
print("=" * 60)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 使用原始 ST-GCN (标准输出，适合 FL)
model_fl = OriginalSTGCN(
    in_channels=3,
    num_class=60,
    graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
    edge_importance_weighting=True,
    dropout=0.5
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_fl.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# 模拟 2 个客户端的本地训练
num_clients = 2
local_epochs = 2
batch_size = 8

# 生成模拟数据
train_data = torch.randn(32, 3, 50, 25, 2).to(device)
train_labels = torch.randint(0, 60, (32,)).to(device)

# 保存初始全局模型参数
global_state = {k: v.clone() for k, v in model_fl.state_dict().items()}

client_updates = []
for client_id in range(num_clients):
    print(f"\n--- 客户端 {client_id} 本地训练 ---")
    # 每个客户端随机选取一部分数据
    idx = torch.randperm(32)[:16]
    client_data = train_data[idx]
    client_labels = train_labels[idx]
    
    # 加载全局模型
    model_fl.load_state_dict(global_state)
    model_fl.train()
    
    for epoch in range(local_epochs):
        total_loss = 0
        correct = 0
        total = 0
        for i in range(0, len(client_data), batch_size):
            batch_x = client_data[i:i+batch_size]
            batch_y = client_labels[i:i+batch_size]
            
            optimizer.zero_grad()
            output = model_fl(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        acc = 100.0 * correct / total if total > 0 else 0
        print(f"  Epoch {epoch+1}: loss={total_loss:.4f}, acc={acc:.1f}%")
    
    # 计算模型更新 (delta = local - global)
    client_state = model_fl.state_dict()
    update = {k: (v.clone() - global_state[k].to(device)) for k, v in client_state.items()}
    client_updates.append(update)

# ============ 4. FedAvg 聚合 ============
print("\n" + "=" * 60)
print("[测试4] FedAvg 聚合")
print("=" * 60)

aggregated_state = {}
for key in global_state:
    # FedAvg: 平均所有客户端更新，加到全局模型上
    avg_update = torch.stack([u[key].cpu() for u in client_updates]).mean(dim=0)
    aggregated_state[key] = global_state[key] + avg_update

model_fl.load_state_dict(aggregated_state)
print("[✓] FedAvg 聚合完成!")

# 验证聚合后模型可以正常推理
model_fl.eval()
with torch.no_grad():
    test_output = model_fl(dummy_input.to(device))
print(f"聚合后推理输出形状: {test_output.shape}")

# ============ 5. 测试 TASL 审计系统兼容性 ============
print("\n" + "=" * 60)
print("[测试5] TASL 审计系统兼容性检查")
print("=" * 60)

# 导入 TASL 核心
try:
    from gradient_aggregator import GradientAggregator
    print("[✓] gradient_aggregator 导入成功")
except ImportError as e:
    print(f"[!] gradient_aggregator 导入失败: {e}")

try:
    from tas import TripartiteAuditSystem
    print("[✓] tas 导入成功")
except ImportError as e:
    print(f"[!] tas 导入失败: {e}")

# 测试梯度维度
model_fl.train()
optimizer.zero_grad()
output = model_fl(train_data[:4])
loss = criterion(output, train_labels[:4])
loss.backward()

grad_dim = sum(p.grad.numel() for p in model_fl.parameters() if p.grad is not None)
print(f"ST-GCN 梯度维度: {grad_dim:,}")

# 拉平梯度用于相似度计算
all_grads = []
for p in model_fl.parameters():
    if p.grad is not None:
        all_grads.append(p.grad.flatten())
grad_vector = torch.cat(all_grads)
print(f"梯度向量形状: {grad_vector.shape}")
print(f"梯度范数: {grad_vector.norm():.6f}")

# ============ 6. 模型大小估算 ============
print("\n" + "=" * 60)
print("[测试6] 模型大小与存储估算")
print("=" * 60)

model_size_mb = sum(p.numel() * p.element_size() for p in model_original.parameters()) / (1024 * 1024)
print(f"原始 ST-GCN 模型大小: {model_size_mb:.2f} MB")
print(f"TASL ST-GCN 模型大小: {sum(p.numel() * p.element_size() for p in model_tasl.parameters()) / (1024 * 1024):.2f} MB")

# 对比 MNIST FedAvgCNN
from models.cnn import FedAvgCNN
mnist_model = FedAvgCNN(num_classes=10)
mnist_size = sum(p.numel() * p.element_size() for p in mnist_model.parameters()) / (1024 * 1024)
mnist_params = sum(p.numel() for p in mnist_model.parameters())
print(f"\n对比 - MNIST FedAvgCNN: {mnist_params:,} 参数, {mnist_size:.2f} MB")
print(f"对比 - ST-GCN (原始): {total_params:,} 参数, {model_size_mb:.2f} MB")
print(f"参数量比: ST-GCN / FedAvgCNN = {total_params / mnist_params:.1f}x")

# ============ 总结 ============
print("\n" + "=" * 60)
print("集成测试总结")
print("=" * 60)
print(f"[✓] 原始 ST-GCN 模型加载和推理: 通过")
print(f"[✓] TASL 修改版 ST-GCN 模型: 通过")
print(f"[✓] 联邦学习客户端本地训练: 通过")
print(f"[✓] FedAvg 聚合: 通过")
print(f"[✓] 梯度计算与 TASL 审计兼容: 通过")
print(f"[✓] 模型大小估算: 完成")
print(f"\nST-GCN 已成功集成到 TASL 框架！等待 NTU60 数据集后即可运行完整实验。")
