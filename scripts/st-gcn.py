"""
基于NPC Deputy的联邦学习训练脚本（FedAvg）
整合区块链、权益管理、相似度计算、模型聚合等功能
使用FedAvg算法：客户端进行本地训练更新参数，服务器聚合模型参数并更新全局模型
使用NTU60数据集和CrosSCLR模型
"""
import os
import sys
import random
import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# 添加 CrosSCLR 项目路径
# 当前文件在 Blockchain/scripts/，需要往上两级到达项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))  # Blockchain/scripts/
blockchain_dir = os.path.dirname(script_dir)  # Blockchain/
project_root = os.path.dirname(blockchain_dir)  # CrosSCLR项目根目录
sys.path.insert(0, project_root)

# 添加 core 目录到路径
core_dir = os.path.join(blockchain_dir, 'core')
sys.path.insert(0, core_dir)

from blockchain_node import BlockchainNetwork
from deputy_blockchain_manager import DeputyBlockchainManager
from stake_manager import StakeManager
from npc_deputy import NPCDeputyManager
from gradient_aggregator import GradientAggregator

# 导入ST-GCN模型（用于监督学习，fedavg_npc_deputy.py使用ST-GCN）
from net.st_gcn import Model as STGCNModel

# 导入攻击模块
from attack_module import (
    AttackConfig,
    apply_random_attack,
    apply_label_flip_attack,
    apply_model_poison_attack,
    apply_scale_attack,
    apply_backdoor_attack
)

# 客户端数据集路径（在Blockchain/data/下）
Datas_path = os.path.join(blockchain_dir, 'data', 'Client_datasets_10')
# NTU60 测试数据路径（在项目根目录的data/下）
test_data_path = os.path.join(project_root, 'data', 'NTU60_frame50', 'xview', 'val_position.npy')
test_label_path = os.path.join(project_root, 'data', 'NTU-RGB-D', 'xview', 'val_label.pkl')


def set_random_seed(seed_value=100):
    """设置随机种子以确保可重复性"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NTU60Dataset(Dataset):
    """NTU60骨架数据集类（支持数据增强）"""
    def __init__(self, data_path, label_path, indices=None, mmap=True, 
                 use_augmentation=True, shear_amplitude=0.5, temperal_padding_ratio=6):
        """
        初始化NTU60数据集
        :param data_path: 数据文件路径 (.npy)
        :param label_path: 标签文件路径 (.pkl) 或标签数组
        :param indices: 数据索引列表（用于客户端数据），如果为None则使用全部数据
        :param mmap: 是否使用内存映射模式
        :param use_augmentation: 是否使用数据增强（训练时建议True，测试时False）
        :param shear_amplitude: 剪切增强幅度（0表示不使用，推荐0.5）
        :param temperal_padding_ratio: 时间裁剪比例（0表示不使用，推荐6）
        """
        self.data_path = data_path
        self.indices = indices
        self.use_augmentation = use_augmentation
        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        
        # 导入数据增强工具
        feeder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'feeder')
        sys.path.insert(0, feeder_path)
        try:
            from tools import shear, temperal_crop
            self.shear_func = shear
            self.temperal_crop_func = temperal_crop
        except ImportError:
            print("[WARNING] 无法导入数据增强工具，将不使用数据增强")
            self.use_augmentation = False
        
        # 加载数据（使用内存映射以节省内存）
        if mmap:
            self.data = np.load(data_path, mmap_mode='r')
        else:
            self.data = np.load(data_path)
        
        # 加载标签
        if isinstance(label_path, str):
            with open(label_path, 'rb') as f:
                sample_name, labels = pickle.load(f)
            self.labels = np.array(labels)
        else:
            self.labels = np.array(label_path)
        
        # 如果指定了索引，只使用索引对应的数据
        if indices is not None:
            self.len = len(indices)
        else:
            self.len = len(self.labels)

    def _augment(self, data_numpy):
        """应用数据增强"""
        if not self.use_augmentation:
            return data_numpy
        
        # 确保数据是float32类型
        if data_numpy.dtype != np.float32:
            data_numpy = data_numpy.astype(np.float32)
        
        # 确保数据形状为 (C, T, V, M)
        if len(data_numpy.shape) == 3:
            data_numpy = np.expand_dims(data_numpy, axis=-1)
        
        # 时间裁剪增强
        if self.temperal_padding_ratio > 0:
            try:
                data_numpy = self.temperal_crop_func(data_numpy, self.temperal_padding_ratio)
                # 确保增强后仍然是float32
                if data_numpy.dtype != np.float32:
                    data_numpy = data_numpy.astype(np.float32)
            except:
                pass  # 如果增强失败，使用原始数据
        
        # 剪切增强
        if self.shear_amplitude > 0:
            try:
                data_numpy = self.shear_func(data_numpy, self.shear_amplitude)
                # 确保增强后仍然是float32
                if data_numpy.dtype != np.float32:
                    data_numpy = data_numpy.astype(np.float32)
            except:
                pass  # 如果增强失败，使用原始数据
        
        return data_numpy

    def __getitem__(self, index):
        if self.indices is not None:
            # 使用 indices 时，actual_index 用于访问原始数据集
            actual_index = self.indices[index]
            # 但标签应该使用 index，因为 self.labels 已经是客户端子集的标签
            label = int(self.labels[index])
        else:
            # 不使用 indices 时，index 和 actual_index 相同
            actual_index = index
            label = int(self.labels[index])
        
        # 获取数据，形状为 (C, T, V, M)
        data = np.array(self.data[actual_index], dtype=np.float32)
        
        # 应用数据增强（训练时）
        data = self._augment(data)
        
        # 转换为torch tensor，并确保形状为 (C, T, V, M)
        # 如果数据是 (C, T, V)，需要添加 M 维度
        if len(data.shape) == 3:
            data = np.expand_dims(data, axis=-1)  # 添加 M 维度
        
        # 确保数据是float32类型（避免类型不匹配错误）
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        # 转换为torch tensor，明确指定dtype为float32
        data = torch.from_numpy(data).float()  # 确保是float32类型
        label = torch.tensor(label, dtype=torch.long)
        
        return data, label

    def __len__(self):
        return self.len


def clients_dataloader(batch_size=32, data_source_path=None):
    """
    为所有客户端创建数据加载器（NTU60数据）
    :param batch_size: 批次大小
    :param data_source_path: 原始数据文件路径（如果为None，从客户端文件夹的data_path.txt读取）
    :return: 数据加载器列表
    """
    dataloader_list = []
    dir_path = sorted(os.listdir(Datas_path))
    
    for dir_name in dir_path:
        client_dir = os.path.join(Datas_path, dir_name)
        
        # 读取数据索引和标签
        indices_path = os.path.join(client_dir, "data_indices.npy")
        label_path = os.path.join(client_dir, "label.npy")
        data_path_file = os.path.join(client_dir, "data_path.txt")
        
        if not os.path.exists(indices_path) or not os.path.exists(label_path):
            print(f"警告: 客户端 {dir_name} 数据文件不完整，跳过")
            continue
        
        # 读取数据索引
        indices = np.load(indices_path)
        
        # 读取标签
        labels = np.load(label_path)
        
        # 读取原始数据路径
        if data_source_path is None:
            # 计算项目根目录（用于构建数据路径）
            script_dir_local = os.path.dirname(os.path.abspath(__file__))
            blockchain_dir_local = os.path.dirname(script_dir_local)
            project_root_local = os.path.dirname(blockchain_dir_local)
            
            if os.path.exists(data_path_file):
                with open(data_path_file, 'r') as f:
                    data_source_path = f.read().strip()
                    # 如果是相对路径，转换为绝对路径（从项目根目录开始）
                    if not os.path.isabs(data_source_path):
                        data_source_path = os.path.join(project_root_local, data_source_path.lstrip('./'))
            else:
                # 默认路径（使用项目根目录的data/）
                data_source_path = os.path.join(project_root_local, 'data', 'NTU60_frame50', 'xview', 'train_position.npy')
        
        # 创建数据集（训练时使用数据增强）
        dataset = NTU60Dataset(
            data_path=data_source_path,
            label_path=labels,
            indices=indices,
            mmap=True,
            use_augmentation=True,  # 训练时启用数据增强
            shear_amplitude=0.5,   # 剪切增强幅度
            temperal_padding_ratio=6  # 时间裁剪比例
        )
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset, 
            shuffle=True, 
            batch_size=batch_size,
            num_workers=0,  # 使用内存映射时，num_workers应该为0
            pin_memory=False  # 内存映射数据不支持pin_memory
        )
        dataloader_list.append(dataloader)
        print(f"客户端 {dir_name}: {len(dataset)} 个样本")
    
    return dataloader_list


def client_train(client_id, E, model_parameter, dataloader, device, criterion, 
                 learning_rate=0.1, weight_decay=1e-4, nesterov=False, momentum=0.9,
                 attack_config=None, global_model_params=None):
    """
    客户端本地训练（FedAvg标准流程）
    客户端接收服务器下发的模型参数，进行E轮本地训练，更新参数
    :param client_id: 客户端ID
    :param E: 本地训练轮数
    :param model_parameter: 全局模型参数（由deputy下发）
    :param dataloader: 客户端数据加载器
    :param device: 设备（CPU/GPU）
    :param criterion: 损失函数
    :param learning_rate: 本地学习率
    :param attack_config: 攻击配置对象（AttackConfig）
    :param global_model_params: 全局模型参数（用于某些攻击类型）
    :return: (更新后的模型参数字典, 样本数量)
    """
    # 初始化ST-GCN模型
    client_model = STGCNModel(
        in_channels=3,
        hidden_channels=16,
        hidden_dim=256,
        num_class=60,
        graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
        edge_importance_weighting=True,
        dropout=0.5
    ).to(device)
    
    client_model.load_state_dict(model_parameter)
    client_model.train()
    
    # 检查是否为恶意客户端
    is_malicious = attack_config is not None and attack_config.is_malicious(client_id)
    attack_type = attack_config.get_attack_type(client_id) if attack_config else 'none'
    
    if is_malicious:
        print(f'[攻击] 客户端 {client_id} 执行 {attack_type} 攻击')
    
    # 创建优化器（与原始CrosSCLR完全一致）
    # 原始CrosSCLR配置: SGD, momentum=0.9, nesterov=False, weight_decay=1e-4
    optimizer = torch.optim.SGD(
        client_model.parameters(), 
        lr=learning_rate, 
        momentum=0.9, 
        nesterov=False,  # 与原始CrosSCLR配置一致
        weight_decay=1e-4  # 与原始CrosSCLR配置一致
    )
    
    sample_count = 0
    
    for epoch in range(E):
        correct = 0
        total = 0
        
        for data, label in dataloader:
            # 数据形状: (N, C, T, V, M)
            # 确保数据类型是float32（避免类型不匹配错误）
            if data.dtype != torch.float32:
                data = data.float()
            train_data_value = data.to(device)  # (N, C, T, V, M)
            train_data_label = label.to(device)
            
            # ========== 攻击处理 ==========
            # 标签翻转攻击：在训练前翻转标签
            if attack_type == 'label_flip' and attack_config.label_flip_map:
                train_data_label = apply_label_flip_attack(train_data_label, attack_config.label_flip_map)
            
            # 后门攻击：在数据中植入后门模式
            if attack_type == 'backdoor' and attack_config.backdoor_pattern is not None:
                train_data_value, train_data_label = apply_backdoor_attack(
                    train_data_value, train_data_label,
                    attack_config.backdoor_pattern,
                    attack_config.backdoor_target_label,
                    attack_ratio=attack_config.attack_strength
                )
            # ========== 攻击处理结束 ==========
            
            # 前向传播
            train_data_label_pred = client_model(train_data_value)
            
            loss = criterion(train_data_label_pred, train_data_label)
            
            # 清零梯度
            optimizer.zero_grad()
            # 反向传播计算梯度
            loss.backward()
            # 更新参数（FedAvg：本地更新）
            optimizer.step()
            
            total += train_data_label.size(0)
            _, predicted = torch.max(train_data_label_pred, 1)
            correct += (predicted == train_data_label).sum().item()
            sample_count = total
        
        accuracy = 100 * correct / total if total > 0 else 0
        attack_marker = f'[攻击:{attack_type}]' if is_malicious else ''
        print(f'客户端 {client_id} {attack_marker}, Epoch {epoch+1}/{E}, 准确率: {accuracy:.2f}%, 损失: {loss.item():.4f}')
    
    # 返回更新后的模型参数（而不是梯度）
    updated_params = client_model.state_dict()
    
    # ========== 攻击后处理 ==========
    # 对模型参数进行攻击（随机攻击、模型投毒、缩放攻击）
    if is_malicious:
        if attack_type == 'random':
            updated_params = apply_random_attack(updated_params, device, attack_config.attack_strength)
        elif attack_type == 'model_poison' and global_model_params is not None:
            updated_params = apply_model_poison_attack(
                updated_params, global_model_params, device, attack_config.attack_strength
            )
        elif attack_type == 'scale' and global_model_params is not None:
            updated_params = apply_scale_attack(
                updated_params, global_model_params, device, attack_config.scale_factor
            )
    # ========== 攻击后处理结束 ==========
    
    return updated_params, sample_count


def train_npc_deputy_round(network, deputy_manager, npc_deputy, stake_manager, 
                          gradient_aggregator, client_num, C, E, client_dataloader_list, 
                          device, criterion, round_number, 
                          base_lr=0.1, weight_decay=1e-4, nesterov=False, momentum=0.9,
                          attack_config=None):
    """
    基于NPC Deputy的FedAvg训练一轮
    :param network: 区块链网络
    :param deputy_manager: Deputy区块链管理器
    :param npc_deputy: NPC Deputy管理器
    :param stake_manager: 权益管理器
    :param gradient_aggregator: 梯度聚合器（用于相似度计算和模型聚合）
    :param client_num: 客户端总数
    :param C: 参与训练的客户端比例
    :param E: 每个客户端本地训练的轮数（FedAvg中E>1）
    :param client_dataloader_list: 客户端数据加载器列表
    :param device: 设备
    :param criterion: 损失函数
    :param round_number: 当前轮次
    :param learning_rate: 本地学习率（用于客户端本地训练）
    :return: 更新后的全局模型
    """
    print(f"\n{'='*60}")
    print(f"轮次 {round_number}: 开始训练")
    print(f"{'='*60}")
    
    # 步骤1: 客户端从区块链获取最新的全局模型参数
    print(f"\n[步骤1] 客户端从区块链获取最新全局模型参数...")
    
    # 先同步所有节点的区块链（确保客户端有最新的区块链状态）
    # 从deputy节点同步到所有客户端节点
    deputy_ids = npc_deputy.get_deputies()
    if deputy_ids:
        # 使用第一个deputy节点的区块链作为源
        deputy_node = deputy_manager.get_deputy_node(deputy_ids[0])
        if deputy_node:
            deputy_blockchain = deputy_node.blockchain
            # 同步所有客户端节点
            for client_id in range(client_num):
                client_node = network.get_node(client_id)
                client_node.sync_blockchain(deputy_blockchain)
            print(f"[INFO] 已从deputy节点同步区块链到所有客户端节点")
    
    # 从任意客户端节点获取最新模型（所有节点都有区块链账本）
    client_node = network.get_node(0)  # 使用节点0获取模型
    
    # 调试信息：检查区块链状态
    print(f"[DEBUG] 客户端节点0的区块链状态:")
    print(f"  - 链长度: {len(client_node.blockchain.chain)}")
    print(f"  - 总轮次数: {client_node.blockchain.total_blocks}")
    latest_block = client_node.blockchain.get_latest_block()
    print(f"  - 最新区块索引: {latest_block.index}, 时间戳: {latest_block.timestamp}")
    print(f"  - 最新区块aggregated_model: {latest_block.aggregated_model is not None}")
    
    latest_model_params = client_node.get_latest_model()
    
    if latest_model_params is None:
        print("[WARNING] 未找到全局模型，创建新模型")
        global_model = STGCNModel(
            in_channels=3,
            hidden_channels=16,
            hidden_dim=256,
            num_class=60,
            graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
            edge_importance_weighting=True,
            dropout=0.5
        ).to(device)
        latest_model_params = global_model.state_dict()
    else:
        print("[INFO] 成功从区块链获取最新全局模型参数")
        # 确保所有参数都是torch.Tensor格式
        if isinstance(latest_model_params, dict):
            for key, value in latest_model_params.items():
                if not isinstance(value, torch.Tensor):
                    latest_model_params[key] = torch.tensor(value, dtype=torch.float32)
                else:
                    latest_model_params[key] = value.to(device)
        global_model = STGCNModel(
            in_channels=3,
            hidden_channels=16,
            hidden_dim=256,
            num_class=60,
            graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
            edge_importance_weighting=True,
            dropout=0.5
        ).to(device)
        global_model.load_state_dict(latest_model_params)
    
    # 步骤2: 随机选择参与训练的客户端
    selected_clients = random.sample(range(client_num), int(client_num * C))
    print(f"\n[步骤2] 选中 {len(selected_clients)}/{client_num} 个客户端 (比例 {C:.1%})")
    print(f"选中的客户端: {selected_clients}")
    
    # 步骤3: Deputy下发模型参数给客户端，客户端进行本地训练（FedAvg标准流程）
    print(f"\n[步骤3] Deputy下发模型参数，客户端进行本地训练...")
    client_models = {}
    client_sample_counts = {}
    
    # 清空所有节点的pending_updates（准备接收新轮次的更新，避免内存累积）
    print(f"[步骤3.0] 清空所有节点的pending_updates...")
    for client_id in range(client_num):
        client_node = network.get_node(client_id)
        # 释放旧update的内存
        for update in client_node.blockchain.pending_updates:
            if update.model_params:
                update.model_params.clear()
        client_node.blockchain.pending_updates = []
    
    # 清空所有deputy节点的pending_updates
    for deputy_id in npc_deputy.get_deputies():
        deputy_node = deputy_manager.get_deputy_node(deputy_id)
        if deputy_node:
            for update in deputy_node.blockchain.pending_updates:
                if update.model_params:
                    update.model_params.clear()
            deputy_node.blockchain.pending_updates = []
    
    print(f"[INFO] 所有节点的pending_updates已清空")
    
    for idx, client_id in enumerate(selected_clients, 1):
        print(f"\n[{idx}/{len(selected_clients)}] 客户端 {client_id} 开始训练...")
        dataloader = client_dataloader_list[client_id]
        
        # 客户端从自己的区块链账本获取最新模型参数
        client_node = network.get_node(client_id)
        client_model_params = client_node.get_latest_model()
        
        # 如果客户端节点没有模型，使用全局模型参数
        if client_model_params is None:
            print(f"[WARNING] 客户端 {client_id} 的区块链中没有模型，使用全局模型参数")
            client_model_params = latest_model_params
        else:
            # 确保所有参数都是torch.Tensor格式
            if isinstance(client_model_params, dict):
                for key, value in client_model_params.items():
                    if not isinstance(value, torch.Tensor):
                        client_model_params[key] = torch.tensor(value, dtype=torch.float32)
        
        # FedAvg流程：客户端从区块链获取模型参数，进行E轮本地训练，返回更新后的模型参数
        # 使用与原始CrosSCLR完全一致的训练参数
        updated_model_params, sample_count = client_train(
            client_id=client_id,
            E=E,
            model_parameter=client_model_params,  # 从区块链获取的模型参数
            dataloader=dataloader,
            device=device,
            criterion=criterion,
            learning_rate=base_lr,  # 与原始CrosSCLR一致：base_lr=0.1
            weight_decay=weight_decay,  # 与原始CrosSCLR一致：1e-4
            nesterov=nesterov,  # 与原始CrosSCLR一致：False
            momentum=momentum,  # 与原始CrosSCLR一致：0.9
            attack_config=attack_config,  # 攻击配置
            global_model_params=latest_model_params  # 全局模型参数（用于某些攻击类型）
        )
        
        client_models[client_id] = updated_model_params
        client_sample_counts[client_id] = sample_count
        
        # 将客户端更新后的模型参数提交到区块链（用于区块保存）
        client_node = network.get_node(client_id)
        # 将模型参数转换为可序列化格式
        model_params_dict = {}
        for key, value in updated_model_params.items():
            if isinstance(value, torch.Tensor):
                model_params_dict[key] = value.cpu().detach().numpy().tolist()
            else:
                model_params_dict[key] = value
        
        # 提交模型更新到客户端节点的区块链
        client_node.submit_model_update(
            round_number=round_number,
            model_params=model_params_dict
        )
        
        print(f"客户端 {client_id} 本地训练完成，样本数: {sample_count}")
    
    # 步骤4: 执行deputy聚合模型参数（FedAvg）
    print(f"\n[步骤4] 执行deputy聚合模型参数（FedAvg）...")
    executor_deputy_id = npc_deputy.get_executor_deputy(round_number)
    print(f"执行deputy: {executor_deputy_id}")
    
    # 计算模型参数差值（用于相似度计算）
    # 将模型参数差值作为"伪梯度"用于相似度检测
    client_gradients = {}
    for client_id, model_params in client_models.items():
        # 计算模型参数与全局模型的差值（作为梯度）
        gradient = {}
        for key in model_params:
            latest_param = latest_model_params[key].to(device)
            gradient[key] = model_params[key] - latest_param
        client_gradients[client_id] = gradient
    
    # 计算全局梯度（用于相似度计算）
    global_gradient = gradient_aggregator._compute_average_gradient(client_gradients)
    
    # 执行聚合：归一化 -> 相似度过滤 -> FedAvg聚合
    # 注意：归一化是必要的，用于检测恶意客户端（通过余弦相似度）
    aggregated_gradient, similarities = gradient_aggregator.aggregate(
        client_gradients=client_gradients,
        sample_counts=client_sample_counts,
        global_gradient=global_gradient,
        normalize=True,  # 归一化用于恶意客户端检测
        filter_by_similarity=True,
        similarity_threshold=0.1  # 降低阈值，避免过滤掉太多客户端
    )
    
    # FedAvg聚合：直接聚合模型参数（而不是梯度）
    # 只聚合通过相似度检测的客户端
    valid_clients = [cid for cid in selected_clients if similarities.get(cid, 0.0) >= 0.1]
    if not valid_clients:
        print("[WARNING] 没有客户端通过相似度检测，使用所有客户端")
        valid_clients = selected_clients
    
    valid_client_models = {cid: client_models[cid] for cid in valid_clients}
    valid_sample_counts = {cid: client_sample_counts[cid] for cid in valid_clients}
    
    # 使用FedAvg聚合模型参数
    aggregated_model = gradient_aggregator.fedavg_aggregate(
        gradients=valid_client_models,  # 这里传入模型参数
        sample_counts=valid_sample_counts
    )
    
    # 步骤5: 计算得分并更新权益
    print(f"\n[步骤5] 计算得分并更新权益...")
    for client_id in selected_clients:
        similarity = similarities.get(client_id, 0.0)
        score = stake_manager.calculate_score(
            client_id=client_id,
            similarity=similarity,
            round_number=round_number,
            similarity_weight=0.7,
            stake_weight=0.3
        )
        print(f"客户端 {client_id}: 相似度={similarity:.4f}, 得分={score:.4f}")
    
    stake_manager.update_stakes_from_scores(round_number, reward_factor=0.1, penalty_factor=0.05)
    
    # 步骤6: 执行deputy起草奖罚报告并提交验证
    print(f"\n[步骤6] 执行deputy起草奖罚报告并提交验证...")
    # FedAvg更新：直接使用聚合后的模型参数（已经通过相似度过滤和加权平均）
    # aggregated_model已经是聚合后的模型参数，不需要再计算
    
    transaction_data = {
        'round_number': round_number,
        'aggregated_model': aggregated_model,
        'similarities': similarities,
        'scores': {cid: stake_manager.round_scores[round_number].get(cid, 0.0) 
                  for cid in selected_clients},
        'executor_deputy': executor_deputy_id
    }
    
    npc_deputy.submit_transaction_for_verification(round_number, transaction_data)
    
    # 步骤7: 验证deputy验证报告（2/3M共识）
    print(f"\n[步骤7] 验证deputy验证报告（2/3M共识）...")
    deputy_ids = npc_deputy.get_deputies()
    verify_deputies = [did for did in deputy_ids if did != executor_deputy_id]
    
    # 模拟验证deputy投票（实际应用中需要真实验证）
    for verify_deputy_id in verify_deputies[:len(verify_deputies)]:
        # 简单验证：检查聚合模型是否有效
        is_valid = True  # 实际应用中需要更复杂的验证逻辑
        npc_deputy.vote_on_transaction(round_number, verify_deputy_id, is_valid)
    
    # 检查共识
    consensus_reached, approve_votes, total_votes = npc_deputy.check_transaction_consensus(
        round_number, threshold_ratio=2.0/3.0, executor_deputy_id=executor_deputy_id
    )
    
    if consensus_reached:
        print(f"[INFO] 共识达成！通过票数: {approve_votes}/{total_votes}")
        
        # 步骤8: 保存结果到区块链
        print(f"\n[步骤8] 保存结果到区块链...")
        executor_node = deputy_manager.get_deputy_node(executor_deputy_id)
        if executor_node:
            # 将聚合模型转换为可序列化格式
            aggregated_dict = {}
            for key, value in aggregated_model.items():
                if isinstance(value, torch.Tensor):
                    aggregated_dict[key] = value.cpu().detach().numpy().tolist()
                else:
                    aggregated_dict[key] = value
            
            # 步骤8.1: 保存客户端模型参数到当前轮次的区块（block_j）
            # 先清空pending_updates，然后收集所有客户端的模型更新
            executor_node.blockchain.pending_updates = []
            
            # 收集所有客户端的模型更新（这些已经在步骤3中提交到各自的节点）
            # 注意：创建新的ModelUpdate对象，但model_params已经是序列化格式，可以直接使用
            from blockchain import ModelUpdate
            for client_id in selected_clients:
                client_node = network.get_node(client_id)
                # 查找当前轮次的更新
                for update in client_node.blockchain.pending_updates:
                    if update.round_number == round_number:
                        # 创建新的ModelUpdate对象
                        # model_params已经是序列化后的格式（list），可以直接使用引用
                        # 因为创建区块后会被清空，不会导致内存泄漏
                        new_update = ModelUpdate(
                            client_id=update.client_id,
                            round_number=update.round_number,
                            model_params=update.model_params,  # 直接使用引用，创建区块后会清空
                            timestamp=update.timestamp,
                            previous_hash=update.previous_hash
                        )
                        executor_node.blockchain.pending_updates.append(new_update)
                        break  # 每个客户端在当前轮次只有一个update
            
            # 准备奖赏信息（从stake_manager获取）
            rewards_dict = {}
            scores_dict = {}
            for client_id in selected_clients:
                # 获取客户端得分（作为奖赏）
                score = stake_manager.round_scores[round_number].get(client_id, 0.0)
                scores_dict[client_id] = score
                # 奖赏可以根据得分计算（这里简化处理，使用得分作为奖赏）
                rewards_dict[client_id] = score
            
            # 创建当前轮次的区块（包含：客户端模型参数、聚合后的模型参数、奖赏、相似度、得分）
            pending_updates_count = len(executor_node.blockchain.pending_updates)
            if executor_node.blockchain.pending_updates:
                # 清理CUDA缓存（如果有GPU）
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                
                result = executor_node.validate_and_seal_block(
                    aggregated_model=aggregated_dict,  # 聚合后的模型参数
                    round_number=round_number,
                    rewards=rewards_dict,  # 客户端奖赏
                    similarities=similarities,  # 客户端相似度
                    scores=scores_dict  # 客户端得分
                )
                if result:
                    print(f"[INFO] 轮次 {round_number} 的区块创建成功")
                    print(f"  - 客户端模型参数: {pending_updates_count} 个更新")
                    print(f"  - 聚合模型: 已保存")
                    print(f"  - 奖赏信息: {len(rewards_dict)} 个客户端")
                    print(f"  - 相似度信息: {len(similarities)} 个客户端")
                    print(f"  - 得分信息: {len(scores_dict)} 个客户端")
                    print(f"  - 区块保存位置: node_{executor_deputy_id} (deputy使用自己的区块链)")
                    
                    # 创建区块后，executor_node的pending_updates已被清空（在create_block中）
                    # 但为了安全，再次确认清空
                    executor_node.blockchain.pending_updates = []
                else:
                    print(f"[WARNING] 轮次 {round_number} 的区块创建失败")
            
            # 步骤8.2: 更新执行deputy的创世块全局模型参数
            print(f"[步骤8.2] 更新执行deputy的创世块全局模型参数...")
            executor_node.blockchain.update_genesis_block_model(aggregated_dict)
            print(f"[INFO] 执行deputy节点 {executor_deputy_id} 的创世块全局模型已更新")
            
            # 步骤8.3: 同步所有deputy节点（让其他deputy也获得最新区块）
            print(f"[步骤8.3] 同步所有deputy节点...")
            deputy_manager.sync_deputy_blockchains()
            print(f"[INFO] 所有deputy节点已同步")
            
            # 步骤8.4: 执行deputy将已验证的区块分发给所有客户端，直接覆盖他们的区块链
            print(f"[步骤8.4] 执行deputy分发已验证区块到所有客户端...")
            executor_blockchain = executor_node.blockchain
            
            # 清空所有客户端节点的pending_updates（避免内存累积）
            for client_id in range(client_num):
                client_node = network.get_node(client_id)
                # 清空pending_updates，释放内存
                for update in client_node.blockchain.pending_updates:
                    if update.model_params:
                        update.model_params.clear()
                client_node.blockchain.pending_updates = []
            
            # 直接覆盖所有客户端节点的区块链（使用执行deputy的区块链）
            for client_id in range(client_num):
                client_node = network.get_node(client_id)
                storage_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'storage', 'blockchain_storage')
                client_storage_dir = os.path.join(storage_base, f"node_{client_id}")
                
                # 先保存执行deputy的区块到客户端存储目录（覆盖旧文件）
                if executor_node.blockchain.disk_storage:
                    # 确保存储目录存在
                    os.makedirs(client_storage_dir, exist_ok=True)
                    # 保存所有区块到客户端节点的存储目录（覆盖模式）
                    for i, block in enumerate(executor_node.blockchain.chain):
                        if i == 0:
                            # 创世块
                            block_file = os.path.join(client_storage_dir, "block_0.json")
                        elif executor_node.blockchain.circular_mode:
                            # 数据区块，使用位置索引
                            position = i
                            block_file = os.path.join(client_storage_dir, f"block_{position}.json")
                        else:
                            # 普通模式
                            block_file = os.path.join(client_storage_dir, f"block_{block.index}.json")
                        
                        # 直接保存到文件（覆盖旧文件）
                        with open(block_file, 'w') as f:
                            import json
                            json.dump(block.to_dict(), f, indent=2, default=str)
                
                # 同步区块链结构（不深拷贝大对象，而是同步链结构）
                # 方法：创建一个新的区块链实例，然后从磁盘加载区块（避免内存翻倍）
                from blockchain import Blockchain
                client_node.blockchain = Blockchain(
                    validators=executor_node.blockchain.validators,
                    lightweight_mode=executor_node.blockchain.lightweight_mode,
                    max_blocks=executor_node.blockchain.max_blocks,
                    keep_recent_blocks=executor_node.blockchain.keep_recent_blocks,
                    disk_storage=True,
                    storage_dir=client_storage_dir,
                    in_memory_blocks=executor_node.blockchain.in_memory_blocks,
                    circular_mode=executor_node.blockchain.circular_mode,
                    circular_length=executor_node.blockchain.circular_length
                )
                # 更新total_blocks
                client_node.blockchain.total_blocks = executor_node.blockchain.total_blocks
                
                # 从磁盘加载区块到内存（只加载必要的区块）
                if executor_node.blockchain.circular_mode:
                    # 循环模式：加载创世块和最新的数据区块
                    genesis_block = client_node.blockchain._load_block_from_disk(0, position_in_chain=0)
                    if genesis_block:
                        client_node.blockchain.chain[0] = genesis_block
                    # 加载最新的数据区块
                    latest_position = executor_node.blockchain.total_blocks % (executor_node.blockchain.circular_length - 1)
                    if latest_position == 0:
                        latest_position = executor_node.blockchain.circular_length - 1
                    latest_block = client_node.blockchain._load_block_from_disk(latest_position, position_in_chain=latest_position)
                    if latest_block and latest_position < len(client_node.blockchain.chain):
                        client_node.blockchain.chain[latest_position] = latest_block
                else:
                    # 普通模式：只加载创世块和最新区块
                    genesis_block = client_node.blockchain._load_block_from_disk(0, position_in_chain=None)
                    if genesis_block:
                        client_node.blockchain.chain[0] = genesis_block
                
                print(f"[INFO] 客户端 {client_id} 的区块链已更新（覆盖自执行deputy {executor_deputy_id}，已优化内存使用）")
            
            print(f"[INFO] 所有客户端节点的区块链已从执行deputy {executor_deputy_id} 同步并覆盖")
        
        # 更新全局模型
        global_model.load_state_dict(aggregated_model)
        
        # 步骤9: 更新下一轮deputy
        print(f"\n[步骤9] 更新下一轮deputy...")
        npc_deputy.update_deputies_from_scores(
            round_number=round_number,
            performance_weight=0.6,
            stake_weight=0.4
        )
        
        # 更新deputy管理器
        new_deputy_ids = npc_deputy.get_deputies()
        deputy_manager.update_deputies(new_deputy_ids)
        
        # 清理投票记录
        npc_deputy.clear_transaction_votes(round_number)
        
    else:
        print(f"[WARNING] 共识未达成！通过票数: {approve_votes}/{total_votes}")
        # 共识未达成，不更新模型
        print(f"[WARNING] 本轮训练结果无效，不更新模型")
    
    print(f"\n轮次 {round_number} 完成")
    return global_model


def test(model, testloader, device):
    """测试模型"""
    print("\n[测试] 开始模型测试...")
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for testdata in testloader:
            test_data_value, test_data_label = testdata
            test_data_value, test_data_label = test_data_value.to(device), test_data_label.to(device)
            test_data_label_pred = model(test_data_value)
            _, test_predicted = torch.max(test_data_label_pred.data, dim=1)
            test_total += test_data_label.size(0)
            test_correct += (test_predicted == test_data_label).sum().item()
    
    test_acc = round(100 * test_correct / test_total, 3)
    print(f'[测试] 测试准确率: {test_acc:.2f}%')
    return test_acc


def main():
    """主训练函数"""
    # 设置随机种子
    set_random_seed(100)
    
    # 配置参数（与原始CrosSCLR保持一致）
    # 原始CrosSCLR配置:
    #   batch_size: 128
    #   base_lr: 0.1
    #   optimizer: SGD
    #   weight_decay: 1e-4
    #   nesterov: False
    #   num_epoch: 300 (预训练)
    
    C = 0.5  # 参与训练的客户端比例（联邦学习特有参数）
    E = 5    # 每个客户端本地训练的轮数（联邦学习特有参数，FedAvg标准设置）
    B = 128  # BatchSize大小（与原始CrosSCLR一致：batch_size=128）
    client_num = 10  # 客户端总数
    epochs = 300  # 全局训练轮数（可根据需要调整，原始预训练是300轮）
    num_deputies = 3  # Deputy节点数量（M，区块链特有参数）
    
    # 训练超参数（与原始CrosSCLR完全一致）
    base_lr = 0.1  # 基础学习率（与原始CrosSCLR一致）
    weight_decay = 1e-4  # 权重衰减（与原始CrosSCLR一致）
    nesterov = False  # Nesterov动量（与原始CrosSCLR一致）
    momentum = 0.9  # 动量（与原始CrosSCLR一致）
    
    # ========== 攻击配置 ==========
    # 创建攻击配置对象
    attack_config = AttackConfig()
    
    # 是否启用攻击（设置为True以启用攻击测试）
    attack_config.enable_attack = False  # 默认关闭攻击
    
    # 攻击类型选择：
    #   'none': 无攻击（正常训练）
    #   'random': 随机攻击（返回随机模型参数）
    #   'label_flip': 标签翻转攻击（翻转数据标签）
    #   'model_poison': 模型投毒攻击（返回反向梯度）
    #   'scale': 缩放攻击（放大模型参数更新）
    #   'backdoor': 后门攻击（在数据中植入后门模式）
    attack_config.attack_type = 'none'
    
    # 恶意客户端ID列表（这些客户端将执行攻击）
    # 例如：[0, 1] 表示客户端0和1为恶意客户端
    attack_config.malicious_clients = [0, 1]  # 示例：客户端0和1为恶意客户端
    
    # 攻击强度（0.0-1.0，1.0为最强攻击）
    attack_config.attack_strength = 1.0
    
    # 标签翻转攻击配置：标签映射 {原标签: 新标签}
    # 例如：{0: 1, 1: 0} 表示将标签0翻转为1，标签1翻转为0
    attack_config.label_flip_map = {0: 1, 1: 0, 2: 3, 3: 2}  # 示例映射
    
    # 缩放攻击配置：模型参数缩放因子
    # 负值表示反向攻击（例如：-1.0表示完全反向）
    attack_config.scale_factor = -1.0
    
    # 后门攻击配置：后门模式（与数据形状相同的tensor）
    # 例如：创建一个小的后门模式添加到数据中
    attack_config.backdoor_pattern = None  # 可以设置为torch.Tensor，形状为(1, C, T, V, M)
    attack_config.backdoor_target_label = 0  # 后门目标标签
    # ========== 攻击配置结束 ==========
    
    # 解决libiomp5md.dll错误
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    print("=" * 70)
    print("基于NPC Deputy的联邦学习训练（FedAvg）")
    print("=" * 70)
    print(f"配置参数:")
    print(f"  - 客户端数量: {client_num}")
    print(f"  - Deputy数量(M): {num_deputies}")
    print(f"  - 参与训练比例(C): {C}")
    print(f"  - 本地训练轮数(E): {E}")
    print(f"  - 全局训练轮数: {epochs}")
    if attack_config.enable_attack:
        print(f"  - 攻击模式: {attack_config.attack_type}")
        print(f"  - 恶意客户端: {attack_config.malicious_clients}")
        print(f"  - 攻击强度: {attack_config.attack_strength}")
    print("=" * 70)
    
    # 初始化权益管理器
    print("\n[初始化] 权益管理器...")
    stake_manager = StakeManager(num_clients=client_num, random_init=True)
    
    # 初始化NPC Deputy管理器
    print("[初始化] NPC Deputy管理器...")
    npc_deputy = NPCDeputyManager(
        num_deputies=num_deputies,
        stake_manager=stake_manager
    )
    
    # 初始化区块链网络（循环模式：M+1轮）
    print("[初始化] 区块链网络（循环模式：M+1轮）...")
    network = BlockchainNetwork(
        num_nodes=client_num,
        validators=list(range(client_num)),
        lightweight_mode=True,
        disk_storage=True,
        storage_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'storage', 'blockchain_storage'),
        circular_mode=True,
        circular_length=num_deputies + 1  # M+1
    )
    
    # 初始化Deputy区块链管理器（M份账本）
    print("[初始化] Deputy区块链管理器（M份账本）...")
    initial_deputy_ids = npc_deputy.elect_deputies(round_number=0)
    deputy_manager = DeputyBlockchainManager(
        num_deputies=num_deputies,
        deputy_ids=initial_deputy_ids,
        blockchain_network=network
    )
    
    # 初始化梯度聚合器
    print("[初始化] 梯度聚合器...")
    gradient_aggregator = GradientAggregator(similarity_threshold=0.3)
    
    # 初始化客户端数据加载器
    print("\n[初始化] 加载客户端数据...")
    client_dataloader_list = clients_dataloader(B)
    print(f"已加载 {len(client_dataloader_list)} 个客户端的数据")
    
    # 加载测试数据集（NTU60）
    print("\n[初始化] 加载测试数据集...")
    # 使用全局变量
    global test_data_path, test_label_path
    
    # 检查并设置测试数据路径
    # 计算项目根目录（用于构建数据路径）
    script_dir_local = os.path.dirname(os.path.abspath(__file__))
    blockchain_dir_local = os.path.dirname(script_dir_local)
    project_root_local = os.path.dirname(blockchain_dir_local)
    
    if not os.path.exists(test_data_path) or not os.path.exists(test_label_path):
        print(f"[WARNING] 测试数据文件不存在，尝试使用默认路径")
        # 使用项目根目录的绝对路径
        default_test_data_path = os.path.join(project_root_local, 'data', 'NTU60_frame50', 'xview', 'val_position.npy')
        default_test_label_path = os.path.join(project_root_local, 'data', 'NTU-RGB-D', 'xview', 'val_label.pkl')
        
        if os.path.exists(default_test_data_path) and os.path.exists(default_test_label_path):
            test_data_path = default_test_data_path
            test_label_path = default_test_label_path
            print(f"[DEBUG] 使用默认路径: {test_data_path}")
        else:
            print(f"[ERROR] 测试数据文件不存在！")
            print(f"  请检查以下路径:")
            print(f"  - {test_data_path}")
            print(f"  - {test_label_path}")
            print(f"  - {default_test_data_path}")
            print(f"  - {default_test_label_path}")
            raise FileNotFoundError("测试数据文件不存在")
    
    Test_dataset = NTU60Dataset(
        data_path=test_data_path,
        label_path=test_label_path,
        indices=None,
        mmap=True,
        use_augmentation=False,  # 测试时不使用数据增强
        shear_amplitude=0.0,
        temperal_padding_ratio=0
    )
    # 测试batch size与原始CrosSCLR一致：test_batch_size=128
    testloader = DataLoader(Test_dataset, shuffle=False, batch_size=128, num_workers=0)
    print(f"测试集大小: {len(Test_dataset)} 个样本")
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[初始化] 使用设备: {device}")
    
    # 初始化模型和损失函数（NTU60，60个类别）- 使用ST-GCN模型
    model = STGCNModel(
        in_channels=3,
        hidden_channels=16,
        hidden_dim=256,
        num_class=60,
        graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
        edge_importance_weighting=True,
        dropout=0.5
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    
    # 将初始模型保存到区块链
    print("\n[初始化] 保存初始模型到区块链...")
    initial_params = model.state_dict()
    initial_dict = {}
    for key, value in initial_params.items():
        initial_dict[key] = value.cpu().detach().numpy().tolist()
    
    # 将初始全局模型保存到创世块（所有deputy节点和客户端节点）
    print("\n[初始化] 将初始全局模型保存到创世块...")
    executor_node = deputy_manager.get_deputy_node(initial_deputy_ids[0])
    if executor_node:
        # 更新执行deputy节点的创世块
        executor_node.blockchain.update_genesis_block_model(initial_dict)
        print(f"[INFO] 执行deputy节点 {initial_deputy_ids[0]} 的创世块已更新")
        
        # 同步所有deputy节点
        deputy_manager.sync_deputy_blockchains()
        print(f"[INFO] 所有deputy节点已同步初始模型")
        
        # 重要：同步所有客户端节点的区块链（让客户端能获取全局模型）
        # 从deputy节点同步到所有客户端节点
        executor_deputy_id = initial_deputy_ids[0]
        executor_deputy_blockchain = executor_node.blockchain
        
        for client_id in range(client_num):
            client_node = network.get_node(client_id)
            if client_id != executor_deputy_id:  # deputy节点已经同步过了
                client_node.sync_blockchain(executor_deputy_blockchain)
                # 确保客户端节点的创世块也被更新
                client_node.blockchain.update_genesis_block_model(initial_dict)
        
        print(f"[INFO] 所有客户端节点已同步初始模型到创世块")
    else:
        print(f"[WARNING] 无法获取执行deputy节点")
    
    # 验证初始模型是否正确保存（从客户端节点验证）
    test_client_node = network.get_node(0)
    test_model = test_client_node.get_latest_model()
    if test_model is None:
        print("[WARNING] 验证失败：无法从客户端节点获取初始模型")
        # 尝试从deputy节点获取
    test_model = deputy_manager.get_latest_model_from_deputy()
    if test_model is None:
            print("[WARNING] 也无法从deputy节点获取初始模型")
    else:
        print("[INFO] 验证成功：初始模型已正确保存到区块链，客户端可以获取")
    
    # 开始训练
    test_accuracies = []
    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70)
    
    for round_num in range(1, epochs + 1):
        # 训练（使用与原始CrosSCLR完全一致的参数）
        model = train_npc_deputy_round(
            network=network,
            deputy_manager=deputy_manager,
            npc_deputy=npc_deputy,
            stake_manager=stake_manager,
            gradient_aggregator=gradient_aggregator,
            client_num=client_num,
            C=C,
            E=E,
            client_dataloader_list=client_dataloader_list,
            device=device,
            criterion=criterion,
            round_number=round_num,
            base_lr=base_lr,  # 与原始CrosSCLR一致：0.1
            weight_decay=weight_decay,  # 与原始CrosSCLR一致：1e-4
            nesterov=nesterov,  # 与原始CrosSCLR一致：False
            momentum=momentum,  # 与原始CrosSCLR一致：0.9
            attack_config=attack_config  # 攻击配置
        )
        
        # 测试
        test_acc = test(model, testloader, device)
        test_accuracies.append(test_acc)
        
        # 每10轮显示一次统计信息
        if round_num % 10 == 0:
            print(f"\n[统计] 轮次 {round_num}:")
            print(f"  - 当前Deputy: {npc_deputy.get_deputies()}")
            print(f"  - 权益前3名: {stake_manager.get_top_stake_clients(3)}")
            storage_info = deputy_manager.get_storage_info()
            print(f"  - 账本份数: {len(storage_info['deputy_storage'])}")
    
    # 绘制测试准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), test_accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel('轮次')
    plt.ylabel('测试准确率 (%)')
    plt.title(f'基于NPC Deputy的FedAvg - 测试准确率曲线 (C={C}, E={E}, B={B}, M={num_deputies})')
    plt.grid(True)
    plt.savefig('npc_deputy_fedavg_accuracy.png')
    plt.show()
    
    # 保存结果
    print("\n[完成] 训练完成，保存结果...")
    result_file = f'Train_result/NPC_Deputy_C{C}B{B}E{E}M{num_deputies}.npy'
    os.makedirs('Train_result', exist_ok=True)
    np.save(result_file, test_accuracies)
    print(f"结果已保存到: {result_file}")
    print(f"最终测试准确率: {test_accuracies[-1]:.2f}%")
    print(f"最佳测试准确率: {max(test_accuracies):.2f}% (轮次 {test_accuracies.index(max(test_accuracies))+1})")


if __name__ == '__main__':
    main()

