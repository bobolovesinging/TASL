"""
基于区块链的FedAvg训练脚本
使用区块链管理模型更新，实现去中心化的联邦学习
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from blockchain_node import BlockchainNetwork

from model import CNN

Datas_path = 'Client_datasets_10'

# 设置随机种子
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


class CustomDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, data):
        self.len = len(data)
        self.x_data = torch.from_numpy(np.array(list(map(lambda x: x[0], data)), dtype=np.float32))
        self.y_data = torch.from_numpy(np.array(list(map(lambda x: x[-1], data)))).squeeze().long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def clients_dataloader(batch_size=60):
    """为所有客户端创建数据加载器"""
    dataloader_list = []
    dir_path = os.listdir(Datas_path)
    
    for dir in dir_path:
        data_path = os.path.join(Datas_path, dir, "data.npy")
        label_path = os.path.join(Datas_path, dir, "label.npy")
        
        data = np.load(data_path)
        label = np.load(label_path)
        dataset = [[i, j] for i, j in zip(data, label)]
        
        dataloader = DataLoader(CustomDataset(dataset), shuffle=True, batch_size=batch_size)
        dataloader_list.append(dataloader)
    
    return dataloader_list


def client_update_blockchain(client_id, E, model_parameter, dataloader, device, criterion):
    """
    客户端本地训练并返回模型更新（区块链版本）
    :param client_id: 客户端ID
    :param E: 本地训练轮数
    :param model_parameter: 全局模型参数
    :param dataloader: 客户端数据加载器
    :param device: 设备（CPU/GPU）
    :param criterion: 损失函数
    :return: 训练后的模型参数
    """
    client_model = CNN().to(device)
    client_model.load_state_dict(model_parameter)
    client_model.train()
    
    optimizer = torch.optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
    
    for epoch in range(E):
        correct = 0
        total = 0
        
        for data, label in dataloader:
            train_data_value, train_data_label = data.to(device), label.to(device)
            train_data_label_pred = client_model(train_data_value)
            
            loss = criterion(train_data_label_pred, train_data_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(train_data_label_pred, 1)
            total += train_data_label.size(0)
            correct += (predicted == train_data_label).sum().item()
        
        accuracy = 100 * correct / total
        print(f'客户端 {client_id}, Epoch {epoch+1}/{E}, 准确率: {accuracy:.2f}%, 损失: {loss.item():.4f}')
    
    return client_model.state_dict()


def train_blockchain(network, client_num, C, E, client_dataloader_list, device, criterion, round_number):
    """
    基于区块链的FedAvg训练
    :param network: 区块链网络
    :param client_num: 客户端总数
    :param C: 参与训练的客户端比例
    :param E: 每个客户端本地训练轮数
    :param client_dataloader_list: 客户端数据加载器列表
    :param device: 设备
    :param criterion: 损失函数
    :param round_number: 当前轮次
    :return: 更新后的全局模型
    """
    # 获取最新的全局模型
    print(f"\n[DEBUG] 轮次 {round_number}: 获取最新全局模型...")
    latest_model_params = network.get_node(0).get_latest_model()
    
    # 如果没有全局模型，初始化一个
    if latest_model_params is None:
        print("[WARNING] 未找到全局模型，创建新模型")
        global_model = CNN().to(device)
        latest_model_params = global_model.state_dict()
    else:
        print("[DEBUG] 成功获取最新全局模型")
        global_model = CNN().to(device)
        global_model.load_state_dict(latest_model_params)
    
    # 随机选择参与训练的客户端
    selected_clients = random.sample(range(client_num), int(client_num * C))
    print(f"[DEBUG] 轮次 {round_number}: 选中 {len(selected_clients)}/{client_num} 个客户端 (比例 {C:.1%})")
    print(f"[DEBUG] 选中的客户端: {selected_clients}")
    
    # 客户端本地训练并提交更新到区块链
    print(f"\n[DEBUG] 开始客户端本地训练...")
    pending_updates_before = network.get_node(0).get_pending_updates_count()
    print(f"[DEBUG] 提交前待处理更新数: {pending_updates_before}")
    
    for idx, client_id in enumerate(selected_clients, 1):
        print(f"\n[DEBUG] [{idx}/{len(selected_clients)}] 客户端 {client_id} 开始训练...")
        dataloader = client_dataloader_list[client_id]
        print(f"[DEBUG] 客户端 {client_id} 数据集大小: {len(dataloader.dataset)} 个样本")
        
        # 本地训练
        updated_params = client_update_blockchain(
            client_id=client_id,
            E=E,
            model_parameter=latest_model_params,
            dataloader=dataloader,
            device=device,
            criterion=criterion
        )
        
        # 提交模型更新到区块链
        print(f"[DEBUG] 客户端 {client_id} 提交模型更新到区块链...")
        node = network.get_node(client_id)
        update = node.submit_model_update(round_number=round_number, model_params=updated_params)
        print(f"[DEBUG] 客户端 {client_id} 更新已提交，哈希: {update.hash[:16]}...")
        print(f"[DEBUG] 当前待处理更新数: {node.get_pending_updates_count()}")
    
    # 聚合更新并由验证者封装区块
    print(f"\n[DEBUG] 开始聚合模型更新...")
    pending_updates_after = network.get_node(0).get_pending_updates_count()
    print(f"[DEBUG] 待聚合的更新数: {pending_updates_after}")
    
    aggregated_params = network.aggregate_and_seal(
        round_number=round_number,
        selected_clients=selected_clients
    )
    
    print(f"[DEBUG] 模型聚合完成")
    print(f"[DEBUG] 聚合后的模型参数键数量: {len(aggregated_params)}")
    
    # 更新全局模型
    global_model.load_state_dict(aggregated_params)
    
    chain_length = network.get_node(0).blockchain.get_chain_length()
    memory_info = network.get_node(0).blockchain.get_memory_info()
    print(f"\n[DEBUG] 轮次 {round_number} 完成")
    print(f"[DEBUG] 区块链长度: {chain_length}")
    print(f"[DEBUG] 内存中区块: {memory_info['in_memory_blocks']}, 磁盘中区块: {memory_info['disk_blocks']}")
    print(f"[DEBUG] 内存占用: {memory_info['estimated_memory_bytes']/1024/1024:.2f}MB")
    
    return global_model


def test(model, testloader, device):
    """测试模型"""
    print("\n[DEBUG] 开始模型测试...")
    model.eval()
    test_correct = 0
    test_total = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, testdata in enumerate(testloader, 1):
            test_data_value, test_data_label = testdata
            test_data_value, test_data_label = test_data_value.to(device), test_data_label.to(device)
            test_data_label_pred = model(test_data_value)
            _, test_predicted = torch.max(test_data_label_pred.data, dim=1)
            test_total += test_data_label.size(0)
            test_correct += (test_predicted == test_data_label).sum().item()
            batch_count = batch_idx
            
            if batch_idx % 10 == 0:
                current_acc = 100 * test_correct / test_total
                print(f"  [DEBUG] 测试进度: {batch_idx} 批次, 当前准确率: {current_acc:.2f}%")
    
    test_acc = round(100 * test_correct / test_total, 3)
    print(f"[DEBUG] 测试完成: {batch_count} 批次, {test_total} 个样本")
    print(f'[DEBUG] 测试准确率: {test_acc:.2f}%')
    return test_acc


def main():
    """主训练函数"""
    # 设置随机种子
    set_random_seed(100)
    
    # 配置参数
    C = 0.5  # 参与训练的客户端比例
    E = 5    # 每个客户端本地训练的轮数
    B = 600  # BatchSize大小
    client_num = 10  # 客户端总数
    epochs = 100  # 全局训练轮数
    
    # 解决libiomp5md.dll错误
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    # 初始化区块链网络（启用内存优化 + 磁盘存储）
    print("=" * 70)
    print("[DEBUG] 初始化区块链网络...")
    print(f"[DEBUG] 配置参数:")
    print(f"  - 客户端数量: {client_num}")
    print(f"  - 轻量级模式: True")
    print(f"  - 磁盘存储: True")
    print(f"  - 内存中保留区块数: 5")
    print(f"  - 最大区块数: 50")
    print("=" * 70)
    
    # 启用轻量级模式、自动修剪和磁盘存储以节省内存
    # lightweight_mode=True: 创建区块后自动清理详细更新
    # disk_storage=True: 旧区块保存到磁盘，内存中只保留最近5个
    # max_blocks=50: 最多保留50个区块，超过后自动修剪
    network = BlockchainNetwork(
        num_nodes=client_num,
        lightweight_mode=True,  # 启用轻量级模式
        max_blocks=50,  # 最多保留50个区块
        keep_recent_blocks=50,
        disk_storage=True,  # 启用磁盘存储
        storage_dir="blockchain_storage",  # 存储目录
        in_memory_blocks=5  # 内存中只保留最近5个区块
    )
    print(f"[DEBUG] 区块链网络已创建，包含 {client_num} 个节点")
    print(f"[DEBUG] 验证者列表: {network.validators}")
    print("[DEBUG] 内存优化: 轻量级模式 + 磁盘存储已启用")
    print("  - 内存中保留: 最近5个区块")
    print("  - 磁盘存储: 旧区块自动保存到 blockchain_storage/")
    print("  - 最多保留: 50个区块（超过后自动删除旧文件）")
    
    # 初始化客户端数据加载器
    print("\n[DEBUG] 加载客户端数据...")
    client_dataloader_list = clients_dataloader(B)
    print(f"[DEBUG] 已加载 {len(client_dataloader_list)} 个客户端的数据")
    for i, dataloader in enumerate(client_dataloader_list):
        print(f"  - 客户端 {i+1}: {len(dataloader.dataset)} 个样本")
    
    # 加载测试数据集
    print("\n[DEBUG] 加载测试数据集...")
    data_test = np.load('Test_dataset/MNIST_test_data.npy')
    label_test = np.load('Test_dataset/MNIST_test_label.npy')
    test_dataset = [[i, j] for i, j in zip(data_test, label_test)]
    Test_dataset = CustomDataset(test_dataset)
    testloader = DataLoader(Test_dataset, shuffle=True, batch_size=256)
    print(f"[DEBUG] 测试集大小: {len(Test_dataset)} 个样本")
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEBUG] 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"[DEBUG] GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"[DEBUG] GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 初始化模型和损失函数
    model = CNN(in_channels=1, classes=10)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    
    # 将初始模型保存到区块链（作为第一个区块）
    print("\n[DEBUG] 初始化区块链...")
    print("[DEBUG] 创建初始模型参数...")
    initial_params = model.state_dict()
    print(f"[DEBUG] 模型参数数量: {len(initial_params)} 个层")
    print(f"[DEBUG] 模型参数键: {list(initial_params.keys())[:5]}...")  # 只显示前5个
    
    print(f"\n[DEBUG] 提交初始模型更新到区块链（所有 {client_num} 个节点）...")
    for i in range(client_num):
        node = network.get_node(i)
        update = node.submit_model_update(round_number=0, model_params=initial_params)
        print(f"  [DEBUG] 节点 {i} 已提交更新，哈希: {update.hash[:16]}...")
    
    # 创建初始区块
    print("\n[DEBUG] 聚合初始模型参数...")
    initial_aggregated = {}
    for key, value in initial_params.items():
        initial_aggregated[key] = value.cpu().detach().numpy().tolist()
    
    print("[DEBUG] 验证并封装初始区块...")
    result = network.get_node(0).validate_and_seal_block(aggregated_model=initial_aggregated)
    if result:
        print("[DEBUG] 初始区块创建成功")
    else:
        print("[WARNING] 初始区块创建失败！")
    
    print("[DEBUG] 同步所有节点...")
    network.sync_all_nodes()
    
    chain_length = network.get_node(0).blockchain.get_chain_length()
    memory_info = network.get_node(0).blockchain.get_memory_info()
    print(f"\n[DEBUG] 初始模型已保存到区块链")
    print(f"[DEBUG] 区块链长度: {chain_length}")
    print(f"[DEBUG] 内存中区块: {memory_info['in_memory_blocks']}, 磁盘中区块: {memory_info['disk_blocks']}")
    
    # 开始训练
    test_accuracies = []
    print("\n开始训练...")
    print("=" * 60)
    
    for round_num in range(1, epochs + 1):
        print(f"\n{'='*60}")
        print(f"全局轮次: {round_num}/{epochs}")
        print(f"{'='*60}")
        
        # 训练
        model = train_blockchain(
            network=network,
            client_num=client_num,
            C=C,
            E=E,
            client_dataloader_list=client_dataloader_list,
            device=device,
            criterion=criterion,
            round_number=round_num
        )
        
        # 测试
        test_acc = test(model, testloader, device)
        test_accuracies.append(test_acc)
        
        # 每10轮保存一次区块链并显示内存信息
        if round_num % 10 == 0:
            print(f"\n[DEBUG] 轮次 {round_num}: 执行检查点保存...")
            blockchain_file = f'blockchain_checkpoint_round_{round_num}.json'
            network.get_node(0).save_blockchain(blockchain_file)
            print(f"[DEBUG] 区块链已保存到: {blockchain_file}")
            
            # 显示内存使用情况
            memory_info = network.get_node(0).blockchain.get_memory_info()
            print(f"\n[DEBUG] ========== 内存使用情况 ==========")
            print(f"[DEBUG] 总区块数: {memory_info['total_blocks']}")
            print(f"[DEBUG] 内存中区块: {memory_info['in_memory_blocks']}")
            print(f"[DEBUG] 磁盘中区块: {memory_info['disk_blocks']}")
            print(f"[DEBUG] 轻量级区块: {memory_info['lightweight_blocks']}")
            print(f"[DEBUG] 完整区块: {memory_info['full_blocks']}")
            print(f"[DEBUG] 内存占用: {memory_info['estimated_memory_bytes']/1024/1024:.2f} MB")
            print(f"[DEBUG] 磁盘占用: {memory_info['disk_size_bytes']/1024/1024:.2f} MB")
            print(f"[DEBUG] ====================================")
    
    # 绘制测试准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), test_accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel('轮次')
    plt.ylabel('测试准确率 (%)')
    plt.title(f'基于区块链的FedAvg - 测试准确率曲线 (C={C}, E={E}, B={B})')
    plt.grid(True)
    plt.savefig('blockchain_fedavg_accuracy.png')
    plt.show()
    
    # 保存结果
    print("\n[DEBUG] 训练完成，保存结果...")
    result_file = f'Train_result/Blockchain_C{C}B{B}E{E}.npy'
    np.save(result_file, test_accuracies)
    print(f"[DEBUG] 结果已保存到: {result_file}")
    print(f"[DEBUG] 最终测试准确率: {test_accuracies[-1]:.2f}%")
    print(f"[DEBUG] 最佳测试准确率: {max(test_accuracies):.2f}% (轮次 {test_accuracies.index(max(test_accuracies))+1})")
    
    # 保存最终区块链
    print("\n[DEBUG] 保存最终区块链...")
    network.get_node(0).save_blockchain('blockchain_final.json')
    print("[DEBUG] 最终区块链已保存到: blockchain_final.json")
    
    # 验证区块链
    print("\n[DEBUG] 验证区块链完整性...")
    is_valid = network.get_node(0).blockchain.validate_chain()
    chain_length = network.get_node(0).blockchain.get_chain_length()
    memory_info = network.get_node(0).blockchain.get_memory_info()
    
    print(f"[DEBUG] ========== 最终统计 ==========")
    print(f"[DEBUG] 区块链验证结果: {'✓ 有效' if is_valid else '✗ 无效'}")
    print(f"[DEBUG] 区块链总长度: {chain_length}")
    print(f"[DEBUG] 内存中区块: {memory_info['in_memory_blocks']}")
    print(f"[DEBUG] 磁盘中区块: {memory_info['disk_blocks']}")
    print(f"[DEBUG] 内存占用: {memory_info['estimated_memory_bytes']/1024/1024:.2f} MB")
    print(f"[DEBUG] 磁盘占用: {memory_info['disk_size_bytes']/1024/1024:.2f} MB")
    print(f"[DEBUG] =============================")


if __name__ == '__main__':
    main()

