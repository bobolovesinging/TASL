"""
基于区块链的FedAvg服务器
使用区块链技术共享数据分布信息以解决non-IID问题
"""
import time
import copy
import os
import hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from flcore.clients.clientour import clientAVG
from flcore.servers.serveravg import FedAvg
from flcore.blockchain.blockchain_node import BlockchainNetwork
from flcore.blockchain.data_distribution import DataDistributionStats, aggregate_distributions
from flcore.blockchain.distribution_update import DistributionUpdate, DistributionBlock
from utils.model_utils import shear, crop, random_rotate, random_spatial_flip
import json


class FedAvgBlockchain(FedAvg):
    """基于区块链的FedAvg服务器，支持数据分布共享"""
    
    def __init__(self, args, times):
        # 先调用父类初始化（会设置客户端和send_pseudo_labels等）
        super().__init__(args, times)
        
        # 初始化区块链网络（用于数据分布共享）
        self.blockchain_network = BlockchainNetwork(
            num_nodes=self.num_clients,
            lightweight_mode=True,
            max_blocks=100,
            keep_recent_blocks=50,
            disk_storage=True,
            storage_dir="blockchain_storage_distribution",
            in_memory_blocks=10
        )
        
        # 数据分布相关
        self.global_distribution = None  # 全局数据分布
        self.client_distributions = {}  # 客户端数据分布缓存
        
        # 为每个客户端设置全局分布（初始为None）
        for client in self.clients:
            client.global_distribution = None
        
        print("\n[Blockchain] 区块链功能已启用")
        print(f"[Blockchain] 客户端数量: {self.num_clients}")
        print(f"[Blockchain] 区块链存储目录: blockchain_storage_distribution")
        
        # 服务器端加权损失函数（基于全局分布）
        self.weighted_loss = None
        
        # TensorBoard 可视化
        tb_log_dir = self.get_tb_log_dir(f"{args.algorithm}_blockchain")
        os.makedirs(tb_log_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(tb_log_dir)
        print(f"[TensorBoard] 日志目录: {tb_log_dir}")
    
    def get_global_class_weights(self):
        """
        根据全局分布生成类别权重，用于加权损失函数
        返回: torch.Tensor 形状为 (num_classes,)，如果全局分布不可用则返回None
        """
        if self.global_distribution is None:
            return None
        
        global_weights = self.global_distribution.get('global_class_weights', {})
        if not global_weights:
            return None
        
        # 计算反频率权重（inverse frequency weighting）
        # 类别在全局分布中越少，权重越大
        class_weights = []
        for i in range(self.num_classes):
            weight = global_weights.get(i, 1e-6)  # 如果类别不存在，使用很小的值
            # 反频率权重：1 / frequency，归一化
            inv_weight = 1.0 / (weight + 1e-6)
            class_weights.append(inv_weight)
        
        # 转换为tensor并归一化
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        # 归一化：使平均权重为1
        class_weights_tensor = class_weights_tensor / class_weights_tensor.mean() * self.num_classes
        
        return class_weights_tensor.to(self.device)
    
    def get_weighted_loss(self):
        """
        获取加权损失函数（如果全局分布可用）
        返回: nn.CrossEntropyLoss 对象
        """
        # 如果 weighted_loss 已设置且全局分布未变化，直接返回
        if self.weighted_loss is not None and self.global_distribution is not None:
            return self.weighted_loss
        
        # 尝试获取全局类别权重
        class_weights = self.get_global_class_weights()
        
        if class_weights is not None:
            # 使用加权损失函数
            self.weighted_loss = torch.nn.CrossEntropyLoss(weight=class_weights).to(self.device)
            print(f"[Blockchain] 服务器使用全局分布加权损失函数")
            print(f"[Blockchain] 类别权重范围: [{class_weights.min():.3f}, {class_weights.max():.3f}]")
            print(f"[Blockchain] 少数类别（权重>1.5）: {torch.sum(class_weights > 1.5).item()} 个")
        else:
            # 使用标准损失函数
            self.weighted_loss = self.crossentropyloss
        
        return self.weighted_loss
    
    def collect_client_distributions(self):
        """
        收集所有客户端的数据分布并存储到区块链
        """
        print("\n[Blockchain] 开始收集客户端数据分布...")
        
        distributions = []
        distribution_updates = []
        
        for client in self.selected_clients:
            # 计算客户端数据分布
            print(f"[Blockchain] 计算客户端 {client.id} 的数据分布...")
            
            # 创建分布统计对象
            stats = DataDistributionStats(
                client_id=client.id,
                num_classes=self.num_classes
            )
            
            # 尝试多种方式获取标签
            labels = None
            
            # 方法1: 从trainloader数据集获取
            if hasattr(client, 'trainloader') and client.trainloader is not None:
                try:
                    dataset = client.trainloader.dataset
                    if hasattr(dataset, 'targets'):
                        labels = dataset.targets
                    elif hasattr(dataset, 'labels'):
                        labels = dataset.labels
                    elif isinstance(dataset, list):
                        labels = [item[1] for item in dataset]
                except Exception as e:
                    print(f"[Blockchain] 警告: 无法从trainloader获取标签: {e}")
            
            # 方法2: 从文件读取数据
            if labels is None:
                try:
                    from utils.data_utils import read_client_data, read_client_traindata
                    # 尝试读取训练数据
                    train_data = read_client_data(self.dataset, client.id, is_train=True)
                    if isinstance(train_data, list):
                        # 列表格式: [(data, label), ...]
                        labels = [item[1].item() if torch.is_tensor(item[1]) else item[1] for item in train_data]
                    elif isinstance(train_data, dict) and 'y' in train_data:
                        # 字典格式: {'x': ..., 'y': ...}
                        y_data = train_data['y']
                        if torch.is_tensor(y_data):
                            labels = y_data.cpu().numpy().tolist()
                        else:
                            labels = y_data.tolist() if hasattr(y_data, 'tolist') else list(y_data)
                except Exception as e:
                    print(f"[Blockchain] 警告: 无法从文件读取数据: {e}")
            
            # 方法3: 从数据加载器计算（需要遍历数据）
            if labels is None and hasattr(client, 'trainloader') and client.trainloader is not None:
                try:
                    print(f"[Blockchain] 从数据加载器计算分布...")
                    stats.compute_from_data(client.trainloader, model=None, device=self.device)
                except Exception as e:
                    print(f"[Blockchain] 警告: 无法从数据加载器计算分布: {e}")
            
            # 如果找到了标签，使用标签计算（更快）
            if labels is not None:
                try:
                    stats.compute_from_labels(labels)
                except Exception as e:
                    print(f"[Blockchain] 警告: 计算标签分布失败: {e}")
                    # 如果失败，尝试从数据加载器计算
                    if hasattr(client, 'trainloader') and client.trainloader is not None:
                        stats.compute_from_data(client.trainloader, model=None, device=self.device)
            
            # 保存分布信息
            self.client_distributions[client.id] = stats
            distributions.append(stats)
            
            # 创建分布更新交易
            dist_dict = stats.to_dict()
            distribution_update = DistributionUpdate(
                client_id=client.id,
                round_number=self.current_round if hasattr(self, 'current_round') else 0,
                distribution_stats=dist_dict,
                timestamp=time.time(),
                previous_hash=self._get_latest_distribution_hash()
            )
            distribution_updates.append(distribution_update)
            
            print(f"[Blockchain] 客户端 {client.id} 分布统计:")
            print(f"  - 总样本数: {stats.total_samples}")
            print(f"  - 类别分布: {dict(list(stats.class_distribution.items())[:5])}...")  # 只显示前5个类别
        
        # 聚合全局分布
        if distributions:
            print("\n[Blockchain] 聚合全局数据分布...")
            self.global_distribution = aggregate_distributions(distributions)
            
            print(f"[Blockchain] 全局分布统计:")
            print(f"  - 总样本数: {self.global_distribution['total_samples']}")
            print(f"  - 参与客户端数: {self.global_distribution['num_clients']}")
            print(f"  - 全局类别权重示例: {dict(list(self.global_distribution['global_class_weights'].items())[:5])}...")
            
            # 将全局分布发送给所有客户端
            self.broadcast_global_distribution()
            
            # 重置服务器端的加权损失函数，使其根据新的全局分布重新计算
            self.weighted_loss = None
            
            # 可选：将分布信息存储到区块链（如果需要持久化）
            # self._store_distributions_to_blockchain(distribution_updates)
        
        return self.global_distribution
    
    def _get_latest_distribution_hash(self) -> str:
        """获取最新的分布区块哈希"""
        # 简化实现：使用时间戳作为previous_hash
        return hashlib.sha256(str(time.time()).encode()).hexdigest()
    
    def broadcast_global_distribution(self):
        """将全局分布广播给所有客户端"""
        if self.global_distribution is None:
            return
        
        print(f"\n[Blockchain] 广播全局分布给 {len(self.clients)} 个客户端...")
        for client in self.clients:
            client.global_distribution = self.global_distribution
            # 重置加权损失函数，使其根据新的全局分布重新计算
            if hasattr(client, 'weighted_loss'):
                client.weighted_loss = None
            print(f"[Blockchain] 客户端 {client.id} 已接收全局分布")
    
    def train(self):
        """训练主循环（扩展自FedAvg，添加区块链数据分布共享）"""
        mode = 1
        start_epoch = 0
        
        for i in range(start_epoch, self.global_rounds):
            g_rounds = i
            s_t = time.time()
            self.selected_clients = self.select_clients()
            
            # 发送模型参数给客户端
            self.send_models()
            
            # 评估
            if i % self.eval_gap == 0:
                print("\n[Blockchain] 评估全局模型")
                self.evaluate()
                
                # 只记录准确率到 TensorBoard
                if hasattr(self, 'tb_writer') and hasattr(self, 'rs_test_acc') and len(self.rs_test_acc) > 0:
                    self.tb_writer.add_scalar('Accuracy', self.rs_test_acc[-1], i)
            
            num_samples = []
            
            print("\n-------------Round number: {}-------------".format(i))
            
            # 区块链功能：收集和共享数据分布（解决non-IID问题）
            if i == 0 or i % max(1, self.eval_gap) == 0:
                print("\n[Blockchain] 收集客户端数据分布...")
                self.collect_client_distributions()
            
            if i < self.avgstage:
                for index, client in enumerate(self.selected_clients):
                    loss, ns = client.pretrain_avg(i)
                self.receive_models()
                self.aggregate_parameters()
                self.Budget.append(time.time() - s_t)
                print('-'*25, 'time cost', '-'*25, self.Budget[-1])
                continue
            
            if i > self.avgstage:
                # 广播所有客户端的特征表示
                self.sendv()
            
            cdata_all = []
            clabel_all = []
            total_loss = 0
            
            for index, client in enumerate(self.selected_clients):
                ns, cdata, clabel = client.pretrain(i)
                num_samples.append(ns)
                cdata_all.append(cdata)
                clabel_all.append(clabel)
            
            # 继续使用父类的训练逻辑
            self.receive_condessed() # 收集压缩数据
            if self.feature_augment == "True":
                agg_covmatrix = self.receive_covmatrix()
            else:
                agg_covmatrix = {}
            
            cdata_all = torch.cat(cdata_all, dim=0).cpu()
            clabel_all = torch.cat(clabel_all, dim=0)
            cdata_dataset = TensorDataset(cdata_all, clabel_all)
            cdata_dataloader = DataLoader(cdata_dataset, batch_size=self.batch_size, shuffle=True)
            
            # 用于骨架动作识别中的姿态聚类初始化与轨迹矩阵聚合，用于数据增强中的时间变换
            # 边界姿态提取（聚类）
            if self.bkg_pose_list == []:
                self.load_pose_cluster(cdata_all)
                self.receive_traj_mat()
                client_traj_mat_list = np.concatenate(self.uploaded_w_matrix, axis=0)
                server_traj_mat_list = self.traj_mat_list
                self.traj_mat_list = np.concatenate((client_traj_mat_list, server_traj_mat_list), axis=0)
            
            self.global_model.train()
            self.global_model.cuda()
            
            feature_d_all = []
            label_d_all = []
            feature_a_all = []
            label_a_all = []
            
            for gl in range(1):
                for i_batch, (datas, label) in enumerate(cdata_dataloader):
                    data = datas.type(torch.FloatTensor).cuda()
                    n, c, v, t, m = data.size()
                    
                    if self.data_augment == "True":
                        input_aug = self.apply_W_all(data, recover_tag='ABL_linearextrap_beta', resample_tag='cropresize0.7')
                        input_aug = self.merge_raw_and_aug(data, input_aug, augratio=0.75)
                        input_aug = torch.cat([input_aug], 0)
                    else:
                        input_aug = data
                    
                    input1 = shear(crop(input_aug))
                    input1 = random_rotate(input1)
                    input1 = random_spatial_flip(input1)
                    
                    input2 = shear(crop(input_aug))
                    input2 = random_rotate(input2)
                    input2 = random_spatial_flip(input2)
                    
                    # 注意：服务器端训练主要是对比学习任务
                    # logits 的形状是 (batch_size, 1+queue_size)，target 是正样本索引（全0）
                    # 因此主损失使用标准损失函数
                    # 但在特征增强（aug_labels）时，如果有分类标签，可以使用加权损失函数
                    
                    if self.feature_augment == "True":
                        stgcn_logits, logits, target, aug_logits, aug_labels = self.global_model(
                            input1, input2, server=True, covmatrix=agg_covmatrix,
                            scnnumber=self.server_cnum, label=label, fnum=self.feature_number
                        )
                        aug_loss = 0
                        # 注意：aug_logits 是对比学习的输出（形状为 batch_size x 32769），不是分类任务
                        # aug_labels 是全0的正样本索引，不是真正的类别标签
                        # 因此这里使用标准损失函数，不使用加权损失函数
                        for i_aug, (logit, aglabel) in enumerate(zip(aug_logits, aug_labels)):
                            aug_loss += self.crossentropyloss(logit, aglabel)
                        loss = self.crossentropyloss(logits, target) + aug_loss
                    else:
                        stgcn_logits, logits, target = self.global_model(
                            input1, input2, server=True, covmatrix=agg_covmatrix,
                            scnnumber=self.server_cnum, label=label
                        )
                        loss = self.crossentropyloss(logits, target)
                    
                    self.model_optimizer.zero_grad()
                    loss.backward()
                    self.model_optimizer.step()
                    total_loss += loss.item()
            
            print(f'epoch avg loss = {total_loss}')
            
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])
            
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        
        print("\n[Blockchain] 训练完成")
        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        
        # 关闭 TensorBoard writer
        if hasattr(self, 'tb_writer'):
            self.tb_writer.close()
            print(f"[TensorBoard] 日志已保存到: {self.tb_writer.log_dir}")
        
        self.save_results()
        
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print("\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
    
    def _store_distributions_to_blockchain(self, distribution_updates):
        """
        将数据分布更新存储到区块链（可选功能）
        注意：这是一个简化实现，实际应用中可能需要更复杂的区块链结构
        """
        # 这里可以实现将分布信息存储到专门的分布区块链
        # 目前先跳过，因为主要目标是共享分布信息而不是持久化
        pass
    
    def get_distribution_info(self) -> dict:
        """获取当前数据分布信息"""
        return {
            'global_distribution': self.global_distribution,
            'client_distributions': {
                cid: dist.to_dict() 
                for cid, dist in self.client_distributions.items()
            }
        }

