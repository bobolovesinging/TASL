"""
智能合约协调器
替代传统"服务器"概念，负责协调联邦学习的训练流程
所有协调逻辑都通过智能合约执行，实现完全去中心化
"""
import copy
import torch
import numpy as np
import time
from typing import List, Dict, Optional, Any
from collections import OrderedDict

from .smart_contract import SmartContract, SmartContractNetwork, NPCDeputySmartContract
from .p2p_network import GlobalP2PNetwork
from .decentralized_executor import DecentralizedContractExecutor


class SmartContractCoordinator:
    """
    智能合约协调器：替代传统服务器
    负责协调联邦学习的训练流程，但不参与模型聚合（聚合由智能合约完成）
    """
    
    def __init__(self, args, times):
        """
        初始化智能合约协调器
        :param args: 配置参数
        :param times: 运行次数
        """
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        
        # 全局模型（初始状态，后续由智能合约聚合更新）
        self.global_model = copy.deepcopy(args.model)
        self.global_encoder = copy.deepcopy(args.model.encoder_q)
        self.global_classifier = copy.deepcopy(args.classifier)
        
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break
        
        # P2P网络
        self.p2p_enabled = getattr(args, "p2p_enabled", False)
        self.p2p_mode = getattr(args, "p2p_mode", "local")
        self.p2p_auto_publish = getattr(args, "p2p_auto_publish", True)
        self.p2p_broadcast_timeout = getattr(args, "p2p_broadcast_timeout", 2.0)
        self.coordinator_node_id = -1  # 协调器节点ID（特殊ID）
        self.p2p_network = GlobalP2PNetwork.get_instance() if self.p2p_enabled else None
        if self.p2p_network is not None:
            self.p2p_network.register_node(self.coordinator_node_id)
        
        # 智能合约网络
        self.smart_contract_network = SmartContractNetwork()
        self.use_smart_contract = getattr(args, 'use_smart_contract', False)
        if isinstance(self.use_smart_contract, str):
            self.use_smart_contract = self.use_smart_contract.lower() in ['true', '1', 'yes']
        self.use_smart_contract = bool(self.use_smart_contract)
        
        self.aggregation_method = getattr(args, 'aggregation_method', 'fedavg')
        min_clients_ratio = getattr(args, 'smart_contract_min_clients_ratio', 0.5)
        max_clients = getattr(args, 'smart_contract_max_clients', None)
        
        if self.use_smart_contract:
            min_clients = max(1, int(self.num_join_clients * min_clients_ratio))
            if max_clients is None:
                max_clients = self.num_clients
            self.smart_contract_min_clients = min_clients
            
            # 部署默认智能合约
            self.smart_contract = self.smart_contract_network.deploy_contract(
                contract_id='fedavg_main',
                aggregation_method=self.aggregation_method,
                min_clients=min_clients,
                max_clients=max_clients,
                set_as_default=True
            )
            print(f"[SmartContractCoordinator] ✓ 已部署智能合约 (方法: {self.aggregation_method}, 最小客户端: {min_clients})")
            
            if self.p2p_enabled:
                self.decentralized_executor = DecentralizedContractExecutor(
                    node_id=self.coordinator_node_id,
                    smart_contract=self.smart_contract,
                    p2p_network=self.p2p_network,
                    min_required=min_clients,
                    total_nodes=self.num_clients,
                )
        else:
            self.smart_contract = None
            self.smart_contract_min_clients = max(1, int(self.num_join_clients * min_clients_ratio))
        
        # 客户端列表
        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []
        
        # 训练状态
        self.current_round = 0
        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate
        
        # 评估结果
        self.rs_test_acc = []
        self.knn_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.Budget = []
        
        # 其他配置
        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client
        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch = args.fine_tuning_epoch
        self.semi = args.semi_data_ratio
        
        # 区块链存储（可选）
        self.blockchain_storage_enabled = getattr(args, "blockchain_storage_enabled", False)
        self.blockchain_lightweight = getattr(args, "blockchain_lightweight", True)
        self.blockchain_disk_storage = getattr(args, "blockchain_disk_storage", False)
        self.blockchain_keep_recent = getattr(args, "blockchain_keep_recent", 5)
        self.blockchain_storage_dir = getattr(args, "blockchain_storage_dir", "blockchain_storage")
        self.blockchain_storage_mode = getattr(args, "blockchain_storage_mode", "summary")
        
        print(f"[SmartContractCoordinator] 智能合约协调器已初始化 (节点ID: {self.coordinator_node_id})")
    
    def set_clients(self, clientObj):
        """设置客户端列表（由智能合约协调器管理）"""
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            from utils.data_utils import read_client_data
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)
    
    def select_slow_clients(self, slow_rate):
        """随机选择慢速客户端"""
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True
        return slow_clients
    
    def set_slow_clients(self):
        """设置慢速客户端"""
        self.train_slow_clients = self.select_slow_clients(self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(self.send_slow_rate)
    
    def select_clients(self):
        """选择参与训练的客户端（智能合约协调）"""
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
        return selected_clients
    
    def notify_clients_round(self, round_number, clients=None):
        """通知客户端当前轮次（智能合约协调）"""
        self.current_round = round_number
        target_clients = clients or self.clients
        for client in target_clients:
            if hasattr(client, "set_round"):
                client.set_round(round_number)
    
    def send_models(self):
        """发送全局模型给客户端（智能合约协调）"""
        assert (len(self.clients) > 0)
        
        for client in self.clients:
            start_time = time.time()
            if hasattr(client, "cache_model_state"):
                client.cache_model_state(self.global_model)
            if hasattr(client, "set_parameters"):
                client.set_parameters(self.global_model, self.global_encoder)
            if hasattr(client, "send_time_cost"):
                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
    
    def get_global_model(self):
        """获取全局模型（从智能合约聚合结果）"""
        return self.global_model
    
    def update_global_model_from_contract(self, aggregated_params: Dict[str, torch.Tensor]):
        """从智能合约聚合结果更新全局模型"""
        if aggregated_params is None:
            return
        
        for name, param in self.global_model.named_parameters():
            if name in aggregated_params:
                param.data = aggregated_params[name].clone()
        
        print(f"[SmartContractCoordinator] 全局模型已从智能合约更新")
    
    def evaluate(self):
        """评估全局模型（智能合约协调）"""
        # 评估逻辑由具体实现类提供
        pass
    
    def save_results(self):
        """保存结果"""
        import h5py
        import os
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

