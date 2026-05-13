import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import h5py
import copy
import time
import random
import wandb
from typing import List, Optional, Dict
from utils.data_utils import read_client_data
from utils.dlg import DLG
import math
from collections import OrderedDict
from flcore.blockchain.smart_contract import SmartContract, SmartContractNetwork, ModelUpdateSubmission, NPCDeputySmartContract
from flcore.blockchain.p2p_network import GlobalP2PNetwork
from flcore.blockchain.decentralized_executor import DecentralizedContractExecutor
from flcore.blockchain.blockchain_node import BlockchainNetwork
from flcore.blockchain.stake_manager import StakeManager
from flcore.blockchain.npc_deputy import NPCDeputyManager
from flcore.blockchain.similarity_calculator import SimilarityCalculator


class Server(object):
    """
    智能合约协调器（SmartContractCoordinator）
    
    注意：虽然类名仍为Server（保持向后兼容），但实际功能是智能合约协调器。
    在完全去中心化的架构中，这个类负责：
    1. 协调训练流程（选择客户端、通知轮次等）
    2. 管理智能合约网络
    3. 不参与模型聚合（聚合由智能合约完成）
    
    未来版本可能会完全重命名为SmartContractCoordinator。
    """
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.global_encoder = copy.deepcopy(args.model.encoder_q)

        # self.global_memodel = copy.deepcopy(args.memodel)
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

        # 去中心化 / P2P
        self.p2p_enabled = getattr(args, "p2p_enabled", False)
        self.p2p_mode = getattr(args, "p2p_mode", "local")
        self.p2p_auto_publish = getattr(args, "p2p_auto_publish", True)
        self.p2p_broadcast_timeout = getattr(args, "p2p_broadcast_timeout", 2.0)
        self.coordinator_node_id = -1  # 智能合约协调器节点ID（替代server_node_id）
        self.server_node_id = self.coordinator_node_id  # 保持向后兼容
        self.p2p_network = GlobalP2PNetwork.get_instance() if self.p2p_enabled else None
        if self.p2p_network is not None:
            self.p2p_network.register_node(self.coordinator_node_id)
        self.decentralized_executor = None

        self.blockchain_storage_enabled = getattr(args, "blockchain_storage_enabled", False)
        self.blockchain_lightweight = getattr(args, "blockchain_lightweight", True)
        self.blockchain_disk_storage = getattr(args, "blockchain_disk_storage", False)
        self.blockchain_keep_recent = getattr(args, "blockchain_keep_recent", 5)
        self.blockchain_storage_dir = getattr(args, "blockchain_storage_dir", "blockchain_storage")
        self.blockchain_storage_mode = getattr(args, "blockchain_storage_mode", "summary")
        self.blockchain_circular_mode = getattr(args, "blockchain_circular_mode", False)
        self.blockchain_circular_length = getattr(args, "blockchain_circular_length", None)
        self.blockchain_network = None
        if self.blockchain_storage_enabled:
            storage_dir = os.path.join(self.blockchain_storage_dir, f"server_{times}")
            # 如果启用循环模式且未指定长度，使用num_deputies+1（如果启用NPC Deputy）
            circular_length = self.blockchain_circular_length
            if self.blockchain_circular_mode and circular_length is None:
                if hasattr(self, 'use_npc_deputy') and self.use_npc_deputy:
                    num_deputies = getattr(args, 'num_deputies', 3)
                    circular_length = num_deputies + 1
                    print(f"[Blockchain] 循环模式：自动设置circular_length={circular_length} (num_deputies={num_deputies} + 1)")
                else:
                    circular_length = self.blockchain_keep_recent + 1
                    print(f"[Blockchain] 循环模式：自动设置circular_length={circular_length} (keep_recent_blocks={self.blockchain_keep_recent} + 1)")
            
            self.blockchain_network = BlockchainNetwork(
                num_nodes=self.num_clients,
                lightweight_mode=self.blockchain_lightweight,
                disk_storage=self.blockchain_disk_storage,
                keep_recent_blocks=self.blockchain_keep_recent,
                storage_dir=storage_dir,
                circular_mode=self.blockchain_circular_mode,
                circular_length=circular_length
            )
            print(f"[Blockchain] 已初始化链上存储，目录: {storage_dir}, 循环模式: {self.blockchain_circular_mode}, 循环长度: {circular_length}")

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        self.uploaded_encoders = []
        self.uploaded_sample_counts = []  # 保存客户端样本数量（用于智能合约）

        self.rs_test_acc = []
        self.knn_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch = args.fine_tuning_epoch
        # self.clmode =  args.contrastloss
        self.semi = args.semi_data_ratio 
        self.current_round = 0
        
        # 智能合约网络（替代集中式聚合）
        self.smart_contract_network = SmartContractNetwork()
        self.use_smart_contract = getattr(args, 'use_smart_contract', False)
        # 确保布尔值正确（处理字符串类型的配置）
        if isinstance(self.use_smart_contract, str):
            self.use_smart_contract = self.use_smart_contract.lower() in ['true', '1', 'yes']
        self.use_smart_contract = bool(self.use_smart_contract)
        
        self.aggregation_method = getattr(args, 'aggregation_method', 'fedavg')
        min_clients_ratio = getattr(args, 'smart_contract_min_clients_ratio', 0.5)
        max_clients = getattr(args, 'smart_contract_max_clients', None)
        
        print(f"[SmartContractCoordinator] use_smart_contract={self.use_smart_contract} (type: {type(self.use_smart_contract).__name__})")
        
        if self.use_smart_contract:
            # 计算最小客户端数
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
            print(f"[SmartContractCoordinator] ✓ 已启用智能合约聚合 (方法: {self.aggregation_method}, 最小客户端: {min_clients}, 最大客户端: {max_clients})")
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
            print(f"[SmartContractCoordinator] 使用传统集中式聚合 (use_smart_contract={self.use_smart_contract})")
            self.smart_contract_min_clients = max(1, int(self.num_join_clients * min_clients_ratio))
        
        # NPC Deputy机制（权益管理和deputy选举）
        self.use_npc_deputy = getattr(args, 'use_npc_deputy', False)
        if isinstance(self.use_npc_deputy, str):
            self.use_npc_deputy = self.use_npc_deputy.lower() in ['true', '1', 'yes']
        self.use_npc_deputy = bool(self.use_npc_deputy)
        
        if self.use_npc_deputy:
            num_deputies = getattr(args, 'num_deputies', 3)
            initial_stake = getattr(args, 'initial_stake', 100.0)
            random_init_stake = getattr(args, 'random_init_stake', True)
            stake_range_min = getattr(args, 'stake_range_min', 50.0)
            stake_range_max = getattr(args, 'stake_range_max', 150.0)
            
            # 服务器端初始化权益管理器（只初始化一次，使用random_init）
            self.stake_manager = StakeManager(
                self.num_clients, 
                initial_stake,
                random_init=random_init_stake,
                stake_range=(stake_range_min, stake_range_max)
            )
            
            # 初始化DPOS共识机制（采用FIFO轮换，只保留最近M轮数据）
            keep_recent_rounds = getattr(args, 'npc_deputy_keep_recent_rounds', None)  # None表示使用默认值M（num_deputies）
            self.deputy_manager = NPCDeputyManager(
                num_deputies, 
                self.stake_manager, 
                keep_recent_rounds=keep_recent_rounds
            )
            self.similarity_calculator = SimilarityCalculator()
            
            # 初始化DPOS共识（如果需要真正的区块链）
            if self.use_smart_contract:
                from flcore.blockchain.block import Blockchain
                from flcore.blockchain.dpos_consensus import DPOSConsensus
                self.blockchain = Blockchain()
                self.dpos_consensus = DPOSConsensus(
                    blockchain=self.blockchain,
                    stake_manager=self.stake_manager,
                    num_deputies=num_deputies,
                    block_time=5.0  # 出块时间间隔（秒）
                )
                self.dpos_consensus.initialize_blockchain()
                print(f"[DPOS] ✓ 已初始化DPOS共识机制")
            
            # 使用NPC Deputy智能合约
            if self.use_smart_contract:
                # 获取存储配置（默认2轮，与区块链保持一致）
                keep_recent_rounds = getattr(self.args, "npc_deputy_keep_recent_rounds", 2)
                gradient_storage_dir = getattr(self.args, "npc_deputy_storage_dir", "blockchain_gradient_storage")
                # 为服务器创建独立的存储目录
                server_storage_dir = os.path.join(gradient_storage_dir, "server")
                
                self.smart_contract = NPCDeputySmartContract(
                    aggregation_method=self.aggregation_method,
                    min_clients=min_clients,
                    max_clients=max_clients,
                    max_gradients=num_deputies,  # 只保存M份梯度
                    keep_recent_rounds=keep_recent_rounds,  # 内存中保留最近轮次（默认2轮）
                    storage_dir=server_storage_dir  # 本地存储目录
                )
                self.smart_contract_network.contracts['fedavg_main'] = self.smart_contract
                self.smart_contract_network.default_contract = self.smart_contract
            
            print(f"[NPCDeputy] ✓ 已启用NPC Deputy机制 (deputy数量: {num_deputies}, 初始权益: {initial_stake})")
        else:
            self.stake_manager = None
            self.deputy_manager = None
            self.similarity_calculator = None
        
    def _tb_status_suffix(self):
        attack_cfg = getattr(self.args, "attack_config", {}) or {}
        if attack_cfg.get("enabled"):
            return "attack"
        return "normal"

    def get_tb_log_dir(self, label=None):
        label_part = label or self.algorithm
        status = self._tb_status_suffix()
        tag = f"{label_part}_{status}_{self.times}"
        return os.path.join("runs", tag)
        

    def load_gm_weights(self, model, weights_path, ignore_weights=None):
        ignore_weights =['feat_queue',"feat_queue_ptr"]
        if ignore_weights is None:
            ignore_weights = []
        if isinstance(ignore_weights, str):
            ignore_weights = [ignore_weights]

        print('Load weights from {}.'.format(weights_path))
        weights = torch.load(weights_path)
        weights = OrderedDict([[k.split('module.')[-1],
                                v.cpu()] for k, v in weights.items()])

        # filter weights
        for i in ignore_weights:
            ignore_name = list()
            for w in weights:
                if w.find(i) == 0:
                    ignore_name.append(w)
            for n in ignore_name:
                weights.pop(n)
                print('Filter [{}] remove weights [{}].'.format(i,n))

        # for w in weights:
        #     self.print('Load weights [{}].'.format(w))

        try:
            model.load_state_dict(weights)
        except (KeyError, RuntimeError):
            state = model.state_dict()
            diff = list(set(state.keys()).difference(set(weights.keys())))
            for d in diff:
                print('Can not find weights [{}].'.format(d))
            state.update(weights)
            model.load_state_dict(state)
        return model
    
    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            if self.blockchain_storage_enabled and self.blockchain_network:
                client.blockchain_node = self.blockchain_network.get_node(i)
                client.blockchain_storage_enabled = True
                client.blockchain_storage_mode = self.blockchain_storage_mode
            
            # NPC Deputy机制：将服务器的stake_manager和deputy_manager同步给客户端
            if self.use_npc_deputy and hasattr(self, 'stake_manager') and self.stake_manager:
                client.stake_manager = self.stake_manager
                client.deputy_manager = self.deputy_manager
                client.similarity_calculator = self.similarity_calculator
            
            self.clients.append(client)
            
            #print("client:{} sample:{}".format(i,client))

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        """
        选择参与训练的客户端
        NPC Deputy机制：排除deputy节点，从剩余客户端中选择join_ratio比例的客户端（向上取整）
        """
        import math
        
        # NPC Deputy机制：先获取deputy节点列表
        deputies = []
        if self.use_npc_deputy and self.deputy_manager is not None:
            deputies = self.deputy_manager.get_deputies()
        
        # 排除deputy节点，获取可训练的客户端列表
        trainable_clients = [c for c in self.clients if c.id not in deputies]
        
        if not trainable_clients:
            # 如果没有可训练的客户端（所有都是deputy），返回空列表
            print(f"[Warning] 所有客户端都是deputy节点，无法选择训练客户端")
            return []
        
        # 计算从可训练客户端中选择的数量（向上取整）
        if self.random_join_ratio:
            # 从可训练客户端数量范围内随机选择
            max_selectable = len(trainable_clients)
            min_selectable = max(1, math.ceil(max_selectable * self.join_ratio))
            self.current_num_join_clients = np.random.choice(
                range(min_selectable, max_selectable + 1), 1, replace=False)[0]
        else:
            # 从可训练客户端中选择join_ratio比例（向上取整）
            self.current_num_join_clients = max(1, math.ceil(len(trainable_clients) * self.join_ratio))
        
        # 从可训练客户端中随机选择
        selected_clients = list(np.random.choice(trainable_clients, 
                                                min(self.current_num_join_clients, len(trainable_clients)), 
                                                replace=False))
        
        if deputies:
            print(f"[NPCDeputy] 已排除deputy节点 {deputies}，从剩余 {len(trainable_clients)} 个客户端中选择 {len(selected_clients)} 个参与训练（向上取整）")

        return selected_clients

    def notify_clients_round(self, round_number, clients=None):
        self.current_round = round_number
        target_clients = clients or self.clients
        for client in target_clients:
            if hasattr(client, "set_round"):
                client.set_round(round_number)

    def send_models(self):
        assert (len(self.clients) > 0)
        
        # NPC Deputy机制：保存当前全局模型状态，用于下一轮的恶意梯度检测
        # 注意：不再保存上一轮全局模型状态，因为格式不匹配会导致相似度计算问题
        # 改为在每一轮都使用当前轮次所有梯度的平均/中位数作为参考
        # 这样可以避免格式不匹配的问题，同时仍然能够检测恶意梯度
        if self.use_npc_deputy:
            # 不再保存全局模型状态（避免格式不匹配问题）
            # 每一轮都使用当前轮次所有梯度的平均/中位数作为参考
            pass
        
        for client in self.clients:
            start_time = time.time()
            if hasattr(client, "cache_model_state"):
                client.cache_model_state(self.global_model)
            
            # NPC Deputy机制：不再保存上一轮全局模型状态
            # 改为在每一轮都使用当前轮次所有梯度的平均/中位数作为参考
            # if self.use_npc_deputy and hasattr(client, 'previous_global_model_state'):
            #     client.previous_global_model_state = global_model_state.copy()
            
            client.set_parameters(self.global_model)
            # client.set_classifiers(self.global_classifier)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
    
    # 广播所有客户端的特征表示
    def sendv(self):
        for client in self.clients:
            client.otherv = np.stack(self.uploaded_Vfm, axis=0) 
    
    # 收集压缩数据
    def receive_condessed(self):
        assert (len(self.selected_clients) > 0)
    
        # active_clients = random.sample(
        #     self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))
        active_clients =  self.selected_clients
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_Vfm=[0]*len(active_clients) # 特征分布
        self.uploaded_Rsum=[0]*len(active_clients) # 软标签分布
        self.uploaded_Sxdata=[0]*len(active_clients) # 压缩数据特征
        self.uploaded_Sydata=[0]*len(active_clients) # 压缩数据标签
    
        for i, client in enumerate(active_clients):
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                #tot_samples += (torch.sum(similarity_matrix[i]) * client.train_samples)
                # tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                # self.uploaded_weights.append(similarity_matrix[i]/(active_num_clients * num_layers))
                # self.uploaded_weights.append(client.train_samples)
                if hasattr(client, "apply_model_poisoning"):
                    client.apply_model_poisoning()
                self.uploaded_models.append(client.model)

                
                self.uploaded_Vfm[client.id]=client.Vk
                self.uploaded_Rsum[client.id]=client.Rk

                self.uploaded_Sxdata[client.id]=client.Skdata['x'].cpu().numpy()
                self.uploaded_Sydata[client.id]=client.Skdata['y'].cpu().numpy()
                # print(client.Skdata['x'].shape)
                # print(client.Skdata['y'].shape)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)
    
        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))
        
        self.uploaded_ids = []  # 保存客户端ID（用于智能合约）
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_sample_counts = []  # 保存客户端样本数量（用于智能合约）
        tot_samples = 0
        for i, client in enumerate(active_clients):
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)  # 保存客户端ID
                self.uploaded_sample_counts.append(client.train_samples)  # 保存样本数量
                self.uploaded_weights.append(client.train_samples)
                if hasattr(client, "publish_model_update"):
                    client.publish_model_update(force=True)
                if hasattr(client, "apply_model_poisoning"):
                    client.apply_model_poisoning()
                self.uploaded_models.append(client.model)
        
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

            
    def aggregate_parameters(self):
        """聚合参数：使用智能合约或传统方式"""
        assert (len(self.uploaded_models) > 0)
  
        # 调试信息
        print(f"[Debug] use_smart_contract={self.use_smart_contract}, smart_contract={self.smart_contract is not None}")
        
        if self.use_smart_contract and self.smart_contract:
            # 使用智能合约聚合
            print(f"[SmartContractCoordinator] 使用智能合约聚合 (方法: {self.aggregation_method})")
            self._aggregate_via_smart_contract()
        else:
            # 传统集中式聚合
            if not self.use_smart_contract:
                print(f"[SmartContractCoordinator] 智能合约未启用，使用传统聚合")
            elif not self.smart_contract:
                print(f"[SmartContractCoordinator] 智能合约对象不存在，使用传统聚合")
            self._aggregate_traditional()
        
        if self.p2p_enabled:
            self.sync_decentralized_clients(self.current_round, self.selected_clients)
    
    def _aggregate_traditional(self):
        """传统集中式聚合"""
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
    
    def _aggregate_via_smart_contract(self):
        """通过智能合约聚合"""
        # 优先使用P2P去中心化执行器
        if self.p2p_enabled and self.decentralized_executor is not None:
            aggregated_params = self.decentralized_executor.run_round(self.current_round)
            if aggregated_params is not None:
                node_count = len(self.p2p_network.list_nodes()) if self.p2p_network else 0
                self._apply_aggregated_parameters(aggregated_params)
                print(f"[SmartContractCoordinator] 去中心化聚合完成（P2P节点数量: {node_count})")
                return
            else:
                print("[SmartContractCoordinator] 去中心化聚合未达成共识，本轮跳过")
                return
        print("[SmartContractCoordinator] 去中心化聚合未启用或执行器缺失，无法聚合")

    def _apply_aggregated_parameters(self, aggregated_params, template_model=None):
        """将聚合好的参数写入全局模型（支持summary模式和全量模式）"""
        if template_model is None:
            template_model = self.global_model
        else:
            self.global_model = copy.deepcopy(template_model)

        for name, param in self.global_model.named_parameters():
            if name in aggregated_params:
                param_value = aggregated_params[name]
                # 检查是否是summary模式（字典格式）
                if isinstance(param_value, dict):
                    # Summary模式：无法直接应用到模型，跳过（这种情况不应该发生）
                    print(f"[Warning] 聚合参数 {name} 是summary格式，无法应用到模型，跳过")
                    continue
                elif isinstance(param_value, torch.Tensor):
                    # 全量模式：直接应用
                    param.data = param_value.clone()
                else:
                    # 其他类型：跳过
                    print(f"[Warning] 聚合参数 {name} 格式不支持，跳过")
                    continue

    def sync_decentralized_clients(self, round_number: int, clients=None):
        """通知客户端本地执行去中心化聚合，保持状态一致"""
        if not self.p2p_enabled:
            return
        target_clients = clients or self.clients
        for client in target_clients:
            if hasattr(client, "run_decentralized_aggregation"):
                client.run_decentralized_aggregation(round_number)
    
    # ========== NPC Deputy机制相关方法 ==========
    def elect_deputies(self, round_number: int) -> List[int]:
        """选举NPC deputy（基于权益）"""
        if not self.use_npc_deputy or self.deputy_manager is None:
            return []
        return self.deputy_manager.elect_deputies(round_number)
    
    def collect_gradients_from_deputies(self, round_number: int, selected_clients):
        """
        收集deputy节点的梯度（只保存M份）
        :param round_number: 轮次
        :param selected_clients: 选中的客户端列表
        """
        if not self.use_npc_deputy or self.deputy_manager is None:
            return
        
        deputies = self.deputy_manager.get_deputies()
        if not deputies:
            return
        
        # 收集deputy节点的梯度
        deputy_gradients = {}
        for client in selected_clients:
            if client.id in deputies:
                # 获取客户端梯度
                if hasattr(client, '_capture_model_and_gradients'):
                    params, grads = client._capture_model_and_gradients()
                    if grads is not None:
                        deputy_gradients[client.id] = grads
        
        # 提交到智能合约（只保存M份）
        if isinstance(self.smart_contract, NPCDeputySmartContract):
            for client_id, grads in deputy_gradients.items():
                # 创建ModelUpdateSubmission
                client = next((c for c in selected_clients if c.id == client_id), None)
                if client is None:
                    continue
                
                # 获取模型参数
                model_params = {}
                if hasattr(client, 'model'):
                    for name, param in client.model.named_parameters():
                        model_params[name] = param.detach().cpu().clone()
                
                submission = ModelUpdateSubmission(
                    client_id=client_id,
                    round_number=round_number,
                    model_params=model_params,
                    gradient_params=grads,
                    sample_count=client.train_samples if hasattr(client, 'train_samples') else 0,
                    train_loss=getattr(client, 'last_train_loss', 0.0),
                    timestamp=time.time()
                )
                
                # 提交到智能合约（只保存M份）
                self.smart_contract.submit_gradient(submission, deputies, round_number)
    
    def calculate_similarities_and_scores(self, round_number: int):
        """
        计算余弦相似度和得分
        :param round_number: 轮次
        """
        if not self.use_npc_deputy or self.deputy_manager is None:
            return
        
        # 获取存储的梯度
        if isinstance(self.smart_contract, NPCDeputySmartContract):
            stored_gradients = self.smart_contract.get_stored_gradients(round_number)
            if not stored_gradients:
                return
            
            # 提取梯度字典
            client_gradients = {}
            for submission in stored_gradients:
                if submission.gradient_params:
                    client_gradients[submission.client_id] = submission.gradient_params
            
            if not client_gradients:
                return
            
            # 计算全局梯度（平均）
            global_gradient = None
            if len(client_gradients) > 1:
                # 计算平均梯度作为全局梯度
                first_grad = list(client_gradients.values())[0]
                global_gradient = {}
                for name in first_grad.keys():
                    if isinstance(first_grad[name], dict):
                        # Summary模式
                        global_gradient[name] = {
                            "mean": np.mean([client_gradients[cid][name].get("mean", 0.0) 
                                            for cid in client_gradients if name in client_gradients[cid]]),
                            "std": np.mean([client_gradients[cid][name].get("std", 0.0) 
                                           for cid in client_gradients if name in client_gradients[cid]]),
                            "l2": np.mean([client_gradients[cid][name].get("l2", 0.0) 
                                          for cid in client_gradients if name in client_gradients[cid]])
                        }
                    else:
                        # 全量模式
                        stacked = torch.stack([
                            client_gradients[cid][name] 
                            for cid in client_gradients 
                            if name in client_gradients[cid]
                        ])
                        global_gradient[name] = torch.mean(stacked, dim=0)
            
            # 计算每个客户端的余弦相似度
            similarities = self.similarity_calculator.calculate_similarities_to_global(
                client_gradients, global_gradient
            )
            
            # 计算得分并更新权益
            similarity_weight = getattr(self.args, 'similarity_weight', 0.7)
            stake_weight = getattr(self.args, 'stake_weight', 0.3)
            
            for client_id, similarity in similarities.items():
                score = self.stake_manager.calculate_score(
                    client_id, similarity, round_number,
                    similarity_weight, stake_weight
                )
                print(f"[NPCDeputy] Client {client_id}: 相似度={similarity:.4f}, 得分={score:.4f}")
            
            # 根据得分更新权益
            reward_factor = getattr(self.args, 'reward_factor', 0.1)
            penalty_factor = getattr(self.args, 'penalty_factor', 0.05)
            self.stake_manager.update_stakes_from_scores(round_number, reward_factor, penalty_factor)
            
            # 打印权益更新
            print(f"[NPCDeputy] Round {round_number} 权益更新:")
            for client_id in similarities.keys():
                stake = self.stake_manager.get_stake(client_id)
                print(f"  Client {client_id}: 权益={stake:.2f}")
    
    def aggregate_top_scoring_clients(self, round_number: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        聚合得分前几的客户端梯度
        :param round_number: 轮次
        :return: 聚合后的模型参数
        """
        if not self.use_npc_deputy or self.deputy_manager is None:
            return None
        
        # 获取得分最高的客户端（数量等于deputy数量）
        top_clients = self.stake_manager.get_top_score_clients(
            round_number, self.deputy_manager.num_deputies
        )
        
        if not top_clients:
            return None
        
        # 获取这些客户端的梯度
        if isinstance(self.smart_contract, NPCDeputySmartContract):
            stored_gradients = self.smart_contract.get_stored_gradients(round_number)
            top_submissions = [
                s for s in stored_gradients 
                if s.client_id in top_clients
            ]
            
            if not top_submissions:
                return None
            
            # 使用智能合约聚合这些梯度
            aggregated = self.smart_contract.aggregate(round_number)
            
            # 更新下一轮deputy（得分前几的客户端成为下一轮deputy）
            self.deputy_manager.update_deputies_from_scores(round_number)
            
            return aggregated
        
        return None


    # def aggregate_parameters(self):

    def send_3ddata(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.get_3ddata(self.g3d_data)
     
    def load_train_data(self, batch_size=None ,get_data=False):
        if batch_size == None:
            batch_size = self.batch_size
        sxdata = self.uploaded_Sxdata
        sydata = self.uploaded_Sydata
        train_xdata = np.concatenate(sxdata,axis = 0)
        train_ydata = np.concatenate(sydata,axis = 0)
        X_train = torch.Tensor(train_xdata ).type(torch.float32)
        y_train = torch.Tensor(train_ydata).type(torch.int64)
        train_data = [(x,y) for x, y in zip(X_train , y_train)]
        # print(X_train.shape)
        # train_data = read_client_data(self.dataset,0)
        # print(len(train_data))
        if get_data:
            return DataLoader(train_data, batch_size, drop_last=False, shuffle=False),X_train,y_train
        else:
            return DataLoader(train_data, batch_size, drop_last=False, shuffle=False)

    #def add_parameters(self, w, client_model):
    #    for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
    #        server_param.data += client_param.data.clone() * w
    
    def add_parameters(self, w, client_model):
     
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            params = client_param.data.clone()
            psize = params.size()
            if len(psize) < 2:
                params = params.reshape(-1, 1)
            else:
                params = params.reshape(client_param.data.shape[0], -1)

            params = w * params
            params = params.reshape(psize)
            server_param.data += params

    def add_encoder_parameters(self, w, client_model):
     
        for server_param, client_param in zip(self.global_encoder.parameters(), client_model.parameters()):
            params = client_param.data.clone()
            psize = params.size()
            if len(psize) < 2:
                params = params.reshape(-1, 1)
            else:
                params = params.reshape(client_param.data.shape[0], -1)

            params = w * params
            params = params.reshape(psize)
            server_param.data += params
            
    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)
    
    def save_model(self, model, name):
        path = "/opt/data/private/fedlearning/NoCGA/system/log/server"
        model_path = path+"epoch"+str(name)+"_model.pt"
        # if not os.path.exists(model_path):
        #     os.makedirs(model_path)
        state_dict = model.state_dict()
        weights = OrderedDict([[''.join(k.split('module.')),
                                v.cpu()] for k, v in state_dict.items()])
        torch.save(weights, model_path)
        print('The model has been saved as {}.'.format(model_path))


    # def load_model(self):
    #     model_path = os.path.join("models", self.dataset)
    #     model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
    #     assert (os.path.exists(model_path))
    #     self.global_model = torch.load(model_path)

    def load_model(self, checkpoint_path):
        # 加载模型的state_dict
        state_dict = torch.load(checkpoint_path)

        # 如果你之前移除了'module.'前缀，现在可以直接加载
        # 否则，如果模型是使用DataParallel/DistributedDataParallel包装的，需要重新添加'module.'前缀
        # 例如：state_dict = OrderedDict([('module.' + k, v) for k, v in state_dict.items()])
        
        self.global_model.load_state_dict(state_dict)
        print('The model has been loaded from {}.'.format(checkpoint_path))
     
    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
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

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def knn_test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        tot_correct = []
        # self.logger.info("Evaluate Linear_test start !") 
        for c in self.clients:
            acc = c.clustering_knn_acc()
            tot_correct.append(acc*1.0)


        ids = [c.id for c in self.clients]
        # self.logger.info("Evaluate Linear_test end !") 
        return ids, tot_correct

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:

            cl, ns = c.lineartrain()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]
        return ids, num_samples, losses
    


    def semitrain_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:

            cl, ns = c.semitrain(self.semi)
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]
    
        return ids, num_samples, losses    
    
    def pfl_test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        # self.logger.info("Evaluate Linear_test start !") 
        for c in self.clients:
            ct, ns, auc = c.lineartest()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]
        # self.logger.info("Evaluate Linear_test end !") 
        return ids, num_samples, tot_correct, tot_auc
    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        
        stats = self.test_metrics()
        if self.semi == 1.0:
            stats_train = self.train_metrics()
        else:
            stats_train = self.semitrain_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        

        print("Linear Test Accurancy: {}".format(accs))
        print("Linear STATs: {}".format(stats))
        print("Linear Train STATs: {}".format(stats_train))
        print("Linear Averaged Test Accurancy: {:.4f}".format(test_acc))
        # print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        # print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        # print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def knn_evaluate(self, acc=None, loss=None):
        
        stats = self.knn_test_metrics()
    
        accs = stats[1]
        test_acc =sum(accs)/len(accs)
        if acc == None:
            self.knn_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        

        print("KNN Test Accurancy: {}".format(accs))
        print("KNN STATs: {}".format(stats))
        print("KNN Averaged Test Accurancy: {:.4f}".format(test_acc))

    def pfl_evaluate(self, acc=None, loss=None):
        
        stats = self.pfl_test_metrics()
        

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        

        print("Test Accurancy: {}".format(accs))
        print("STATs: {}".format(stats))
        
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
    
    def linear_train(self,loss=None):
        stats_train = self.train_metrics()
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)
        print("Linear Averaged Train Loss: {:.4f}".format(train_loss))


    def receive_classifiers(self):
        assert (len(self.selected_clients) > 0)
    
        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))
        
        self.uploaded_classifiers = []

        for i, client in enumerate(active_clients):
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                self.uploaded_classifiers.append(client.classifier_lt.cpu())


    def aggregate_classifiers(self):
        # print(len(self.uploaded_classifiers))
        assert (len(self.uploaded_classifiers) > 0)
  
        self.global_classifier = copy.deepcopy(self.uploaded_classifiers[0])
        for param in self.global_classifier.parameters():
            param.data.zero_()
       
        for w,client_model in zip(self.uploaded_weights, self.uploaded_classifiers):
            self.add_classifiers(w, client_model)

    def add_classifiers(self,w,client_model):

        for server_param, client_param in zip(self.global_classifier.parameters(), client_model.parameters()):
            params = client_param.data.clone()
            psize = params.size()
            if len(psize) < 2:
                params = params.reshape(-1, 1)
            else:
                params = params.reshape(client_param.data.shape[0], -1)

            params = w * params
            params = params.reshape(psize)
            server_param.data += params






    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models , self.uploaded_encoders):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
            # self.logger.info('PSNR value is : %.2f', (psnr_val / cnt)) 
        
        else:
            print('PSNR error')
            # self.logger.info('PSNR error') 

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
    
    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            try:
                ct, ns, auc = c.lineartest()
                tot_correct.append(ct*1.0)
                tot_auc.append(auc*ns)
                num_samples.append(ns)
            except (ValueError, Exception) as e:
                # 如果评估失败，尝试使用简单的测试方法
                print(f"警告: 客户端 {c.id} lineartest 评估失败: {e}")
                try:
                    # 尝试使用test方法
                    acc, ns = c.test()
                    tot_correct.append(acc*ns)
                    tot_auc.append(0.0)  # AUC设为0
                    num_samples.append(ns)
                except Exception as e2:
                    print(f"警告: 客户端 {c.id} test 也失败: {e2}")
                    # 使用默认值
                    tot_correct.append(0.0)
                    tot_auc.append(0.0)
                    num_samples.append(c.test_samples if hasattr(c, 'test_samples') else 0)
        #         for c in self.clients:
        # ct, ns, auc = self.clients[0].lineartestall()
        # tot_correct.append(ct*1.0)
        # tot_auc.append(auc*ns)
        # num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

