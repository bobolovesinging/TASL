import copy
import torch
import torch.nn as nn
import numpy as np
import os
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize, normalize
from sklearn import metrics
from typing import Optional, Dict, Any, List
from utils.data_utils import read_client_data,read_client_traindata
# from sklearn.neighbors import KNeighborsClassifier
from torch.utils.tensorboard import SummaryWriter

from .attack_utils import AttackManager
from flcore.blockchain.smart_contract import SmartContract, NPCDeputySmartContract
from flcore.blockchain.p2p_network import GlobalP2PNetwork
from flcore.blockchain.decentralized_executor import DecentralizedContractExecutor
from flcore.blockchain.stake_manager import StakeManager
from flcore.blockchain.npc_deputy import NPCDeputyManager
from flcore.blockchain.similarity_calculator import SimilarityCalculator


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples,**kwargs):
        self.args = args  # 保存args引用，供NPC deputy使用
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.current_round = 0
        self.attack_manager = AttackManager(getattr(args, "attack_config", {}), num_classes=self.num_classes)

        # 去中心化 / P2P 配置
        self.p2p_enabled = getattr(args, "p2p_enabled", False)
        self.p2p_auto_publish = getattr(args, "p2p_auto_publish", True)
        self.p2p_network = None
        self.decentralized_executor = None
        self.local_smart_contract = None
        self._last_published_round = None
        self.last_decentralized_global = None

        self.blockchain_storage_enabled = getattr(args, "blockchain_storage_enabled", False)
        self.blockchain_node = None
        self.blockchain_storage_mode = getattr(args, "blockchain_storage_mode", "summary")
        self.cached_global_state = None
        self.captured_gradients = None  # 存储训练过程中捕获的梯度
        self.previous_global_model_state = None  # 存储上一轮的全局模型状态（用于恶意梯度检测）

        # NPC Deputy机制（完全去中心化，客户端自主管理）
        self.use_npc_deputy = getattr(args, 'use_npc_deputy', False)
        if isinstance(self.use_npc_deputy, str):
            self.use_npc_deputy = self.use_npc_deputy.lower() in ['true', '1', 'yes']
        self.use_npc_deputy = bool(self.use_npc_deputy)
        
        if self.use_npc_deputy:
            num_deputies = getattr(args, 'num_deputies', 3)
            
            # 客户端不在这里初始化权益管理器，而是等待服务器同步
            # 如果服务器已经设置了stake_manager，就使用服务器的；否则创建空的（等待同步）
            if not hasattr(self, 'stake_manager') or self.stake_manager is None:
                # 临时创建，等待服务器同步
                initial_stake = getattr(args, 'initial_stake', 100.0)
                self.stake_manager = StakeManager(
                    args.num_clients, 
                    initial_stake,
                    random_init=False  # 客户端不随机初始化，等待服务器同步
                )
                # 采用FIFO轮换，只保留最近M轮数据（M = num_deputies）
                keep_recent_rounds = getattr(args, "npc_deputy_keep_recent_rounds", None)  # None表示使用默认值M
                self.deputy_manager = NPCDeputyManager(
                    num_deputies, 
                    self.stake_manager,
                    keep_recent_rounds=keep_recent_rounds
                )
                self.similarity_calculator = SimilarityCalculator()
            
            # 使用NPC Deputy智能合约
            if self.p2p_enabled:
                min_ratio = getattr(args, "smart_contract_min_clients_ratio", 0.5)
                min_clients = max(1, int(args.num_clients * min_ratio))
                max_clients = getattr(args, "smart_contract_max_clients", None)
                aggregation_method = getattr(args, "aggregation_method", "fedavg")
                
                # 获取存储配置（默认2轮，与区块链保持一致）
                keep_recent_rounds = getattr(args, "npc_deputy_keep_recent_rounds", 2)
                gradient_storage_dir = getattr(args, "npc_deputy_storage_dir", "blockchain_gradient_storage")
                # 为每个客户端创建独立的存储目录
                client_storage_dir = os.path.join(gradient_storage_dir, f"client_{self.id}")
                
                self.local_smart_contract = NPCDeputySmartContract(
                    aggregation_method=aggregation_method,
                    min_clients=min_clients,
                    max_clients=max_clients,
                    max_gradients=num_deputies,  # 只保存M份梯度
                    keep_recent_rounds=keep_recent_rounds,  # 内存中保留最近轮次（默认2轮）
                    storage_dir=client_storage_dir  # 本地存储目录
                )
            
            print(f"[NPCDeputy][Client {self.id}] ✓ 已启用NPC Deputy机制 (deputy数量: {num_deputies})")
        else:
            self.stake_manager = None
            self.deputy_manager = None
            self.similarity_calculator = None

        if self.p2p_enabled:
            self._init_decentralized_components(args)

        # # check BatchNorm
        # self.has_BatchNorm = False
        # for layer in self.model.children():
        #     if isinstance(layer, nn.BatchNorm2d):
        #         self.has_BatchNorm = True
        #         break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma


        # tensorboard（可选，如果不需要客户端日志可以注释掉）
        # self.work_dir = os.path.join("runs", "client_" + str(id) + "_trainlog")
        # os.makedirs(self.work_dir, exist_ok=True)
        # self.train_writer = SummaryWriter(self.work_dir)
    
    
    def load_semitrain_data(self, semi=None):
        
        # if batch_size == None:
        batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True,semi=semi)
        train_data = self._apply_data_poisoning(train_data)
        return DataLoader(train_data, batch_size, drop_last=False, shuffle=False)
   
    def load_train_data(self, batch_size=None):
        
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        train_data = self._apply_data_poisoning(train_data)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)
    
    # def load_train_data(self, batch_size=None):
    #     if batch_size == None:
    #         batch_size = self.batch_size
    #     train_data = read_client_data(self.dataset, self.id, is_train=True)
    #     return 

    def load_test_data(self, id= None, batch_size=128):
        if id == None:
            id = self.id
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)
        
    def set_parameters(self, model, encoder):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
        for new_param, old_param in zip(encoder.parameters(), self.encoder.parameters()):
            old_param.data = new_param.data.clone()
    

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))
    
    def load_modelpt(self):
        loadname="270"
        print("class {}:load 270 ground data".format(self.id))
        if self.clmode == "contrast":
            loadpath= "/opt/data/private/fedlearning/base/system/log/md_contrast/"+str(self.id)+"/"
        else:
            loadpath= "/opt/data/private/fedlearning/base/system/log/mdpath/"+str(self.id)+"/"
        self.model = torch.load(os.path.join(loadpath, "client_" + str(self.id) + "_" + loadname + ".pt"))
    # @staticmethod 
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))

    def get_mixdataloader(self):

        batch_size = self.batch_size

        train_x,train_y= read_client_traindata(self.dataset, self.id, is_train=True)
        
        # print("train_data:{},{}".format(type(train_x),train_x.shape))
        train_mix =torch.cat((train_x, self.train_g3d), dim=0)
        zero_tensor = torch.zeros(train_mix.size(0))
        train_data = [(x, y) for x, y in zip(train_mix, zero_tensor)]
        train_data = self._apply_data_poisoning(train_data)
        # print("train_data:{},{}".format(type(train_mix),train_mix.shape))
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)
    
    def get_pseudo_loader(self,pseudo_labels):

        batch_size =self.batch_size

        train_x,train_y= read_client_traindata(self.dataset, self.id, is_train=True)
        #保证新标签与旧标签一一对应
        label_mapping = dict(zip(self.clabel, pseudo_labels))
        # print(pseudo_labels)
        new_y = [label_mapping[label] for label in self.y_plabel]
        train_y = torch.Tensor(new_y).type(torch.int64)
        # print("train_data:{},{}".format(type(train_x),train_x.shape))

        train_data = [(x, y) for x, y in zip(train_x, train_y)]
        train_data = self._apply_data_poisoning(train_data)
        # print("train_data:{}".format(len(train_data)))
        return DataLoader(train_data, batch_size, drop_last=False, shuffle=False)

    def set_round(self, round_number: int):
        self.current_round = round_number
        if self._last_published_round != round_number:
            self._last_published_round = None

    def _apply_data_poisoning(self, dataset):
        if not dataset or not hasattr(self, "attack_manager"):
            return dataset
        return self.attack_manager.poison_dataset(
            dataset, getattr(self, "current_round", 0), self.id
        )

    def apply_model_poisoning(self):
        if hasattr(self, "attack_manager"):
            self.attack_manager.poison_model(
                getattr(self, "model", None), getattr(self, "current_round", 0), self.id
            )

    # ======== 去中心化辅助方法 ========
    def _init_decentralized_components(self, args):
        try:
            self.p2p_network = GlobalP2PNetwork.get_instance()
            self.p2p_network.register_node(self.id)
            
            # 计算min_clients（无论是否已有合约都需要）
            min_ratio = getattr(args, "smart_contract_min_clients_ratio", 0.5)
            min_clients = max(1, int(args.num_clients * min_ratio))
            
            # 如果已经初始化了NPC Deputy智能合约，就使用它；否则创建普通智能合约
            if not hasattr(self, 'local_smart_contract') or self.local_smart_contract is None:
                max_clients = getattr(args, "smart_contract_max_clients", None)
                aggregation_method = getattr(args, "aggregation_method", "fedavg")
                self.local_smart_contract = SmartContract(
                    aggregation_method=aggregation_method,
                    min_clients=min_clients,
                    max_clients=max_clients,
                )
            
            self.decentralized_executor = DecentralizedContractExecutor(
                node_id=self.id,
                smart_contract=self.local_smart_contract,
                p2p_network=self.p2p_network,
                min_required=min_clients,
                total_nodes=args.num_clients,
            )
        except Exception as exc:
            print(f"[P2P] 客户端 {self.id} 初始化去中心化组件失败: {exc}")
            self.p2p_enabled = False

    def publish_model_update(self, force: bool = False):
        if (
            not self.p2p_enabled
            or (not self.p2p_auto_publish and not force)
            or self.decentralized_executor is None
            or not hasattr(self, "model")
        ):
            return
        if self._last_published_round == self.current_round and not force:
            return
        
        # NPC Deputy机制：需要包含梯度信息
        if self.use_npc_deputy:
            # 获取梯度和模型参数
            params, grads = self._capture_model_and_gradients()
            if grads is not None:
                # 转换为可序列化的格式
                model_state = {}
                gradient_state = {}
                summary_mode = getattr(self, "blockchain_storage_mode", "summary") == "summary"
                
                for name, param in self.model.named_parameters():
                    if summary_mode:
                        flat = param.detach().cpu().view(-1)
                        model_state[name] = {
                            "mean": float(flat.mean()),
                            "std": float(flat.std(unbiased=False)),
                            "l2": float(torch.norm(flat, p=2))
                        }
                    else:
                        model_state[name] = param.detach().cpu().numpy().tolist()
                
                for name, grad in grads.items():
                    if summary_mode:
                        if isinstance(grad, dict):
                            gradient_state[name] = grad
                        else:
                            flat_grad = grad.view(-1) if isinstance(grad, torch.Tensor) else grad
                            gradient_state[name] = {
                                "mean": float(flat_grad.mean()) if isinstance(flat_grad, torch.Tensor) else float(np.mean(flat_grad)),
                                "std": float(flat_grad.std(unbiased=False)) if isinstance(flat_grad, torch.Tensor) else float(np.std(flat_grad)),
                                "l2": float(torch.norm(flat_grad, p=2)) if isinstance(flat_grad, torch.Tensor) else float(np.linalg.norm(flat_grad))
                            }
                    else:
                        gradient_state[name] = grad.detach().cpu().numpy().tolist() if isinstance(grad, torch.Tensor) else grad
                
                # 广播包含梯度的更新
                payload = {
                    "client_id": self.id,
                    "round": self.current_round,
                    "sample_count": self.train_samples,
                    "model_state": model_state,
                    "gradient_state": gradient_state,
                    "train_loss": 0.0,  # 可以从finalize_local_training传入
                    "timestamp": time.time(),
                }
                
                self.p2p_network.broadcast(
                    sender_id=self.id,
                    payload=payload,
                    message_type="model_update",
                    round_number=self.current_round,
                    exclude_self=True,
                )
                self._last_published_round = self.current_round
                return
        
        # 非NPC Deputy机制：使用原有逻辑
        self.decentralized_executor.submit_local_update(
            round_number=self.current_round,
            model=self.model,
            sample_count=self.train_samples,
        )
        self._last_published_round = self.current_round

    def run_decentralized_aggregation(self, round_number: Optional[int] = None):
        if not self.p2p_enabled or self.decentralized_executor is None:
            return None
        target_round = round_number if round_number is not None else self.current_round
        result = self.decentralized_executor.run_round(target_round)
        if result is not None:
            self.last_decentralized_global = result
        return result

    # ---------- 区块链存储相关 ----------
    def cache_model_state(self, source_model):
        """缓存服务器下发的全局参数，供之后计算梯度"""
        self.cached_global_state = {}
        for name, param in source_model.named_parameters():
            self.cached_global_state[name] = param.detach().cpu().clone()

    def capture_gradients_during_training(self):
        """
        在训练过程中捕获梯度（在loss.backward()之后调用）
        这是更准确的梯度计算方式，而不是使用参数差值
        """
        if not hasattr(self, "model"):
            return None
        
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.detach().cpu().clone()
            else:
                # 如果没有梯度（例如某些层被冻结），使用零梯度
                gradients[name] = torch.zeros_like(param.detach().cpu())
        
        self.captured_gradients = gradients
        return gradients
    
    def _capture_model_and_gradients(self):
        """
        获取模型参数和梯度
        优先使用训练过程中捕获的梯度，如果没有则使用参数差值作为近似
        """
        if not hasattr(self, "model"):
            return None, None
        
        current_state = {}
        gradient_state = {}
        summary_mode = getattr(self, "blockchain_storage_mode", "summary") == "summary"
        
        # NPC Deputy机制：model_params必须使用全量模式（Tensor），因为需要聚合应用到模型
        # 只有gradient_params可以使用summary模式（用于相似度计算）
        use_full_model_params = self.use_npc_deputy if hasattr(self, 'use_npc_deputy') else False
        
        # 获取当前模型参数
        for name, param in self.model.named_parameters():
            curr = param.detach().cpu().clone()
            
            # model_params：NPC Deputy机制使用全量模式，其他情况根据summary_mode决定
            if use_full_model_params:
                current_state[name] = curr
            elif summary_mode:
                flat = curr.view(-1)
                current_state[name] = {
                    "mean": float(flat.mean()),
                    "std": float(flat.std(unbiased=False)),
                    "l2": float(torch.norm(flat, p=2))
                }
            else:
                current_state[name] = curr
        
        # 优先使用训练过程中捕获的梯度
        if self.captured_gradients is not None:
            # 使用训练过程中捕获的真实梯度
            for name, grad in self.captured_gradients.items():
                if summary_mode:
                    flat_grad = grad.view(-1)
                    gradient_state[name] = {
                        "mean": float(flat_grad.mean()),
                        "std": float(flat_grad.std(unbiased=False)),
                        "l2": float(torch.norm(flat_grad, p=2))
                    }
                else:
                    gradient_state[name] = grad
        elif self.cached_global_state is not None:
            # 回退到参数差值方法（兼容旧代码）
            for name, param in self.model.named_parameters():
                prev = self.cached_global_state.get(name)
                if prev is not None:
                    curr = param.detach().cpu().clone()
                    diff = curr - prev
                    if summary_mode:
                        flat_diff = diff.view(-1)
                        gradient_state[name] = {
                            "mean": float(flat_diff.mean()),
                            "std": float(flat_diff.std(unbiased=False)),
                            "l2": float(torch.norm(flat_diff, p=2))
                        }
                    else:
                        gradient_state[name] = diff
        else:
            # 如果没有缓存的全局状态，返回None
            return current_state, None
        
        return current_state, gradient_state

    def _store_update_on_blockchain(self, train_loss: float = 0.0, sample_count: Optional[int] = None):
        if not self.blockchain_storage_enabled or self.blockchain_node is None:
            return
        params, grads = self._capture_model_and_gradients()
        if params is None:
            return
        scount = sample_count if sample_count is not None else self.train_samples
        self.blockchain_node.submit_model_update(
            round_number=self.current_round,
            model_params=params,
            gradient_params=grads,
            sample_count=scount,
            train_loss=float(train_loss)
        )

    def finalize_local_training(self, train_loss: float = 0.0, sample_count: Optional[int] = None):
        """
        客户端本地训练结束后调用：将梯度上传到NPC deputy管理的区块链
        Deputy节点不参与训练，因此不会调用此方法
        """
        # Deputy节点不参与训练，直接返回
        if self.use_npc_deputy and self.deputy_manager is not None:
            if self.deputy_manager.is_deputy(self.id):
                # Deputy节点不训练，只负责接收和处理梯度
                return
        
        self._store_update_on_blockchain(train_loss, sample_count)
        
        # NPC Deputy机制：非deputy客户端将梯度上传到deputy管理的区块链（保证数据不变）
        if self.use_npc_deputy and self.deputy_manager is not None:
            # 获取梯度
            params, grads = self._capture_model_and_gradients()
            if grads is not None:
                # 通过P2P将梯度发送给deputy节点，deputy负责上链
                self._upload_gradient_to_deputy_blockchain(
                    round_number=self.current_round,
                    model_params=params,
                    gradient_params=grads,
                    train_loss=train_loss,
                    sample_count=sample_count if sample_count is not None else self.train_samples
                )
        
        # 非NPC Deputy机制：使用原有逻辑
        elif self.p2p_enabled:
            self.publish_model_update(force=True)
        
        self.cached_global_state = None
        self.last_decentralized_global = None
    
    def _upload_gradient_to_deputy_blockchain(self, round_number: int, model_params: Dict[str, Any], 
                                             gradient_params: Dict[str, Any], train_loss: float, sample_count: int):
        """
        将梯度上传到NPC deputy管理的区块链
        :param round_number: 轮次
        :param model_params: 模型参数
        :param gradient_params: 梯度参数
        :param train_loss: 训练损失
        :param sample_count: 样本数量
        """
        if not self.p2p_enabled:
            return
        
        # 获取当前deputy节点列表
        deputies = self.deputy_manager.get_deputies() if self.deputy_manager else []
        if not deputies:
            print(f"[NPCDeputy][Client {self.id}] 警告：未找到deputy节点，无法上传梯度")
            return
        
        # 准备梯度数据（转换为可序列化格式）
        model_state_serialized = {}
        gradient_state_serialized = {}
        summary_mode = getattr(self, "blockchain_storage_mode", "summary") == "summary"
        
        # NPC Deputy机制：model_params必须使用全量模式（Tensor），因为需要聚合应用到模型
        # 只有gradient_params可以使用summary模式（用于相似度计算）
        use_full_model_params = True  # NPC Deputy机制中始终使用全量模式
        
        for name, value in model_params.items():
            if isinstance(value, torch.Tensor):
                # 始终将model_params序列化为list（全量模式）
                model_state_serialized[name] = value.detach().cpu().numpy().tolist()
            elif isinstance(value, dict):
                # 如果已经是字典（summary模式），保留（但这种情况不应该在NPC Deputy中出现）
                model_state_serialized[name] = value
            else:
                model_state_serialized[name] = value
        
        for name, value in gradient_params.items():
            if summary_mode and isinstance(value, dict):
                # gradient_params可以使用summary模式
                gradient_state_serialized[name] = value
            elif isinstance(value, torch.Tensor):
                gradient_state_serialized[name] = value.detach().cpu().numpy().tolist()
            else:
                gradient_state_serialized[name] = value
        
        # 将梯度发送给所有deputy节点（deputy负责上链）
        payload = {
            "client_id": self.id,
            "round": round_number,
            "model_params": model_state_serialized,
            "gradient_params": gradient_state_serialized,
            "sample_count": sample_count,
            "train_loss": float(train_loss),
            "timestamp": time.time(),
        }
        
        # 发送给所有deputy节点
        for deputy_id in deputies:
            self.p2p_network.send(
                sender_id=self.id,
                target_id=deputy_id,
                payload=payload,
                message_type="gradient_upload",
                round_number=round_number
            )
        
        print(f"[NPCDeputy][Client {self.id}] 已将梯度上传给deputy节点 {deputies}（等待上链）")
    
    def _deputy_receive_and_store_gradients_only(self, round_number: int):
        """
        Deputy节点：只接收客户端上传的梯度，保存到区块链（保证数据不变）
        不进行聚合，聚合由第一个deputy节点统一完成
        优化：加强内存管理，限制内存池大小
        """
        if not self.p2p_enabled or not isinstance(self.local_smart_contract, NPCDeputySmartContract):
            return
        
        # 接收客户端上传的梯度，保存到区块链
        messages = self.p2p_network.receive(self.id)
        received_count = 0
        remaining_messages = []  # 保存非当前轮次梯度上传的消息
        
        # 内存管理：限制处理的梯度数量（避免内存溢出）
        max_gradients_per_round = getattr(self.args, 'max_gradients_per_round', 100) if hasattr(self, 'args') else 100
        
        for message in messages:
            if message.type == "gradient_upload" and message.round == round_number:
                # 内存保护：如果已接收的梯度数量超过限制，跳过后续梯度
                if received_count >= max_gradients_per_round:
                    print(f"[NPCDeputy][Deputy {self.id}] 警告：已接收梯度数量达到上限 ({max_gradients_per_round})，跳过后续梯度")
                    remaining_messages.append(message)
                    continue
                
                # 处理当前轮次的梯度上传消息
                payload = message.payload
                client_id = payload.get("client_id")
                if client_id is None:
                    continue
                
                # 将payload转换为ModelUpdateSubmission并保存到区块链
                from flcore.blockchain.smart_contract import ModelUpdateSubmission
                model_state = payload.get("model_params", {})
                gradient_state = payload.get("gradient_params", {})
                
                if not model_state:
                    continue
                
                # 转换为tensor格式（使用summary模式减少内存占用）
                tensor_model_state = {}
                for name, value in model_state.items():
                    if isinstance(value, dict):
                        tensor_model_state[name] = value
                    else:
                        # 如果值太大，转换为summary模式
                        if isinstance(value, list) and len(value) > 10000:
                            # 大tensor转换为summary模式
                            temp_tensor = self._tensorize_gradient_value(value)
                            flat = temp_tensor.view(-1)
                            tensor_model_state[name] = {
                                "mean": float(flat.mean()),
                                "std": float(flat.std(unbiased=False)),
                                "l2": float(torch.norm(flat, p=2))
                            }
                            del temp_tensor, flat
                        else:
                            tensor_model_state[name] = self._tensorize_gradient_value(value)
                
                tensor_gradient_state = {}
                if gradient_state:
                    for name, value in gradient_state.items():
                        if isinstance(value, dict):
                            tensor_gradient_state[name] = value
                        else:
                            tensor_gradient_state[name] = self._tensorize_gradient_value(value)
                
                submission = ModelUpdateSubmission(
                    client_id=client_id,
                    round_number=round_number,
                    model_params=tensor_model_state,
                    gradient_params=tensor_gradient_state if tensor_gradient_state else None,
                    sample_count=int(payload.get("sample_count", 0)),
                    train_loss=float(payload.get("train_loss", 0.0)),
                    timestamp=payload.get("timestamp", time.time())
                )
                
                # 保存到区块链（所有梯度都上链，保证不可篡改）
                # 注意：权益更新在区块确认后执行，这里只保存梯度
                deputies = self.deputy_manager.get_deputies()
                if self.local_smart_contract.submit_gradient(submission, deputies, round_number):
                    received_count += 1
                    print(f"[NPCDeputy][Deputy {self.id}] 已将客户端 {client_id} 的梯度保存到区块链（权益更新将在区块确认后执行）")
                
                # 释放临时变量
                del tensor_model_state, tensor_gradient_state, submission
                
                # 定期清理内存
                if received_count % 10 == 0:
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
            else:
                # 保留非当前轮次梯度上传的消息
                remaining_messages.append(message)
        
        if received_count > 0:
            print(f"[NPCDeputy][Deputy {self.id}] 本轮共接收并保存了 {received_count} 个梯度到区块链")
        
        # 清理已处理的P2P消息，释放内存
        # 注意：receive()已经取出了所有消息，所以需要清空队列并重新放入未处理的消息
        if self.p2p_enabled:
            # 清空队列（因为receive已经取出了所有消息）
            self.p2p_network.clear(self.id)
            # 重新放入未处理的消息（非当前轮次的梯度上传消息）
            for msg in remaining_messages:
                self.p2p_network.send(
                    sender_id=msg.sender_id,
                    target_id=self.id,
                    payload=msg.payload,
                    message_type=msg.type,
                    round_number=msg.round
                )
        
        # 释放临时变量
        del messages, remaining_messages
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    
    def _deputy_receive_and_store_gradients(self, round_number: int):
        """
        Deputy节点：接收客户端上传的梯度，保存到区块链（保证数据不变）
        然后从区块链读取所有梯度，根据相似度选择M份进行聚合
        （已废弃：现在使用_deputy_receive_and_store_gradients_only和_deputy_select_and_aggregate_by_similarity分离）
        """
        # 先接收并存储梯度
        self._deputy_receive_and_store_gradients_only(round_number)
        # 然后进行聚合
        return self._deputy_select_and_aggregate_by_similarity(round_number)
    
    def _deputy_create_transaction(self, round_number: int) -> Optional[Dict[str, Any]]:
        """
        权益最大的Deputy节点：创建交易（聚合梯度）
        :param round_number: 轮次
        :return: 交易数据（包含聚合结果、相似度信息等），如果失败返回None
        """
        if not isinstance(self.local_smart_contract, NPCDeputySmartContract):
            return None
        
        # 从区块链读取所有梯度（保证数据不变）
        all_stored_gradients = self.local_smart_contract.get_stored_gradients(round_number)
        if not all_stored_gradients:
            return None
        
        print(f"[NPCDeputy][Deputy {self.id}] 创建交易：从区块链读取到 {len(all_stored_gradients)} 个梯度")
        
        # 提取梯度字典
        client_gradients = {}
        client_submissions = {}  # {client_id: submission}
        for submission in all_stored_gradients:
            if submission.gradient_params:
                client_gradients[submission.client_id] = submission.gradient_params
                client_submissions[submission.client_id] = submission
        
        if not client_gradients:
            return None
        
        # 步骤1：计算初始全局梯度（用于恶意梯度检测）
        # 注意：不使用上一轮的全局模型（因为格式不匹配），而是使用当前轮次所有梯度的平均/中位数
        # 这样可以避免格式不匹配的问题，同时仍然能够检测恶意梯度
        initial_global_gradient = self._compute_robust_global_gradient(client_gradients)
        print(f"[NPCDeputy][Deputy {self.id}] 使用鲁棒聚合方法（中位数）计算初始全局梯度用于恶意梯度检测")
        
        # 步骤2：计算每个客户端与全局梯度的余弦相似度
        similarities = self.similarity_calculator.calculate_similarities_to_global(
            client_gradients, initial_global_gradient
        )
        
        # 打印所有相似度值（用于调试）
        print(f"[NPCDeputy][Deputy {self.id}] 所有客户端相似度:")
        for cid, sim in similarities.items():
            print(f"  - 客户端 {cid}: 相似度={sim:.4f}")
        
        # 步骤3：设置相似度阈值，剔除恶意梯度（相似度过低的）
        similarity_threshold = getattr(self.args, 'similarity_threshold', 0.3) if hasattr(self, 'args') else 0.3
        
        valid_clients = []
        malicious_clients = []
        for client_id, similarity in similarities.items():
            if similarity >= similarity_threshold:
                valid_clients.append(client_id)
            else:
                malicious_clients.append(client_id)
        
        print(f"[NPCDeputy][Deputy {self.id}] 相似度检测结果:")
        print(f"  - 阈值: {similarity_threshold}")
        print(f"  - 有效客户端 ({len(valid_clients)}): {valid_clients}")
        if malicious_clients:
            print(f"  - 恶意客户端 ({len(malicious_clients)}): {malicious_clients} (已剔除)")
            for cid in malicious_clients:
                print(f"    * 客户端 {cid}: 相似度={similarities[cid]:.4f}")
        
        if not valid_clients:
            print(f"[NPCDeputy][Deputy {self.id}] 警告：没有通过相似度检测的客户端，无法聚合")
            return None
        
        # 步骤4：聚合所有通过相似度检测的梯度（数量不固定）
        valid_submissions = [client_submissions[cid] for cid in valid_clients if cid in client_submissions]
        
        # 使用智能合约聚合有效梯度
        original_history = self.local_smart_contract.update_history.copy()
        original_min_clients = self.local_smart_contract.min_clients
        
        # 临时调整min_clients为实际有效客户端数量（至少为1）
        self.local_smart_contract.update_history = valid_submissions
        self.local_smart_contract.min_clients = max(1, len(valid_submissions))
        
        aggregated = self.local_smart_contract.aggregate(round_number)
        
        # 恢复原始历史记录和min_clients
        self.local_smart_contract.update_history = original_history
        self.local_smart_contract.min_clients = original_min_clients
        
        if aggregated is None:
            print(f"[NPCDeputy][Deputy {self.id}] 聚合失败")
            return None
        
        print(f"[NPCDeputy][Deputy {self.id}] 聚合完成，共聚合了 {len(valid_clients)} 个客户端的梯度")
        
        # 步骤5：根据聚合后的全局梯度，重新计算相似度
        aggregated_gradient = self._model_params_to_gradient(aggregated, client_gradients)
        final_similarities = self.similarity_calculator.calculate_similarities_to_global(
            client_gradients, aggregated_gradient
        )
        
        # 创建交易数据（包含聚合结果和相关信息）
        # 注意：需要序列化tensor数据以便P2P传输
        serialized_aggregated = {}
        if aggregated:
            for name, value in aggregated.items():
                if isinstance(value, torch.Tensor):
                    serialized_aggregated[name] = value.detach().cpu().numpy().tolist()
                else:
                    serialized_aggregated[name] = value
        
        # 序列化aggregated_gradient（用于验证）
        serialized_aggregated_gradient = {}
        if aggregated_gradient:
            for name, value in aggregated_gradient.items():
                if isinstance(value, torch.Tensor):
                    serialized_aggregated_gradient[name] = value.detach().cpu().numpy().tolist()
                elif isinstance(value, dict):
                    serialized_aggregated_gradient[name] = value
                else:
                    serialized_aggregated_gradient[name] = value
        
        # 序列化client_gradients（用于验证）
        serialized_client_gradients = {}
        if client_gradients:
            for cid, grad_dict in client_gradients.items():
                serialized_client_gradients[cid] = {}
                for name, value in grad_dict.items():
                    if isinstance(value, torch.Tensor):
                        serialized_client_gradients[cid][name] = value.detach().cpu().numpy().tolist()
                    elif isinstance(value, dict):
                        serialized_client_gradients[cid][name] = value
                    else:
                        serialized_client_gradients[cid][name] = value
        
        transaction_data = {
            "round_number": round_number,
            "creator_deputy_id": self.id,
            "aggregated_params": serialized_aggregated,  # 序列化后的聚合参数
            "valid_clients": valid_clients,
            "malicious_clients": malicious_clients,
            "final_similarities": final_similarities,  # 字典，可以直接序列化
            "aggregated_gradient": serialized_aggregated_gradient,  # 序列化后的聚合梯度
            "client_gradients": serialized_client_gradients,  # 序列化后的客户端梯度（用于验证）
            "timestamp": time.time()
        }
        
        # 释放临时变量（但保留transaction_data中的数据）
        if 'initial_global_gradient' in locals():
            del initial_global_gradient
        if 'similarities' in locals():
            del similarities
        del client_gradients, client_submissions, valid_submissions
        
        # 清理CUDA缓存
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        return transaction_data
    
    def _deputy_verify_transaction(self, round_number: int, transaction_data: Dict[str, Any]) -> bool:
        """
        其他Deputy节点：验证交易的有效性
        :param round_number: 轮次
        :param transaction_data: 交易数据
        :return: 交易是否有效
        """
        if not isinstance(self.local_smart_contract, NPCDeputySmartContract):
            return False
        
        print(f"[NPCDeputy][Deputy {self.id}] 开始验证Round {round_number}的交易（创建者: Deputy {transaction_data.get('creator_deputy_id', -1)}）")
        
        # 验证1：检查轮次是否匹配
        if transaction_data.get("round_number") != round_number:
            print(f"[NPCDeputy][Deputy {self.id}] 验证失败：轮次不匹配")
            return False
        
        # 验证2：从区块链读取所有梯度，验证交易中的梯度是否一致
        all_stored_gradients = self.local_smart_contract.get_stored_gradients(round_number)
        if not all_stored_gradients:
            print(f"[NPCDeputy][Deputy {self.id}] 验证失败：区块链中没有梯度数据")
            return False
        
        # 验证3：检查有效客户端列表是否合理
        valid_clients = transaction_data.get("valid_clients", [])
        malicious_clients = transaction_data.get("malicious_clients", [])
        
        # 验证4：重新计算相似度，验证交易中的相似度是否正确
        serialized_client_gradients = transaction_data.get("client_gradients", {})
        serialized_aggregated_gradient = transaction_data.get("aggregated_gradient")
        
        if not serialized_client_gradients or not serialized_aggregated_gradient:
            print(f"[NPCDeputy][Deputy {self.id}] 验证失败：交易数据不完整")
            return False
        
        # 反序列化client_gradients和aggregated_gradient（从list转回tensor）
        client_gradients = {}
        for cid, grad_dict in serialized_client_gradients.items():
            client_gradients[int(cid)] = {}
            for name, value in grad_dict.items():
                if isinstance(value, list):
                    client_gradients[int(cid)][name] = torch.tensor(value, dtype=torch.float32)
                elif isinstance(value, dict):
                    client_gradients[int(cid)][name] = value
                else:
                    client_gradients[int(cid)][name] = value
        
        aggregated_gradient = {}
        for name, value in serialized_aggregated_gradient.items():
            if isinstance(value, list):
                aggregated_gradient[name] = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, dict):
                aggregated_gradient[name] = value
            else:
                aggregated_gradient[name] = value
        
        # 重新计算相似度
        recalculated_similarities = self.similarity_calculator.calculate_similarities_to_global(
            client_gradients, aggregated_gradient
        )
        
        # 验证相似度是否一致（允许小的浮点误差）
        final_similarities = transaction_data.get("final_similarities", {})
        similarity_threshold = getattr(self.args, 'similarity_threshold', 0.3) if hasattr(self, 'args') else 0.3
        
        # 验证相似度计算是否正确（检查有效客户端和恶意客户端的分类）
        for client_id, similarity in recalculated_similarities.items():
            if client_id in final_similarities:
                expected_similarity = final_similarities[client_id]
                # 允许小的浮点误差（0.01）
                if abs(similarity - expected_similarity) > 0.01:
                    print(f"[NPCDeputy][Deputy {self.id}] 验证失败：客户端 {client_id} 的相似度不一致 "
                          f"(计算值={similarity:.4f}, 交易值={expected_similarity:.4f})")
                    return False
        
        # 验证5：检查聚合结果是否合理
        aggregated_params = transaction_data.get("aggregated_params")
        if aggregated_params is None:
            print(f"[NPCDeputy][Deputy {self.id}] 验证失败：聚合结果为空")
            return False
        
        # 验证6：检查有效客户端数量是否合理
        if len(valid_clients) == 0:
            print(f"[NPCDeputy][Deputy {self.id}] 验证失败：没有有效客户端")
            return False
        
        # 验证通过
        print(f"[NPCDeputy][Deputy {self.id}] 交易验证通过")
        
        # 释放临时变量（包括反序列化的梯度数据）
        if 'client_gradients' in locals():
            for cid, grad_dict in client_gradients.items():
                if isinstance(grad_dict, dict):
                    grad_dict.clear()
            client_gradients.clear()
        if 'aggregated_gradient' in locals():
            if isinstance(aggregated_gradient, dict):
                aggregated_gradient.clear()
        del client_gradients, aggregated_gradient, recalculated_similarities, all_stored_gradients
        
        # 清理CUDA缓存
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        return True
    
    def _deputy_select_and_aggregate_by_similarity(self, round_number: int):
        """
        Deputy节点：从区块链读取所有梯度，剔除恶意梯度（相似度过低的），聚合剩余梯度
        然后根据相似度给客户端打分并奖罚
        （已废弃：现在使用_deputy_create_transaction和_deputy_verify_transaction）
        """
        # 兼容旧代码，调用新方法
        transaction_data = self._deputy_create_transaction(round_number)
        if transaction_data is None:
            return None
        
        # 应用交易（打分和奖罚）
        final_similarities = transaction_data.get("final_similarities", {})
        valid_clients = transaction_data.get("valid_clients", [])
        malicious_clients = transaction_data.get("malicious_clients", [])
        
        self._score_and_reward_clients(round_number, final_similarities, valid_clients, malicious_clients)
        
        # 清理历史记录和临时变量，释放内存
        self.local_smart_contract.clear_round_updates(round_number)
        
        # 释放临时变量
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        return transaction_data.get("aggregated_params")
    
    def _model_params_to_gradient(self, model_params: Dict[str, torch.Tensor], 
                                  reference_gradients: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        将聚合后的模型参数转换为梯度格式（用于相似度计算）
        这里简化处理：使用模型参数本身作为"梯度"（实际应该是相对于上一轮的差值）
        """
        if not reference_gradients:
            return None
        
        # 使用第一个客户端的梯度结构作为参考
        first_grad = list(reference_gradients.values())[0]
        gradient_format = {}
        
        for name in model_params.keys():
            if name in first_grad:
                # 保持与原始梯度相同的格式
                if isinstance(first_grad[name], dict):
                    # Summary模式
                    param_tensor = model_params[name]
                    if isinstance(param_tensor, torch.Tensor):
                        flat = param_tensor.view(-1)
                        gradient_format[name] = {
                            "mean": float(flat.mean()),
                            "std": float(flat.std(unbiased=False)),
                            "l2": float(torch.norm(flat, p=2))
                        }
                    else:
                        gradient_format[name] = first_grad[name]  # 保持原格式
                else:
                    # 全量模式
                    gradient_format[name] = model_params[name]
        
        return gradient_format
    
    def _score_and_reward_clients(self, round_number: int, similarities: Dict[int, float],
                                  valid_clients: List[int], malicious_clients: List[int]):
        """
        根据相似度给客户端打分并进行奖罚（在区块确认后执行）
        :param round_number: 轮次
        :param similarities: 每个客户端与聚合后全局梯度的相似度
        :param valid_clients: 通过相似度检测的客户端
        :param malicious_clients: 被剔除的恶意客户端
        """
        if not similarities:
            return
        
        # 步骤1：给予所有上传梯度的客户端参与奖励（区块确认后执行）
        # 注意：这里不需要重新加载梯度数据，因为valid_clients和malicious_clients已经包含了所有上传梯度的客户端
        if isinstance(self.local_smart_contract, NPCDeputySmartContract):
            upload_reward = getattr(self.args, 'client_upload_reward', 0.5) if hasattr(self, 'args') else 0.5
            
            # 从valid_clients和malicious_clients中获取所有上传梯度的客户端（避免重复加载梯度数据）
            all_uploaded_clients = set(valid_clients) | set(malicious_clients)
            for client_id in all_uploaded_clients:
                # 区块确认后给予参与奖励
                self.stake_manager.update_stake(client_id, upload_reward)
                print(f"[NPCDeputy][Deputy {self.id}] 客户端 {client_id} 上传梯度，获得参与奖励 +{upload_reward:.2f}（区块确认后）")
        
        # 计算平均相似度
        valid_similarities = [similarities[cid] for cid in valid_clients if cid in similarities]
        avg_similarity = np.mean(valid_similarities) if valid_similarities else 0.0
        
        print(f"[NPCDeputy][Deputy {self.id}] 相似度统计:")
        print(f"  - 平均相似度: {avg_similarity:.4f}")
        
        # 对有效客户端：根据相似度打分并奖罚
        similarity_weight = getattr(self.args, 'similarity_weight', 0.7) if hasattr(self, 'args') else 0.7
        stake_weight = getattr(self.args, 'stake_weight', 0.3) if hasattr(self, 'args') else 0.3
        
        for client_id in valid_clients:
            if client_id not in similarities:
                continue
            
            similarity = similarities[client_id]
            # 计算得分（基于相似度和权益）
            score = self.stake_manager.calculate_score(
                client_id, similarity, round_number,
                similarity_weight, stake_weight
            )
            
            # 根据相似度与平均值的差异进行奖罚
            if similarity > avg_similarity:
                # 相似度高于平均值，奖励
                reward = (similarity - avg_similarity) * getattr(self.args, 'reward_factor', 0.1)
                self.stake_manager.update_stake(client_id, reward)
                print(f"  - 客户端 {client_id}: 相似度={similarity:.4f} (高于平均), 奖励+{reward:.2f}")
            else:
                # 相似度低于平均值，轻微惩罚
                penalty = (avg_similarity - similarity) * getattr(self.args, 'penalty_factor', 0.05)
                self.stake_manager.update_stake(client_id, -penalty)
                print(f"  - 客户端 {client_id}: 相似度={similarity:.4f} (低于平均), 惩罚-{penalty:.2f}")
        
        # 对恶意客户端：严重惩罚
        for client_id in malicious_clients:
            if client_id in similarities:
                similarity = similarities[client_id]
                # 恶意客户端严重惩罚
                severe_penalty = (avg_similarity - similarity) * getattr(self.args, 'malicious_penalty_factor', 0.2)
                self.stake_manager.update_stake(client_id, -severe_penalty)
                print(f"  - 客户端 {client_id}: 相似度={similarity:.4f} (恶意), 严重惩罚-{severe_penalty:.2f}")
        
        # DPOS奖励机制：奖励deputy节点（完成聚合工作）
        deputies = self.deputy_manager.get_deputies() if self.deputy_manager else []
        block_reward = getattr(self.args, 'deputy_block_reward', 5.0) if hasattr(self, 'args') else 5.0
        
        for deputy_id in deputies:
            if deputy_id == self.id:  # 当前deputy节点（执行聚合的节点）
                # 出块奖励：完成聚合和验证工作
                self.stake_manager.reward_deputy(deputy_id, block_reward, f"Round {round_number} 出块奖励")
            else:
                # 其他deputy节点：参与存储梯度，给予较小奖励
                storage_reward = block_reward * 0.3  # 存储奖励是出块奖励的30%
                self.stake_manager.reward_deputy(deputy_id, storage_reward, f"Round {round_number} 存储奖励")
        
        # DPOS惩罚机制：检查deputy验证失败
        failure_threshold = getattr(self.args, 'deputy_validation_failure_threshold', 3) if hasattr(self, 'args') else 3
        failure_penalty = getattr(self.args, 'deputy_failure_penalty', 2.0) if hasattr(self, 'args') else 2.0
        
        for deputy_id in deputies:
            if self.deputy_manager.should_penalize_deputy(deputy_id, failure_threshold):
                failure_count = self.deputy_manager.get_validation_failure_count(deputy_id)
                self.stake_manager.penalize_deputy_failure(
                    deputy_id, 
                    failure_penalty * failure_count, 
                    f"验证失败 {failure_count} 次"
                )
        
        # DPOS Slashing机制：检查deputy严重作恶
        slash_threshold = getattr(self.args, 'deputy_slash_threshold', 0.5) if hasattr(self, 'args') else 0.5
        slash_ratio = getattr(self.args, 'deputy_slash_ratio', 0.1) if hasattr(self, 'args') else 0.1
        
        # 检查是否有deputy在恶意客户端列表中（如果deputy作恶）
        for deputy_id in deputies:
            if deputy_id in malicious_clients:
                # Deputy节点作恶，严重惩罚（Slashing）
                self.stake_manager.slash_deputy(
                    deputy_id, 
                    slash_ratio, 
                    f"Deputy节点作恶（恶意梯度）"
                )
        
        # 更新权益后，选举下一轮deputy（综合考虑表现和权益，保留至少一半的上一轮deputy）
        performance_weight = getattr(self.args, 'deputy_performance_weight', 0.6) if hasattr(self, 'args') else 0.6
        stake_weight = getattr(self.args, 'deputy_stake_weight', 0.4) if hasattr(self, 'args') else 0.4
        self.deputy_manager.update_deputies_from_scores(
            round_number, 
            performance_weight=performance_weight,
            stake_weight=stake_weight
        )
    
    
    def _tensorize_gradient_value(self, value):
        """将梯度值转换为tensor"""
        if isinstance(value, torch.Tensor):
            return value.clone()
        elif isinstance(value, list):
            return torch.tensor(value, dtype=torch.float32)
        else:
            return torch.tensor([value], dtype=torch.float32)
    
    def _compute_robust_global_gradient(self, client_gradients: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """计算鲁棒全局梯度（使用中位数，更抗异常值）"""
        if not client_gradients:
            return None
        
        first_grad = list(client_gradients.values())[0]
        global_gradient = {}
        
        for name in first_grad.keys():
            if isinstance(first_grad[name], dict):
                # Summary模式：计算统计量的中位数
                means = [client_gradients[cid][name].get("mean", 0.0) 
                        for cid in client_gradients if name in client_gradients[cid]]
                stds = [client_gradients[cid][name].get("std", 0.0) 
                       for cid in client_gradients if name in client_gradients[cid]]
                l2s = [client_gradients[cid][name].get("l2", 0.0) 
                      for cid in client_gradients if name in client_gradients[cid]]
                global_gradient[name] = {
                    "mean": float(np.median(means)),
                    "std": float(np.median(stds)),
                    "l2": float(np.median(l2s))
                }
            else:
                # 全量模式：计算中位数（更鲁棒）
                stacked = torch.stack([
                    client_gradients[cid][name] 
                    for cid in client_gradients 
                    if name in client_gradients[cid]
                ])
                # 使用中位数而不是平均值（更鲁棒）
                global_gradient[name] = torch.median(stacked, dim=0)[0]
                # 释放临时tensor
                del stacked
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
        
        return global_gradient
    
    def _compute_global_gradient(self, client_gradients: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """计算全局梯度（平均）"""
        if not client_gradients:
            return None
        
        first_grad = list(client_gradients.values())[0]
        global_gradient = {}
        
        for name in first_grad.keys():
            if isinstance(first_grad[name], dict):
                # Summary模式：计算统计量的平均值
                global_gradient[name] = {
                    "mean": np.mean([client_gradients[cid][name].get("mean", 0.0) 
                                    for cid in client_gradients if name in client_gradients[cid]]),
                    "std": np.mean([client_gradients[cid][name].get("std", 0.0) 
                                   for cid in client_gradients if name in client_gradients[cid]]),
                    "l2": np.mean([client_gradients[cid][name].get("l2", 0.0) 
                                  for cid in client_gradients if name in client_gradients[cid]])
                }
            else:
                # 全量模式：计算平均值
                stacked = torch.stack([
                    client_gradients[cid][name] 
                    for cid in client_gradients 
                    if name in client_gradients[cid]
                ])
                global_gradient[name] = torch.mean(stacked, dim=0)
                # 释放临时tensor
                del stacked
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
        
        return global_gradient
    
    def sync_stake_and_deputy_from_p2p(self, round_number: int):
        """从P2P网络同步权益和deputy信息（完全去中心化）"""
        if not self.use_npc_deputy or not self.p2p_enabled:
            return
        
        # 接收P2P消息中的权益和deputy信息
        messages = self.p2p_network.receive(self.id)
        for message in messages:
            if message.type == "stake_update":
                # 同步权益信息（只更新，不重置）
                stake_data = message.payload.get("stakes", {})
                for client_id, stake_value in stake_data.items():
                    # 重要：使用update_stake而不是set_stake，避免重置权益
                    # 但如果本地没有该客户端的权益记录，则设置初始值
                    current_stake = self.stake_manager.get_stake(client_id)
                    if current_stake == 0.0 and stake_value > 0:
                        # 如果本地权益为0且收到非零权益，说明是首次同步，使用set_stake
                        self.stake_manager.set_stake(client_id, stake_value)
                    elif abs(current_stake - stake_value) > 0.01:  # 只同步差异较大的权益
                        # 计算差异并更新
                        delta = stake_value - current_stake
                        self.stake_manager.update_stake(client_id, delta)
            
            elif message.type == "deputy_election":
                # 同步deputy选举结果
                deputies = message.payload.get("deputies", [])
                round_num = message.payload.get("round", round_number)
                if round_num == round_number:
                    self.deputy_manager.current_deputies = deputies
                    self.deputy_manager.deputy_history[round_num] = deputies.copy()
            
            elif message.type == "transaction_proposal":
                # 接收交易提案（执行deputy创建的交易）
                transaction_data = message.payload.get("transaction", {})
                round_num = message.payload.get("round", round_number)
                executor_deputy_id = transaction_data.get("creator_deputy_id", -1)
                
                # 只有deputy节点才能验证和投票，且执行deputy不参与投票
                # 检查是否已经投票过（防止重复投票）
                if round_num == round_number and self.deputy_manager.is_deputy(self.id) and self.id != executor_deputy_id:
                    # 检查是否已经投票过
                    votes = self.deputy_manager.transaction_votes.get(round_number, {})
                    if self.id in votes:
                        # 已经投票过，跳过
                        pass
                    else:
                        print(f"[NPCDeputy][Deputy {self.id}] 收到Round {round_number}的交易提案（执行deputy: Deputy {executor_deputy_id}）")
                        # 验证交易
                        is_valid = self._deputy_verify_transaction(round_number, transaction_data)
                        # 投票（执行deputy不参与投票，vote_on_transaction内部会检查重复投票）
                        self.deputy_manager.vote_on_transaction(round_number, self.id, is_valid)
                        # 广播投票结果
                        self.p2p_network.broadcast(
                            sender_id=self.id,
                            payload={"round": round_number, "deputy_id": self.id, "vote": is_valid},
                            message_type="transaction_vote",
                            round_number=round_number
                        )
                        print(f"[NPCDeputy][Deputy {self.id}] 已投票: {'通过' if is_valid else '拒绝'}")
                elif round_num == round_number and self.id == executor_deputy_id:
                    # 执行deputy收到自己的交易提案（不应该发生，但为了安全起见）
                    print(f"[NPCDeputy][Deputy {self.id}] 警告：执行deputy收到自己的交易提案，忽略（执行deputy不参与投票）")
            
            elif message.type == "transaction_vote":
                # 接收其他deputy的投票
                vote_round = message.payload.get("round", round_number)
                deputy_id = message.payload.get("deputy_id", -1)
                vote = message.payload.get("vote", False)
                if vote_round == round_number:
                    self.deputy_manager.vote_on_transaction(round_number, deputy_id, vote)
            
            elif message.type == "transaction_commit":
                # 接收已确认的交易（大于等于2/3验证通过）
                transaction_data = message.payload.get("transaction", {})
                round_num = message.payload.get("round", round_number)
                if round_num == round_number and isinstance(self.local_smart_contract, NPCDeputySmartContract):
                    # 反序列化聚合参数（从list转回tensor）
                    serialized_aggregated = transaction_data.get("aggregated_params", {})
                    aggregated_params = {}
                    for name, value in serialized_aggregated.items():
                        if isinstance(value, list):
                            aggregated_params[name] = torch.tensor(value, dtype=torch.float32)
                        else:
                            aggregated_params[name] = value
                    
                    # 将交易保存到本地区块链（这里可以添加区块创建逻辑）
                    print(f"[NPCDeputy][Deputy {self.id}] 已接收并保存确认的交易到本地区块链（Round {round_number}）")
    
    def broadcast_transaction_proposal(self, round_number: int, transaction_data: Dict[str, Any]):
        """广播交易提案（权益最大的deputy调用）"""
        if not self.use_npc_deputy or not self.p2p_enabled:
            return
        
        # 广播交易提案给所有deputy节点
        deputies = self.deputy_manager.get_deputies() if self.deputy_manager else []
        for deputy_id in deputies:
            if deputy_id != self.id:  # 不发送给自己
                self.p2p_network.send(
                    sender_id=self.id,
                    target_id=deputy_id,
                    payload={"transaction": transaction_data, "round": round_number},
                    message_type="transaction_proposal",
                    round_number=round_number
                )
        
        print(f"[NPCDeputy][Deputy {self.id}] 已广播交易提案给其他deputy节点")
    
    def broadcast_confirmed_transaction(self, round_number: int, transaction_data: Dict[str, Any]):
        """广播已确认的交易（大于等于2/3验证通过后，广播到所有deputy的区块链）"""
        if not self.use_npc_deputy or not self.p2p_enabled:
            return
        
        # 广播已确认的交易给所有deputy节点
        deputies = self.deputy_manager.get_deputies() if self.deputy_manager else []
        for deputy_id in deputies:
            self.p2p_network.send(
                sender_id=self.id,
                target_id=deputy_id,
                payload={"transaction": transaction_data, "round": round_number},
                message_type="transaction_commit",
                round_number=round_number
            )
        
        print(f"[NPCDeputy][Deputy {self.id}] 已广播确认的交易到所有deputy节点")
    
    def broadcast_stake_and_deputy(self, round_number: int):
        """广播权益和deputy信息到P2P网络"""
        if not self.use_npc_deputy or not self.p2p_enabled:
            return
        
        # 广播权益信息
        stakes = self.stake_manager.get_all_stakes()
        self.p2p_network.broadcast(
            sender_id=self.id,
            payload={"stakes": stakes, "round": round_number},
            message_type="stake_update",
            round_number=round_number
        )
        
        # 如果是deputy节点，广播deputy选举结果
        if self.deputy_manager.is_deputy(self.id):
            deputies = self.deputy_manager.get_deputies()
            self.p2p_network.broadcast(
                sender_id=self.id,
                payload={"deputies": deputies, "round": round_number},
                message_type="deputy_election",
                round_number=round_number
            )
    
    def elect_deputies_locally(self, round_number: int):
        """本地选举deputy（基于权益）"""
        if not self.use_npc_deputy or self.deputy_manager is None:
            return []
        return self.deputy_manager.elect_deputies(round_number)
    
    def calculate_similarities_and_scores_locally(self, round_number: int):
        """本地计算余弦相似度和得分"""
        if not self.use_npc_deputy or self.deputy_manager is None:
            return
        
        if not isinstance(self.local_smart_contract, NPCDeputySmartContract):
            return
        
        # 获取存储的梯度
        stored_gradients = self.local_smart_contract.get_stored_gradients(round_number)
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
            first_grad = list(client_gradients.values())[0]
            global_gradient = {}
            for name in first_grad.keys():
                if isinstance(first_grad[name], dict):
                    global_gradient[name] = {
                        "mean": np.mean([client_gradients[cid][name].get("mean", 0.0) 
                                        for cid in client_gradients if name in client_gradients[cid]]),
                        "std": np.mean([client_gradients[cid][name].get("std", 0.0) 
                                       for cid in client_gradients if name in client_gradients[cid]]),
                        "l2": np.mean([client_gradients[cid][name].get("l2", 0.0) 
                                      for cid in client_gradients if name in client_gradients[cid]])
                    }
                else:
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
        similarity_weight = getattr(self.args, 'similarity_weight', 0.7) if hasattr(self, 'args') else 0.7
        stake_weight = getattr(self.args, 'stake_weight', 0.3) if hasattr(self, 'args') else 0.3
        
        for client_id, similarity in similarities.items():
            score = self.stake_manager.calculate_score(
                client_id, similarity, round_number,
                similarity_weight, stake_weight
            )
        
        # 根据得分更新权益
        reward_factor = getattr(self.args, 'reward_factor', 0.1) if hasattr(self, 'args') else 0.1
        penalty_factor = getattr(self.args, 'penalty_factor', 0.05) if hasattr(self, 'args') else 0.05
        self.stake_manager.update_stakes_from_scores(round_number, reward_factor, penalty_factor)
        
        # 更新下一轮deputy
        self.deputy_manager.update_deputies_from_scores(round_number)


