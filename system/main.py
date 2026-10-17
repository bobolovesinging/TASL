#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import wandb
import json
import random
from flcore.servers.serveravg import FedAvg
from flcore.servers.serveravgbase import FedBaseline
from flcore.servers.serverblockchain import FedAvgBlockchain
from flcore.trainmodel.GRU import *
from flcore.trainmodel.models import *
from flcore.trainmodel.st_gcn import Model
from flcore.trainmodel.net.st_gcn import Model as oModel
from flcore.trainmodel.skeletonclr import SkeletonCLR

from utils.result_utils import average_data
from utils.mem_utils import MemReporter
from utils.DSTformer import *
from utils.model_action import *
from utils.model_utils import *
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
import os
import sys
# sys.path.append('/opt/data/private/gjj/condessed/system/')
# print(sys.path)
from datetime import datetime
import pytz


os.chdir(sys.path[0]) 

os.environ["WANDB_DISABLED"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.simplefilter("ignore")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)


def load_experiment_config(path: str):
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"[Config] 未找到配置文件: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:
            print(f"[Config] 解析配置文件失败: {exc}")
            return {}

#torch.manual_seed(0)

# hyper-params for Text tasks
# vocab_size = 98635
# max_len=200
# emb_dim=32

def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print("\n============= Running time: {}th =============".format(i))
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "gru":
            if "ntu60" in args.dataset:
                args.model = BIGRU(en_input_size=150,en_hidden_size=1024,en_num_layers=3,num_class=128).to(args.device)
            else:
                raise NotImplementedError
        elif model_str == "st-gcn":
            if "ntu" in args.dataset:
                args.model = Model(in_channels=3, hidden_channels=16,
                                   hidden_dim=256, dropout=0.5, 
                                   graph_args={
                                   "layout" : 'ntu-rgb+d',
                                   "strategy" : 'spatial'
                                   },
                                   edge_importance_weighting=True,
                                   )
        elif model_str == "skeletonclr":
             args.model = SkeletonCLR(
                                   base_encoder=None,
                                   pretrain=True,
                                   feature_dim=256,
                                   queue_size=32768,
                                   momentum=0.999,
                                   Temperature=0.07,
                                   mlp=False,
                                   in_channels=3, 
                                   hidden_channels=16,
                                   hidden_dim=256, #128
                                   num_class=60,
                                   dropout=0.5, 
                                   graph_args={
                                   "layout" : 'ntu-rgb+d',
                                   "strategy" : 'spatial'
                                   },
                                   edge_importance_weighting=True,
                        
                                   )
             classifier = Linear(hidden_size=256,dataset=args.dataset)
             args.classifier = classifier


        else:
            raise NotImplementedError(
                f"不支持的模型类型: '{model_str}'. "
                f"支持的模型类型: 'gru', 'st-gcn', 'skeletonclr'. "
                f"请使用 -m 参数指定模型类型，例如: -m skeletonclr"
            )

        # print(args.model)

        # select Fed_algorithm
        if args.algorithm == "FedAvg":
            # args.head = copy.deepcopy(args.model.fc)
            # args.model.fc = nn.Identity()
            # args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)
        elif args.algorithm == "FedProx":
            # args.head = copy.deepcopy(args.model.fc)
            # args.model.fc = nn.Identity()
            # args.model = BaseHeadSplit(args.model, args.head)
            server = FedProx(args, i)
        elif args.algorithm == "FedBN":
            server = FedBN(args, i)
        elif args.algorithm == "MOON":
            server = FedMOON(args, i)
        elif args.algorithm == "FedDyn":
            server = FedDyn(args, i)
        elif args.algorithm == "baseline":
            server = FedBaseline(args, i)
        elif args.algorithm == "FedAvgBlockchain" or args.algorithm == "Blockchain":
            server = FedAvgBlockchain(args, i)
        else:
            raise NotImplementedError
        server.train()

        time_list.append(time.time()-start)

    print("\nAverage time cost:{} s.".format(round(np.average(time_list), 2)))
    

    # Global average
    #从result文件夹读取结果，计算最佳准确率的均值和标准差
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")
    # 内存使用情况
    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="small")
    parser.add_argument('-nb', "--num_classes", type=int, default=60)
    parser.add_argument('-m', "--model", type=str, default="skeletonclr")
    parser.add_argument('-lbs', "--batch_size", type=int, default=128)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,
                            help="Local learning rate")
    parser.add_argument('-lrpr', "--pretrain_lr", type=float, default=0.1,
                        help="Pretrain local learning rate")
    parser.add_argument('-lrli', "--linear_lr", type=float, default=0.001,
                        help="Linear local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=True)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=10)
    parser.add_argument('-ls', "--local_epochs", type=int, default=2, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=10,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-fte', "--fine_tuning_epoch", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-sslr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP
    parser.add_argument('-bt', "--beta", type=float, default=0.0,
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
                        or L1 regularization weight of FedTransfer")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.1,
                        help="Proximal rate for FedProx")
    parser.add_argument('-nu', "--nu", type=float, default=0.1,
                        help="Proximal rate for FedIGA")
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    
    parser.add_argument('-hsize', "--hsize", type=int, default=256,
                        help="hidden size")
    parser.add_argument('-m_size', "--m_size", type=int, default=300,
                        help="memory size")
    # parser.add_argument('-N_centroids', "--N_centroids", type=int, default=256,
    #                     help="global centroids size")
    parser.add_argument('-N_local', "--N_local", type=int, default=256,
                        help="local centroids size")
    
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0, 
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_steps", type=int, default=1)
    # MOON
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fts', "--fine_tuning_steps", type=int, default=10)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    # SCAFFOLD
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    parser.add_argument('-base', "--baseline", type=bool, default=False)

    parser.add_argument('-stage', "--avgstage", type=int, default=300)
    parser.add_argument('-inpo', "--interpolate", type=str, default="None",help='数据插值方式')
    parser.add_argument('-cp', "--cepoch", type=int, default=0)
    parser.add_argument('-ccn', "--client_cnum", type=int, default=100)
    parser.add_argument('-scn', "--server_cnum", type=int, default=100)
    parser.add_argument('-ssr', "--server_condensegrate", type=float, default=0.001)
    # 插值系数：
    parser.add_argument('-it', "--inter_t", type=float, default=0.5,help='插值系数')
    # 插值个数：
    parser.add_argument('-inum', "--inter_num", type=int, default=10,help='插值个数')
    parser.add_argument('-semi', "--semi_data_ratio", type=float, default=1.0)
    parser.add_argument('-init', "--init_method", type=str, default="avg")
    parser.add_argument('-jw', "--jwaloss", type=str, default="jwcdc")
    parser.add_argument('-skl', "--skloss", type=bool, default=True)
    # CL_LOSS
    # parser.add_argument('-cl', "--conloss", type=bool, default=False)
    parser.add_argument('-fa', "--feature_augment", type=str, default="True")
    parser.add_argument('-da', "--data_augment", type=str, default="True")
    parser.add_argument('-fn', "--feature_number", type=int, default=10)
    parser.add_argument('-cfg', "--config_path", type=str, default="config/attack_config.json",
                        help="Path to JSON config for experiment/attack settings")
    
    # 智能合约相关参数
    parser.add_argument('-sc', "--use_smart_contract", type=bool, default=False,
                        help="是否使用智能合约进行去中心化聚合")
    parser.add_argument('-am', "--aggregation_method", type=str, default="fedavg",
                        choices=["fedavg", "weighted_avg", "median", "krum"],
                        help="智能合约聚合方法")
    parser.add_argument('--p2p_enabled', type=bool, default=False,
                        help="是否启用P2P通信层")
    parser.add_argument('--p2p_mode', type=str, default="local",
                        choices=["local", "grpc", "websocket"],
                        help="P2P通信模式（默认local，模拟消息队列）")
    parser.add_argument('--p2p_auto_publish', type=bool, default=True,
                        help="客户端训练完成后是否自动广播模型更新")
    parser.add_argument('--blockchain_storage_enabled', type=bool, default=False,
                        help="是否启用区块链存储梯度/参数")
    parser.add_argument('--blockchain_lightweight', type=bool, default=True,
                        help="区块链是否启用轻量模式（仅存元数据）")
    parser.add_argument('--blockchain_disk_storage', type=bool, default=False,
                        help="区块链是否写入磁盘文件")
    parser.add_argument('--blockchain_keep_recent', type=int, default=5,
                        help="区块链内存中保留的最近区块数量")
    parser.add_argument('--blockchain_storage_dir', type=str, default="blockchain_storage",
                        help="区块链存储目录")
    parser.add_argument('--blockchain_storage_mode', type=str, default="summary",
                        choices=["full", "summary"],
                        help="区块链存储内容：full=完整参数，summary=统计摘要")
    
    args = parser.parse_args()

    config_payload = load_experiment_config(args.config_path)
    if config_payload:
        federated_cfg = config_payload.get("federated", {})
        for key, value in federated_cfg.items():
            if hasattr(args, key):
                setattr(args, key, value)
        
        # 加载智能合约配置
        smart_contract_cfg = config_payload.get("smart_contract", {})
        if smart_contract_cfg:
            if "enabled" in smart_contract_cfg:
                # 确保布尔值正确转换
                enabled_value = smart_contract_cfg["enabled"]
                if isinstance(enabled_value, bool):
                    args.use_smart_contract = enabled_value
                elif isinstance(enabled_value, str):
                    args.use_smart_contract = enabled_value.lower() in ['true', '1', 'yes']
                else:
                    args.use_smart_contract = bool(enabled_value)
            if "aggregation_method" in smart_contract_cfg:
                args.aggregation_method = smart_contract_cfg["aggregation_method"]
            if "min_clients_ratio" in smart_contract_cfg:
                args.smart_contract_min_clients_ratio = smart_contract_cfg["min_clients_ratio"]
            if "max_clients" in smart_contract_cfg:
                args.smart_contract_max_clients = smart_contract_cfg["max_clients"]
            print(f"[Config] 智能合约配置: enabled={args.use_smart_contract} (type: {type(args.use_smart_contract)}), method={args.aggregation_method}")
        else:
            print("[Config] 未找到智能合约配置，使用默认值")
        
        # 加载去中心化配置
        decentralized_cfg = config_payload.get("decentralized", {})
        if decentralized_cfg:
            if "p2p_enabled" in decentralized_cfg:
                p2p_enabled = decentralized_cfg["p2p_enabled"]
                if isinstance(p2p_enabled, bool):
                    args.p2p_enabled = p2p_enabled
                elif isinstance(p2p_enabled, str):
                    args.p2p_enabled = p2p_enabled.lower() in ['true', '1', 'yes']
                else:
                    args.p2p_enabled = bool(p2p_enabled)
            if "p2p_mode" in decentralized_cfg:
                args.p2p_mode = decentralized_cfg["p2p_mode"]
            if "broadcast_timeout" in decentralized_cfg:
                args.p2p_broadcast_timeout = decentralized_cfg["broadcast_timeout"]
            if "auto_publish" in decentralized_cfg:
                args.p2p_auto_publish = decentralized_cfg["auto_publish"]
            print(f"[Config] 去中心化配置: p2p_enabled={args.p2p_enabled}, mode={getattr(args, 'p2p_mode', 'local')}")
        else:
            print("[Config] 未找到去中心化配置，使用默认P2P设置")
        
        # 加载区块链存储配置
        blockchain_cfg = config_payload.get("blockchain_storage", {})
        if blockchain_cfg:
            if "enabled" in blockchain_cfg:
                enabled_value = blockchain_cfg["enabled"]
                if isinstance(enabled_value, bool):
                    args.blockchain_storage_enabled = enabled_value
                elif isinstance(enabled_value, str):
                    args.blockchain_storage_enabled = enabled_value.lower() in ['true', '1', 'yes']
                else:
                    args.blockchain_storage_enabled = bool(enabled_value)
            if "lightweight" in blockchain_cfg:
                args.blockchain_lightweight = blockchain_cfg["lightweight"]
            if "disk_storage" in blockchain_cfg:
                args.blockchain_disk_storage = blockchain_cfg["disk_storage"]
            if "keep_recent_blocks" in blockchain_cfg:
                args.blockchain_keep_recent = blockchain_cfg["keep_recent_blocks"]
            if "storage_dir" in blockchain_cfg:
                args.blockchain_storage_dir = blockchain_cfg["storage_dir"]
            if "storage_mode" in blockchain_cfg:
                args.blockchain_storage_mode = blockchain_cfg["storage_mode"]
            if "circular_mode" in blockchain_cfg:
                args.blockchain_circular_mode = blockchain_cfg["circular_mode"]
            if "circular_length" in blockchain_cfg:
                args.blockchain_circular_length = blockchain_cfg["circular_length"]
            print(f"[Config] 区块链存储配置: enabled={args.blockchain_storage_enabled}, lightweight={args.blockchain_lightweight}, circular_mode={getattr(args, 'blockchain_circular_mode', False)}")
        else:
            print("[Config] 未找到区块链存储配置，使用默认值")
        
        attack_cfg = config_payload.get("attack", {})
        if attack_cfg and not attack_cfg.get("malicious_ids") and attack_cfg.get("num_malicious"):
            num_malicious = int(attack_cfg["num_malicious"])
            attack_cfg["malicious_ids"] = list(range(num_malicious))
        args.attack_config = attack_cfg
        
        # 加载NPC Deputy配置
        npc_deputy_cfg = config_payload.get("npc_deputy", {})
        if npc_deputy_cfg:
            if "enabled" in npc_deputy_cfg:
                enabled_value = npc_deputy_cfg["enabled"]
                if isinstance(enabled_value, bool):
                    args.use_npc_deputy = enabled_value
                elif isinstance(enabled_value, str):
                    args.use_npc_deputy = enabled_value.lower() in ['true', '1', 'yes']
                else:
                    args.use_npc_deputy = bool(enabled_value)
            if "num_deputies" in npc_deputy_cfg:
                args.num_deputies = npc_deputy_cfg["num_deputies"]
            if "initial_stake" in npc_deputy_cfg:
                args.initial_stake = npc_deputy_cfg["initial_stake"]
            if "similarity_weight" in npc_deputy_cfg:
                args.similarity_weight = npc_deputy_cfg["similarity_weight"]
            if "stake_weight" in npc_deputy_cfg:
                args.stake_weight = npc_deputy_cfg["stake_weight"]
            if "reward_factor" in npc_deputy_cfg:
                args.reward_factor = npc_deputy_cfg["reward_factor"]
            if "penalty_factor" in npc_deputy_cfg:
                args.penalty_factor = npc_deputy_cfg["penalty_factor"]
            if "similarity_threshold" in npc_deputy_cfg:
                args.similarity_threshold = npc_deputy_cfg["similarity_threshold"]
            if "malicious_penalty_factor" in npc_deputy_cfg:
                args.malicious_penalty_factor = npc_deputy_cfg["malicious_penalty_factor"]
            if "random_init_stake" in npc_deputy_cfg:
                args.random_init_stake = npc_deputy_cfg["random_init_stake"]
            if "stake_range_min" in npc_deputy_cfg:
                args.stake_range_min = npc_deputy_cfg["stake_range_min"]
            if "stake_range_max" in npc_deputy_cfg:
                args.stake_range_max = npc_deputy_cfg["stake_range_max"]
            if "deputy_performance_weight" in npc_deputy_cfg:
                args.deputy_performance_weight = npc_deputy_cfg["deputy_performance_weight"]
            if "deputy_stake_weight" in npc_deputy_cfg:
                args.deputy_stake_weight = npc_deputy_cfg["deputy_stake_weight"]
            if "keep_recent_rounds" in npc_deputy_cfg:
                args.npc_deputy_keep_recent_rounds = npc_deputy_cfg["keep_recent_rounds"]
            if "gradient_storage_dir" in npc_deputy_cfg:
                args.npc_deputy_storage_dir = npc_deputy_cfg["gradient_storage_dir"]
            # DPOS新增配置参数
            if "deputy_max_tenure" in npc_deputy_cfg:
                args.deputy_max_tenure = npc_deputy_cfg["deputy_max_tenure"]
            if "deputy_block_reward" in npc_deputy_cfg:
                args.deputy_block_reward = npc_deputy_cfg["deputy_block_reward"]
            if "deputy_storage_reward_ratio" in npc_deputy_cfg:
                args.deputy_storage_reward_ratio = npc_deputy_cfg["deputy_storage_reward_ratio"]
            if "deputy_validation_failure_threshold" in npc_deputy_cfg:
                args.deputy_validation_failure_threshold = npc_deputy_cfg["deputy_validation_failure_threshold"]
            if "deputy_failure_penalty" in npc_deputy_cfg:
                args.deputy_failure_penalty = npc_deputy_cfg["deputy_failure_penalty"]
            if "deputy_slash_threshold" in npc_deputy_cfg:
                args.deputy_slash_threshold = npc_deputy_cfg["deputy_slash_threshold"]
            if "deputy_slash_ratio" in npc_deputy_cfg:
                args.deputy_slash_ratio = npc_deputy_cfg["deputy_slash_ratio"]
            print(f"[Config] NPC Deputy配置: enabled={getattr(args, 'use_npc_deputy', False)}, num_deputies={getattr(args, 'num_deputies', 3)}, similarity_threshold={getattr(args, 'similarity_threshold', 0.3)}, random_init_stake={getattr(args, 'random_init_stake', True)}, keep_recent_rounds={getattr(args, 'npc_deputy_keep_recent_rounds', 5)}, max_tenure={getattr(args, 'deputy_max_tenure', 5)}, block_reward={getattr(args, 'deputy_block_reward', 5.0)}")
        else:
            args.use_npc_deputy = False
            print("[Config] 未找到NPC Deputy配置，使用默认值（禁用）")
        
        print(f"[Config] 已加载配置文件: {args.config_path}")
    else:
        args.attack_config = {}
        # 设置默认智能合约配置
        if not hasattr(args, 'smart_contract_min_clients_ratio'):
            args.smart_contract_min_clients_ratio = 0.5
        if not hasattr(args, 'smart_contract_max_clients'):
            args.smart_contract_max_clients = None
        print("[Config] 未使用外部配置文件或文件为空。")
    
    # 全局默认值（确保属性存在）
    if not hasattr(args, 'smart_contract_min_clients_ratio'):
        args.smart_contract_min_clients_ratio = 0.5
    if not hasattr(args, 'smart_contract_max_clients'):
        args.smart_contract_max_clients = None
    if not hasattr(args, 'p2p_broadcast_timeout'):
        args.p2p_broadcast_timeout = 2.0
    if not hasattr(args, 'blockchain_storage_enabled'):
        args.blockchain_storage_enabled = False
    if not hasattr(args, 'blockchain_lightweight'):
        args.blockchain_lightweight = True
    if not hasattr(args, 'blockchain_disk_storage'):
        args.blockchain_disk_storage = False
    if not hasattr(args, 'blockchain_keep_recent'):
        args.blockchain_keep_recent = 5
    if not hasattr(args, 'blockchain_storage_dir'):
        args.blockchain_storage_dir = "blockchain_storage"
    if not hasattr(args, 'blockchain_storage_mode'):
        args.blockchain_storage_mode = "summary"
    if not hasattr(args, 'blockchain_circular_mode'):
        args.blockchain_circular_mode = False
    if not hasattr(args, 'blockchain_circular_length'):
        args.blockchain_circular_length = None
    if not hasattr(args, 'use_npc_deputy'):
        args.use_npc_deputy = False
    if not hasattr(args, 'num_deputies'):
        args.num_deputies = 3
    if not hasattr(args, 'initial_stake'):
        args.initial_stake = 100.0
    if not hasattr(args, 'similarity_weight'):
        args.similarity_weight = 0.7
    if not hasattr(args, 'stake_weight'):
        args.stake_weight = 0.3
    if not hasattr(args, 'reward_factor'):
        args.reward_factor = 0.1
    if not hasattr(args, 'penalty_factor'):
        args.penalty_factor = 0.05
    if not hasattr(args, 'similarity_threshold'):
        args.similarity_threshold = 0.3
    if not hasattr(args, 'malicious_penalty_factor'):
        args.malicious_penalty_factor = 0.2
    if not hasattr(args, 'random_init_stake'):
        args.random_init_stake = True
    if not hasattr(args, 'stake_range_min'):
        args.stake_range_min = 50.0
    if not hasattr(args, 'stake_range_max'):
        args.stake_range_max = 150.0
    if not hasattr(args, 'deputy_performance_weight'):
        args.deputy_performance_weight = 0.6
    if not hasattr(args, 'deputy_stake_weight'):
        args.deputy_stake_weight = 0.4
    if not hasattr(args, 'npc_deputy_keep_recent_rounds'):
        args.npc_deputy_keep_recent_rounds = 5
    if not hasattr(args, 'npc_deputy_storage_dir'):
        args.npc_deputy_storage_dir = "blockchain_gradient_storage"
    # DPOS新增配置参数默认值
    if not hasattr(args, 'deputy_max_tenure'):
        args.deputy_max_tenure = 5
    if not hasattr(args, 'deputy_block_reward'):
        args.deputy_block_reward = 5.0
    if not hasattr(args, 'deputy_storage_reward_ratio'):
        args.deputy_storage_reward_ratio = 0.3
    if not hasattr(args, 'deputy_validation_failure_threshold'):
        args.deputy_validation_failure_threshold = 3
    if not hasattr(args, 'deputy_failure_penalty'):
        args.deputy_failure_penalty = 2.0
    if not hasattr(args, 'deputy_slash_threshold'):
        args.deputy_slash_threshold = 0.5
    if not hasattr(args, 'deputy_slash_ratio'):
        args.deputy_slash_ratio = 0.1

    beijing_tz = pytz.timezone('Asia/Shanghai')
    timestamp="run_{date}".format(date=datetime.now(beijing_tz).strftime('%Y-%m-%d_%H-%M'))
    log_name = str(args.algorithm) + timestamp
    print("log_name:{}".format(log_name))
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="flskl_base",

    #     # track hyperparameters and run metadata
    #     config={
    #     "architecture": "stgcn",
    #     "dataset": "ntu_D",
    #     },
    #     settings=dict(init_timeout=120),
    #     mode="offline",
    #     id=log_name
    # )

    

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    print("Componenr: fedAvg25  +  condessed  +  halp  + G_contrast")
    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    print("Pretrain local learing rate: {}".format(args.pretrain_lr))
    print("Linear local learing rate: {}".format(args.linear_lr))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Using DP: {}".format(args.privacy))
    if args.privacy:
        print("Sigma for DP: {}".format(args.dp_sigma))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("Feature Augment: {}".format(args.feature_augment))
    print("Data Augment: {}".format(args.data_augment))
    print("Feature Augment Number : {}".format(args.feature_number))
    print("Total number of new clients: {}".format(args.num_new_clients))
    print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch))
    print("=" * 50)

    run(args)