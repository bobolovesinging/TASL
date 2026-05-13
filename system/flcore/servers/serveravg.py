import time
import copy
import os
from sklearn.manifold import TSNE
from flcore.clients.clientour import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
from utils.model_utils import *
from utils.DSTformer import *
from utils.model_action import *
import wandb
import logging
from sklearn.preprocessing import StandardScaler
from flcore.blockchain.smart_contract import NPCDeputySmartContract


from scipy.interpolate import griddata, RectBivariateSpline
from scipy.interpolate import CubicSpline
from scipy.interpolate import Rbf
from scipy.interpolate import interp1d

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
matplotlib.use('Agg')

from ..trainmodel.w_transforms  import (f_croppad_torch_batch, f_cropresize_torch_batch, f2_linear)
from ..trainmodel.linear_extrapolation import get_first_frame_condition

class FedAvg(Server):
    """
    FedAvg算法实现
    使用智能合约协调器（SmartContractCoordinator）进行训练协调
    模型聚合由智能合约完成，协调器只负责流程协调
    """
    def __init__(self, args, times):
        super().__init__(args, times)
        self.avgstage = args.avgstage
        self.inpo =args.interpolate
        self.in_t =args.inter_t
        self.in_num =args.inter_num
        self.skl = args.skloss
        self.pth_path = '../log/mdpath/'+str(self.in_t)+"_"+str(self.in_num)+"/"

        # logging.basicConfig(level=logging.INFO, filename='log/base.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
        # self.logger = logging.getLogger("base_log")
        
        self.c3d =False
        self.crossentropyloss = nn.CrossEntropyLoss()
        self.jw =args.jwaloss
        self.model_optimizer = torch.optim.SGD(
                self.global_model.parameters(),
                lr=0.01,
                weight_decay=0.0005,
                momentum=0.9,
            )
        self.feature_augment = args.feature_augment
        self.data_augment = args.data_augment
        self.feature_number = args.feature_number
        print("fa:{}".format(self.feature_augment))
        print("da:{}".format(self.data_augment ))
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        # model: optimizer


        # self.optimizer_c = torch.optim.Adam(
        #     self.global_classifier.parameters(),
        #     lr=1e-4)
        # self.learning_rate_scheduler_c = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_c, args.global_rounds)
        
        
        
        self.set_slow_clients()
        self.set_clients(clientAVG)
        

        # self.global_centroids = nn.Linear(args.hsize, args.N_centroids, bias=False).cuda()
        # self.set_global_centroids()

        print("\nJoin ratio / total clients: {} / {}".format(self.join_ratio,self.num_clients))
        print("Finished creating server and clients.")

        # self.load_model()
        # load models:
        self.cepoch = args.cepoch

        self.Budget = []
        self.work_dir="server"
        
        # TensorBoard 可视化（只记录准确率）
        tb_log_dir = self.get_tb_log_dir(self.algorithm)
        os.makedirs(tb_log_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(tb_log_dir)
        print(f"[TensorBoard] 日志目录: {tb_log_dir}")

        self.server_cnum = args.server_cnum 
        self.server_learningrate= args.server_condensegrate
        pre_lr1 = self.server_learningrate # batchsize(512=4*128 sqrt(4))
        self.optimizer = torch.optim.SGD(
            self.global_model.parameters(),
            weight_decay=1e-4,
            lr=pre_lr1,
            momentum=0.9,
            nesterov=False)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
        self.send_pseudo_labels()

        self.bkg_pose_list = []


    def send_pseudo_labels(self):
        server_cdata = []
        self.selected_clients = self.select_clients()
        server_clabel = []
        # 获取每个客户端的聚类数据：
        for client in self.selected_clients:
            sdata =  client.init_condensed_data()
            # print("sdata:{}".format(sdata))
            s_cx = sdata['x']
            s_cy = sdata['y']
            server_cdata.append(s_cx)
            server_clabel.append(s_cy)
            # print("client:{} cluster".format(client.id))
        
        server_data = torch.concat(server_cdata, dim=0)
        # sf:服务器聚类中心，s_label：服务器聚类标签,cp_label：得到的每个客户端聚类中心的聚类标签
        cnum = self.server_cnum
        sf,s_label,cp_plabel = self.kmeans(server_data, cnum)
        # sf与s_label的长度是一样的
        # print("s_label:{}".format(len(s_label)))
        # print("sf:{}".format(len(sf)))
        lengths = [len(a) for a in server_clabel]
        # print("lengths:{}".format(len(lengths)))
        # 把客户端的聚类中心的聚类标签传回给每个客户端
        split_arrays = np.split(cp_plabel, np.cumsum(lengths)[:-1])
        for index,client in enumerate(self.selected_clients):
            client.get_pseudo_labels(split_arrays[index])
            # print("client:{} get伪标签".format(client.id))
        
    def train(self):
        mode = 1
        #self.para_frozen(self.global_model , mode)

        start_epoch = 0

        # self.load_model(self.cepoch)
        # print(self.cepoch)
        # start_epoch =self.cepoch+1 
        # for index, client in enumerate(self.selected_clients):
        #     client.load_opacl(self.cepoch)
        for i in range(start_epoch ,self.global_rounds):
            g_rounds = i
            s_t = time.time()
            self.current_round = i  # 更新当前轮次（用于智能合约）
            
            # NPC Deputy机制：获取当前deputy列表
            # Round 0: 初始选举（基于权益）
            # Round > 0: 使用上一轮更新后的deputy列表（如果存在）
            if self.use_npc_deputy and self.deputy_manager is not None:
                if i == 0:
                    # 第一轮：初始选举deputy（基于权益）
                    deputies = self.elect_deputies(i)
                    print(f"[NPCDeputy] Round {i}: 初始选举出 {len(deputies)} 个NPC deputy: {deputies}")
                else:
                    # 后续轮次：使用上一轮更新后的deputy列表
                    deputies = self.deputy_manager.get_deputies()
                    if not deputies:
                        # 如果没有deputy列表，重新选举（不应该发生）
                        deputies = self.elect_deputies(i)
                        print(f"[NPCDeputy] Round {i}: 未找到deputy列表，重新选举: {deputies}")
                    else:
                        # 确保deputy历史记录已更新
                        if i not in self.deputy_manager.deputy_history:
                            self.deputy_manager.deputy_history[i] = deputies.copy()
                        print(f"[NPCDeputy] Round {i}: 使用deputy列表: {deputies}")
            
            # 选择参与训练的客户端（排除deputy节点）
            self.selected_clients = self.select_clients()
            self.notify_clients_round(i, self.selected_clients)
            
            # 通知所有客户端（包括deputy）当前轮次，但只有非deputy客户端参与训练
            if self.use_npc_deputy and self.deputy_manager is not None:
                deputies = self.deputy_manager.get_deputies()
                # Deputy节点也需要接收模型参数（用于后续处理梯度）
                for client in self.clients:
                    if client.id in deputies:
                        # Deputy节点只接收模型，不参与训练
                        if hasattr(client, "cache_model_state"):
                            client.cache_model_state(self.global_model)
                        client.set_parameters(self.global_model)
                        client.set_round(i)
                        print(f"[NPCDeputy] Deputy节点 {client.id} 已接收模型参数（不参与训练）")
    
            self.send_models()
            # if i%5 == 0:
            #     self.save_model(self.global_model,i)
            
            # if self.c3d:
            #     if i!=0:
            #         self.send_3ddata()
            # contrast loss
            if i%self.eval_gap == 0:                
                print("\nEvaluate global model")
                # self.knn_evaluate()
                # self.linear_train()
                self.evaluate()
                
                # 记录准确率到 TensorBoard
                if hasattr(self, 'tb_writer') and hasattr(self, 'rs_test_acc') and len(self.rs_test_acc) > 0:
                    self.tb_writer.add_scalar('Accuracy', self.rs_test_acc[-1], i)

            num_samples = []
            # losses = []

            print("\n-------------Round number: {}-------------".format(i))

            if i<self.avgstage:
                # 非deputy客户端进行训练
                for index, client in enumerate(self.selected_clients):
                    loss,ns= client.pretrain_avg(i)  
                
                # NPC Deputy机制：DPOS共识流程
                if self.use_npc_deputy and self.deputy_manager is not None:
                    deputies = self.deputy_manager.get_deputies()
                    if not deputies:
                        print(f"[NPCDeputy] Round {i}: 没有deputy节点，回退到传统聚合")
                        self.receive_models()
                        self.aggregate_parameters()
                    else:
                        # 步骤1：所有deputy节点接收并存储梯度（只存储，不聚合）
                        for deputy_id in deputies:
                            deputy_client = next((c for c in self.clients if c.id == deputy_id), None)
                            if deputy_client and hasattr(deputy_client, '_deputy_receive_and_store_gradients_only'):
                                deputy_client._deputy_receive_and_store_gradients_only(i)
                        
                        # 步骤2：选择执行deputy创建交易（第0轮权益最高，之后FIFO轮换）
                        executor_deputy_id = self.deputy_manager.get_executor_deputy(i)
                        if executor_deputy_id is None:
                            print(f"[NPCDeputy] Round {i}: 无法确定权益最大的deputy，回退到传统聚合")
                            self.receive_models()
                            self.aggregate_parameters()
                        else:
                            creator_deputy_client = next((c for c in self.clients if c.id == executor_deputy_id), None)
                            transaction_data = None
                            
                            if creator_deputy_client and hasattr(creator_deputy_client, '_deputy_create_transaction'):
                                transaction_data = creator_deputy_client._deputy_create_transaction(i)
                                
                                if transaction_data is not None:
                                    print(f"[NPCDeputy] Round {i}: Deputy节点 {executor_deputy_id}（执行deputy）创建交易")
                                    print(f"[NPCDeputy] Round {i}: 执行deputy不参与投票，由其他deputy节点验证和投票")
                                    
                                    # 步骤3：广播交易提案给其他deputy节点（执行deputy不参与投票）
                                    creator_deputy_client.broadcast_transaction_proposal(i, transaction_data)
                                    
                                    # 步骤4：等待其他deputy验证和投票（带超时和重试机制）
                                    max_wait_time = getattr(self.args, 'consensus_timeout', 5.0)  # 默认5秒超时
                                    max_retries = getattr(self.args, 'consensus_max_retries', 2)  # 默认最多重试2次
                                    consensus_reached = False
                                    approve_votes = 0
                                    total_votes = 0
                                    
                                    for retry in range(max_retries + 1):
                                        if retry > 0:
                                            print(f"[NPCDeputy] Round {i}: 共识未达成，重试 {retry}/{max_retries}")
                                        
                                        # 让其他deputy节点处理消息（接收transaction_proposal并投票）
                                        for deputy_id in deputies:
                                            if deputy_id != executor_deputy_id:  # 除了执行deputy
                                                deputy_client = next((c for c in self.clients if c.id == deputy_id), None)
                                                if deputy_client and hasattr(deputy_client, 'sync_stake_and_deputy_from_p2p'):
                                                    # 处理P2P消息（包括transaction_proposal）
                                                    deputy_client.sync_stake_and_deputy_from_p2p(i)
                                        
                                        # 等待deputy验证和投票（排除执行deputy）
                                        start_wait = time.time()
                                        while time.time() - start_wait < max_wait_time:
                                            # 检查是否达到共识（排除执行deputy的投票）
                                            consensus_reached, approve_votes, total_votes = self.deputy_manager.check_transaction_consensus(
                                                i, threshold_ratio=2.0/3.0, executor_deputy_id=executor_deputy_id
                                            )
                                            if consensus_reached:
                                                break
                                            time.sleep(0.1)  # 每100ms检查一次
                                        
                                        if consensus_reached:
                                            break
                                        
                                        # 如果未达成共识，检查是否有deputy未投票（视为故障，排除执行deputy）
                                        if retry < max_retries:
                                            deputies = self.deputy_manager.get_deputies()
                                            votes = self.deputy_manager.transaction_votes.get(i, {})
                                            voting_deputies = [d for d in deputies if d != executor_deputy_id]
                                            missing_votes = [d for d in voting_deputies if d not in votes]
                                            if missing_votes:
                                                print(f"[NPCDeputy] Round {i}: 以下deputy未投票: {missing_votes}，记录为故障")
                                                # 记录故障deputy（可以后续惩罚）
                                                for deputy_id in missing_votes:
                                                    self.deputy_manager.record_validation_failure(deputy_id)
                                    
                                    # 步骤5：最终检查共识结果（排除执行deputy）
                                    if not consensus_reached:
                                        consensus_reached, approve_votes, total_votes = self.deputy_manager.check_transaction_consensus(
                                            i, threshold_ratio=2.0/3.0, executor_deputy_id=executor_deputy_id
                                        )
                                    
                                    if consensus_reached:
                                        print(f"[NPCDeputy] Round {i}: 交易达成共识 ({approve_votes}/{total_votes} 通过)，确认交易")
                                        
                                        # 步骤7：广播确认的交易到所有deputy的区块链（先确认，再执行奖励/惩罚）
                                        creator_deputy_client.broadcast_confirmed_transaction(i, transaction_data)
                                        
                                        # 步骤8：所有deputy接收并保存交易到本地区块链
                                        for deputy_id in deputies:
                                            deputy_client = next((c for c in self.clients if c.id == deputy_id), None)
                                            if deputy_client and hasattr(deputy_client, 'sync_stake_and_deputy_from_p2p'):
                                                deputy_client.sync_stake_and_deputy_from_p2p(i)
                                        
                                        # 步骤9：等待所有deputy接收确认的交易后，再执行奖励/惩罚
                                        time.sleep(0.2)  # 给deputy时间接收和保存交易
                                        
                                        # 步骤10：应用交易（打分和奖罚）- 在区块链确认后执行
                                        final_similarities = transaction_data.get("final_similarities", {})
                                        valid_clients = transaction_data.get("valid_clients", [])
                                        malicious_clients = transaction_data.get("malicious_clients", [])
                                        creator_deputy_client._score_and_reward_clients(i, final_similarities, valid_clients, malicious_clients)
                                        
                                        # 步骤9：应用聚合结果到全局模型
                                        serialized_aggregated = transaction_data.get("aggregated_params", {})
                                        if serialized_aggregated:
                                            # 反序列化聚合参数（从list转回tensor）
                                            aggregated_params = {}
                                            for name, value in serialized_aggregated.items():
                                                if isinstance(value, list):
                                                    aggregated_params[name] = torch.tensor(value, dtype=torch.float32)
                                                else:
                                                    aggregated_params[name] = value
                                            
                                            self._apply_aggregated_parameters(aggregated_params)
                                            print(f"[NPCDeputy] Round {i}: 已应用聚合结果到全局模型")
                                            
                                            # 释放临时变量
                                            del aggregated_params
                                        
                                        # 步骤11：立即清理梯度数据，释放内存（区块确认后）
                                        if isinstance(creator_deputy_client.local_smart_contract, NPCDeputySmartContract):
                                            creator_deputy_client.local_smart_contract.clear_round_updates(i)
                                            print(f"[NPCDeputy] Round {i}: 已清理梯度数据，释放内存")
                                        
                                        # 清理所有deputy客户端的上一轮全局模型状态（释放大量内存）
                                        for client in self.clients:
                                            if hasattr(client, 'previous_global_model_state') and client.previous_global_model_state is not None:
                                                client.previous_global_model_state.clear()
                                                client.previous_global_model_state = None
                                        
                                        # 同步权益和deputy信息到所有客户端
                                        creator_deputy_client.broadcast_stake_and_deputy(i)
                                        for client in self.clients:
                                            if client.id != executor_deputy_id:
                                                client.sync_stake_and_deputy_from_p2p(i)
                                        
                                        # 清理P2P消息队列和投票记录，释放内存
                                        self.deputy_manager.clear_transaction_votes(i)
                                        if hasattr(creator_deputy_client, 'p2p_network'):
                                            for client in self.clients:
                                                if hasattr(client, 'p2p_network'):
                                                    client.p2p_network.clear(client.id)
                                        
                                        # 释放transaction_data中的大对象（序列化的梯度数据）
                                        if 'client_gradients' in transaction_data:
                                            del transaction_data['client_gradients']
                                        if 'aggregated_gradient' in transaction_data:
                                            del transaction_data['aggregated_gradient']
                                        
                                        # 清理所有deputy客户端的梯度存储（只保留当前轮和上一轮）
                                        for client in self.clients:
                                            if isinstance(client.local_smart_contract, NPCDeputySmartContract):
                                                # 强制清理旧轮次数据
                                                if hasattr(client.local_smart_contract, 'gradient_storage'):
                                                    current_round = i
                                                    rounds_to_remove = [r for r in client.local_smart_contract.gradient_storage.keys() 
                                                                      if r < current_round - 1]  # 只保留当前轮和上一轮
                                                    for round_num in rounds_to_remove:
                                                        if round_num in client.local_smart_contract.gradient_storage:
                                                            gradients = client.local_smart_contract.gradient_storage[round_num]
                                                            for grad in gradients:
                                                                if grad.gradient_params:
                                                                    grad.gradient_params.clear()
                                                                if grad.model_params:
                                                                    grad.model_params.clear()
                                                            gradients.clear()
                                                            del gradients
                                                            del client.local_smart_contract.gradient_storage[round_num]
                                        
                                        # 清理CUDA缓存
                                        if hasattr(torch.cuda, 'empty_cache'):
                                            torch.cuda.empty_cache()
                                        
                                        # 强制垃圾回收
                                        import gc
                                        gc.collect()
                                    else:
                                        print(f"[NPCDeputy] Round {i}: 交易未达成共识 ({approve_votes}/{total_votes} 通过)，回退到传统聚合")
                                        self.receive_models()
                                        self.aggregate_parameters()
                                else:
                                    print(f"[NPCDeputy] Round {i}: Deputy节点 {executor_deputy_id} 创建交易失败，回退到传统聚合")
                                    self.receive_models()
                                    self.aggregate_parameters()
                            else:
                                print(f"[NPCDeputy] Round {i}: 无法找到执行deputy客户端，回退到传统聚合")
                                self.receive_models()
                                self.aggregate_parameters()
                else:
                    self.receive_models()
                    self.aggregate_parameters()
                
                self.Budget.append(time.time() - s_t)
                print('-'*25, 'time cost', '-'*25, self.Budget[-1])
                continue

            if i>self.avgstage:
                self.sendv()
            cdata_all=[]
            clabel_all=[]
            total_loss=0
            
            for index, client in enumerate(self.selected_clients):
                ns, cdata, clabel = client.pretrain(i)
                num_samples.append(ns)
                cdata_all.append(cdata)
                clabel_all.append(clabel)

            
            self.receive_condessed()
            if self.feature_augment == "True":
                agg_covmatrix = self.receive_covmatrix()
                # clients_covmatrix = self.uploaded_class_client_covmatrix
            else:
                agg_covmatrix = {}

            
            cdata_all = torch.cat(cdata_all, dim=0).cpu()
            clabel_all = torch.cat(clabel_all, dim=0)                
            cdata_dataset = TensorDataset(cdata_all, clabel_all)
            cdata_dataloader = DataLoader(cdata_dataset, batch_size=self.batch_size, shuffle=True)
            # 边界姿态提取（聚类）
            if self.bkg_pose_list == []: 
                self.load_pose_cluster(cdata_all)
                self.receive_traj_mat()
                client_traj_mat_list= np.concatenate(self.uploaded_w_matrix, axis=0)
                server_traj_mat_list= self.traj_mat_list
                self.traj_mat_list = np.concatenate((client_traj_mat_list,server_traj_mat_list),axis=0)
            # print("self.traj_mat_list.shape:{}".format(self.traj_mat_list.shape))
            self.global_model.train()
            self.global_model.cuda()
            # model_optimizer = torch.optim.Adam(self.global_model.parameters(), lr=0.001, weight_decay=1e-4)
            # self.model_optimizer.zero_grad()
            
            feature_d_all = []
            label_d_all = []
            
            feature_a_all = []
            label_a_all = []

            for gl in range(1):
                for i, (datas ,label) in enumerate(cdata_dataloader):
                    data = datas.type(torch.FloatTensor).cuda()
                    # print(1)
                    # data = torch.rand_like(datas, dtype=torch.float).cuda()
                    # data = torch.zeros_like(data)
                    n,c,v,t,m = data.size()
                    # print(data.size())
                    # input1
                    if self.data_augment == "True":
                        input_aug = self.apply_W_all(data,recover_tag='ABL_linearextrap_beta', resample_tag='cropresize0.7')
                        input_aug = self.merge_raw_and_aug(data,input_aug, augratio=0.75)
                        input_aug = torch.cat([input_aug],0)
                    else:
                        input_aug = data
                    # print(input_aug.shape)
                    # print(data_add.shape)
                    input1 = shear(crop(input_aug))
                    # input1 = shear(crop(data))
                    input1 = random_rotate(input1)
                    input1 = random_spatial_flip(input1)
                    #feat1 = self.encoder(input1)
                    
                    
                    # input2
                    input2 = shear(crop(input_aug))
                    # input2 = shear(crop(data))
                    input2 = random_rotate(input2)
                    input2 = random_spatial_flip(input2)
                    
                    # with torch.no_grad():
                        
                    #     feat_data, _, _= self.global_model.encoder_q(data_add)
                    #     feat_augdata, _, _= self.global_model.encoder_q(input_aug)
                        
                    #     feature_d_all.append(feat_data.detach())
                    #     label_d_all.append(label.detach())       
                    #     feature_a_all.append(feat_augdata.detach())
                    #     label_a_all.append(label.detach())   

                    # input3 new aug:

                    if self.feature_augment == "True":
                        stgcn_logits, logits, target ,aug_logits,aug_labels = self.global_model(input1, input2,server=True,covmatrix=agg_covmatrix,scnnumber=self.server_cnum,label=label,fnum=self.feature_number)
                        aug_loss = 0
                        for i, (logit, aglabel) in enumerate(zip(aug_logits,aug_labels)):
                            aug_loss+= self.crossentropyloss(logit, aglabel) 
                        loss = self.crossentropyloss(logits, target) + aug_loss
                    else:
                        stgcn_logits, logits, target  = self.global_model(input1, input2,server=True,covmatrix=agg_covmatrix,scnnumber=self.server_cnum,label=label)
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

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        
        
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        # self.save_global_model()
        
        # 关闭 TensorBoard writer
        if hasattr(self, 'tb_writer'):
            self.tb_writer.close()
            print(f"[TensorBoard] 日志已保存到: {self.tb_writer.log_dir}")

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print("\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
    
    def receive_covmatrix(self):
        assert (len(self.selected_clients) > 0)
    
        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))
        
        self.uploaded_class_client_covmatrix = []
        
        tot_samples = 0
        self.class_counts = []
        for i, client in enumerate(active_clients):
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.class_counts = client.class_counts
                self.uploaded_class_client_covmatrix.append(client.class_covmatrix)
       
        
        aggregate_covmatrix = {}
        for class_idx in range(self.server_cnum):

            
            sum_mean = 0
            total_samples = np.sum([client.class_counts[class_idx] for i,client in enumerate(active_clients) if class_idx in client.class_counts ] )
            combined_mean  = np.sum([client.class_counts[class_idx] * client.mean_matrix[class_idx] for i,client in enumerate(active_clients) if class_idx in client.class_counts ]  , axis=0) / total_samples
            sum_np = 0
    
            sum_matrix = [ (client.class_counts[class_idx] * client.class_covmatrix[class_idx].cpu())   for i,client in enumerate(active_clients) if class_idx in client.class_counts ]
            stack = torch.stack((sum_matrix ),dim=0)
            weighted_matrix =  torch.sum( stack ,dim = 0)
            combined_cov_matrix = weighted_matrix / total_samples
            combined_cov_matrix += np.sum([client.class_counts[class_idx] * np.outer(client.mean_matrix[class_idx]   - combined_mean, client.mean_matrix[class_idx]  - combined_mean) for i,client in enumerate(active_clients) if class_idx in client.class_counts], axis=0) / total_samples
            aggregate_covmatrix[class_idx]= combined_cov_matrix

        return aggregate_covmatrix 
    
    def receive_traj_mat(self):
        assert (len(self.selected_clients) > 0)
    
        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))
        
        self.uploaded_w_matrix = []
        
        tot_samples = 0
        self.class_counts = []
        for i, client in enumerate(active_clients):
            self.uploaded_w_matrix.append(client.traj_mat_list)

    def apply_W_all(self, data_src, recover_tag, resample_tag):
        '''
        W_recover, W_resample is a random variable, not a fixed matrix.
        so we have to sample a W for each input.
        '''
        data_src = self.apply_W_recover(data_src, recover_tag)
        data_src = self.apply_W_resample(data_src, resample_tag)
        return data_src
        
    def apply_W_recover(self, data_src, recover_tag):
        if recover_tag=='none':
            return data_src
        elif recover_tag=='croppad3':
            return f_croppad_torch_batch(data_src, 3)
        elif recover_tag=='croppad6':
            return f_croppad_torch_batch(data_src, 6)

        elif recover_tag=='ABL_linearextrap_beta':

            N,C,T,V,M = data_src.shape
            t0 = int(np.random.beta(0.1,0.1)*(T//2))
            data_output_interp = self.get_f0_and_linear_interp_given_t0(data_src, t0)
            data_output_interp = f2_linear(data_output_interp, self.traj_mat_list, None)
            return data_output_interp

        # do abl 
        elif recover_tag=='abl_linear':
            data_output_interp = f2_linear(data_src, self.traj_mat_list, None)
            return data_output_interp

        elif recover_tag=='abl_extrapbeta':

            N,C,T,V,M = data_src.shape
            t0 = int(np.random.beta(0.1,0.1)*(T//2))
            data_output_interp = self.get_f0_and_linear_interp_given_t0(data_src, t0)
            return data_output_interp

        else:
            raise ValueError()

    def apply_W_resample(self, data_src, resample_tag):
        if resample_tag=='none':
            return data_src
        elif resample_tag=='cropresize0.7':
            return f_cropresize_torch_batch(data_src, [0.7,1.0])
        elif resample_tag=='cropresize0.75':
            return f_cropresize_torch_batch(data_src, [0.75,1.0])

        elif resample_tag=='cropresize0.9':
            return f_cropresize_torch_batch(data_src, [0.9,1.0])

        elif resample_tag=='cropresize0.7_nonuni':
            return f_cropresize_torch_batch_nonuni(data_src, [0.7,1.0])


        else:
            raise ValueError()

    def mix_raw_and_aug(self, data_src, data_src_prior, augratio):
        bs = 96
        final_bs = 128  # 总样本数 = 原始 + 增强部分
        res = torch.zeros(final_bs, *data_src.shape[1:], device=data_src.device)

        # 保留所有原始数据
        res[:bs] = data_src

        # 从 data_src_prior 中随机选取 augratio * bs 条数据添加到后面
        aug_n = final_bs - bs
        idx_list = torch.randperm(data_src_prior.shape[0])[:aug_n].to(data_src.device)
        res[bs:] = data_src_prior[idx_list]

        return res,res[bs:]

    def merge_raw_and_aug(self, data_src, data_src_prior, augratio):
        res = torch.zeros_like(data_src)
        bs = data_src.shape[0]
        # assert bs==64
        order_idx_list = torch.randperm(bs).to(data_src.device)

        raw_data_ratio = 1-augratio 
        raw_data_n = int( (1-augratio)*bs )

        idx_list_1 = order_idx_list[0:raw_data_n]
        idx_list_2 = order_idx_list[raw_data_n:bs]
        

        res[idx_list_1] = data_src[idx_list_1]
        res[idx_list_2]  = data_src_prior[idx_list_2]
        
        return res 

    def get_f0_and_linear_interp_given_t0(self, data_src, t0):
        frame_bkg_estimate = get_first_frame_condition(data_src, self.bkg_pose_list)
        data_input = data_src
        N,C,T,V,M = data_src.shape
        
        st_int=0
        ed_int=T
        if t0 != 0:
            data_output_2frames = torch.cat([
                frame_bkg_estimate, data_input[:,:,st_int:st_int+1,:,:] ], 2)
            assert data_output_2frames.shape==(N,C,2,V,M)
            data_output_1st = self.resize_torch_interp_batch(data_output_2frames, t0)
            data_output_2nd = self.resize_torch_interp_batch(data_input[:,:,st_int:ed_int,:,:], T-t0)
            data_output = self.resize_torch_interp_batch(
                torch.cat([data_output_1st,data_output_2nd],2), T
            )
        else:
            data_output = data_input
        return data_output


    def get_f0_and_nn_pred_given_t0(self, data_src, t0):
        frame_bkg_estimate = get_first_frame_condition(data_src, self.bkg_pose_list)
        data_input = data_src
        N,C,T,V,M = data_src.shape
        
        st_int=0
        ed_int=T
        if t0 != 0:
            data_output_2frames = torch.cat([
                frame_bkg_estimate, data_input[:,:,st_int:st_int+1,:,:] ], 2)
            assert data_output_2frames.shape==(N,C,2,V,M)
            data_output = torch.zeros_like(data_src)
            
            data_output[:,:,0:1,:,:] = frame_bkg_estimate
            data_output[:,:,t0:T,:,:] = self.resize_torch_interp_batch(data_input[:,:,st_int:ed_int,:,:], T-t0)
            data_output.requires_grad=False 
            self.model_prior.eval()
            with torch.no_grad():
                # data_output_pred = self.model_prior(data_output)
                data_output_pred = self.model_prior.forward_eval_2p(data_output)
            num_person=1
            data_output[:,:,1:t0,:,:num_person] = data_output_pred[:,:,1:t0,:,:num_person]

            valid_idx_list_second_person = torch.where(data_input[:,:,:,:,1].sum(dim=[3,2,1]))[0]
            data_output[valid_idx_list_second_person,:,1:t0,:,1:2] = data_output_pred[valid_idx_list_second_person,:,1:t0,:,1:2]
            # data_output[:,:,1:t0,:,:] = data_output_pred[:,:,1:t0,:,:]
        else:
            data_output = data_input
        return data_output
        

    def load_pose_cluster(self,data):
        res_all = []
        st = 0
        data_all = data[:,:,st:st+1,:,:1]
        cluster_res_single = self.cluster_per_person(data_all,n_cluster=10)
        cluster_res_single = np.concatenate([cluster_res_single, cluster_res_single*0], 4)
        res_all=cluster_res_single
        bkg_clusters=res_all[:,:,:,:,:1]
        pose_clusters = np.concatenate([bkg_clusters, np.zeros_like(bkg_clusters)], 4)
        bkg_pose_list = torch.FloatTensor(pose_clusters)
        self.bkg_pose_list = bkg_pose_list.cuda()


        segment_list = [
        (0.0, 1.0), 
        (0.0, 0.5), (0.5, 1.0), (0.25, 0.75), (0.125, 0.625), (0.375, 0.875),
        (0.0, 0.75), (0.25, 1.0)
        ]

        conf_mat_list1, traj_mat_list1, traj_weight_list1= self.do_w_clustering(data,n_clusters=50, T1=0.1, topk=32, use_bkg=False, save_name=None, 
                        use_sample=False, segment_list=segment_list, bkg_pose_list=None)
        conf_mat_list, traj_mat_list, traj_weight_list = \
            conf_mat_list1, traj_mat_list1, traj_weight_list1
        conf_mat_list_bkg, traj_mat_list_bkg, traj_weight_list_bkg = conf_mat_list, traj_mat_list, traj_weight_list
        self.conf_mat_list = conf_mat_list
        self.traj_mat_list = traj_mat_list
        self.traj_weight_list = traj_weight_list

        self.conf_mat_list_bkg = conf_mat_list_bkg
        self.traj_mat_list_bkg = traj_mat_list_bkg
        self.traj_weight_list_bkg = traj_weight_list_bkg

        
    def cluster_per_person(self,data_all,n_cluster=5):
        n_cluster_bkg =  n_cluster
        # data = flatten_nctvm(data)
        bs = data_all.shape[0]
        x = data_all
        x = x.reshape((bs,-1))
        # get_cluster_dist:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters= n_cluster_bkg , random_state=0).fit(x)
        cluster_labels = kmeans.labels_
        cluster_labels = np.array(cluster_labels)
        # c = Counter(cluster_labels)
        # print(c)
        res=[]
        class_weight_list = []
        for c in np.unique(cluster_labels):
            selected_idx_list = np.where(cluster_labels==c)[0]
            data_c = data_all[selected_idx_list].mean(axis=0, keepdims=True)
            res.append(data_c)
            class_weight_list.append(selected_idx_list.shape[0])
        res = np.concatenate(res,0)
        # print('cluster result = ', res.shape)
        cluster_res=res
        # class_weight_list = np.array(class_weight_list)
        # class_weight_list = class_weight_list/np.sum(class_weight_list)
        # class_weight_list = np.round(class_weight_list,2).astype(str)

        return res



    def do_w_clustering(self, data,n_clusters=5, T1=0.1, T2=0.1, topk=32, use_bkg=False, save_name=None, 
        use_sample=False, segment_list=None, bkg_pose_list=None):



        from sklearn.cluster import KMeans
        from collections import Counter

       
        # print('training prior data.shape = ', data.shape)
        N0,C0,T0,V0,M0 = data.shape

        sample_list = np.arange(data.shape[0])

        M_list = []
        tr_list = []
        if segment_list is None:
            segment_list = [
                (0.0, 1.0), 
                (0.0, 0.5), (0.5, 1.0), (0.375, 0.875),
                (0.25, 0.75), (0.125, 0.625), 
                (0.0, 0.75), (0.25, 1.0)
            ]
        
        for i in sample_list:
            data_input = torch.FloatTensor(data[i:i+1])

            # sample here 
            N,C,T,V,M = data_input.shape
            raw_data_st=0
            raw_data_ed=1
            raw_data_st_int = int(raw_data_st*T)
            raw_data_ed_int = int(raw_data_ed*T)
            data_input = resize_torch_interp_batch(data_input[:,:,raw_data_st_int:raw_data_ed_int,:,:], 50)

            for (st,ed) in segment_list:
                M, _ = get_gt_frame_mapping(data_input, st, ed, T1)
                M = M.numpy()

                tr_list.append( M.copy() )
            
        tr_list = np.stack(tr_list)
        # print('tr_list.shape=', tr_list.shape)

        n = tr_list.shape[0]
        tr_list = tr_list.reshape((n,-1))
        # print(tr_list.shape, tr_list.dtype)
        
        
        X = tr_list
        kmeans = KMeans(n_clusters, random_state=0).fit(X)
        cluster_labels = kmeans.labels_
        c = Counter(cluster_labels)
        # print(c)

        # get each cluster center
        tr_cluster_center_list = []
        tr_cluster_weight_list = []
        cluster_labels = np.array(cluster_labels)
        n_cluster_label = np.unique(cluster_labels)
        for cluster_label in n_cluster_label:
            selected_idx_list = np.where(cluster_labels==cluster_label)[0]
            # print(f'cluster_label={cluster_label}, number={selected_idx_list.shape}')

            tr_cluster_weight_list.append(selected_idx_list.shape[0])
            tr_cluster_center_list.append( tr_list[selected_idx_list].mean(axis=0) )
        
        # append center for all.
        tr_cluster_center_list.append( tr_list.mean(axis=0) )

        tr_cluster_center_list = np.stack(tr_cluster_center_list,0)
        tr_cluster_center_list = tr_cluster_center_list.reshape((tr_cluster_center_list.shape[0],T0,T0))
        # print("adding mean(all samples) as last cluster center.")
        # print(tr_cluster_center_list.shape)

        
        tr_cluster_center_list = torch.FloatTensor(tr_cluster_center_list).unsqueeze(1)
        tr_cluster_center_list_npy = tr_cluster_center_list.clone().numpy()
        



        tr_cluster_weight_list = np.array(tr_cluster_weight_list)
        tr_cluster_weight_list = tr_cluster_weight_list/tr_cluster_weight_list.sum()
        tr_cluster_weight_list = np.append(tr_cluster_weight_list, 1.0)

        n_clusters_append = tr_cluster_center_list_npy.shape[0]

        conf_mat_list = []
        traj_mat_list = []

        for bs in range(tr_cluster_center_list_npy.shape[0]):
            # print(tr_cluster_center_list_npy[bs,0].shape)s
            conf_mat = tr_cluster_center_list_npy[bs,0]

            conf_mat = conf_mat / conf_mat.sum(axis=1, keepdims=True)
            # print(conf_mat.sum(axis=1))

            conf_mat = set_mask_topk(conf_mat, topk=topk).numpy()
            conf_mat = conf_mat / (conf_mat.sum(axis=1, keepdims=True) + 1e-5)

            coord_mat = np.repeat(np.arange(T0)[None], T0, axis=0)

            conf_mat_masked = conf_mat.copy()
            mean_coord_mat = (conf_mat_masked * coord_mat).sum(axis=1).astype(int)
            traj_mat = np.zeros_like(conf_mat)
            traj_mat[np.arange(T0), mean_coord_mat] = 1.0

            conf_mat_list.append(conf_mat)
            traj_mat_list.append(traj_mat)


        
        
        conf_mat_list = np.stack(conf_mat_list,0)
        traj_mat_list = np.stack(traj_mat_list,0)
        

        return conf_mat_list[:-1], traj_mat_list[:-1], tr_cluster_weight_list[:-1]

    def resize_torch_interp_batch(self, data, target_frame):

        window = target_frame

        n,c,t,v,m = data.shape 
        data = data.permute(0,1,3,4,2).contiguous().view(n,c*v*m,t)
        data = data[:,:,:,None]
        data = F.interpolate(data, size=(window, 1), mode='bilinear',align_corners=False).squeeze(dim=3)
        data = data.reshape(n,c,v,m,window)
        data = data.permute(0,1,4,2,3).contiguous()
        return data 
    
    def server_train(self):
        # chazhi 
        if self.inpo != "None":
            train_data_loader = self.interpolate_data()
        else:
            train_data_loader = self.load_train_data()
        # rk
        
        r_c = torch.stack(self.uploaded_Rsum, dim=0)
        r_c = r_c.mean(dim=0)

        self.global_model.cuda()
        self.global_model.train()
        # self.global_classifier.cuda()
        # self.global_classifier.train()
        losses = 0
        train_num =0
        server_epoch = 1
        for e in range(server_epoch):
            for i, (datas ,label) in enumerate(train_data_loader):
                data = datas.type(torch.FloatTensor).cuda()
                # print(1)
                # data = torch.rand_like(datas, dtype=torch.float).cuda()
                # data = torch.zeros_like(data)
                n,c,v,t,m = data.size()
                # print(data.size())
                # input1
                input1 = shear(crop(data))
                input1 = random_rotate(input1)
                input1 = random_spatial_flip(input1)
                #feat1 = self.encoder(input1)
                
                # input2
                input2 = shear(crop(data))
                input2 = random_rotate(input2)
                input2 = random_spatial_flip(input2)
                
                
               
                
                stgcn_logits, logits, target  = self.global_model(input1, input2,server=True)
                ce_loss = self.crossentropyloss(logits, target)
                # _, output1, _ = self.global_model.encoder_q(input1)
                # _, output2, _ = self.global_model.encoder_q(input1 + 1e-3 * torch.randn_like(input1))
                # print("update_wave:{}".format((output1.detach() - output2.detach()).abs().mean()))  # 若差异小，可能模型未更新

                logits_mean = self.compute_mean(stgcn_logits,label)
                # logits_mean = logits_mean.transpose(0, 1)
                tau = 0.07
                r_s = F.softmax(logits_mean/tau, dim=0)
                r_s= torch.tensor(r_s).cuda()
                # print("r_s:{}".format(r_s.shape))
                # print("r_c:{}".format(r_c.shape))
                r_c= torch.tensor(r_c).cuda()
                # k1 = F.kl_div(F.log_softmax(r_c, dim=-1), F.softmax(r_s, dim=-1), reduction='batchmean')
                kl_loss = 0.5 * (self.kl_divergence(r_c, r_s) + self.kl_divergence(r_s, r_c))
                lamda_g = 0.01
                # loss = ce_loss 
                if self.skl == False:
                    loss = ce_loss 
                else:
                    loss = ce_loss + lamda_g*kl_loss

                self.optimizer.zero_grad()
                ce_loss.backward() 
                # grad_norm = gradient_norm(self.global_model)
                # print(f"Gradient norm: {grad_norm}")
                self.optimizer.step()
                
                losses = loss.detach()

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()
            
            # print("server_batchsize_loss:{}".format(loss.detach()))
        print("serverloss:{}".format(losses))
        # wandb.log({"server_loss:{}":losses/train_num})
    
    def interpolate_data(self):
        train_data_loader ,x_data ,y_data=  self.load_train_data(batch_size=100, get_data=True)
        # prototypes
        total_data = x_data.type(torch.FloatTensor)
        N,C,T,V,M = total_data.size()
        # total_data = total_data.view(N*M, C*T*V)
        centroids = self.ckmeans(total_data,20)
        centroids_t_list = [torch.tensor(c) if not isinstance(c, torch.Tensor) else c for c in centroids]
        c_d =torch.stack(centroids_t_list,dim=0)
        # norms = np.linalg.norm(queue_features, axis=1, keepdims=True)
        # # 对每一行进行 L2 归一化
        # queue_features_normalized = queue_features / norms
    
        savepath = "before_inpo"
        self.skeleton_visual(x_data,y_data,savepath)
        generate = []
        generate_y = []
        for i, (data ,label) in enumerate(train_data_loader):
            data = data.type(torch.FloatTensor).cuda()
            N,C,V,T,M = data.size()
            anchor = data.view(N*M, C*T*V)
            # B:
            B,D =anchor.shape
            inpola_num = self.in_num * M
            anchor_indices = torch.randint(0, anchor.shape[0], (inpola_num,))
            anchor_i =  anchor[anchor_indices]
            b,c,v,t,m =c_d.size()
            nc_d = c_d.view(b*m, c*t*v)
            indices = torch.randint(0, nc_d.shape[0], (inpola_num,))
            known = nc_d[indices]
            anchor_in = anchor_i.cpu().numpy()
            known_in = known.cpu().numpy()
            # 线性插值
            if self.inpo == "linear":
                # t_values = np.linspace(0, 1, 3750)
                t_values = np.linspace(0, 1, B).reshape(-1, 1)  # 生成形状为 (128, 1) 的插值系数

                # 扩展 t_values 以匹配 A 和 B 的形状
                t_matrix = np.tile(t_values, (1, anchor.shape[1]))
                # 生成插值结果
                
                interpolated_points = (1 - t_matrix) * anchor_in + t_matrix * known_in
                interpolated_points = torch.from_numpy(interpolated_points)
                interpolated_points = interpolated_points.contiguous().view(N, C, V,T,M)
                generate.append(interpolated_points)
                generate_y.append(label)
            # 拉东基函数  
            elif self.inpo == "radial":
                # 假设 z 是对应于 x 和 y 的函数值
                z = np.sin(np.pi * anchor_in) * np.cos(np.pi * known_in)

                # 将 x, y, z 展平
                x_flat = anchor_in.ravel()
                y_flat = known_in.ravel()
                z_flat = z.ravel()

                # 创建 RBF 插值器
                rbf = Rbf(x_flat, y_flat, z_flat, function='multiquadric')

                # 创建新的插值点网格
                x_new = np.linspace(anchor_in.min(), anchor_in.max(), 300)
                y_new = np.linspace(known_in.min(), known_in.max(), 300)
                X_new, Y_new = np.meshgrid(x_new, y_new)

                # 进行插值
                Z_new = rbf(X_new, Y_new)



                interpolated_points = Z_new.reshape(n,c,v,t,m)
                generate.append(interpolated_points)
                generate_y.append(label)
            elif self.inpo == "slerp":
                t=self.in_t

                for j in range(0,inpola_num):
                    # print(j)
                    dot_product = np.dot(anchor_in[j], known_in[j])
                    # 生成贝塞尔插值
                    if dot_product < 0.0:
                        known_in[j] = -known_in[j]
                        dot_product = -dot_product

                    if dot_product > 0.9995:
                        interpolated_points = anchor_in[j] + t * (known_in[i] -anchor_in[j])
                        interpolated_points = torch.from_numpy(interpolated_points)
                        interpolated_points = interpolated_points.contiguous().view(C,V,T)
                        generate.append(interpolated_points)
                        inscn = self.server_cnum
                        im = inpola_num/inscn
                        idx = (j)/im +j%im
                        if idx>=inscn :
                            idx = inscn -1
                        generate_y.append([label[int(idx)]])
                        continue
                    else:
                        theta_0 = np.arccos(dot_product)
                        
                        sin_theta_0 = np.sin(theta_0)
                        theta = theta_0 * t
                        sin_theta = np.sin(theta)
                        s0 = np.cos(theta) - dot_product * sin_theta / sin_theta_0
                        s1 = sin_theta / sin_theta_0
                        interpolated_points = s0 * anchor_in[j] + s1 * known_in[j]
                        interpolated_points = torch.from_numpy(interpolated_points)
                        interpolated_points = interpolated_points.contiguous().view(C,V,T)
                        generate.append(interpolated_points)
                        inscn = self.server_cnum
                        im = inpola_num/inscn
                        idx = (j)/im +j%im
                        if idx>=inscn :
                            idx = inscn -1
                        generate_y.append([label[int(idx)]])
                

        ixdata = generate
        iydata = generate_y
       
        if self.inpo == "slerp":
            train_xdata = np.stack(ixdata, axis=0)

        else:     
            train_xdata = np.concatenate(ixdata,axis = 0)
        train_ydata = np.concatenate(iydata,axis = 0)
        # 可视化
        savepath = "after_inpo"
        
        self.skeleton_visual(train_xdata,train_ydata,savepath)

        # assert False,"stop"

        X_train = torch.Tensor(train_xdata).type(torch.float32)
        y_train = torch.Tensor(train_ydata).type(torch.int64)
        if self.inpo == "slerp":
            X_train = X_train.contiguous().view(-1,C,V,T,M)
        concatX = torch.cat((x_data,X_train), dim=0)
        concatY = torch.cat((y_data,y_train.squeeze()), dim=0)
        train_data = [(x,y) for x, y in zip(concatX , concatY)]
        print("concatX.shape:{}".format(concatX.shape))
        return DataLoader(train_data, self.batch_size, drop_last=True, shuffle=False)

    def func(self,x, y, z):
        return x*(1-x)*np.cos(4*np.pi*x) * (np.sin(4*np.pi*y**2)**2)*z
    
    def ckmeans(self,X, k=60, max_iters=100):
        # 1. 随机初始化聚类中心
        X=X.detach()
        X=X.to("cuda")
        # centers = X[np.random.choice(range(X.shape[0]), size=k, replace=False)]
        random_indices = random.sample(range(X.size(0)), k)

        # 根据随机索引选择对应的张量
        centers = X[random_indices]

        for _ in range(max_iters):
            
            
            new_centroids=[ [] for _ in range(k)]

            # 将每个样本分配到最近的中心
            for idx in range(X.size(0)):
                # 计算每个样本到各个中心的距离
                c_idx = min(range(len(centers)), key=lambda index: self.compute_distance(X[idx], centers[index]))
                
                new_centroids[c_idx].append(X[idx])
            
            # 保证聚类中心不为0
            for i in range(len(new_centroids)):
                if len(new_centroids[i])!=0:
                    new_centroids[i] = torch.stack(new_centroids[i])
                
            # # 4. 更新聚类中心
            new_centers = [torch.mean(center, dim=0) for center in new_centroids if len(center) != 0]
            if new_centers:
                new_centers  = torch.stack(new_centers )
    
            # 5. 检查是否收敛
            if centers.shape != new_centers.shape: 
                centers = new_centers
            else:
                if torch.eq(centers ,new_centers).all():
                    break
                else:
                    centers = new_centers


        return centers 

    def compute_distance(self,a ,b):
        return torch.sqrt(torch.sum((a - b) ** 2))
    
    def cosine_interpolation(self,A, B, t):
        """
        使用余弦插值在两个点 A 和 B 之间插值。
        
        参数:
        A : float or numpy array
            第一个已知点的值。
        B : float or numpy array
            第二个已知点的值。
        t : float or numpy array
            插值参数，范围在 [0, 1] 之间。

        返回:
        interpolated_value : float or numpy array
            插值后的值。
        """
        # 计算余弦插值因子
        if A.is_cuda:
            A = A.cpu()
        if B.is_cuda:
            B = B.cpu()
        t2 = (1 - np.cos(t * np.pi)) / 2
        # 计算插值结果
        return A * (1 - t2) + B * t2
    
    def kl_divergence(self,p, q):
        return F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')

    def para_frozen(self,server_model, frozen_mode):
        if frozen_mode == -1:
            # frozen all
            for name, param in server_model.named_parameters():
                param.requires_grad = False
        elif frozen_mode == 1:
            # midmlp2
            for name, param in server_model.named_parameters():
                if 'stgcn_end' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def compute_distance(self,a ,b):
        return torch.sqrt(torch.sum((a - b) ** 2))

    def kmeans(self,X, k=100, max_iters=100):
        # 1. 随机初始化聚类中心
        X=X.detach()
        X=X.to("cuda")
        
        # 确保 X 是 2D 张量 (N, D)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        elif X.dim() > 2:
            # 如果是多维张量，展平除第一维外的所有维度
            X = X.view(X.size(0), -1)
        
        # 确保 k 不超过样本数
        k = min(k, X.size(0))
        random_indices = random.sample(range(X.size(0)), k)
        centers = X[random_indices]  # (k, D)

        for iteration in range(max_iters):
            # 向量化计算所有样本到所有中心的距离 (N, k)
            # X: (N, D), centers: (k, D)
            # 计算 (X - centers)^2 然后求和得到距离
            distances = torch.cdist(X, centers, p=2)  # (N, k)
            
            # 找到每个样本最近的中心索引
            _, assignments = torch.min(distances, dim=1)  # (N,)
            
            # 更新聚类中心（向量化），确保始终有 k 个中心
            new_centers = []
            for i in range(k):
                mask = (assignments == i)
                if mask.sum() > 0:
                    new_centers.append(X[mask].mean(dim=0))
                else:
                    # 如果某个聚类没有样本，保持原中心
                    new_centers.append(centers[i])
            
            new_centers = torch.stack(new_centers)  # (k, D)
    
            # 检查是否收敛
            if torch.allclose(centers, new_centers, atol=1e-6):
                    break
                    centers = new_centers

        # 为所有样本分配聚类标签（向量化）
        distances = torch.cdist(X, centers, p=2)  # (N, k)
        _, y = torch.min(distances, dim=1)
        y = y.cpu().tolist()
        
        c = list(range(centers.size(0)))
        print("Cluster X :{}".format(X.shape))
        print("Cluster Y :{}".format(len(y)))
        print("Cluster centers :{}".format(centers.shape))
        print("Cluster cY :{}".format(len(c)))
        return centers , c , y
    
    def compute_mean(self,features,labels):
        class_num  = self.server_cnum
        class_means = torch.zeros((class_num  , features.size(1)))
        for i in range(class_num ):
            class_features = features[labels == i]  # 选择标签为c的所有特征向量
            if class_features.size(0) > 0:  # 确保该类有样本
                class_means[i] = class_features.mean(dim=0)  # 计算该类的特征均值

        # 60*256
        return class_means
    
    def create_pose(self,ax,plots,vals, update=False):
        connect = [
            (1,2),(2,21),(21,3),(3,4), (9,21),
            (9,10),(10,11),(11,12),(12,24),(12,25),
            (1,17), (17,18), (18,19), (19,20), (5, 21),
            (5,6), (6,7), (7,8), (8,22),(8,23),
            (1,13), (13,14),(14,15),(15,16)
            ]
        connect = [(i - 1, j - 1) for (i, j) in connect]

        LR =[
            False, False, False, False, True,
            True, True, True, True, True,
            True, True, True, True, False,
            False, False, False, False, False,
            False,False,False,False
        ] #这里可以设置哪些骨架是左边，哪些是右边，有时候不准可以全设置为True

        # Start and endpoints of our representation
        I   = np.array([touple[0] for touple in connect])
        J   = np.array([touple[1] for touple in connect])

        lcolor = "#3660ac" #设置骨架颜色，左右颜色
        rcolor = "#3660ac"

        for i in np.arange( len(I) ):
            x = np.array( [vals[I[i], 0], vals[J[i], 0]] )
            z = np.array( [vals[I[i], 1], vals[J[i], 1]] )
            y = np.array( [vals[I[i], 2], vals[J[i], 2]] )
            if not update:
                plots.append(ax.plot(x, y, z, lw=2,linestyle='-' ,c=lcolor if LR[i] else rcolor))

            elif update:
                plots[i][0].set_xdata(x)
                plots[i][0].set_ydata(y)
                plots[i][0].set_3d_properties(z)
                plots[i][0].set_color(lcolor if LR[i] else rcolor)
        
        return plots
         # ax.legend(loc='lower left')

    def update_gt(self,num, data_gt, plots_gt, fig, ax):
        gt_vals = data_gt[num]
        # gt_vals[:,1] = gt_vals[:,1]+0.2*num
        plots_gt = self.create_pose(ax, plots_gt, gt_vals, update=True)

        r = 0.75
        xroot, zroot, yroot = gt_vals[0, 0], gt_vals[0, 1], gt_vals[0, 2]
        ax.set_xlim3d([-r + xroot, r + xroot])
        ax.set_ylim3d([-r + yroot, r + yroot])
        ax.set_zlim3d([-r + zroot, r + zroot])

        return plots_gt

    def visualize(self, input, frame_indices, save_path):
        # input dim : t, v, c
        data_gt = input
        # print(data_gt.shape)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # 使用 add_subplot 创建 3D 子图
        ax.view_init(elev=20, azim=-40)  # 视角转换

        # 绘制指定帧的骨架
        for frame_index in range(0,frame_indices,5):
            vals = data_gt[frame_index]
            xroot, zroot, yroot = vals[0, 0], vals[0, 1],vals[0, 2]
            # 获取指定帧的骨架数据
            vals[:,2] = vals[:,2]+0.3*frame_index
            r = 0.75
            ry =10
            # xroot, zroot, yroot = vals[0, 0], vals[0, 1],vals[0, 2]
            ax.set_xlim3d([-r + xroot, r + xroot])
            ax.set_ylim3d([-0.25 + yroot, ry + yroot])
            ax.set_zlim3d([-r + zroot, r + zroot])
            gt_plots=[]
            # gt_plots = self.update_gt(frame_index, data_gt, plots_gt, fig, ax)
            gt_plots = self.create_pose(ax, gt_plots, vals, update=False)
        
        ax.axis('off') # 关闭坐标轴
        ax.grid(b=None) #关闭网格
        # # 设置坐标轴标签和范围
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # ax.set_xlim3d([-1, 1.5])
        # ax.set_ylim3d([-1, 1.5])
        # ax.set_zlim3d([0.0, 1.5])
        # ax.set_title('多帧骨架叠加图')

        # 保存或显示图像
        plt.savefig(save_path + '/multi_frame_overlay1.png')

    def svisualize(self,input ,vision_num, save_path):
        # input dim : t, v, c
        data_gt = input
        print(data_gt.shape)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.view_init(elev=20, azim=-40) #视角转换

        vals = np.zeros((25, 3)) # or joints_to_consider
        gt_plots=[]

        gt_plots=self.create_pose(ax,gt_plots,vals,update=False)

        #以下是关于坐标轴的设置，如果需要可以打开
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(loc='lower left')
        ax.set_xlim3d([-1, 1.5])
        ax.set_xlabel('X')
        ax.set_ylim3d([-1, 1.5])
        ax.set_ylabel('Y')
        ax.set_zlim3d([0.0, 1.5])
        ax.set_zlabel('Z')
        ax.set_title('设置标题')

        line_anim_gt = animation.FuncAnimation(fig, self.update_gt,vision_num,  fargs=(data_gt,gt_plots,fig,ax),interval=70, blit=False)

        #下面是关闭坐标轴和网格，如果需要可以取消注释，即可去掉坐标轴和网格
        # ax.axis('off') # 关闭坐标轴
        # ax.grid(b=None) #关闭网格
        # ax.view_init(elev=15, azim=60) 
        
        # dataone=[]
        # data =input[:,0,:]
        # # plt.show()
        # for i in range(vision_num):
        #    d = data[:,1]+ input[:,i,]

        gt_name = 'test.gif' #保存名称

        line_anim_gt.save(save_path + '/' + gt_name,writer='pillow')

    def get_data(self,file_name):
        max_V = 25 #节点数
        max_M = 2 #骨架数量
        with open(file_name, 'r') as fr:
            frame_num = int(fr.readline())
            point = np.zeros((3, frame_num, 25, 2))
            for frame in range(frame_num):
                person_num = int(fr.readline())
                for person in range(person_num):
                    fr.readline()
                    joint_num = int(fr.readline())
                    for joint in range(joint_num):
                        v = fr.readline().split(' ')
                        if joint < max_V and person < max_M:
                            point[0,frame,joint,person] = float(v[0])#一个关节的一个坐标
                            point[1,frame,joint,person] = float(v[1])
                            point[2,frame,joint,person] = float(v[2])
        point = point[:,:,:,0] #只取一个人
        # point = np.expand_dims(point, -1)
        return point

    def get_gif_imgs(self,gif_path, save_path):
        """
        提取动图中的每一帧图片,并保存到文件夹中
        """
        new_path = save_path
        gif = Image.open(gif_path)
        try:
            gif.save(f"{new_path}/{gif.tell()}"+"_frame.png")
            while True:
                gif.seek(gif.tell() + 1)
                gif.save(f'{new_path}/{gif.tell()}'+"_frame.png")
        except Exception as e:
            print("处理结束")

    def skeleton_visual(self,data=None,ydata=None,savepath=None):
        #(100, 3, 50, 25)
        # print(train_xdata)
        file_name = '/opt/data/private/dataset/pku2_original/skeleton/A01N01-M.txt' #
        points = np.loadtxt(file_name, dtype=np.float64)
    
        # print("points:{}".format(points.shape)) (898, 150) ，898：frames
        # data :batchsize,3,50,25,2
        # row = data.shape[2]
        # point = (data[:, :75]).reshape(row, 25, 3)  # frames*25*3
        # print(point.shape)

        # point = data[0,:,0,:,0]
        # print(point.shape)
        # for i in range(25):
        #     print(point[:,i])
        if data.shape[0] > 0:
            for i in range(data.shape[0]):
                input = data[i] # data : 3,50,25,2
                 # data : 3,50,25
                # print(input.shape)
                # input = input.permute(1,2,0)
                # input = input[:,:,1]+0.2*i
                vision_num = 50 #要可视化多少帧
                # print('视频帧数共{}帧，需要可视化{}帧'.format(input.shape[0], vision_num))
                # if vision_num > input.shape[0]:
                #     print('要可视化的帧数大于视频总帧数！')

                save_path = './results'+'/server/'+savepath+"/"+ str(ydata[i])
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                                
                # input = data[0]
                # # print(input.shape)
                # input = input[:,:,:,0] 
                # # print(input.shape)
                # input = input.permute(1,2,0)
                # input = input[:,:,1]+0.2*i
                # print(input.shape)
                

                # point:frames*25*3
                # input : 3,50,25
                point = input
                
                if not torch.is_tensor(input):
                    point = torch.from_numpy(point)
                xmax = torch.max(point[0,:, :])
                xmin = torch.min(point[0,:, :]) - 0.5
                ymax = torch.max(point[1,:, :]) + 0.2
                ymin = torch.min(point[1,:, :]) - 0.2
                zmax = torch.max(point[2,:, :])
                zmin = torch.min(point[2,:, :]) 

            
                # 相邻各节点列表，用来画节点之间的连接线
                arm = [21, 7, 6, 5, 4, 20, 8, 9, 10, 11, 23]
                rightHand = [22, 7]
                leftHand = [24, 11]
                bodyLeg = [3, 2, 20, 1, 0, 12, 13, 14, 15]
                leftLeg = [0, 16, 17, 18, 19]
                
                n = 0   # 从第n帧开始展示
                m = 50
                plt.figure()
                # plt.ion()
                

                    # print(pointo[20,i,:])
                
                shift = 0
                shift_flag = 0

                # point:frames*25*3
                # input : 3,50,25
                for j in range(n, m,2):
                    effective_frame_flag =self.check_frame_validity(point[:,j,:])
                    if effective_frame_flag == True and not self.check_dot(point[:,j,:]):
                        # plt.cla()

                        plt.scatter(point[0,j, :]+shift, point[1,j, :], c='darkblue', s=40.0)
                        plt.plot(point[0,j, arm]+shift, point[1,j, arm], c='blue', lw=2.0)
                        plt.plot(point[0,j, rightHand]+shift, point[1,j, rightHand], c='blue', lw=2.0)
                        plt.plot(point[0,j, leftHand]+shift, point[1,j, leftHand], c='blue', lw=2.0)
                        plt.plot(point[0,j, bodyLeg]+shift, point[1,j, bodyLeg], c='blue', lw=2.0)
                        plt.plot(point[0,j, leftLeg]+shift, point[1,j, leftLeg], c='blue', lw=2.0)
                        # plt.text(xmax-0.8, ymax-0.2, 'frame: {}/{}'.format(i, row))
                        # plt.text(xmax-0.8, ymax-0.4, 'label: ' + str(label[i]))
                        # plt.savefig('pku1_plot'+str(shift)+'.png')
                        shift_flag +=1
                        shift = 0.6*shift_flag
                        

                plt.xlim(xmin, xmax+shift_flag*0.6)
                plt.ylim(ymin, ymax)
                # plt.pause(0.01)

                # plt.ioff()
                plt.show()
                
                plt.savefig(save_path+'/'+str(i)+'pku2_plot.png')
                plt.close()


    def check_dot(self,frame):
        """检查单帧数据是否为原点"""
        x_flag = torch.all(frame[0,:] < 0.2)
        y_flag = torch.all(frame[1,:] < 0.2)
        if x_flag and y_flag:
            return True
        else:
            return False

    def check_frame_validity(self,frame):
        """检查单帧数据有效性"""
        # print(frame.shape)
        z=np.zeros(frame.shape, dtype=float)
        zero_joints = np.allclose(frame, z)
        zero_positions = np.where(zero_joints) 
        invalid_flags = {
            'has_nan': np.isnan(frame).any(),
            'has_inf': np.isinf(frame).any(),
            'all_zero':np.allclose(frame, 0),
            'empty': frame.size == 0
        }
        # print("zero_positions:{}".format(zero_joints))
        
        if any(invalid_flags.values()):
            # print(f"无效数据警告: {invalid_flags}")
            return False
        return True

    
    def receive_models(self):
        assert (len(self.selected_clients) > 0)
        # 随机采样，模拟客户端掉线
        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))
        
        # 缺点，模型传输了整个模型
        self.uploaded_ids = []  # 保存客户端ID（用于智能合约）
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_sample_counts = []  # 保存客户端样本数量（用于智能合约）
        tot_samples = 0
        for i, client in enumerate(active_clients):
            try:
                # 时延过滤
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)  # 保存客户端ID
                self.uploaded_sample_counts.append(client.train_samples)  # 保存样本数量
                self.uploaded_weights.append(client.train_samples)
                if hasattr(client, "apply_model_poisoning"):
                    client.apply_model_poisoning()
                self.uploaded_models.append(client.model)
        # 权重归一化
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples


    def save_model(self, epoch):
        
        checkpoint = {
        'epoch': epoch,
        'model_state_dict': self.global_model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        }

        # 保存检查点到文件
        
        os.makedirs(self.pth_path, exist_ok=True)
        save_dir = self.pth_path
        save_path = os.path.join(save_dir, str(epoch)+"model_weights.pth")
        torch.save(checkpoint,save_path)
        print("已保存:{}".format(save_path))




    def load_model(self, epoch):
                    # 恢复模型和优化器的状态
        save_path = os.path.join(self.pth_path, str(epoch)+"model_weights.pth")
        checkpoint = torch.load(save_path )
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.global_model.to('cuda')
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
    
    # def test_metrics(self):
    #     num_samples = []
    #     tot_correct = []
    #     tot_auc = []
        
    #     ct, ns, auc = self.clients[0].lineartestall()
    #     tot_correct.append(ct*1.0)
    #     tot_auc.append(auc*ns)
    #     num_samples.append(ns)

    #     ids = [c.id for c in self.clients]

    #     return ids, num_samples, tot_correct, tot_auc
def gradient_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.norm(2).item()  # L2 范数
            total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5  # 总梯度范数
    
    return total_norm


def resize_torch_interp_batch(data, target_frame):

    window = target_frame

    n,c,t,v,m = data.shape 
    data = data.permute(0,1,3,4,2).contiguous().view(n,c*v*m,t)
    data = data[:,:,:,None]
    data = F.interpolate(data, size=(window, 1), mode='bilinear',align_corners=False).squeeze(dim=3)
    data = data.reshape(n,c,v,m,window)
    data = data.permute(0,1,4,2,3).contiguous()
    return data 

def get_gt_frame_mapping(data_input, st, ed, temp=0.1):
    '''
    map [st,ed] -> [0,1]
    data_input: (n,c,t,v,m), torch.tensor
    return data_output
    '''
    N,C,T,V,M = data_input.shape
    res = torch.zeros((T,T))
    st_int = int(T*st)
    ed_int = int(T*ed)

    # [0,1,...,T-1] -> [st_int,..., ed_int]
    # [best_fit_index] -> [0, st_int-1], [ed_int+1,T]
    data_output = resize_torch_interp_batch(data_input[:,:,st_int:ed_int,:,:], T)


    # distance matrix, (n,n)
    distance_matrix = get_dist_matrix(data_input, data_output)
    _, min_dist_idx_list = distance_matrix.min(dim=1)

    
    res = F.softmax(-distance_matrix/temp, dim=1)

    return res, data_output

def get_dist_matrix(x, y):
    '''
    x,y: (n,c,t,v,m), torch.tensor
    '''
    # (n,t,c,v,m)
    assert x.shape==y.shape
    n,c,t,v,m = x.shape
    x = x.permute(0,2,1,3,4).contiguous()
    y = y.permute(0,2,1,3,4).contiguous()
    x = x.reshape(n,t,c*v*m)[0]
    y = y.reshape(n,t,c*v*m)[0]

    x = x.unsqueeze(1)
    y = y.unsqueeze(0)

    dist = ((x-y)**2).sum(dim=2)
    # print(dist.shape)
    return dist


def set_mask_topk(coef_mat, topk):
    coef_mat = torch.FloatTensor(coef_mat)
    T0,_ = coef_mat.shape 
    _,topk_idx = coef_mat.topk(k=topk, dim=1)

    res = torch.zeros_like(coef_mat)
    for row in range(T0):
        res[row, topk_idx[row]] = coef_mat[row, topk_idx[row]]
    return res
