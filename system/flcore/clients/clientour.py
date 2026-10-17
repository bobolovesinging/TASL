import torch
import torch.nn as nn
import numpy as np
import time
import copy
import wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader
from flcore.clients.clientbase import Client
from utils.model_utils import *
from utils.DSTformer import *
from utils.model_action import *
from sklearn.preprocessing import label_binarize, normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from collections import OrderedDict
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent, PerturbedGradientDescenth
import os
from datetime import datetime
from utils.data_utils import read_client_data,read_client_traindata
import pickle

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
matplotlib.use('Agg')

class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.model = copy.deepcopy(args.model)
        self.encoder = copy.deepcopy( self.model.encoder_q)
        self.client_num = args.num_clients
        # print(args)
        self.server_num = args.server_cnum
        # self.memodel = copy.deepcopy(args.memodel)
        self.hsize = 256
        self.jw =args.jwaloss
        self.posC = []
        self.negC = []
        self.conden =True
        self.lamda_loc = 0.1  # 0.0001
        self.classifier = copy.deepcopy(args.classifier).cuda()
        self.otherv = []
        self.tau = 0.07
        self.in_t =args.inter_t
        self.in_num =args.inter_num
        self.init_method = args.init_method

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
        self.global_step = 0
        self.init_method = args.init_method
        self.pre_lr1 = args.pretrain_lr # batchsize(512=4*128 sqrt(4))
        self.optimizerp1 = torch.optim.SGD(
            self.model.parameters(),
            weight_decay=1e-4,
            lr=self.pre_lr1,
            momentum=0.9,
            nesterov=False)
        #self.optimizerp1 = PerturbedGradientDescent(
        #    self.encoder.parameters(), lr=pre_lr1, mu=self.mu)



        # learning_rate_scheduler
        self.learning_rate_scheduler1 = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizerp1, 
            gamma=args.learning_rate_decay_gamma
        )

        


        # lineartrain
        # self.classifier_lt = Linear(hidden_size=256,dataset=self.dataset)
        self.classifier_lt = Linear(hidden_size=256,dataset=self.dataset)
        self.optimizerl = torch.optim.Adam(
            self.classifier_lt.parameters(),
            lr=0.0001)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizerl, args.global_rounds)
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss().cuda()
        self.num_classes = args.num_classes
        
        self.pth_path = '../log/mdpath/'+str(self.in_t)+"_"+str(self.in_num)+"/"+'/client'+str(id)
        self.client_cnum = args.client_cnum
        
        # 全局数据分布（用于解决non-IID问题）
        self.global_distribution = None  # 将在服务器端设置
        self.weighted_loss = None  # 加权损失函数（基于全局分布）
        
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
            self.weighted_loss = nn.CrossEntropyLoss(weight=class_weights).to(self.device)
            if self.id == 0:  # 只在第一个客户端打印，避免重复输出
                print(f"[Blockchain] 客户端 {self.id} 使用全局分布加权损失函数")
                print(f"[Blockchain] 类别权重范围: [{class_weights.min():.3f}, {class_weights.max():.3f}]")
                print(f"[Blockchain] 少数类别（权重>1.5）: {torch.sum(class_weights > 1.5).item()} 个")
        else:
            # 使用标准损失函数
            self.weighted_loss = self.loss
        
        return self.weighted_loss
    
    def adjust_lr(self,runningrounds):
        if runningrounds is not None and runningrounds > 80:
            lr = self.llr  * 0.1
            for param_group in self.optimizerl.param_groups:
                param_group['lr'] = lr
            self.llr  = lr   
    
    def pre_adjust_lr(self,runningrounds):
        if runningrounds is not None and runningrounds >= 250:
            lr =self.pre_lr1* 0.1
            for param_group in self.optimizerp1.param_groups:
                param_group['lr'] = lr
            self.pre_lr1  = lr   

    def set_classifiers(self, classifier):
        for new_param, old_param in zip(classifier.parameters(), self.classifier_lt.parameters()):
            old_param.data = new_param.data.clone()

    def init_condensed_data(self):
        
        cluster_data_path = "log/"+str(self.dataset)+"/"+str(self.id)+"/"+str(self.client_cnum)+"/data.pkl"
        train_data, train_label = read_client_traindata(self.dataset,self.id)

        if os.path.exists(cluster_data_path):
            with open(cluster_data_path, 'rb') as file:
                data = pickle.load(file)
            kf = data['c_data']
            y_plabel = data['y_plabel']
            c_label = data['c_label']
            
        else:
            train_data =train_data.type(torch.FloatTensor)
            train_label =train_label.type(torch.IntTensor)
            # print(train_data[-1])
            cnum = self.client_cnum
            # print("id:{} cnum:{}".format(self.id, cnum))
            kf ,c_label ,y_plabel = self.kmeans(train_data,cnum)
            # assert False , "读取客户端聚类数据路径不存在"

            # 将数据存储到 pkl 文件
            data_to_save = {
            'c_data': kf ,
            'y_plabel': y_plabel ,
            'c_label': c_label,
            }
                 # # cluster_data_path = 
            if os.path.exists(cluster_data_path):
                pass
            else:
                os.makedirs(os.path.dirname(cluster_data_path), exist_ok=True)
            with open(cluster_data_path, 'wb') as f:
                pickle.dump(data_to_save, f)
            print("Data has been saved to data.pkl")
        # # else:

        

        # 客户端 变换矩阵提取
        segment_list = [
        (0.0, 1.0), 
        (0.0, 0.5), (0.5, 1.0), (0.25, 0.75), (0.125, 0.625), (0.375, 0.875),
        (0.0, 0.75), (0.25, 1.0)
        ]
        n_cluster_w= 4
        T1_each = 0.1
        conf_mat_list1, traj_mat_list1, traj_weight_list1= self.do_w_clustering(train_data,n_clusters=n_cluster_w, T1=T1_each, topk=32, use_bkg=False, save_name=None, 
                        use_sample=False, segment_list=segment_list, bkg_pose_list=None)
        conf_mat_list, traj_mat_list, traj_weight_list = \
            conf_mat_list1, traj_mat_list1, traj_weight_list1
        conf_mat_list_bkg, traj_mat_list_bkg, traj_weight_list_bkg = conf_mat_list, traj_mat_list, traj_weight_list

        # save_name = 'your_file.npz'  # 替换成你的文件名

        # data = np.load(save_name)

        # conf_mat_list = data['conf_mat_list']
        # traj_mat_list = data['traj_mat_list']
        # traj_weight_list = data['traj_weight_list']

        # data.close()

        self.conf_mat_list = conf_mat_list
        self.traj_mat_list = traj_mat_list
        self.traj_weight_list = traj_weight_list

        save_name = os.path.join("log/"+str(self.dataset)+"/"+str(self.id)+'_tr_cluster_'+str(n_cluster_w)+'_T'+str(T1_each)+'.npy')

        # print(traj_mat_list.shape)
        # print("save to", save_name)
        # np.savez(save_name, conf_mat_list=conf_mat_list, traj_mat_list=traj_mat_list, traj_weight_list=traj_weight_list)

        # print(train_data.shape)
        # savepath = "before_clustering"
        # point = train_data[0,:,0,:,0]
        # print(point.shape)
        # for i in range(25):
        #     print(point[:,i])
        # train_data =train_data.type(torch.FloatTensor)
        # train_label =train_label.type(torch.IntTensor)
        # n,c,v,t,m=cData.shape
        # cData = cData.reshape(n*m, c, v, t)

        # self.skeleton_visual(train_data[:,:,:,:,:],train_label,savepath)
        # print(train_data[-1])
        # cnum = self.client_cnum
        # print("id:{} cnum:{}".format(self.id, cnum))
        # kf ,c_label ,y_plabel = self.kmeans(train_data,cnum)
        # print("kf:{}".format(type(kf)))
        # savepath = "before_training"
        # cData =kf.type(torch.FloatTensor)
        # # n,c,v,t,m=cData.shape
        # # cData = cData.reshape(n*m, c, v, t)
        # self.skeleton_visual(cData,c_label,savepath)
        # cData = cData.reshape(n*m, c, v, t)
        # cData = cData[:40,:,:,:]
        # self.check_frame_validity(cData)
        # self.c_data = kf
        # self.y_plabel =  torch.tensor(y_plabel)
        # self.c_label =torch.tensor(c_label) 
        # # 将数据存储到 pkl 文件
        # data_to_save = {
        # 'c_data': kf,
        # 'y_plabel': y_plabel,
        # 'c_label': c_label
        # }
        # cluster_data_path = "log/"+str(self.dataset)+"/"+str(self.id)+"/data.pkl"
        # if os.path.exists(cluster_data_path):
        #     pass
        # else:
        #     os.makedirs(os.path.dirname(cluster_data_path), exist_ok=True)
        # with open(cluster_data_path, 'wb') as f:
        #     pickle.dump(data_to_save, f)
        # print("Data {} has been saved to data.pkl".format(self.id))



        self.y_plabel= y_plabel
        self.clabel = c_label
        self.kf = kf
        sdata=dict()
        sdata['x']=kf
        sdata['y']=c_label
        # self.image_lr :0.1
        
        return sdata
    


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

    # 获取伪标签，并且把伪标签给客户端Dk
    def get_pseudo_labels(self,plabels):
        # plabels为全局的伪标签
        self.new_plabels = plabels
        print("client:{} load_new_labels".format(self.id))
        print("client labels :{}".format(max(plabels)))
   
  
    def compute_mean(self,features,labels):
        class_num  = self.server_num
        class_means = torch.zeros((class_num  , features.size(1)))
        for i in range(class_num):
            class_features = features[labels == i]  # 选择标签为c的所有特征向量
            if class_features.size(0) > 0:  # 确保该类有样本
                class_means[i] = class_features.mean(dim=0)  # 计算该类的特征均值

        # 60*256
        return class_means
    
    def compute_g_mean(self,features,labels):
        class_num  = self.client_cnum
        class_means = torch.zeros((class_num  , features.size(1),features.size(-1)))
        for i in range(class_num):
            class_features = features[labels == i]  # 选择标签为c的所有特征向量
            if class_features.size(0) > 0:  # 确保该类有样本
                class_means[i] = class_features.mean(dim=0)  # 计算该类的特征均值

        # 60*256*25
        return class_means
        
    def SWD(self,p1, p2, num_projections=100):
        """
        计算两个分布 u 和 v 之间的切片Wasserstein距离（Sliced Wasserstein Distance, SWD）

        参数:
        u (torch.Tensor): 第一个分布的样本, 形状为 (n_samples, n_features)
        v (torch.Tensor): 第二个分布的样本, 形状为 (m_samples, n_features)
        num_projections (int): 生成的随机投影数量

        返回:
        float: 切片Wasserstein距离
        """
        """
        计算两个分布之间的切片沃瑟斯坦距离 (SWD)。
        
        参数：
        p1, p2 - 要比较的两个分布，形状为 (n_samples, n_features)
        num_projections - 投影方向的数量（默认100）
        
        返回：
        SWD 距离
        """
        # 数据的维度
        dim = p1.shape[1]
        num_projections = self.client_cnum
        # 随机生成投影方向
        projections = torch.randn(num_projections, dim).cuda()
        projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
        
        # 投影到这些方向上
        p1_proj = p1 @ projections.T
        p2_proj = p2 @ projections.T
        
        # 对投影结果排序
        p1_proj_sorted, _ = torch.sort(p1_proj, dim=0)
        p2_proj_sorted, _ = torch.sort(p2_proj, dim=0)
        
        # 计算 SWD
        swd = torch.mean(torch.abs(p1_proj_sorted - p2_proj_sorted))
        return swd 
    
    def getUpdate(self, grads, input):
        self.learningRate = 0.01

        return input - grads * self.learningRate
    
    def pretrain_avg(self, round):
  
        trainloader = self.load_train_data()
        
        self.model.train()
        self.model.cuda()
        start_time = time.time()

        max_local_epochs = self.local_epochs
        train_num = 0
        losses = 0

        # print(max_local_epochs)
        for step in range(max_local_epochs):
            for i, (data, _) in enumerate(trainloader):

                n,c,t,v,m = data.shape
                data = data.type(torch.FloatTensor).cuda()
                #data = get_stream(data)
                self.model.to('cuda')
                # input1
                input1 = shear(crop(data))
                input1 = random_rotate(input1)
                input1 = random_spatial_flip(input1)
                #feat1 = self.encoder(input1)
                
                # input2
                input2 = shear(crop(data))
                input2 = random_rotate(input2)
                input2 = random_spatial_flip(input2)
                _,output, target= self.model(input1, input2)

                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output.size(0))
                else:
                    self.model.update_ptr(output.size(0))
                
                # 注意：pretrain_avg 是对比学习任务，不是分类任务
                # output 的形状是 (batch_size, 1+queue_size)，target 是正样本索引（全0）
                # 因此这里不使用加权损失函数，使用标准损失函数
                loss = self.loss(output, target)  


                self.optimizerp1.zero_grad()
                #self.optimizerp2.zero_grad()
                # loss.backward()
                
                loss.backward()
                
                # NPC Deputy机制：在训练过程中捕获梯度（更准确的梯度计算方式）
                if self.use_npc_deputy:
                    self.capture_gradients_during_training()
                
                self.optimizerp1.step()
               

                #self.optimizerp2.step()
                #self.optimizerp1.step(self.global_params, self.old_params, self.device)
                #self.optimizerp2.step(self.head_old_params, other_params, self.device)
                
                train_num += n
                loss_end = loss.item() 

        # if self.learning_rate_decay:
        #     self.learning_rate_scheduler1.step()
        self.pre_adjust_lr(round)
        # print(f'当前学习率: {self.optimizerp1.param_groups[0]["lr"]}')
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        
        self.finalize_local_training(train_loss=float(loss_end), sample_count=train_num)
        
        # TensorBoard 记录已移除，只记录服务器端准确率
        
        return loss_end, train_num

    def pretrain(self, round):
        torch.cuda.empty_cache()
        y = self.new_plabels
        data_trainloader = self.get_pseudo_loader(y)
        c_data = self.kf
        newdata  = self.kf.detach()
        c_label  =  torch.tensor(self.new_plabels)
        self.model.encoder_q.train()
        self.model.encoder_q.cuda()
        for param in self.model.encoder_q.parameters():
            param.requires_grad = False 
        start_time = time.time()
        # savepath = "before_training"
        # cData = c_data.type(torch.FloatTensor)
        # # n,c,v,t,m=cData.shape
        # # cData = cData.reshape(n*m, c, v, t)
        # self.skeleton_visual(cData,c_label,savepath)
        # savepath = "before_training"
        # cData =c_data.type(torch.FloatTensor)
        # n,c,v,t,m=cData.shape
        # cData = cData.reshape(n*m, c, v, t)
        # self.skeleton_visual(cData,c_label,savepath)
        c_data.requires_grad=True
        self.con_optimizer  = torch.optim.SGD([c_data], lr=5, momentum=0.5, weight_decay=0)
        # self.model_optimizer  = torch.optim.SGD([self.model.encoder_q], lr=0.01, momentum=0.9, weight_decay=0.0005)
        self.con_optimizer.zero_grad()
        max_local_epochs = 1
        total_loss = 0
        train_num = 0
        loss = 0
        sk_data= []
        LogitsVk =[]
        LogitsDsum = []
        LogitsLabel = []
        R_Labels = []
        train_num = 0
        cdata = 0

        feature_d_all = []
        label_d_all = []
        
        feature_c_all = []
        label_c_all = []
        # print(c_data.mean(), c_data.std())  # 查看是否在变化
        # print("client :{},before c_data:{},c_label:{}".format(self.id, cData.shape,c_label))
        
        loss = 0
        # allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        # reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        # print(f"1Allocated: {allocated:.2f} MiB")
        # print(f"1Reserved:  {reserved:.2f} MiB")

        for i, (data, plabel) in enumerate(data_trainloader):

            data = data.type(torch.FloatTensor).detach().cuda()
            data.requires_grad = False
            plabel = plabel.type(torch.IntTensor).cuda()
            c_data = c_data.type(torch.FloatTensor).cuda()
            clabel = c_label.type(torch.IntTensor).cuda()


            feat_d, logits_d ,graph_d= self.model.encoder_q(data)
            feat_c, logits_c ,graph_c= self.model.encoder_q(c_data )

            feature_d_all.append(feat_d.detach())
            label_d_all.append(plabel.detach())       
            feature_c_all.append(feat_c.detach())
            label_c_all.append(clabel.detach())   

            feat_mean = self.compute_mean(feat_d,plabel).detach()
            feat_syn = self.compute_mean(feat_c,clabel)

    
            # # 60*256*25
            # g_mean = self.compute_g_mean(graph_d,plabel).detach()
            # g_syn = self.compute_g_mean(graph_c,clabel)

            # part_loss = F.mse_loss(feat_mean,feat_syn) + F.mse_loss(g_mean,g_syn)
            part_loss = F.mse_loss(feat_mean,feat_syn)


            if self.otherv != [] and self.jw == "loss2c":
                # 计算cdcloss需要的logits均值
                # logits_mean = self.compute_mean(logits_d,plabel)
                # print("5:{}".format(cdata.grad is None))
                logits_syn = self.compute_mean(logits_c,clabel).cuda()
                # print("logits_syn:{}".format(logits_syn.requires_grad))
                # print("6:{}".format(cdata.grad is None))
                
                otherv =  torch.tensor(self.otherv).cuda()
                otherv = otherv.mean(dim=0)
                
                # 启用特征对齐损失：计算当前客户端特征与其他客户端平均特征的距离
                F_distance = 0
                if logits_syn.shape[0] == otherv.shape[0]:  # 确保维度匹配
                    for cl in range(otherv.shape[0]):  # 假设 Vk 的形状为 (C, ...)
                        if logits_syn[cl].numel() > 0 and otherv[cl].numel() > 0:
                            # 确保形状正确
                            syn_vec = logits_syn[cl].view(1, -1) if logits_syn[cl].dim() > 0 else logits_syn[cl].unsqueeze(0)
                            other_vec = otherv[cl].view(1, -1) if otherv[cl].dim() > 0 else otherv[cl].unsqueeze(0)
                            F_distance += self.SWD(syn_vec, other_vec)
                
                # 动态调整特征对齐权重：随训练轮次递减
                # 早期训练时更重视特征对齐，后期更重视重构损失
                lambda_f = 0.1  # 特征对齐损失权重系数
                loss = part_loss + lambda_f * F_distance
                
                # 记录特征对齐损失（仅在第一个batch打印，避免过多输出）
                if i == 0 and round % 10 == 0:
                    print(f"[Client {self.id}] Round {round}: part_loss={part_loss.item():.4f}, "
                          f"F_distance={F_distance.item():.4f}, total_loss={loss.item():.4f}")
            

                            
            else: 
                
                loss = part_loss 
                
                    
            
            # print(f"Epoch {round}, Loss: {loss.item()}")
            with torch.no_grad():
                logits_mean = self.compute_mean(logits_d ,plabel)
                # print(logits_mean.shape)
                tau = 0.07 # not sure
                r_labels = F.softmax(logits_mean/ tau, dim=1)
                # print(r_labels.shape)
                    # print(label_d_all_stack)
                    
            self.con_optimizer.zero_grad()
            loss.backward()
            self.con_optimizer.step()
            total_loss += loss.detach().item()     
        
        
        feature_d_all_stack = torch.cat(feature_d_all, dim=0)
        label_d_all_stack = torch.cat(label_d_all, dim=0)
        feature_c_all_stack = torch.cat(feature_c_all, dim=0)
        label_c_all_stack = torch.cat(label_c_all, dim=0)
   
        #     # t-SNE 可视化
        # save_path = './sne'+'/'+str(self.id)+'/'
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # # t-SNE 降维到 2D
        # tsne = TSNE(n_components=2, perplexity=20, random_state=42)
        # reduced_data = tsne.fit_transform(feature_d_all_stack.detach().cpu())
        # # print(v_feat_d)
        # # 可视化

        # plt.scatter(reduced_data[:,0], reduced_data[:,1],c=label_d_all_stack.cpu(), cmap='viridis')
        # # plt.title("t-SNE Visualization")
        # # plt.xlabel("Dimension 1")
        # # plt.ylabel("Dimension 2")
        # plt.xticks([])  # 关闭x轴刻度
        # plt.yticks([]) 
        # plt.savefig(save_path+str(round)+'d_TSNE.png')

        # reduced_data = tsne.fit_transform(feature_c_all_stack.detach().cpu())

        # # 可视化
        # plt.cla()
        # plt.xticks([])  # 关闭x轴刻度
        # plt.yticks([]) 
        # plt.scatter(reduced_data[:,0], reduced_data[:,1],c=label_c_all_stack.cpu() , cmap='viridis')
        # # plt.title("t-SNE Visualization")
        # # plt.xlabel("Dimension 1")
        # # plt.ylabel("Dimension 2")
        # plt.savefig(save_path+str(round)+'c_TSNE.png')

        self.class_covmatrix = {}
        self.class_counts = {}
        self.mean_matrix = {}
        class_features = {}
        # print(f"feature_d_all_stack: {feature_d_all_stack.shape}, label_d_all_stack: {label_d_all_stack.shape}")
        for i in range(len(label_d_all_stack)):
            class_idx = label_d_all_stack[i].item()
            if class_idx in self.class_counts:
                self.class_counts[class_idx] += 1
                class_features[class_idx].append(feature_d_all_stack[i])
            else:
                self.class_counts[class_idx] = 1
                class_features[class_idx] = []
                class_features[class_idx].append(feature_d_all_stack[i])
        

        # for i, (key, value) in enumerate(class_counts.items()):

        # 获取每个伪标签类的特征均值，协方差矩阵，样本数量   
        for class_idx, count in self.class_counts.items():
            self.process_features(class_idx, class_features)


        # client_train_loss = str(self.id)+"PretrainLoss"
        # wandb.log({client_train_loss:losses})
        self.Vk = logits_mean
        self.Rk = r_labels
        # self.kf = cdata.detach()
        self.Skdata = dict()
        self.Skdata['x'] = c_data.detach()
        self.kf = c_data.detach()
        self.Skdata['y'] = clabel.detach()

        
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        print(f'Client {self.id}epoch avg loss = {total_loss}')
        return  total_loss, c_data.detach(), clabel

    def process_features(self,class_idx,class_feature_data): 
        Z =  class_feature_data[class_idx]
        Z = torch.stack(Z)
        mean_Z = Z.cpu().mean().item()
        Z_centered = Z - mean_Z
        cov_matrix = (1 / Z.shape[0]) * torch.mm(Z_centered.T, Z_centered)
        self.class_covmatrix[class_idx] = cov_matrix
        # print(f"class_covmatrix:{self.class_covmatrix[class_idx].shape}")
        self.mean_matrix[class_idx] = mean_Z


    def compute_partloss(self, orign, syn):
        if self.jw == "loss2":
            # two_strategy_1 = [1,2,3,4,5,6,7,8,9,10,11,20,21,22,23,24]
            # two_strategy_2 = [0,12,13,14,15,16,17,18,19]
            # orign1 = orign[:,two_strategy_1,:]
            # orign2 = orign[:,two_strategy_2,:]
            # syn1 = syn[:,two_strategy_1,:]
            # syn2 = syn[:,two_strategy_2,:]
            # part_loss =  F.mse_loss(orign1.mean(dim=2) ,syn1.mean(dim=2) )+F.mse_loss(orign2.mean(dim=2) ,syn2.mean(dim=2) )
            # return part_loss
            strategies = {
                'strategy_1': [1,2,3,4,5,6,7,8,9,10,11,20,21,22,23,24],
                'strategy_2': [0,12,13,14,15,16,17,18,19]
            }

            fmse_results = {}
            total_loss = 0
            for name, indices in strategies.items():
                orig_part = orign[:, indices, :].mean(dim=2)
                syn_part = syn[:, indices, :].mean(dim=2)
                current_fmse = F.mse_loss(orig_part, syn_part)
                fmse_results[name] = current_fmse.item()  # .item() to get Python float
                total_loss += current_fmse
            return total_loss
        elif self.jw == "loss4":
            strategies = {
                'strategy_1': [2, 3, 20],
                'strategy_2': [4, 5, 6, 7, 21, 22],
                'strategy_3': [8, 9, 10, 11, 23, 24],
                'strategy_4': [1, 0],
                'strategy_5': [12, 13, 14, 15],
                'strategy_6': [16, 17, 18, 19]
            }

            fmse_results = {}
            total_loss = 0
            for name, indices in strategies.items():
                orig_part = orign[:, indices, :].mean(dim=2)
                syn_part = syn[:, indices, :].mean(dim=2)
                current_fmse = torch.sum((orig_part-syn_part)**2)
                fmse_results[name] = current_fmse.item()  # .item() to get Python float
                total_loss += current_fmse
            return total_loss
        elif self.jw == "loss6":
            strategies = {
                'strategy_1': [2, 3, 20],
                'strategy_2': [4, 5, 6],
                'strategy_3': [8, 9, 10],
                'strategy_4': [1, 0],
                'strategy_5': [12, 13, 14, 15],
                'strategy_6': [16, 17, 18, 19],
                'strategy_7': [7, 21, 22],
                'strategy_8': [11, 23, 24]
            }

            fmse_results = {}
            total_loss = 0
            for name, indices in strategies.items():
                orig_part = orign[:, indices, :].mean(dim=2)
                syn_part = syn[:, indices, :].mean(dim=2)
                current_fmse = F.mse_loss(orig_part, syn_part)
                fmse_results[name] = current_fmse.item()  # .item() to get Python float
                total_loss += current_fmse
            return total_loss
    
    def FD_reduce(self, z, z1):
        random_tangent_points = F.normalize(torch.randn(z.shape).to(z))
        random_samples = z1
        swd = 0
        for random_tangent_point in random_tangent_points:
            sphere_model = geoopt.manifolds.Sphere()
            log_random = sphere_model.logmap(random_tangent_point, random_samples)
            log_sample = sphere_model.logmap(random_tangent_point, z)
            swd += self.SWD(log_random, log_sample)
        # return swd / self.num_tangent_space
        return swd / 6
    
    def train_batch(self, data, label):
        
        _,Z ,_= self.model.encoder_q(data)
        Z = Z.detach()
        predict = self.classifier_lt(Z)
        _, pred = torch.max(predict, 1)

        acc = pred.eq(label.view_as(pred)).float().mean()
        
        # 使用全局分布加权损失函数（如果可用）
        class_weights = self.get_global_class_weights()
        if class_weights is not None:
            weighted_loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(self.device)
            cls_loss = weighted_loss_fn(predict, label)
        else:
            cls_loss = self.CrossEntropyLoss(predict, label)
        
        loss = cls_loss
        
        return loss
    
    # separately train the classifier
    def lineartrain(self, runningrounds=None):
        trainloader = self.load_train_data()
        self.model.eval()
        self.model.cuda()
        self.model.encoder_q.eval()
        self.model.encoder_q.cuda()
        self.classifier_lt.train()
        self.classifier_lt.cuda()
        
        losses = 0
        for epoch in range(1,11):
            train_num = 0
            for data,label in trainloader:
                data = data.type(torch.FloatTensor).cuda()
                label = label.type(torch.LongTensor).cuda()
                
                label = label.squeeze()
                #data = get_stream(data)
                #print("label.dim:{}".format(label.dim()))

                # label = label - 60
                # label = torch.argmax(label, dim=1)
                #print("argmax_label.dim:{}".format(label.dim()))

                loss = self.train_batch(data, label)
                self.optimizerl.zero_grad()
                loss.backward()
                self.optimizerl.step()
                
                train_num += label.shape[0]
                losses = loss.item() 
      
            
            self.adjust_lr(epoch)
            
            # TensorBoard 记录已移除，只记录服务器端准确率
        
        return losses, train_num
   
    def pfl_lineartest(self):
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
            
        
        testloaderfull = self.load_test_data(self.id)
        self.model = self.model.cuda()
        self.classifier_lt.cuda()
        self.model.encoder_q.eval()
        self.classifier_lt.eval()
        

        with torch.no_grad():
            for data, label in testloaderfull:
                data = data.type(torch.FloatTensor).cuda()
                label = label.type(torch.LongTensor).cuda()
                #data = get_stream(data)
                
                with torch.no_grad():
                    # print(type(data))
                    # print(data.size())

                    _, Z ,_= self.model.encoder_q(data)
                    
                    predict = self.classifier_lt(Z)
                    # print(data.shape)
                    # print(predict.shape)
                    # print(label.shape)
                test_acc += (torch.sum(torch.argmax(predict, dim=1) == label)).item()
                # print("(torch.sum(torch.argmax(predict, dim=1) == label)).item(): {})".format((torch.sum(torch.argmax(predict, dim=1) == label)).item())) 
                # print("test_acc +=: {})".format(test_acc)) 
                test_num += label.shape[0]
                # print("label.shape[0] : {})".format(label.shape[0])) 
                # print("test_num += : {})".format(test_num)) 
                y_prob.append(predict.detach().cpu().numpy())
                nc = self.num_classes
                # print(nc)
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(label.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        # print(y_prob.shape)
        # print(y_true.shape)
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc
       
    def lineartest(self):
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
            
        for i in range(1):
            testloaderfull = self.load_test_data(self.id)
            self.model.cuda()
            self.classifier_lt.cuda()
            self.model.eval()
            self.model.encoder_q.eval()
            self.classifier_lt.eval()
            

            with torch.no_grad():
                for data, label in testloaderfull:
                    data = data.type(torch.FloatTensor).cuda()
                    label = label.type(torch.LongTensor).cuda()
                    if label.shape == torch.Size([1]):
                        label = label 
                    else:
                        label = label.squeeze()
                    #data = get_stream(data)
                    
                    with torch.no_grad():
                        # print(type(data))
                        # print(data.size())

                        _, Z ,_= self.model.encoder_q(data)
                        
                        predict = self.classifier_lt(Z)
                        # print(data.shape)
                        # print(predict.shape)
                        # print(label.shape)
                    test_acc += (torch.sum(torch.argmax(predict, dim=1) == label)).item()
                    # print("(torch.sum(torch.argmax(predict, dim=1) == label)).item(): {})".format((torch.sum(torch.argmax(predict, dim=1) == label)).item())) 
                    # print("test_acc +=: {})".format(test_acc)) 
                    # print(label)
                    # print(label.shape)
                    test_num += label.shape[0]
                    # print("label.shape[0] : {})".format(label.shape[0])) 
                    # print("test_num += : {})".format(test_num)) 
                    
                    # 确保 predict 的形状正确：应该是 (batch_size, num_classes)
                    predict_np = predict.detach().cpu().numpy()
                    batch_size_actual = label.shape[0]
                    
                    # 调试信息：检查转换后的形状
                    # if len(y_prob) == 0:  # 只在第一个批次打印
                    #     print(f"[DEBUG] predict_np.shape (numpy): {predict_np.shape}")
                    #     print(f"[DEBUG] batch_size_actual: {batch_size_actual}")
                    
                    # 检查并修复 predict 的形状
                    if len(predict_np.shape) == 1:
                        # 如果被展平了，需要根据 batch_size 和 num_classes reshape
                        if predict_np.shape[0] == batch_size_actual * self.num_classes:
                            predict_np = predict_np.reshape(batch_size_actual, self.num_classes)
                        else:
                            # 如果无法 reshape，说明有问题，跳过这个批次
                            print(f"警告: predict 形状异常 {predict_np.shape}, batch_size={batch_size_actual}, num_classes={self.num_classes}，跳过此批次")
                            continue
                    elif len(predict_np.shape) == 2:
                        # 正常的 2D 形状，检查第一维是否匹配 batch_size
                        if predict_np.shape[0] != batch_size_actual:
                            # 如果第一维不匹配，尝试 reshape
                            if predict_np.shape[0] * predict_np.shape[1] == batch_size_actual * self.num_classes:
                                predict_np = predict_np.reshape(batch_size_actual, self.num_classes)
                            else:
                                print(f"警告: predict 形状不匹配 {predict_np.shape}, batch_size={batch_size_actual}, num_classes={self.num_classes}，跳过此批次")
                                continue
                    elif len(predict_np.shape) > 2:
                        # 如果有额外维度，reshape 为 (batch_size, num_classes)
                        # 保留第一维（batch_size），展平其他维度
                        if predict_np.shape[0] == batch_size_actual:
                            predict_np = predict_np.reshape(batch_size_actual, -1)
                            # 如果展平后的第二维不等于 num_classes，可能需要进一步处理
                            if predict_np.shape[1] != self.num_classes:
                                if predict_np.shape[1] > self.num_classes:
                                    predict_np = predict_np[:, :self.num_classes]
                                else:
                                    print(f"警告: predict 展平后维度 {predict_np.shape[1]} 小于 num_classes {self.num_classes}，跳过此批次")
                                    continue
                        else:
                            print(f"警告: predict 形状异常 {predict_np.shape}, batch_size={batch_size_actual}，跳过此批次")
                            continue
                    
                    # 最终验证：确保形状正确
                    if predict_np.shape[0] != batch_size_actual or predict_np.shape[1] != self.num_classes:
                        print(f"警告: predict 最终形状 {predict_np.shape} 不正确，期望 ({batch_size_actual}, {self.num_classes})，跳过此批次")
                        continue
                    
                    y_prob.append(predict_np)
                    nc = self.num_classes
                    # print(nc)
                    if self.num_classes == 2:
                        nc += 1
                    lb = label_binarize(label.detach().cpu().numpy(), classes=np.arange(nc))
                    if self.num_classes == 2:
                        lb = lb[:, :2]
                    
                    y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        # print(y_prob.shape)
        # print(y_true.shape)
        
        # 检查形状是否匹配
        if y_prob.shape[0] != y_true.shape[0]:
            print(f"警告: y_prob 和 y_true 长度不匹配 ({y_prob.shape[0]} vs {y_true.shape[0]})，跳过AUC计算")
            auc = 0.0
        else:
            try:
                auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
            except (ValueError, Exception) as e:
                print(f"警告: AUC计算失败: {e}，使用默认值0.0")
                auc = 0.0
        
        return test_acc, test_num, auc



    def lineartestall(self):
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
            
        for i in range(10):
            testloaderfull = self.load_test_data(i)
            self.model.cuda()
            self.classifier_lt.cuda()
            self.model.eval()
            self.model.encoder_q.eval()
            self.classifier_lt.eval()
            
            with torch.no_grad():
                for data, label in testloaderfull:
                    data = data.type(torch.FloatTensor).cuda()
                    label = label.type(torch.LongTensor).cuda()
                    if label.shape == torch.Size([1]):
                        label = label 
                    else:
                        label = label.squeeze()
                    #data = get_stream(data)
                    
                    with torch.no_grad():
                        _, Z ,_= self.model.encoder_q(data)
                        
                        predict = self.classifier_lt(Z)

                    test_acc += (torch.sum(torch.argmax(predict, dim=1) == label)).item()

                    test_num += label.shape[0]

                    y_prob.append(predict.detach().cpu().numpy())
                    nc = self.num_classes
                    # print(nc)
                    if self.num_classes == 2:
                        nc += 1
                    lb = label_binarize(label.detach().cpu().numpy(), classes=np.arange(nc))
                    if self.num_classes == 2:
                        lb = lb[:, :2]
                    
                    y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc
       
    def semitrain(self,data_raito=None):
        trainloader = self.load_semitrain_data(data_raito)
        # self.model.eval()
        self.model.encoder_q.eval()
        self.model.encoder_q.cuda()
        self.classifier_lt.train()
        self.classifier_lt.cuda()
        
        losses = 0
        for epoch in range(1,21):

            train_num = 0
            for data,label in trainloader:
                data = data.type(torch.FloatTensor).cuda()
                label = label.type(torch.LongTensor).cuda()
                if label.shape == torch.Size([1]):
                    label = label 
                else:
                    label = label.squeeze()
                #data = get_stream(data)
                #print("label.dim:{}".format(label.dim()))

                # label = label - 60
                # label = torch.argmax(label, dim=1)
                #print("argmax_label.dim:{}".format(label.dim()))
                loss = self.train_batch(data, label)
                self.optimizerl.zero_grad()
                loss.backward()
                self.optimizerl.step()
                
                train_num += label.shape[0]
                losses = loss.item() 
      
            
            self.adjust_lr(epoch)  
        return losses, train_num

    def knn(self, data_train, data_test, label_train, label_test, nn=9):
        label_train = np.asarray(label_train)
        label_test = np.asarray(label_test)
        # print("Number of KNN Neighbours = ", nn)
        # print("training feature and labels", data_train.shape, len(label_train))
        # print("test feature and labels", data_test.shape, len(label_test))

        Xtr_Norm = normalize(data_train)
        Xte_Norm = normalize(data_test)

        knn = KNeighborsClassifier(n_neighbors=nn,
                                metric='cosine')  # , metric='cosine'#'mahalanobis', metric_params={'V': np.cov(data_train)})
        knn.fit(Xtr_Norm, label_train)
        pred = knn.predict(Xte_Norm)
        acc = metrics.accuracy_score(pred, label_test)

        return acc

    def clustering_knn_acc(self, knn_neighbours=1):
        data_train = self.load_train_data()
        data_test = self.load_test_data()
        self.model.encoder_q.eval()
        self.model.encoder_q.cuda()
        # print("Extracting training features")
        label_train_list =[]
        hidden_array_train_list =[]
        for ith, (data, label) in enumerate(data_train):
            data = data.cuda()
            # feat, logits, target
            feat, _ , _ = self.model.encoder_q(data)
            feat = feat.squeeze()
            #print("encoder size",en_hi.size())

            label_train_list.append(label)
            hidden_array_train_list.append(feat[:, :].detach().cpu().numpy())
        label_train = np.hstack(label_train_list)
        hidden_array_train = np.vstack(hidden_array_train_list)

        # print("Extracting validation features")
        label_eval_list = []
        hidden_array_eval_list = []
        for ith, (ith_data,  label) in enumerate(data_test):

            input_tensor = ith_data.cuda()

            en_hi, _ , _= self.model.encoder_q(input_tensor)
            en_hi = en_hi.squeeze()

            label_eval_list.append(label)
            hidden_array_eval_list.append(en_hi[:, :].detach().cpu().numpy())
        label_eval = np.hstack(label_eval_list)
        hidden_array_eval = np.vstack(hidden_array_eval_list)
        # print(hi_train.shape)

        knn_acc_1 = self.knn(hidden_array_train, hidden_array_eval, label_train, label_eval, nn=knn_neighbours)

        
        return knn_acc_1
  
    def kmeans(self,X, k=60, max_iters=100):
        # 1. 随机初始化聚类中心
        X=X.detach()
        X=X.cpu()
        # centers = X[np.random.choice(range(X.shape[0]), size=k, replace=False)]
        random_indices = random.sample(range(X.size(0)), k)

        # 根据随机索引选择对应的张量
        centers = X[random_indices]
        # print(len(centers))
        for _ in range(max_iters):
            
            
            new_centroids=[ [] for _ in range(k)]
            dmx = -1
            di = None
            # 将每个样本分配到最近的中心
            for idx in range(X.size(0)):
                # 计算每个样本到各个中心的距离
                c_idx = min(range(len(centers)), key=lambda index: self.compute_distance(X[idx], centers[index]))
                
                cmx = max(range(len(centers)), key=lambda index: self.compute_distance(X[idx], centers[index]))
                max_distance = self.compute_distance(X[idx], centers[cmx])
                if dmx < max_distance: 
                    dmx = max_distance
                    di = cmx
                new_centroids[c_idx].append(X[idx])
            # print("1",len(new_centroids))
            # 保证聚类中心不为0
            
            for i in range(len(new_centroids)):
                if len(new_centroids[i])!=0:
                    new_centroids[i] = torch.stack(new_centroids[i])

            # print("2",len(new_centroids))    
            # # 4. 更新聚类中心
            new_centers = []

            # print(len(new_centroids))
            # print(len(new_centroids[i]))
            for id, center in enumerate(new_centroids):
                if len(center) != 0:
                    new_centers.append(torch.mean(center, dim=0))
                else:
                    new_centers.append(X[di])
            
            # new_centers = [torch.mean(center, dim=0) for center in new_centroids if len(center) != 0]
            # print("3",len(new_centers))   
            if new_centers:
                new_centers  = torch.stack(new_centers )
            
            # print("4",len(new_centers))   
            # 5. 检查是否收敛
            if  torch.eq(centers ,new_centers).all(): 
                break           
            else:
                centers = new_centers

        # 为聚类中心设置聚类标签
        y = []
        for idx in range(X.size(0)):
            c_idx = min(range(len(centers)), key=lambda index: self.compute_distance(X[idx], centers[index]))
            y.append(c_idx) 
        c = []
        for idx in range(centers.size(0)):
            c.append(idx)
        print("Cluster X :{}".format(X.shape))
        print("Cluster Y :{}".format(len(y)))
        print("Cluster centers :{}".format(centers.shape))
        print("Cluster cY :{}".format(len(c)))
        return centers , c , y
    def compute_distance(self,a ,b):
        return torch.sqrt(torch.sum((a - b) ** 2))

    def set_parameters(self, model):
        # for (nn, np), (on, op) in zip(model.named_parameters(), self.model.named_parameters()):
        #     # if 'bn' not in nn or self.init_method == "avg":
        #     #     op.data = np.data.clone()
        #     op.data = np.data.clone()
        self.model.load_state_dict(model.state_dict())

    def skeleton_visual(self,data=None,ydata=None,savepath=None):
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
                input = input[:,:,:,0] # data : 3,50,25
                # print(input.shape)
                # input = input.permute(1,2,0)
                # input = input[:,:,1]+0.2*i
                vision_num = 50 #要可视化多少帧
                # print('视频帧数共{}帧，需要可视化{}帧'.format(input.shape[0], vision_num))
                # if vision_num > input.shape[0]:
                #     print('要可视化的帧数大于视频总帧数！')

                save_path = './results'+'/'+str(self.id)+'/'+savepath+"/"+ str(ydata[i])
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
                # print(point.shape)
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

    def test_visualize(self,point,frame_indices, save_path):
        
        row = point.shape[0]
        # print(row)
        # print(f"Input shape: {point.shape}")
        xmax = torch.max(point[:, :, 0])
        xmin = torch.min(point[:, :, 0])
        ymax = torch.max(point[:, :, 1])
        ymin = torch.min(point[:, :, 1])
        zmax = torch.max(point[:, :, 2])
        zmin = torch.min(point[:, :, 2])
        arm = [21, 7, 6, 5, 4, 20, 8, 9, 10, 11, 23]
        rightHand = [22, 7]
        leftHand = [24, 11]
        bodyLeg = [3, 2, 20, 1, 0, 12, 13, 14, 15]
        leftLeg = [0, 16, 17, 18, 19]
        fig = plt.figure()   # 先生成一块画布，然后在画布上添加3D坐标轴
        plt.ion()
        if point.ndim != 3 or point.shape[2] != 3:
            raise ValueError(f"数据应为3维数组 (N, J, 3)，实际形状: {points.shape}")
        for i in range(0, 50):
            ax = Axes3D(fig)
            # self.check_frame_validity(point)
            ax.scatter(point[i, :, 0], point[i, :, 1], point[i, :, 2], c='red', s=40.0)
            ax.plot(point[i, arm, 0], point[i, arm, 1], point[i, arm, 2], c='green', lw=2.0)
            ax.plot(point[i, rightHand, 0], point[i, rightHand, 1], point[i, rightHand, 2], c='green', lw=2.0)
            ax.plot(point[i, leftHand, 0], point[i, leftHand, 1], point[i, leftHand, 2], c='green', lw=2.0)
            ax.plot(point[i, bodyLeg, 0], point[i, bodyLeg, 1], point[i, bodyLeg, 2], c='green', lw=2.0)
            ax.plot(point[i, leftLeg, 0], point[i, leftLeg, 1], point[i, leftLeg, 2], c='green', lw=2.0)
            ax.text(xmax-0.8, ymax-0.2, zmax-0.2, 'frame {}/{}'.format(i, row))
            ax.text(xmax-0.8, ymax-0.4, zmax-0.4, 'label: ' )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_zlim(zmin, zmax)
            plt.savefig(save_path+'/'+str(i)+'multi_frame_plot.png')
        plt.ioff()
        plt.show()
        return

    def visualize(self, input, frame_indices, save_path):
        # input dim : t, v, c
        data_gt = input
        # print(data_gt.shape)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # 使用 add_subplot 创建 3D 子图
        ax.view_init(elev=20, azim=-40)  # 视角转换

        # 绘制指定帧的骨架
        for frame_index in range(0,frame_indices):
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



def model_parameter_vector(model):

    param = [p.view(-1) for p in model.parameters()]
    return torch.cat(param, dim=0)

def mlp_model_parameter_vector(model):
    mlp1_param = []
    mlp2_param = []
    for name,p in model.named_parameters():
        if 'mlp' not in name:
            continue
        elif 'midmlp1' in name:
            #print("mlpname:{}".format(name))
            mlp1_param.append(p.view(-1))
        elif 'midmlp2' in name:
            #print("mlpname:{}".format(name))
            mlp2_param.append(p.view(-1))
    #assert False , "Stop this way show the mlparam"
    # param = [p.view(-1) for p in model.parameters()]
    #mlp_p = torch.cat(mlp_param, dim=0)
    private_para = torch.cat(mlp1_param, dim=0)
    public_para =  torch.cat(mlp2_param, dim=0)  
    return private_para,public_para

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

