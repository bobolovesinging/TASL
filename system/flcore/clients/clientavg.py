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
from collections import OrderedDict
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent, PerturbedGradientDescenth
import os
from datetime import datetime
from utils.data_utils import read_client_data,read_client_traindata
import pickle
class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.clmode =  args.contrastloss
        self.model = copy.deepcopy(args.model)
        self.encoder = copy.deepcopy( self.model.encoder_q)
        self.client_num = args.num_clients
        # self.memodel = copy.deepcopy(args.memodel)
        self.hsize = 256
        self.posC = []
        self.negC = []
        self.conden =True
        self.lamda_loc = 0.0001  # 0.0001
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
        # self.init_condensed_data()
        #self.btwins_head = BTwins(hidden_size=self.hsize,lambd=2e-4,pj_size=6144).cuda()
        #self.dst_head = ActionHeadEmbed(dropout_ratio=0, dim_rep=512, hidden_dim=2048, num_joints=17)
        # self.dstformer = DSTformer(dim_in=3, dim_out=3, dim_feat=512, dim_rep=512, 
        #                            depth=5, num_heads=8, mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        #                            maxlen=50, num_joints=17)
        # self.actionmodel = ActionNet(self.dstformer,dim_rep=512,num_classes=args.num_classes, dropout_ratio=0.5, version='embed', hidden_dim=256, num_joints=17)
        # self.mu = args.mu
        #self.nu = args.nu
        #self.global_params = model_parameter_vector(self.model)
        #self.old_params = model_parameter_vector(self.model)
        
        #self.head_old_params = model_parameter_vector(self.dstformer)
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
        self.classifier_lt = Linear(hidden_size=128,dataset=self.dataset)
        self.optimizerl = torch.optim.Adam(
            self.classifier_lt.parameters(),
            lr=0.0001)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizerl, args.global_rounds)
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss().cuda()
        self.num_classes = args.num_classes
        
        self.pth_path = '../log/mdpath/'+str(self.in_t)+"_"+str(self.in_num)+"/"+'/client'+str(id)
        #self.reset_memory()
        # self.local_clustering()
        # loadname="240"
        # if self.clmode == "contrast":
        #     loadpath= "/opt/data/private/fedlearning/base/system/log/md_contrast/"+str(self.id)+"/"
        # else:
        #     loadpath= "/opt/data/private/fedlearning/base/system/log/mdpath/"+str(self.id)+"/"
        # =self.load_item(loadname,loadpath)
        self.client_cnum = args.client_cnum
    
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
        
        cluster_data_path = "log/"+str(self.dataset)+"/"+str(self.id)+"/data.pkl"

        # if os.path.exists(cluster_data_path):
        #     with open(cluster_data_path, 'rb') as file:
        #         data = pickle.load(file)
        #     kf = data['c_data']
        #     y_plabel = data['y_plabel']
        #     c_label = data['c_label']
            
        # else:
        #     assert False , "读取客户端聚类数据路径不存在"

        # # 将数据存储到 pkl 文件
        # # data_to_save = {
        # # 'c_data': train_data
        # # }
        # # cluster_data_path = 
        # # if os.path.exists(cluster_data_path):
        # #     pass
        # # else:
        # #     os.makedirs(os.path.dirname(cluster_data_path), exist_ok=True)
        # # with open(cluster_data_path, 'wb') as f:
        # #     pickle.dump(data_to_save, f)
        # # print("Data has been saved to data.pkl")
        # # else:
        train_data = read_client_traindata(self.dataset,self.id)
        cnum = self.client_cnum
        print("id:{} cnum:{}".format(self.id, cnum))
        kf ,c_label ,y_plabel = self.kmeans(train_data,cnum)
        print("kf:{}".format(type(kf)))
        
        self.c_data = kf
        self.y_plabel =  torch.tensor(y_plabel)
        self.c_label =torch.tensor(c_label ) 
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
        return sdata
    

    # 获取伪标签，并且把伪标签给客户端Dk
    def get_pseudo_labels(self,plabels):
        # plabels为全局的伪标签
        self.new_plabels = plabels
        print("client:{} load_new_labels".format(self.id))
        print("client labels :{}".format(max(plabels)))


    def generate_random_parameters(self,model):
        random_model = model
        for param in random_model.parameters():
            param.data = torch.randn_like(param)
        return random_model  
        
    # def set_parameters(self, model ):
    #     model.cuda()
    #     self.model.cuda()
    #     # for new_param, param ,rand_param in zip(model.parameters(), self.model.parameters(),rand_model.parameters()):
    #     #     param.data = new_param.data.clone() 
    #     #     # * gama + (1-gama)*rand_param
    #     for new_param, param in zip(model.parameters(), self.model.parameters()):
    #         param.data = new_param.data.clone()

    def save_checkpoint(self,model, optimizer, epoch, model_path='model_state_dict.pth', optimizer_path='optimizer_state_dict.pth', epoch_path='epoch.pth'):
        """
        保存模型和优化器的状态字典以及当前 epoch 信息。
        
        参数:
        - model: PyTorch 模型实例
        - optimizer: PyTorch 优化器实例
        - epoch: 当前训练的 epoch
        - model_path: 模型状态字典文件路径
        - optimizer_path: 优化器状态字典文件路径
        - epoch_path: 当前 epoch 文件路径
        """
            # 创建保存目录
        base_path = "/opt/data/private/fedlearning/gen3d/system/log/"+str(self.id)
        os.makedirs(base_path, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 文件路径
        model_path = os.path.join(base_path, f"model_{timestamp}.pth")
        optimizer_path = os.path.join(base_path, f"optimizer_{timestamp}.pth")
        epoch_path = os.path.join(base_path, f"epoch_{timestamp}.pth")
        
        # 保存模型状态字典
        torch.save(model.state_dict(), model_path)
        print("模型状态字典保存到 {}".format(model_path))

        # 保存优化器状态字典
        torch.save(optimizer.state_dict(), optimizer_path)
        print("优化器状态字典保存到 {}".format(optimizer_path))

        # 保存当前 epoch
        torch.save({'epoch': epoch}, epoch_path)
        print("当前 epoch 保存到 {}".format(epoch_path))

        
    def compute_mean(self,features,labels):
        class_num  = self.client_cnum
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
        self.learningRate = 0.001

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
                loss = self.loss(output, target)  


                self.optimizerp1.zero_grad()
                #self.optimizerp2.zero_grad()
                # loss.backward()
                
                loss.backward()
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
        
        return loss_end, train_num

    def pretrain(self, round):
        # torch.cuda.empty_cache()
        y = self.new_plabels
        data_trainloader = self.get_pseudo_loader(y)
        c_data = self.kf
        c_label  =  torch.tensor(self.new_plabels)
        self.model.encoder_q.eval()
        self.model.encoder_q.cuda()
        start_time = time.time()

        max_local_epochs = 2
        train_num = 0
        losses = 0
        sk_data= []
        LogitsVk =[]
        LogitsDsum = []
        LogitsLabel = []
        R_Labels = []
        train_num = 0
        for step in range(max_local_epochs):
            for i, (data, plabel) in enumerate(data_trainloader):
                
                data = data.type(torch.FloatTensor).detach().cuda()
                data.requires_grad = False
                plabel = plabel.type(torch.IntTensor).cuda()
                cData = c_data.type(torch.FloatTensor).cuda()
                n,c,t,v,m = cData.shape
                clabel = c_label.type(torch.IntTensor).cuda()
                

                cdata = cData.clone().cuda()
                cdata.requires_grad = True
                
                # 显示当前设备已分配的显存量
                # print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024 ** 2} MiB")

                # 显示当前设备保留的显存量（包括已分配的显存）
                # print(f"Reserved Memory: {torch.cuda.memory_reserved() / 1024 ** 2} MiB")

                # 显示当前设备的最大可用显存量
                # print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2} MiB")

                #feat1 = self.encoder(input1)
                feat_d, logits_d ,graph_d= self.model.encoder_q(data)
                feat_c, logits_c ,graph_c= self.model.encoder_q(cdata)
                
                # print("featd:{}".format(feat_d.shape))

                # print("logits_d :{}".format(logits_d .shape))
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(logits_d.size(0))
                    # self.model.module.feat_update_ptr(feat_d.size(0))
                else:
                    self.model.update_ptr(logits_d.size(0))
                    # self.model.feat_update_ptr(feat_d.size(0))
                
                feat_mean = self.compute_mean(feat_d,plabel)
                feat_syn = self.compute_mean(feat_c,clabel)
        

                # start_time = times.time()  # 记录开始时间


               
                # for i in range(graph_d.size(-1)):
                # 60*256*25
                g_mean = self.compute_g_mean(graph_d,plabel)
                g_syn = self.compute_g_mean(graph_c,clabel)
                # 60*256*25
                gcloss =  F.mse_loss(g_mean.mean(dim=2) ,g_syn.mean(dim=2) )
                    # gcloss += g_loss
                # gcloss = gcloss / graph_d.size(-1)
                


                # end_time = time.time()  # 记录结束时间

                # elapsed_time = end_time - start_time  # 计算运行时间
                # print(f"代码1运行时间：{elapsed_time:.6f} 秒")
                cdata.grad = None
                gcloss.backward(retain_graph=True)
                ggs=cdata.grad
                ggsnorms = torch.sqrt(torch.sum(torch.square(ggs),axis=-1))
                ggsnorms = ggsnorms  + 1e-18
                ggsnorms = ggsnorms.unsqueeze(-1)
                ggs = ggs / ggsnorms



                dm_loss = F.mse_loss(feat_mean,feat_syn)
                cdata.grad = None
                dm_loss.backward(retain_graph=True)
                dmgs=cdata.grad
                dmgsnorms = torch.sqrt(torch.sum(torch.square(dmgs),axis=-1))
                dmgsnorms = dmgsnorms + 1e-18
                dmgsnorms = dmgsnorms.unsqueeze(-1)
                dmgs = dmgs / dmgsnorms

                # print(f"Allocated Memory3: {torch.cuda.memory_allocated() / 1024 ** 2} MiB")
                if self.otherv != []:
                    # 计算cdcloss需要的logits均值
                    # logits_mean = self.compute_mean(logits_d,plabel)
                    # print("5:{}".format(cdata.grad is None))
                    logits_syn = self.compute_mean(logits_c,clabel).cuda()
                    # print("logits_syn:{}".format(logits_syn.requires_grad))
                    # print("6:{}".format(cdata.grad is None))
                    
                    otherv =  torch.tensor(self.otherv).cuda()
                    otherv = otherv.mean(dim=0)
                    F_distance =0
                    for cl in range(otherv.shape[0]):  # 假设 Vk 的形状为 (C, ...)
                        F_distance += self.SWD(logits_syn[cl].view(1, -1), otherv[cl].view(1, -1))
                    
                    # cdc_loss = self.SWD(logits_syn,otherv)
                    cdata.grad = None
                    F_distance.backward(retain_graph=True)
                     
                    cdcgs=cdata.grad
                    # print(cdata.grad is None)
                    cdcgs = torch.tensor(cdcgs).cuda()
                    cdcgsnorms = torch.sqrt(torch.sum(torch.square(cdcgs),axis=-1))
                    cdcgsnorms =cdcgsnorms + 1e-18
                    cdcgsnorms = cdcgsnorms.unsqueeze(-1)
                    cdcgs = cdcgs / cdcgsnorms
                    lamda_loc = self.lamda_loc
                    cdata =self.getUpdate(cdcgs * lamda_loc + gcloss  ,cdata)
                    # grads = F_distance * lamda_loc + gcloss  
                    grads = dmgs * lamda_loc + gcloss  
                    
                    

                # # print(f"Allocated Memory4: {torch.cuda.memory_allocated() / 1024 ** 2} MiB")
                
                    
                else:
                    cdata =self.getUpdate(gcloss ,cdata)
                grads = gcloss 

                
                del ggs
                # del ggs
                if 'cdcgs' in vars():
                    del cdcgs
                
                
                with torch.no_grad():
                    logits_mean = self.compute_mean(logits_d ,plabel)
                    # logits_mean =  logits_mean.transpose(0, 1)
                    # LogitsDsum.append(logits_d.cpu())
                    # LogitsLabel.append(plabel.cpu())
                    # LogitsDsum_m = torch.cat(LogitsDsum, dim=0)
                    # LogitsLabels_m = torch.cat(LogitsLabel, dim=0)
                    # Logits_Mean = self.compute_mean(LogitsDsum_m ,LogitsLabels_m)
                    tau = 0.07 # not sure
                    r_labels = F.softmax(logits_mean/ tau, dim=1)

                    
            
                # end_time = time.time()  # 记录结束时间
                # elapsed_time = end_time - start_time  # 计算运行时间
                # print(f"代码运行时间：{elapsed_time:.6f} 秒")
            # normal
            # self.optimizerp1.zero_grad()
            # #self.optimizerp2.zero_grad()
            # # loss.backward()
            
            # loss.backward()
            # self.optimizerp1.step()

            # upload to server
            # Vk = self.compute_mean(LogitsDsum_m,LogitsLabels_m)
            train_num += n
            
            losses += grads.detach().item()

        # print("client:{} train_loss:{}".format(self.id,losses))n
        client_train_loss = str(self.id)+"PretrainLoss"
        # wandb.log({client_train_loss:losses})
        self.Vk = logits_mean
        self.Rk = r_labels
        Sk = dict()
        Sk['x'] = cdata.detach()
        Sk['y'] = clabel.detach()
        self.Skdata = Sk

        
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        item_name = str(round)
        item_path = "log/mdpath/"+str(self.id)
        # if round%20 == 0:
        #     if self.clmode == "contrast":
        #         item_path = "log/md_contrast270/"+str(self.id)
        #     elif self.conden==True:
        #         item_path = "log/condenpath/"+str(self.id)
        #     else:
        #         item_path = "log/mdpath/"+str(self.id)
            
        #     self.save_checkpoint(item_path,self.model,self.optimizerp1,round)
        # wandb.log({str(self.id)+"_pretrain_loss":losses,
        #            str(self.id)+"_g_"+str(round)+"_pretrain_loss":losses
        #            })
        # self.logger.info('Client %s pretrain_loss: %.4f!', self.id ,losses)
        
        return train_num
    
    def add_noise_to_skeleton(skeleton, noise_level=0.01):
        """
        对骨骼点的 x, y, z 坐标添加随机噪声。
        
        参数:
        skeleton (np.ndarray): 骨骼点坐标数组，形状为 (num_points, 3)。
        noise_level (float): 噪声的强度，默认值为 0.01。
        
        返回:
        np.ndarray: 添加了噪声的骨骼点坐标数组。
        """
        noise = np.random.normal(loc=0, scale=noise_level, size=skeleton.shape)
        noisy_skeleton = skeleton + noise
        return noisy_skeleton
    
    def train_batch(self, data, label):
        
        _,Z ,_= self.model.encoder_q(data)
        Z = Z.detach()
        predict = self.classifier_lt(Z)
        _, pred = torch.max(predict, 1)

        acc = pred.eq(label.view_as(pred)).float().mean()
        cls_loss = self.CrossEntropyLoss(predict, label)
        loss = cls_loss
        
        return loss
    
    # separately train the classifier
    def lineartrain(self, runningrounds=None):
        trainloader = self.load_train_data()
        # self.model.eval()
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
        return losses, train_num
    

    def save_opacl(self,epoch):
        checkpoint = {
        'optimizer_state_dict': self.optimizerl.state_dict(),
        'classifer_state_dict': self.classifier_lt.state_dict(),
        'scheduler_state_dict': self.scheduler.state_dict()
        }

        # 保存检查点到文件
        
        os.makedirs(self.pth_path, exist_ok=True)
        save_dir = self.pth_path
        save_path = os.path.join(save_dir, str(epoch)+"model_weights.pth")
        torch.save(checkpoint,save_path)
        print("客户端已保存:{}".format(save_path))

    def load_opacl(self, epoch):
                    # 恢复模型和优化器的状态
        save_path = os.path.join(self.pth_path, str(epoch)+"model_weights.pth")
        checkpoint = torch.load(save_path )

        self.optimizerl.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        for state in self.optimizerl.state.values():
            if isinstance(state, torch.Tensor):
                state.data = state.data.to('cuda')
                if state._grad is not None:
                    state._grad.data = state._grad.data.to('cuda')
            elif isinstance(state, dict):
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to('cuda')
        self.classifier_lt.load_state_dict(checkpoint['classifer_state_dict'])
    
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



    # def lineartestall(self):
    #     test_acc = 0
    #     test_num = 0
    #     y_prob = []
    #     y_true = []
            
    #     for i in range(10):
    #         testloaderfull = self.load_test_data(i)
    #         self.model.cuda()
    #         self.classifier_lt.cuda()
    #         self.model.eval()
    #         self.model.encoder_q.eval()
    #         self.classifier_lt.eval()
            
    #         with torch.no_grad():
    #             for data, label in testloaderfull:
    #                 data = data.type(torch.FloatTensor).cuda()
    #                 label = label.type(torch.LongTensor).cuda()
    #                 if label.shape == torch.Size([1]):
    #                     label = label 
    #                 else:
    #                     label = label.squeeze()
    #                 #data = get_stream(data)
                    
    #                 with torch.no_grad():
    #                     _, Z ,_= self.model.encoder_q(data)
                        
    #                     predict = self.classifier_lt(Z)

    #                 test_acc += (torch.sum(torch.argmax(predict, dim=1) == label)).item()

    #                 test_num += label.shape[0]

    #                 y_prob.append(predict.detach().cpu().numpy())
    #                 nc = self.num_classes
    #                 # print(nc)
    #                 if self.num_classes == 2:
    #                     nc += 1
    #                 lb = label_binarize(label.detach().cpu().numpy(), classes=np.arange(nc))
    #                 if self.num_classes == 2:
    #                     lb = lb[:, :2]
                    
    #                 y_true.append(lb)

    #     y_prob = np.concatenate(y_prob, axis=0)
    #     y_true = np.concatenate(y_true, axis=0)

    #     auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
    #     return test_acc, test_num, auc
    
    def lineartestall(self):
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
            
        
        testloaderfull = self.load_test_data()
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
        for epoch in range(1,11):

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


    def save_checkpoint(self,base_path,model, optimizer, epoch, model_path='model_state_dict.pth', optimizer_path='optimizer_state_dict.pth', epoch_path='epoch.pth'):
        """
        保存模型和优化器的状态字典以及当前 epoch 信息。
        
        参数:
        - model: PyTorch 模型实例
        - optimizer: PyTorch 优化器实例
        - epoch: 当前训练的 epoch
        - model_path: 模型状态字典文件路径
        - optimizer_path: 优化器状态字典文件路径
        - epoch_path: 当前 epoch 文件路径
        """
            # 创建保存目录
        
        os.makedirs(base_path, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 文件路径
        model_path = os.path.join(base_path, f"model_{timestamp}.pth")
        optimizer_path = os.path.join(base_path, f"optimizer_{timestamp}.pth")
        epoch_path = os.path.join(base_path, f"epoch_{timestamp}.pth")
        
        # 保存模型状态字典
        torch.save(model.state_dict(), model_path)
        # print(f"模型状态字典保存到 {model_path}")

        # 保存优化器状态字典
        torch.save(optimizer.state_dict(), optimizer_path)
        # print(f"优化器状态字典保存到 {optimizer_path}")

        # 保存当前 epoch
        torch.save({'epoch': epoch}, epoch_path)
        # print(f"当前 epoch 保存到 {epoch_path}")

    def concatSto25(self,x,rd):
        # n,c,t,v,m:  
        N,C,T,V,M = x.shape
        #  N:batchsize, C:3, T:50, V:25, M:2
        y =  np.zeros((N,C,T,25,M),dtype=float)
        y[:,:,:,0,:] =  x[:,:,:,0,:] 
        y[:,:,:,1,:] =  x[:,:,:,7,:] 
        y[:,:,:,2,:] =  x[:,:,:,8,:] 
        y[:,:,:,3,:] =  x[:,:,:,10,:]
        y[:,:,:,4,:] =  x[:,:,:,11,:]
        y[:,:,:,5,:] =  x[:,:,:,12,:]
        y[:,:,:,6,:] =  x[:,:,:,13,:]
        y[:,:,:,7,:] = rd[:,:,:,0,:]
        y[:,:,:,8,:] =  x[:,:,:,14,:]
        y[:,:,:,9,:] =  x[:,:,:,15,:]
        y[:,:,:,10,:] = x[:,:,:,16,:]
        y[:,:,:,11,:] = rd[:,:,:,1,:]
        y[:,:,:,12,:] = x[:,:,:,4,:]
        y[:,:,:,13,:] = x[:,:,:,5,:]
        y[:,:,:,14,:] = x[:,:,:,6,:]
        y[:,:,:,15,:] = rd[:,:,:,2,:]
        y[:,:,:,16,:] = x[:,:,:,1,:]
        y[:,:,:,17,:] = x[:,:,:,2,:]
        y[:,:,:,18,:] = x[:,:,:,3,:]
        y[:,:,:,19,:] = rd[:,:,:,3,:]
        y[:,:,:,20,:] = rd[:,:,:,4,:]
        y[:,:,:,21,:] = rd[:,:,:,5,:]
        y[:,:,:,22,:] = rd[:,:,:,6,:]
        y[:,:,:,23,:] = rd[:,:,:,7,:]
        y[:,:,:,24,:] = rd[:,:,:,8,:]

        return y 

    # def kmeans(self,X, k=60, max_iters=100):
    #     # 1. 随机初始化聚类中心
    #     X=X.detach()
    #     X=X.to("cuda")
    #     # centers = X[np.random.choice(range(X.shape[0]), size=k, replace=False)]
    #     random_indices = random.sample(range(X.size(0)), k)

    #     # 根据随机索引选择对应的张量
    #     centers = X[random_indices]

    #     for _ in range(max_iters):
            
            
    #         new_centroids=[ [] for _ in range(k)]

    #         # 将每个样本分配到最近的中心
    #         for idx in range(X.size(0)):
    #             # 计算每个样本到各个中心的距离
    #             c_idx = min(range(len(centers)), key=lambda index: self.compute_distance(X[idx], centers[index]))
                
    #             new_centroids[c_idx].append(X[idx])
            
    #         # 保证聚类中心不为0
    #         for i in range(len(new_centroids)):
    #             if len(new_centroids[i])!=0:
    #                 new_centroids[i] = torch.stack(new_centroids[i])
    #             else:
    #                 new_centroids[i] = centers[i].unsqueeze(0)
                
    #         # # 4. 更新聚类中心
    #         new_centers = [torch.mean(center, dim=0) for center in new_centroids if len(center) != 0]
    #         if new_centers:
    #             new_centers  = torch.stack(new_centers )
    
    #         # 5. 检查是否收敛
    #         if centers.shape != new_centers.shape: 
    #             centers = new_centers
    #         else:
    #             if torch.eq(centers ,new_centers).all():
    #                 break
    #             else:
    #                 centers = new_centers

    #     # 为聚类中心设置聚类标签
    #     y = []
    #     for idx in range(X.size(0)):
    #         c_idx = min(range(len(centers)), key=lambda index: self.compute_distance(X[idx], centers[index]))
    #         y.append(c_idx) 
    #     c = []
    #     for idx in range(centers.size(0)):
    #         c.append(idx)
    #     print("Cluster X :{}".format(X.shape))
    #     print("Cluster Y :{}".format(len(y)))
    #     print("Cluster centers :{}".format(centers.shape))
    #     print("Cluster cY :{}".format(len(c)))
    #     return centers , c , y
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

    def ntu25to17_2d(self,x):

        '''
            Input: x (B x M x T x V x C)
            B:batchsize
            ntu: (7,22,21,11,24,23,18,14)
            x                   y
            {0: base spine;     0:root
             1: mid spine       7
             2: neck            8
             3: head            10
             4: L_shoulder      11
             5: L_elbow         12
             6: L-wrist         13
             8: R_shoulder      14
             9: R_elbow         15
             10: R-wrist        16
             12: L-hip          4 :rhip
             13: L-knee         5
             14: L-ankle        6
             16: R-hip          1
             17: R-knee         2
             18: R-ankle        3   
             nose               9 : 0.4*neck+0.6*head


            
            
            H36M:
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'neck',
            9: 'nose',
            10: 'head',
            11: 'lsho',
            12: 'lelb',
            13: 'lwri',
            14: 'rsho',
            15: 'relb',
            16: 'rwri'
        '''
        # n,c,t,v,m:  
        N,C,T,V,M = x.shape
        #  N:batchsize, C:3, T:50, V:25, M:2
        y =  np.zeros((N,3,T,17,M),dtype=float)
        rx = np.zeros((N,C,T,9,M),dtype=float)
        y[:,:,:,0,:] = x[:,:,:,0,:] 
        y[:,:,:,1,:] = x[:,:,:,16,:]
        y[:,:,:,2,:] = x[:,:,:,17,:]
        y[:,:,:,3,:] = x[:,:,:,18,:]
        y[:,:,:,4,:] = x[:,:,:,12,:]
        y[:,:,:,5,:] = x[:,:,:,13,:]
        y[:,:,:,6,:] = x[:,:,:,14,:]
        y[:,:,:,8,:] = x[:,:,:,2,:] 
        y[:,:,:,7,:] = x[:,:,:,1,:] 
        y[:,:,:,9,:] = x[:,:,:,2,:]*0.4 + x[:,:,:,3,:]*0.6
        y[:,:,:,10,:] = x[:,:,:,3,:]
        y[:,:,:,11,:] = x[:,:,:,4,:]
        y[:,:,:,12,:] = x[:,:,:,5,:]
        y[:,:,:,13,:] = x[:,:,:,6,:]
        y[:,:,:,14,:] = x[:,:,:,8,:]
        y[:,:,:,15,:] = x[:,:,:,9,:]
        y[:,:,:,16,:] = x[:,:,:,10,:]
        

        # 剩余的9个关节点
        rx[:,:,:,0,:] = x[:,:,:,7,:] 
        rx[:,:,:,1,:] = x[:,:,:,11,:]
        rx[:,:,:,2,:] = x[:,:,:,15,:]
        rx[:,:,:,3,:] = x[:,:,:,19,:]
        rx[:,:,:,4,:] = x[:,:,:,20,:]
        rx[:,:,:,5,:] = x[:,:,:,21,:]
        rx[:,:,:,6,:] = x[:,:,:,22,:]
        rx[:,:,:,7,:] = x[:,:,:,23,:] 
        rx[:,:,:,8,:] = x[:,:,:,24,:] 


        # y=  torch.from_numpy(y).float().to("cuda")

        return y , rx
    
    def set_parameters(self, model):
        for (nn, np), (on, op) in zip(model.named_parameters(), self.model.named_parameters()):
            op.data = np.data.clone()
    

    


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

def mlp2_model_parameter_vector(model):
    mlp2_param = []
    for name,p in model.named_parameters():
        # if 'stgcn_end' in name:
        #     #print("mlpname:{}".format(name))
        mlp2_param.append(p.view(-1))
    l_para =  torch.cat(mlp2_param, dim=0)  
    return l_para

def flip_data(data):
    """
    horizontal flip
        data: [N, F, 17, D] or [F, 17, D]. X (horizontal coordinate) is the first channel in D.
    Return
        result: same
    """
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]
    flipped_data = copy.deepcopy(data)
    # 将第一维的坐标取反
    flipped_data[..., 0] *= -1     
    # print(flipped_data.shape)                                          # flip x of all joints
    flipped_data[..., left_joints+right_joints, :] = flipped_data[..., right_joints+left_joints, :]
    return flipped_data


