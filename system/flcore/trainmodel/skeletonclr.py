import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flcore.trainmodel.generator import pos_generator
# from torchlight import import_class  # 注释掉，因为torchlight包没有这个函数
from flcore.trainmodel.net.st_gcn import Model
from flcore.trainmodel.net.utils.graph import normalize_undigraph
from geomstats.learning.kmeans import RiemannianKMeans
from geomstats.geometry.hypersphere import Hypersphere
class SkeletonCLR(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=128, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """


        super().__init__()
        #base_encoder = import_class(base_encoder)
        self.pretrain = pretrain

        if not self.pretrain:
            self.encoder_q = Model(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature

            self.encoder_q = Model(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting
                                          )
            self.encoder_k = Model(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting
                                          )
            
            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_k.fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient

            # create the logits queue(origin)
            self.register_buffer("queue", torch.randn(feature_dim, queue_size))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            
       


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys.device.index if keys.device.index is not None else 0
        
        # 计算要更新的队列位置
        start_idx = ptr + batch_size * gpu_index
        end_idx = ptr + batch_size * (gpu_index + 1)
        
        # 处理队列边界情况（循环队列）
        if end_idx <= self.K:
            # 正常情况，没有跨越边界
            self.queue[:, start_idx:end_idx] = keys.T
        else:
            # 跨越边界，需要分段更新
            first_part_size = self.K - start_idx
            self.queue[:, start_idx:] = keys.T[:, :first_part_size]
            self.queue[:, :(end_idx - self.K)] = keys.T[:, first_part_size:]

    @torch.no_grad()
    def update_ptr(self, batch_size):
        # 移除严格断言，允许任意 batch_size
        # 如果 batch_size 不能被 K 整除，我们仍然可以更新指针
        # 只需要确保指针在有效范围内
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K
    

    def forward(self, im_q, im_k=None,server=False, covmatrix={},scnnumber=0,label=None,fnum=0):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """
        if server:
            # compute query features
            # print("server skeletonclr")
            feat_q , q , _= self.encoder_q(im_q)  # queries: NxC
            feat_q = F.normalize(feat_q, dim=1)
            q = F.normalize(q, dim=1)
            # ME+Q ->normalize
            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                feat_k, k ,_= self.encoder_k(im_k)  # keys: NxC
                k = F.normalize(k, dim=1)
                # feat_k = F.normalize(feat_k, dim=1)
            
            if covmatrix!={}:

                augment_feature = []
                generate_num = fnum
                # 增强的feature (根据feat_q的增强版本进行增强):
                for i,(feature,classidx) in enumerate(zip(feat_q,label)):
                    classidx = classidx.cpu().item()
                    feature = feature.cpu().detach()
                    new_feature = generate_new_samples(feature,covmatrix[classidx],generate_num)
                    augment_feature.append(new_feature)
                    # 10*256
                aug_logits = []
                aug_labels = []
          
                augment_feature = torch.stack(augment_feature,dim=0)
                for i in range(generate_num):
                    ax = augment_feature[:,i,:]
                    ax = ax.cuda().to(torch.float32) 
                    ax= self.encoder_k.fc(ax)
                    ax = ax.view(ax.size(0), -1)
                    ax = F.normalize(ax, dim=1)
                    aug_logit ,aug_label= self.aug_forward(q,ax)
                    aug_logits.append(aug_logit)
                    aug_labels.append(aug_label)

            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # f_pos = torch.einsum('nc,nc->n', [feat_q, feat_k]).unsqueeze(-1)
            # negative logits: NxK
            # print("self.q:{}".format(q.shape))
            # print("self.queue:{}".format(self.queue.shape))
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
            # f_neg = torch.einsum('nc,ck->nk', [feat_q, self.feat_queue.clone().detach()])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)
            # feats = torch.cat([f_pos, f_neg], dim=1)

            # apply temperature
            logits /= self.T
            # feats /= self.T
            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            self._dequeue_and_enqueue(k)
            if covmatrix!={}:
                return feat_q, logits, labels ,aug_logits,aug_labels
            else:
                return feat_q, logits, labels 
        else:
            feat_q , q , graph= self.encoder_q(im_q)  # queries: NxC
            feat_q = F.normalize(feat_q, dim=1)
            q = F.normalize(q, dim=1)
            # ME+Q ->normalize
            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                feat_k, k ,_= self.encoder_k(im_k)  # keys: NxC
                k = F.normalize(k, dim=1)
                # feat_k = F.normalize(feat_k, dim=1)

            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # f_pos = torch.einsum('nc,nc->n', [feat_q, feat_k]).unsqueeze(-1)
            # negative logits: NxK
            # print("self.q:{}".format(q.shape))
            # print("self.queue:{}".format(self.queue.shape))
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
            # f_neg = torch.einsum('nc,ck->nk', [feat_q, self.feat_queue.clone().detach()])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)
            # feats = torch.cat([f_pos, f_neg], dim=1)

            # apply temperature
            logits /= self.T
            # feats /= self.T
            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            # dequeue and enqueue
            self._dequeue_and_enqueue(k)
            # self._feat_dequeue_and_enqueue(feat_k)
            

            # 128* queue.size()
            return feat_q,logits, labels

    def aug_forward(self,q,k):
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        self._dequeue_and_enqueue(k)
        return logits,labels

# 保证协方差矩阵为正定矩阵
def nearest_pos_def(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # 计算协方差矩阵的特征值和特征向量
    eigenvalues[eigenvalues < 0] = 0  # 将负特征值设为0，确保所有特征值非负
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T  # 重新构建正定协方差矩阵

# 生成新的样本
def generate_new_samples(feature, cov_matrix, num_generated):
    cov_matrix = nearest_pos_def(cov_matrix)  # 确保协方差矩阵为正定
    jitter = 1e-6  # 初始化抖动值
    while True:
        try:
            B = np.linalg.cholesky(cov_matrix + jitter * np.eye(cov_matrix.shape[0]))  # 计算协方差矩阵的Cholesky分解
            break
        except np.linalg.LinAlgError:
            jitter *= 10  # 如果Cholesky分解失败，增大抖动值重新计算

    new_features = np.random.multivariate_normal(feature, B @ B.T, num_generated)  # 使用多元正态分布生成新样本
    return torch.tensor(new_features)

