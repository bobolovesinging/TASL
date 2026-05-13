import time
import copy
import os
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
from utils.model_utils import *
from utils.DSTformer import *
from utils.model_action import *
import wandb
import logging
from geomstats.learning.kmeans import RiemannianKMeans
from geomstats.geometry.hypersphere import Hypersphere


from scipy.interpolate import griddata, RectBivariateSpline
from scipy.interpolate import CubicSpline
from scipy.interpolate import Rbf
from scipy.interpolate import interp1d

from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
matplotlib.use('Agg')



class FedBaseline(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.avgstage = args.avgstage
        self.inpo =args.interpolate #数据插值方式
        self.in_t =args.inter_t #插值系数
        self.in_num =args.inter_num #插值个数
        self.pth_path = '../log/mdpath/'+str(self.in_t)+"_"+str(self.in_num)+"/"

        # logging.basicConfig(level=logging.INFO, filename='log/base.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
        # self.logger = logging.getLogger("base_log")
        
        self.c3d =False
        self.loss = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        # model: optimizer
        pre_lr1 = 0.001 # batchsize(512=4*128 sqrt(4))
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

  
    def train(self):
        mode = 1
        #self.para_frozen(self.global_model , mode)
        global total_sk_seen
        total_sk_seen = 0
        self.queue_warmup = True
        start_epoch = 0

        # self.load_model(self.cepoch)
        # print(self.cepoch)
        # start_epoch =self.cepoch+1 
        # for index, client in enumerate(self.selected_clients):
        #     client.load_opacl(self.cepoch)
        for i in range(start_epoch ,self.global_rounds):
            
            s_t = time.time()
            self.selected_clients = self.select_clients()
            # for index, client in enumerate(self.selected_clients):
            #     print(client.id)
            self.send_models()
            # if i%5 == 0:
            #     self.save_model(self.global_model,i)
            
            # if self.c3d:
            #     if i!=0:
            #         self.send_3ddata()
            # contrast loss
            self.runningrounds = i

            if i%self.eval_gap == 0:                
                print("\nEvaluate global model")
                # self.knn_evaluate()
                # self.linear_train()
                self.evaluate()

            num_samples = []
            losses = 0

            print("\n-------------Round number: {}-------------".format(i))
            for index, client in enumerate(self.clients):
                loss,ns = client.pretrain_avg(i)
                num_samples.append(ns)
                losses +=loss

            print("Pretrain_loss:{}".format(losses/len(self.clients)))            
            
            # self.pfl_evaluate()
            self.receive_models()
            # self.receive_classifiers()
            self.aggregate_parameters()
            # self.aggregate_classifiers()



            # if self.dlg_eval and i%self.dlg_gap == 0:
            #     self.call_dlg(i)
            
            # self.aggregate_parameters()

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
        # 联邦学习冷启动
        # 在训练完成后，如果有新客户端加入，对新客户端进行微调和评估
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print("\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
    
   