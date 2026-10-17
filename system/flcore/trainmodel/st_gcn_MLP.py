import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.tgcn import ConvTemporalGraphical
from .utils.graph import Graph

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 定义第一个全连接层
        self.fc2 = nn.Linear(hidden_size, output_size)  # 定义第二个全连接层

    def forward(self, x):
        #print("x.shape:{}".format(x.shape))
        #print("self.weight:{}".format(self.fc1.weight.shape))
        x = torch.relu(self.fc1(x))  # 使用ReLU激活函数进行第一层的前向传播
        x = self.fc2(x)  # 进行第二层的前向传播
        return x
    
class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks."""
    
    def __init__(self, in_channels, hidden_channels, hidden_dim,num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1)).cuda()
        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        
        # self.st_gcn_networks1 = nn.ModuleList((
        #     st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0)

        # ))
        #  self.stgcn_start = st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0)

        # #x: 128*32*25*25
        # # server
        # self.mlpsize1 = 20000
        # self.midmlp1= MLP(self.mlpsize1,hidden_channels * 4,self.mlpsize1)
        
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
        ))
        # #
        # self.mlpsize2 = 64*13*25
        # self.midmlp2= MLP(self.mlpsize2,hidden_channels * 4,self.mlpsize2)
        # # self.st_gcn_networks3 = nn.ModuleList((
        # #     st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs)
        # # ))

        self.stgcn_end = st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs)
        

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            #self.edge_importance1 = nn.Parameter(torch.ones(self.A.size()))

            self.edge_importance1 = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
            self.edge_importance2 = nn.Parameter(torch.ones(self.A.size()))

        else:
            # self.edge_importance1 = [1] * len(self.st_gcn_networks1)
            # self.edge_importance1 = [1] 
            self.edge_importance1 = [1] * len(self.st_gcn_networks)
            # self.edge_importance3 = [1] * len(self.st_gcn_networks3)
            self.edge_importance2 = [1] 

    def add_logger(self,logger):
        self.logger = logger
    
    def forward(self, x):


        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        #print(x.shape)
        
        # parameters = self.data_bn.state_dict()
        # weight_size = parameters['weight'].size()
        # self.logger.info('x = self.data_bn(x) before data_bn.weight.size():%s ', weight_size)
        x = self.data_bn(x)

        # parameters = self.data_bn.state_dict()
        # weight_size = parameters['weight'].size()
        # self.logger.info('x = self.data_bn(x) afterdata_bn.weight.size():%s ', weight_size)
        

        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)




        # # for gcn, importance in zip(self.st_gcn_networks1, self.edge_importance1):
        # #     x, _ = gcn(x, self.A * importance)
        # x, _ = self.stgcn_start(x, self.A * self.edge_importance1)
        # # flat x [128,32,25,25]->[128,32*2 5*25]
        # #print("x_mid.shape:{}".format(x_mid.shape))
        
        # x_flat = x.view(x.shape[0], -1)
        # #print("x1.shape:{}".format(x.shape))
        # #print("x1_flat.shape:{}".format(x_flat.shape))
        # x_mid = self.midmlp1(x_flat)
        # #print("x_mid.shape:{}".format(x_mid.shape))
        # #print("x.shape:{}".format(x.shape))
        # # restore 
        # x = x_mid.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])



        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance1):
            x, _ = gcn(x, self.A * importance)
        

        # # nextmlp input
        # x_flat = x.view(x.shape[0], -1)
        # # print("x2.shape:{}".format(x.shape))
        # # print("x2_flat.shape:{}".format(x_flat.shape))

        # x_mid = self.midmlp2(x_flat)
        # # nextmlp restore
        # x = x_mid.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        # for gcn, importance in zip(self.st_gcn_networks3, self.edge_importance3):
        #     x, _ = gcn(x, self.A * importance)
        x, _ = self.stgcn_end(x, self.A * self.edge_importance2)

        x = F.avg_pool2d(x, x.size()[2:])
        #print("F.avg_pool2d(x, x.size()[2:]).shape:{}".format(x.shape))
        x = x.view(N, M, -1).mean(dim=1)
        #print("x.view(N, M, -1).mean(dim=1).shape:{}".format(x.shape))
        

        return x


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
            

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, A):
        #print(self.weight)
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        
        return self.relu(x), A