# The based unit of graph convolutional networks.

import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

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
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        
        rel_reduction=8
        mid_reduction=1

        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # if in_channels == 3 or in_channels == 9:
        #     self.rel_channels = 8
        #     self.mid_channels = 16
        # else:
        #     self.rel_channels = in_channels // rel_reduction
        #     self.mid_channels = in_channels // mid_reduction
        # self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        # self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        # # self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        # self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        # self.tanh = nn.Tanh()
        # # self.alpha = 1

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        dx = x
        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        
        
        # # graph
        # # x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        # x1 = self.conv1(dx)
        # x2 =self.conv2(dx)
        # graph = self.tanh(x1.mean(-2).unsqueeze(-1) - x2.mean(-2).unsqueeze(-2))
        # graph = self.conv4(graph)
        # # graph_c = graph * self.alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        # # y = torch.einsum('ncuv,nctv->nctu', graph_c, x3)

        return x.contiguous(), A