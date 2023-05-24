import math
import pdb
from termios import VINTR

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from torch import Tensor
from typing import Tuple
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class PositionEmbeddingSine(nn.Module):
    def __init__(self, numPositionFeatures: int = 64, temperature: int = 10000, normalize: bool = True,
                 scale: float = None):
        super(PositionEmbeddingSine, self).__init__()

        self.numPositionFeatures = numPositionFeatures
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        N, _, H, W = x.shape

        mask = torch.zeros(N, H, W, dtype=torch.bool, device=x.device)
        notMask = ~mask

        yEmbed = notMask.cumsum(1)
        xEmbed = notMask.cumsum(2)

        if self.normalize:
            epsilon = 1e-6
            yEmbed = yEmbed / (yEmbed[:, -1:, :] + epsilon) * self.scale
            xEmbed = xEmbed / (xEmbed[:, :, -1:] + epsilon) * self.scale

        dimT = torch.arange(self.numPositionFeatures, dtype=torch.float32, device=x.device)
        dimT = self.temperature ** (2 * (dimT // 2) / self.numPositionFeatures)

        posX = xEmbed.unsqueeze(-1) / dimT
        posY = yEmbed.unsqueeze(-1) / dimT

        posX = torch.stack((posX[:, :, :, 0::2].sin(), posX[:, :, :, 1::2].cos()), -1).flatten(3)
        posY = torch.stack((posY[:, :, :, 0::2].sin(), posY[:, :, :, 1::2].cos()), -1).flatten(3)

        return torch.cat((posY, posX), 3).permute(0, 3, 1, 2)
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class TSGCNext_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True,layer_scale_init_value=1e-6,drop_path=0.):
        super(TSGCNext_unit, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        if residual:
            self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=(3,1), padding=(1,0), groups=in_channels) # depthwise conv Twise
        else:
            self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=(4,1), stride=(4,1))
            in_channels = out_channels
        self.norm = LayerNorm(in_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channels, 4 * in_channels) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
       
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        

        self.pwconv2 = nn.Linear(4 * in_channels, in_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((in_channels)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if in_channels != out_channels:
            self.down = nn.Sequential(
                LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(in_channels, out_channels, kernel_size=(2,1), stride=(2,1)),
        )
        else:
            self.down = None

        self.residual = residual
        self.alpha = nn.Parameter(torch.zeros(1))
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x, preA=None):
        if preA!=None:
            A = self.PA
        else:
            A = self.PA
        input = x
        x = self.dwconv(x) # (N,C,T,V)
        x = x.permute(0, 2, 3, 1) # (N, C, T, V) -> (N, T, V, C)
        x = self.norm(x)
        x = self.pwconv1(x)#(N, T, V, 4*C)
        N,T,V,C4 = x.shape
        x = x.reshape(N,T,V,-1,4)
        x1 = x[:,:,:,:,0:3]
        x2 = x[:,:,:,:,3:4]
        if self.training:
            Ak = torch.unsqueeze(A,dim=0).repeat_interleave(N, dim=0)#.permute(0, 3, 2, 1).contiguous().view(-1,V,3)
            x1 = torch.einsum('niuk,ntkci->ntuci', Ak,x1)
        else:
            x1 = torch.einsum('iuk,ntkci->ntuci', A,x1)
        #x1 = torch.einsum('kuv,ntvck->ntuck', A,x1)
        x = torch.cat([x1,x2],dim=-1)
        x = x.reshape(N,T,V,C4)
        x = self.act(x)
        x = self.pwconv2(x)
        #if self.gamma is not None:
        #    x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        if self.residual:
            y = input + self.drop_path(x)
        else:
            y = self.drop_path(x)

        if self.down!=None:
            y = self.down(y)

        return y,A



class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True,unify=None):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25
        if unify == "coco":
            from data.unifyposecode import COCO
            vindex = COCO
            A = A[:,vindex]
            A = A[:,:,vindex]
            num_point = len(vindex)
        elif unify == "ntu":
            from data.unifyposecode import NTU
            vindex = NTU
            A = A[:,vindex]
            A = A[:,:,vindex]
            num_point = len(vindex)
        
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 96
        dp_rates=[x.item() for x in torch.linspace(0, drop_out, 9)]
        stem = nn.Sequential(
            nn.Conv2d(in_channels,  base_channel, kernel_size=(4,1), stride=(4,1)),
            LayerNorm( base_channel, eps=1e-6, data_format="channels_first")
        )
        self.l1  = stem #TSGCNext_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive,drop_path=dp_rates[0])
        # 2 5 2
        self.l2  = TSGCNext_unit(base_channel, base_channel, A, adaptive=adaptive,drop_path=dp_rates[0])
        self.l3  = TSGCNext_unit(base_channel, base_channel, A, adaptive=adaptive,drop_path=dp_rates[1])
        self.l4  = TSGCNext_unit(base_channel, base_channel, A, adaptive=adaptive,drop_path=dp_rates[2])
        self.l5  = TSGCNext_unit(base_channel, base_channel*2, A, adaptive=adaptive,drop_path=dp_rates[3])
        self.l6  = TSGCNext_unit(base_channel*2, base_channel*2, A, adaptive=adaptive,drop_path=dp_rates[4])
        self.l7  = TSGCNext_unit(base_channel*2, base_channel*2, A, adaptive=adaptive,drop_path=dp_rates[5])
        self.l8  = TSGCNext_unit(base_channel*2, base_channel*4, A, adaptive=adaptive,drop_path=dp_rates[6])
        self.l9  = TSGCNext_unit(base_channel*4, base_channel*4, A, adaptive=adaptive,drop_path=dp_rates[7])
        self.l10 = TSGCNext_unit(base_channel*4, base_channel*4, A, adaptive=adaptive,drop_path=dp_rates[8])
        
        self.norm = nn.LayerNorm(base_channel*4, eps=1e-6)
        #self.encode = nn.Linear(base_channel*4, base_channel*4)
        self.fc = nn.Linear(base_channel*4, num_class)
        
        self.in_channels = in_channels
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
    
        

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        if self.in_channels == 2:
            x = x[:,0:2,:,:,:]
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x,A1 = self.l2(x)
        x,A2 = self.l3(x,A1)
        x,A3 = self.l4(x,A2)
        x,A4 = self.l5(x,A3)
        x,A5 = self.l6(x,A4)
        x,A6 = self.l7(x,A5)
        x,A7 = self.l8(x,A6)
        x,A8 = self.l9(x,A7)
        x,A9 = self.l10(x,A8)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.norm(x)
        #x = self.encode(x)
        return self.fc(x)
