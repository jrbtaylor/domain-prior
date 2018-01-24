"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2018-2019
"""

import numpy as np
import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, img_size, n_in, base_f, n_layers, n_out):
        super(VGG,self).__init__()

        # automatically insert pooling so final feature map is at least 4x4
        n_pools = int(np.log2(img_size)-2)
        n_scales = n_pools+1
        layers_per_scale = n_layers//n_scales
        deeper_scales = n_layers%n_scales
        n_feat = [[base_f*(2**i)]*(
            layers_per_scale+(i+1>n_layers/layers_per_scale-deeper_scales))
                  for i in range(n_pools+1)]
        n_feat = [n_in]+[f for nf in n_feat for f in nf]

        # conv layers
        self.conv_layers = []
        for i in range(len(n_feat)-1):
            bn = nn.BatchNorm2d(n_feat[i],momentum=0.1)
            conv = nn.Conv2d(n_feat[i],n_feat[i+1],3,padding=1)
            if i>0 and n_feat[i]==n_feat[i+1]//2:
                pool = nn.MaxPool2d(2,2,0)
                layer = nn.Sequential(pool, bn, conv, nn.ReLU())
            else:
                layer = nn.Sequential(bn, conv, nn.ReLU())
            self.conv_layers.append(layer)
        self.conv_layers = nn.Sequential(*self.conv_layers)

        # fully-connected layers
        self.fc_layers = []
        self.n_flatten = n_feat[-1]*int(img_size/2**n_pools)**2
        n_feat = [self.n_flatten,self.n_flatten//4,n_out]
        for i in range(len(n_feat)-1):
            bn = nn.BatchNorm1d(n_feat[i],momentum=0.1)
            proj = nn.Linear(n_feat[i],n_feat[i+1])
            if i==len(n_feat)-2:
                act = nn.LogSoftmax(dim=1)
            else:
                act = nn.ReLU()
            layer = nn.Sequential(bn, proj, act)
            self.fc_layers.append(layer)
        self.fc_layers = nn.Sequential(*self.fc_layers)

    def forward(self,x):
        x = self.conv_layers(x)
        x = x.view(-1,self.n_flatten)
        x = self.fc_layers(x)
        return x


class MaskedConv(nn.Conv2d):
    def __init__(self,mask_type,in_channels,out_channels,kernel_size,stride=1):
        """
        mask_type: 'A' for first layer of network, 'B' for all others
        """
        super(MaskedConv,self).__init__(in_channels,out_channels,kernel_size,
                                        stride,padding=kernel_size//2)
        assert mask_type in ('A','B')
        mask = torch.ones(1,1,kernel_size,kernel_size)
        mask[:,:,kernel_size//2,kernel_size//2+(mask_type=='B'):] = 0
        mask[:,:,kernel_size//2+1:] = 0
        self.register_buffer('mask',mask)

    def forward(self,x):
        self.weight.data *= self.mask
        return super(MaskedConv,self).forward(x)


class GatedRes(nn.Module):
    def __init__(self,in_channels,out_channels,n_classes,kernel_size=3,stride=1,
                 aux_channels=0):
        super(GatedRes,self).__init__()
        self.conv = MaskedConv('B',in_channels,2*out_channels,kernel_size,
                               stride)
        self.y_embed = nn.Linear(n_classes,2*out_channels)
        self.out_channels = out_channels
        if aux_channels!=2*out_channels and aux_channels!=0:
            self.aux_shortcut = nn.Sequential(
                nn.Conv2d(aux_channels,2*out_channels,1),
                nn.BatchNorm2d(2*out_channels,momentum=0.1))
        if in_channels!=out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,1),
                nn.BatchNorm2d(out_channels,momentum=0.1))
        self.batchnorm = nn.BatchNorm2d(out_channels,momentum=0.1)

    def forward(self,x,y):
        # check for aux input from first half of net stacked into x
        if x.dim()==5:
            x,aux = torch.split(x,1,dim=0)
            x = torch.squeeze(x,0)
            aux = torch.squeeze(x,0)
        else:
            aux = None
        x1 = self.conv(x)
        y = torch.unsqueeze(torch.unsqueeze(self.y_embed(y),-1),-1)
        if aux is not None:
            if hasattr(self,'aux_shortcut'):
                if aux.dim()==3:  # bug: batch size of 1 gets removed
                    aux = aux.unsqueeze(0)
                aux = self.aux_shortcut(aux)
            x1 = (x1+aux)/2
        # split for gate (note: pytorch dims are [n,c,h,w])
        xf,xg = torch.split(x1,self.out_channels,dim=1)
        yf,yg = torch.split(y,self.out_channels,dim=1)
        f = torch.tanh(xf+yf)
        g = torch.sigmoid(xg+yg)
        if hasattr(self,'shortcut'):
            x = self.shortcut(x)
        return x+self.batchnorm(g*f)


class PixelCNN(nn.Module):
    def __init__(self,in_channels,n_classes,n_features,n_layers,n_bins,
                 dropout=0.5):
        super(PixelCNN,self).__init__()

        self.layers = nn.ModuleList()
        self.n_layers = n_layers

        # Up pass
        self.input_batchnorm = nn.BatchNorm2d(in_channels,momentum=0.1)
        for l in range(n_layers):
            if l==0:  # start with normal conv
                block = nn.Sequential(
                    MaskedConv('A',in_channels+1,n_features,kernel_size=7),
                    nn.BatchNorm2d(n_features,momentum=0.1),
                    nn.ReLU())
            else:
                block = GatedRes(n_features, n_features, n_classes)
            self.layers.append(block)

        # Down pass
        for _ in range(n_layers):
            block = GatedRes(n_features, n_features,n_classes,
                             aux_channels=n_features)
            self.layers.append(block)

        # Last layer: project to n_bins (output is [-1, n_bins, h, w])
        self.dropout = nn.Dropout2d(dropout)
        self.layers.append(GatedRes(n_features,n_bins,n_classes))
        self.layers.append(nn.LogSoftmax(dim=1))

    def forward(self,x,y):
        # Add channel of ones so network can tell where padding is
        x = nn.functional.pad(x,(0,0,0,0,0,1,0,0),mode='constant',value=1)

        # Up pass
        features = []
        i = -1
        for _ in range(self.n_layers):
            i += 1
            if i>0:
                x = self.layers[i](x,y)
            else:
                x = self.layers[i](x)
            features.append(x)

        # Down pass
        for _ in range(self.n_layers):
            i += 1
            x = self.layers[i](torch.stack((x,features.pop())),y)

        # Last layer
        x = self.dropout(x)
        i += 1
        x = self.layers[i](x,y)
        i += 1
        x = self.layers[i](x)
        assert i==len(self.layers)-1
        assert len(features)==0
        return x
