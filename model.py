"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2018-2019
"""

import numpy as np
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, img_size, n_in, base_f, n_layers, n_out, dropout=0.5):
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
                pool = nn.MaxPool2d(2,2,1)
                layer = nn.Sequential(pool, bn, conv, nn.ReLU())
            else:
                layer = nn.Sequential(bn, conv, nn.ReLU())
            self.conv_layers.append(layer)
        self.conv_layers = nn.Sequential(*self.conv_layers)

        # fully-connected layers
        self.fc_layers = []
        in_shape = int(img_size/2**n_pools)
        n_feat = [n_feat[-1]*in_shape,n_feat[-1]*in_shape//4,n_out]
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
        x = x.view(x.size[0],-1)
        x = self.fc_layers(x)
        return x