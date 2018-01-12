"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2018-2019
"""

import os
import random

import numpy as np
import torch
from torchvision import datasets, transforms


def onehot(n_classes):
    def onehot_fcn(x):
        y = np.zeros((n_classes), dtype='float32')
        y[x] = 1
        return y
    return onehot_fcn


def augment(img_size=28,scale=0.1):
    if random.random()<0.5:  # random crop & upsampling
            resize = transforms.RandomResizedCrop(
                img_size, scale=(1-scale,1.0), ratio=(1-scale,1/(1-scale)))
    else:  # random pad & downsampling
        left = random.randint(0,int(np.round(scale*img_size)))
        right = random.randint(0,int(np.round(scale*img_size)-left))
        top = random.randint(0,int(np.round(scale*img_size)))
        bottom = random.randint(0, int(np.round(scale*img_size)-top))
        resize = transforms.Compose([transforms.Pad((left,top,right,bottom)),
                                     transforms.Resize((img_size,img_size))])
    return transforms.Compose([resize,transforms.ToTensor()])


def loader(dataset, batch_size, n_workers=8):
    assert dataset.lower() in ['mnist','emnist','fashionmnist']

    loader_args = {'batch_size':batch_size,
                   'num_workers':n_workers,
                   'pin_memory':True}
    datapath = os.path.join(os.getenv('HOME'), 'data', dataset.lower())
    dataset_args = {'root':datapath,
                    'download':True,
                    'transform':augment()}

    if dataset.lower()=='mnist':
        dataset_init = datasets.MNIST
        n_classes = 10
    elif dataset.lower()=='emnist':
        dataset_init = datasets.eMNIST
        n_classes = 62
        dataset_args.update({'split':'byclass'})
    else:
        dataset_init = datasets.FashionMNIST
        n_classes = 10
    onehot_fcn = onehot(n_classes)
    dataset_args.update({'target_transform':onehot_fcn})

    train_loader = torch.utils.data.DataLoader(
        dataset_init(train=True, **dataset_args), shuffle=True, **loader_args)
    val_loader = torch.utils.data.DataLoader(
        dataset_init(train=False, **dataset_args), shuffle=False, **loader_args)

    return train_loader, val_loader