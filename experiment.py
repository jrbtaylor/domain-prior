"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2018-2019
"""

import json
import os

import numpy as np
from progressbar import ProgressBar
import torch
from torch.autograd import Variable

import data
from model import VGG
import train

def run(pixelcnn_ckpt, vgg_ckpt=None, adversarial_range=0.2,
        train_dataset='mnist', test_dataset='emnist', n_classes=10, img_size=28,
        vgg_params={'batch_size':64, 'base_f':16, 'n_layers':7, 'dropout':0.5,
                    'optimizer':'adam','learnrate':1e-4, 'dropout':0.5},
        exp_name='domain-prior', exp_dir='~/experiments/domain-prior/',
        cuda=True, resume=False):

    # Set up experiment directory
    exp_name += '_%s-to-%s_vgg%i-%i_adv%.2f'%(
        train_dataset, test_dataset, vgg_params['n_layers'],
        vgg_params['base_f'], adversarial_range)
    exp_dir = os.path.join(os.path.expanduser(exp_dir), exp_name)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)

    # Train a VGG classifier if not already done
    if vgg_ckpt is None:
        with open(os.path.join(exp_dir,'vgg_params.json'),'w') as f:
            json.dump(vgg_params,f)
        vgg = VGG(img_size, 1, vgg_params['base_f'], vgg_params['n_layers'],
                  n_classes, vgg_params['dropout'])
        train_loader, val_loader = data.loader(train_dataset,
                                               vgg_params['batch_size'])
        train.fit(train_loader, val_loader, vgg, exp_dir, torch.nn.NLLLoss(),
                  vgg_params['optimizer'], vgg_params['learnrate'], cuda,
                  resume=resume)
    else:
        vgg = torch.load(vgg_ckpt)

    pixelcnn = torch.load(pixelcnn_ckpt)
    pixelcnn_params = os.path.join(os.path.dirname(pixelcnn_ckpt),'params')
    with open(pixelcnn_params,'r') as f:
        pixelcnn_params = json.load(f)
    n_bins = pixelcnn_params['n_bins']

    # Run the datasets through the networks and calculate 3 pixelcnn losses:
    # 1. Average: mean across the image
    # 2. High-pass filtered: weight by difference to upper- and left- neighbors
    # 3. Saliency: weight by pixel saliency (vgg backprop-to-input)
    _,loader = data.loader(train_dataset,16)
    domain_losses = calc_losses(vgg,pixelcnn,loader,n_bins,cuda)
    _,loader = data.loader(test_dataset,16)
    external_losses = calc_losses(vgg,pixelcnn,loader,n_bins,cuda)


    # TO-DO: add adversarial examples



def calc_losses(vgg, pixelcnn, dataloader, n_bins, cuda=True):
    bar = ProgressBar()
    loss_fcn = torch.nn.NLLLoss2d()
    avg_losses = []
    hpf_losses = []
    sal_losses = []
    for x,y in bar(dataloader):
        pcnn_label = torch.squeeze(
            torch.round((n_bins-1)*x).type(torch.LongTensor), 1)
        if cuda:
            x,y = x.cuda(),y.cuda()
            pcnn_label = pcnn_label.cuda()
            vgg = vgg.cuda()
            pixelcnn = pixelcnn.cuda()
        x,y = Variable(x,requires_grad=True), Variable(y)
        vgg.eval()
        pixelcnn.eval()

        # get the saliency maps
        classifier_output = vgg(x)
        classifier_argmax = np.argmax(classifier_output.data.cpu().numpy(),-1)
        classifer_max = classifier_output[range(len(classifier_argmax)),
                                          classifier_argmax]
        saliency = classifer_max.backward(retain_variables=True)
        saliency = saliency.data.cpu().numpy()

        pcnn_out = pixelcnn(x,classifier_output)
        loss = loss_fcn(pcnn_out,pcnn_label).data.cpu().numpy()

        avg_losses += np.mean(loss,axis=-1).tolist()

        weight = x.data.cpu().numpy()
        weight = np.abs(weight[:,:,1:,1:]-0.5*(weight[:,:,:-1,1:]
                                               +weight[:,:,1:,:-1]))
        weight = weight/np.sum(weight,axis=[1,2,3],keepdims=True)
        hpf_losses += np.sum(loss*weight)

        saliency = saliency/np.sum(saliency,axis=[1,2,3],keepdims=True)
        sal_losses += np.sum(loss*saliency)

    return avg_losses, hpf_losses, sal_losses








