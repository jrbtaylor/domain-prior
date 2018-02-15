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
import model
import train
import vis

def run(pixelcnn_ckpt, vgg_ckpt=None, adversarial_range=0.2,
        train_dataset='mnist', test_dataset='emnist', img_size=28,
        vgg_params={'batch_size':16, 'base_f':16, 'n_layers':9, 'dropout':0.8,
                    'optimizer':'adam','learnrate':1e-4},
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
        train_loader, val_loader, n_classes = data.loader(
            train_dataset,vgg_params['batch_size'])
        if not resume:
            with open(os.path.join(exp_dir,'vgg_params.json'),'w') as f:
                json.dump(vgg_params,f)
            vgg = model.VGG(img_size, 1, vgg_params['base_f'],
                            vgg_params['n_layers'], n_classes,
                            vgg_params['dropout'])
        else:
            vgg = torch.load(os.path.join(exp_dir, 'best_checkpoint'))

        train.fit(train_loader, val_loader, vgg, exp_dir,
                  torch.nn.CrossEntropyLoss(), vgg_params['optimizer'],
                  vgg_params['learnrate'], cuda, resume=resume)
    else:
        vgg = torch.load(vgg_ckpt)

    pixelcnn = torch.load(pixelcnn_ckpt)
    pixelcnn_params = os.path.join(os.path.dirname(pixelcnn_ckpt),'params.json')
    with open(pixelcnn_params,'r') as f:
        pixelcnn_params = json.load(f)
    n_bins = pixelcnn_params['n_bins']

    if cuda:
        vgg = vgg.cuda()
        pixelcnn = pixelcnn.cuda()

    # Run the datasets through the networks and calculate 3 pixelcnn losses:
    # 1. Average: mean across the image
    # 2. High-pass filtered: weight by difference to upper- and left- neighbors
    # 3. Saliency: weight by pixel saliency (vgg backprop-to-input)
    _,loader,_ = data.loader(train_dataset,1)
    print('Calculating losses for '+train_dataset)
    dom_avg,dom_hp,dom_sw,dom_sal,dom_var = calc_losses(
        vgg,pixelcnn,loader,n_bins,cuda)
    print('Calculating losses for adversarial images')
    adv_avg,adv_hp,adv_sw,adv_sal,adv_var = adversarial(
        vgg,pixelcnn,loader,n_bins,adversarial_range,cuda)
    _,loader,_ = data.loader(test_dataset,1)
    print('Calculating losses for '+test_dataset)
    ext_avg,ext_hp,ext_sw,ext_sal,ext_var = calc_losses(
        vgg,pixelcnn,loader,n_bins,cuda)

    # Loss histograms
    n_bins = 100
    all_losses = np.concatenate((dom_avg, adv_avg, ext_avg,
                                 dom_hp, adv_hp, ext_hp,
                                 dom_sw, adv_sw, ext_sw))
    edges = np.linspace(0, np.percentile(all_losses,95), n_bins+1)
    # average loss
    vis.histogram(dom_avg, edges, train_dataset+' average loss', exp_dir)
    vis.histogram(adv_avg, edges, 'adversarial average loss', exp_dir)
    vis.histogram(ext_avg, edges, test_dataset+' average loss', exp_dir)
    # high-pass weighted loss
    vis.histogram(dom_hp, edges, train_dataset+' highpass loss', exp_dir)
    vis.histogram(adv_hp, edges, 'adversarial highpass loss', exp_dir)
    vis.histogram(ext_hp, edges, test_dataset+' highpass loss', exp_dir)
    # saliency weighted loss
    vis.histogram(dom_sw, edges, train_dataset+' saliency loss', exp_dir)
    vis.histogram(adv_sw, edges, 'adversarial saliency loss', exp_dir)
    vis.histogram(ext_sw, edges, test_dataset+' saliency loss', exp_dir)
    # loss variances
    loss_variances = np.concatenate((dom_var,adv_var,ext_var))
    edges = np.linspace(0, np.percentile(loss_variances, 95), n_bins+1)
    vis.histogram(dom_var, edges, train_dataset+' loss variance', exp_dir)
    vis.histogram(adv_var, edges, 'adversarial loss variance', exp_dir)
    vis.histogram(ext_var, edges, test_dataset+' loss variance', exp_dir)

    # Calculate epistemic uncertainties for each dataset for each model
    _, loader, _ = data.loader(train_dataset, 1)
    dom_class_epi = epistemic(vgg, loader, cuda)
    adv_class_epi = epistemic_adversarial(vgg, adversarial_range, loader, cuda)
    _, loader, _ = data.loader(test_dataset, 1)
    ext_class_epi = epistemic(vgg, loader, cuda)

    # Classifier uncertainty histograms
    n_bins = 100
    all_class_epi = dom_class_epi+adv_class_epi+ext_class_epi
    edges = np.linspace(0, np.percentile(all_class_epi,95), n_bins+1)
    vis.histogram(dom_class_epi, edges, train_dataset+' classifier uncertainty',
                  exp_dir)
    vis.histogram(adv_class_epi, edges, 'adversarial classifier uncertainty',
                  exp_dir)
    vis.histogram(ext_class_epi, edges, test_dataset+' classifier uncertainty',
                  exp_dir)

    # ROC curves
    vis.roc(dom_avg, ext_avg, 'out-of-domain: average loss', exp_dir)
    vis.roc(dom_hp, ext_hp, 'out-of-domain: high-pass filtered loss', exp_dir)
    vis.roc(dom_sw, ext_sw, 'out-of-domain: saliency-weighted loss', exp_dir)
    vis.roc(dom_class_epi, ext_class_epi,
            'out-of-domain: epistemic uncertainty', exp_dir)
    vis.roc(dom_avg, adv_avg, 'adversarial: average loss', exp_dir)
    vis.roc(dom_hp, adv_hp, 'adversarial: high-pass filtered loss', exp_dir)
    vis.roc(dom_sw, adv_sw, 'adversarial: saliency-weighted loss', exp_dir)
    vis.roc(dom_class_epi, adv_class_epi,
            'adversarial: epistemic uncertainty', exp_dir)


def calc_losses(vgg, pixelcnn, dataloader, n_bins, cuda=True):
    avg_losses = []
    hpf_losses = []
    sal_losses = []
    var_losses = []
    sal_maps = []

    vgg.eval()
    pixelcnn.eval()

    bar = ProgressBar()
    for x,_ in bar(dataloader):
        pcnn_label = torch.squeeze(
            torch.round((n_bins-1)*x).type(torch.LongTensor), 1)
        if cuda:
            x = x.cuda()
        x = Variable(x, requires_grad=True)

        # get the saliency maps
        classifier_output = vgg(x)
        dx = torch.autograd.grad(classifier_output.sum(),x,create_graph=True)[0]
        saliency = dx.data.cpu().numpy()
        vgg.zero_grad()

        # calculate pixelcnn loss
        pcnn_out = np.squeeze(pixelcnn(x, classifier_output).data.cpu().numpy(),
                              0)
        pcnn_out = np.transpose(pcnn_out, [1, 2, 0])
        pcnn_label = np.squeeze(pcnn_label.numpy(), 0)
        r, c = np.meshgrid(range(pcnn_out.shape[0]), range(pcnn_out.shape[1]),
                           indexing='ij')
        loss = -pcnn_out[r, c, pcnn_label]  # per-pixel negative log-likelihood

        avg_loss = np.mean(loss)
        var_loss = np.var(loss)

        weight = x.data.cpu().numpy()
        weight = np.abs(weight[:, :, 1:, 1:]-0.5*(
            weight[:, :, :-1, 1:]+weight[:, :, 1:, :-1]))
        weight = np.pad(weight, ((0, 0), (0, 0), (1, 0), (1, 0)), 'constant',
                        constant_values=0)
        weight = weight/(np.sum(weight, axis=(1,2,3), keepdims=True)+1e-9)
        hpf_loss = np.sum(loss*weight)

        saliency = np.abs(saliency)
        saliency = saliency/(np.sum(saliency, axis=(1,2,3), keepdims=True)+1e-9)
        sal_loss = np.sum(loss*saliency)

        avg_losses.append(avg_loss)
        hpf_losses.append(hpf_loss)
        sal_losses.append(sal_loss)
        sal_maps.append(saliency)
        var_losses.append(var_loss)
    vis.clearline()

    return avg_losses, hpf_losses, sal_losses, sal_maps, var_losses


def adversarial(vgg, pixelcnn, dataloader, n_bins, adversarial_range,
                cuda=True):
    avg_losses = []
    hpf_losses = []
    sal_losses = []
    var_losses = []
    sal_maps = []

    vgg.eval()
    pixelcnn.eval()

    bar = ProgressBar()
    for x,y in bar(dataloader):
        pcnn_label = torch.squeeze(
            torch.round((n_bins-1)*x).type(torch.LongTensor), 1)
        if cuda:
            x,y = x.cuda(), y.cuda()
        x,y = Variable(x, requires_grad=True), Variable(y)

        # generate adversarial example
        loss = torch.nn.CrossEntropyLoss()(vgg(x),y)
        dx = torch.autograd.grad(loss,x,create_graph=True)[0]
        adv_x = x+adversarial_range*torch.sign(dx)

        # get the saliency maps
        classifier_output = vgg(adv_x)
        dx = torch.autograd.grad(
            classifier_output.sum(),adv_x,create_graph=True)[0]
        saliency = dx.data.cpu().numpy()
        vgg.zero_grad()

        # calculate pixelcnn loss
        pcnn_out = np.squeeze(
            pixelcnn(adv_x, classifier_output).data.cpu().numpy(),0)
        pcnn_out = np.transpose(pcnn_out, [1, 2, 0])
        pcnn_label = np.squeeze(pcnn_label.numpy(), 0)
        r, c = np.meshgrid(range(pcnn_out.shape[0]), range(pcnn_out.shape[1]),
                           indexing='ij')
        loss = -pcnn_out[r, c, pcnn_label]  # per-pixel negative log-likelihood

        avg_loss = np.mean(loss)
        var_loss = np.var(loss)

        weight = adv_x.data.cpu().numpy()
        weight = np.abs(weight[:, :, 1:, 1:]-0.5*(
            weight[:, :, :-1, 1:]+weight[:, :, 1:, :-1]))
        weight = np.pad(weight, ((0, 0), (0, 0), (1, 0), (1, 0)), 'constant',
                        constant_values=0)
        weight = weight/np.sum(weight, axis=(1, 2, 3), keepdims=True)
        hpf_loss = np.sum(loss*weight)

        saliency = saliency/np.sum(saliency, axis=(1, 2, 3), keepdims=True)
        sal_loss = np.sum(loss*saliency)

        avg_losses.append(avg_loss)
        hpf_losses.append(hpf_loss)
        sal_losses.append(sal_loss)
        sal_maps.append(saliency)
        var_losses.append(var_loss)
    vis.clearline()

    return avg_losses, hpf_losses, sal_losses, sal_maps, var_losses


def epistemic(model, dataloader, cuda=True, trials=20):
    # need to set model to train for dropout
    # note: effects batchnorm but ignore for now & hope it's negligible
    model.train()

    uncertainties = []
    bar = ProgressBar()
    for x,_ in bar(dataloader):
        # replicate the input to monte carlo sample the dropout results
        x = x.expand(trials, x.size()[1], x.size()[2], x.size()[3])
        if cuda:
            x = x.cuda()
        x = Variable(x)

        model_output = model(x).data.cpu().numpy()

        # take the mean (over the non-batch/dropout dims) of the variance
        uncertainties.append(np.sqrt(np.mean(np.var(model_output,axis=0))))
    return uncertainties


def epistemic_adversarial(model, adversarial_range, dataloader,
                          cuda=True, trials=20):
    # need to set model to train so dropout
    # note: effects batchnorm but ignore for now & hope it's negligible
    model.train()

    uncertainties = []
    bar = ProgressBar()
    for x,y in bar(dataloader):
        # replicate the input to monte carlo sample the dropout results
        x = x.expand(trials, x.size()[1], x.size()[2], x.size()[3])
        y = y.expand(trials)
        if cuda:
            x,y = x.cuda(), y.cuda()
        x,y = Variable(x, requires_grad=True), Variable(y)

        # find the adversarial version of x
        loss = torch.nn.NLLLoss2d()(model(x), y)
        dx = torch.autograd.grad(loss, x, create_graph=True)[0]
        x = x+adversarial_range*torch.sign(dx)

        model_output = model(x).data.cpu().numpy()
        # take the mean (over the non-batch/dropout dims) of the variance
        uncertainties.append(np.sqrt(np.mean(np.var(model_output, axis=0))))
    return uncertainties