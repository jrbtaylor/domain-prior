"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2018-2019
"""

import json
import os
import time

import numpy as np
from progressbar import ProgressBar
import torch
from torch.autograd import Variable

from vis import plot_stats, clearline


def fit(train_loader, val_loader, model, exp_path, loss_fcn, optimizer='adam',
        learnrate=1e-4, cuda=True, patience=10, max_epochs=200, resume=False):

    if cuda:
        model = model.cuda()

    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)
    statsfile = os.path.join(exp_path,'stats.json')

    optimizer = {'adam':torch.optim.Adam(model.parameters(),lr=learnrate),
                 'sgd':torch.optim.SGD(
                     model.parameters(),lr=learnrate,momentum=0.9),
                 'adamax':torch.optim.Adamax(model.parameters(),lr=learnrate)
                 }[optimizer.lower()]

    if not resume:
        stats = {'loss':{'train':[],'val':[]},
                 'accuracy':{'train':[],'val':[]}}
        best_val = np.inf
        stall = 0
        start_epoch = 0
    else:
        with open(statsfile,'r') as js:
            stats = json.load(js)
        best_val = np.min(stats['loss']['val'])
        stall = len(stats['loss']['val'])-np.argmin(stats['loss']['val'])-1
        start_epoch = len(stats['loss']['val'])-1
        print('Resuming from epoch %i'%start_epoch)

    def epoch(dataloader,training):
        bar = ProgressBar()
        losses = []
        accuracy = []
        for x,y in bar(dataloader):
            if cuda:
                x,y = x.cuda(),y.cuda()
            x,y = Variable(x),Variable(y)
            if training:
                optimizer.zero_grad()
                model.train()
            else:
                model.eval()
            output = model(x)
            loss = loss_fcn(output,y)
            # track accuracy
            output = output.data.cpu().numpy()
            output_argmax = np.argmax(output,axis=-1)
            int_label = y.data.cpu().numpy()
            correct = np.sum(int_label==output_argmax)
            accuracy.append(correct/int_label.shape[0])
            if training:
                loss.backward()
                optimizer.step()
            losses.append(loss.data.cpu().numpy())
        clearline()
        return float(np.mean(losses)), np.mean(accuracy)

    for e in range(start_epoch,max_epochs):
        # Training
        t0 = time.time()
        loss,acc = epoch(train_loader,training=True)
        time_per_example = (time.time()-t0)/len(train_loader.dataset)
        stats['loss']['train'].append(loss)
        stats['accuracy']['train'].append(acc)
        print(('Epoch %3i:    Training loss = %6.4f    accuracy = %2.2f    '
               '%4.2f msec/example')%(e,loss,acc*100,time_per_example*1000))

        # Validation
        t0 = time.time()
        loss,acc= epoch(val_loader,training=False)
        time_per_example = (time.time()-t0)/len(val_loader.dataset)
        stats['loss']['val'].append(loss)
        stats['accuracy']['val'].append(acc)
        print(('            Validation loss = %6.4f    accuracy = %2.2f    '
               '%4.2f msec/example')%(loss,acc*100,time_per_example*1000))

        # Save stats and update training curves
        with open(statsfile,'w') as sf:
            json.dump(stats,sf)
        plot_stats(stats,exp_path)

        # Early stopping
        torch.save(model,os.path.join(exp_path,'last_checkpoint'))
        if loss<best_val:
            best_val = loss
            stall = 0
            torch.save(model,os.path.join(exp_path,'best_checkpoint'))
        else:
            stall += 1
        if stall>=patience:
            break






