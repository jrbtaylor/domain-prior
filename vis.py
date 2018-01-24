"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2018-2019
"""

import json
import os

import imageio
import matplotlib
# Disable Xwindows backend before importing matplotlib.pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_stats(stats,savepath):
    """
    Make all the plots in stats. Stats can be a dict or a path to json (str)
    """
    if type(stats) is str:
        assert os.path.isfile(stats)
        with open(stats,'r') as sf:
            stats = json.load(sf)
    assert type(stats) is dict

    assert type(savepath) is str
    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    def _plot(y,title):
        plt.Figure()
        if type(y) is list:
            plt.plot(range(1,len(y)+1),y)
        elif type(y) is dict:
            for key,z in y.items():
                plt.plot(range(1,len(z)+1),z,label=key)
            plt.legend()
        else:
            raise ValueError
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.title(title)
        plt.savefig(os.path.join(savepath,title.replace(' ','_')+'.png'))
        plt.close()

    # Loop over stats dict and plot. Dicts within stats get plotted together.
    for key,value in stats.items():
        _plot(value,key)


def clearline():
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE+ERASE_LINE+CURSOR_UP_ONE)


def tile_images(imgs,r=0):
    n = len(imgs)
    h = imgs[0].shape[1]
    w = imgs[0].shape[2]
    if r==0:
        r = int(np.floor(np.sqrt(n)))
    while n%r!=0:
        r -= 1
    c = int(n/r)
    imgs = np.squeeze(np.array(imgs),axis=1)
    imgs = np.transpose(imgs,(1,2,0))
    imgs = np.reshape(imgs,[h,w,r,c])
    imgs = np.transpose(imgs,(2,3,0,1))
    imgs = np.concatenate(imgs,1)
    imgs = np.concatenate(imgs,1)
    return imgs


def histogram(y,edges,title,savepath):
    plt.Figure()
    plt.hist(y,edges)
    plt.title(title)
    plt.savefig(os.path.join(savepath,title.replace(' ','_')+'.png'))
    plt.close()

