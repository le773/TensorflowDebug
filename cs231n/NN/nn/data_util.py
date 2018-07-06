#coding=utf-8

import cPickle as pickle
import numpy as np
import os


def load_CIFAR_batch(file_name):
    #"""load one batch of the datset"""
    with open(file_name,'r') as f:
        data_dict = pickle.load(f)
        x = data_dict['data']
        y = data_dict['labels']
        x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        y = np.array(y)
        return x, y

def load_CTFAR10(ROOT):
    # """loar all the dataset"""
     xs = []
     ys = []
     for b in range(1, 6):
         f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
         x, y = load_CIFAR_batch(f)
         xs.append(x)
         ys.append(y)
     x_tr = np.concatenate(xs)
     y_tr = np.concatenate(ys)
     del x, y
     x_te, y_te = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
     return x_tr, y_tr, x_te, y_te
