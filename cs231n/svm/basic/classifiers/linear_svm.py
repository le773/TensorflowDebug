#coding=utf-8

import numpy as np
from random import shuffle

def svm_loss_naive(w, x, y, reg):
    """
    Structured SVM loss function, naive implementation(with loops)
    inputs:
    :param w: C x D array of weights (10x3073)
    :param x: D x N array of data.Data are D-dimensional columns (3073x49000)
    :param y: 1-dimensional array of length N with labels 0...k-1,for k classes(k=10)
    :param reg: (float) regularization strength
    :return:
    a tuple of :
    - loss as single float
    - gradient with respect to weights w;an array of same shape as w
    """
    dw = np.zeros(w.shape)#initialize the gradient as zero
    #compute the loss and the gradient
    num_classes = w.shape[0]  #  10 classes
    num_train = x.shape[1]  # 49000
    loss = 0.0
    for i in xrange(num_train):
        scores = w.dot(x[:, i])
        correct_class_score = scores[y[i]]
        count = 0
        delta = 1
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + delta# note delta equals to 1
            if margin > 0:
                loss += margin
                dw[j, :] += x[:, i]
                count += 1
        dw[y[i], :] -= x[:, i] * count

    #Now,the loss is a sum over all training examples ,but we want it to be an average instead ,so we divide by num_train
    loss /= num_train
    #also use the averages gradient of full batch
    dw /= num_train
    #add regularization to the loss
    loss += 0.5 * reg * np.sum(w ** 2)
    #add reaularization gradient
    dw += reg * w
    return loss, dw


def svm_loss_vectorized(w, x, y, reg):
    """
    Structured SVM loss function ,vectorized implementtion.(without loop)
    inputs and outputs are same with the function svm_loss_naive
    """
    loss = 0.0
    dw = np.zeros(w.shape)
    num_train = x.shape[1]
    scores = w.dot(x)

    #select all the correct class score
    correct_class_score = scores[y, range(num_train)]
    margins = np.maximum(scores - correct_class_score + 1, 0)  #hinge loss(keypoint)
    margins[y, range(num_train)] = 0
    loss = np.sum(margins) / num_train
    loss += 0.5 * reg * np.sum(w ** 2)
    select_wrong = np.zeros(margins.shape)
    select_wrong[margins > 0] = 1
    select_correct = np.zeros(margins.shape)
    select_correct[y, range(num_train)] = np.sum(select_wrong, axis=0)
    dw = select_wrong.dot(x.T)
    dw -= select_correct.dot(x.T)
    dw /= num_train
    dw += reg * w
    return loss, dw