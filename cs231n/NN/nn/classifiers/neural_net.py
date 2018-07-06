#!/usr/bin/env python
#coding=utf-8


import numpy as np
# import matplotlib.pyplot as plt


def init_two_layer_model(input_size, hidden_size, output_size):
    """
    This is a two-layer full connectivity,initialize the weights as little value,and all bias as 0
    the input of net id D-dim,the hidden layer has H neuron,the result class number is C

    :param input_size: D-dim
    :param hidden_size:node_num is H
    :param output_size:C classes
    :return:
    a model of python dict,include below key,the corresponding value is numpy array:
    W1:first layer weights,has shape (D, H)
    b1:first layer bias,has shape (H,)
    W2:second layer weights,has shape(H, C)
    b2:second layer bias,has shape (C,)
    """
    #  initialize the model
    model = {}
    model['W1'] = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (input_size * hidden_size))
    model['b1'] = np.zeros(hidden_size)
    model['W2'] = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / (hidden_size * output_size))
    model['b2'] = np.zeros(output_size)
    return model

def two_layer_net(X, model, y=None, regularization=0.0, verbose=False):
    """
    Same as above;In this function,we use softmax and L2 regularization term.In addition,
    this perceptron takes Relu as the activation function。
    The whole structure as follows:
    Input -> Full-Connectivity -> Relu -> Full-Connectivity -> softmax
    the output of the second FC layer are the scores of each class.
    Inputs:
    :param X: input data of shape (N, D).Each X[i] is a training sample.
    :param model: Dictionary mapping params names to arrays of param values.It should contain the following:
       W1:first layer weights,has shape (D, H)
       b1:first layer bias,has shape (H,)
       W2:second layer weights,has shape(H, C)
       b2:second layer bias,has shape (C,)
    :param y: Vector of traning labels.y[i] is the label for X[i],
    and each y[i] is an integer in the range 0 <= y[i] < C.
    This params is optional,if it is not passed then we only return scores,
    and if it is passed then we instead return the loss and grad.
    :param regularization: Regularization strength.
    :param verbose:boolean.
    :return:
    if not give the y,return an array with N x C,among which,the [i,c]th element means the score on the class of c.
    if give the y,return the following tuple:
    -- loss:the loss (include the regularization term) of the current batch.
    -- grads:the grads of the corresponding model params(that is the dict)
    """
    #  unpack variables from the model dict
    # print type(model)
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    print 'W1 shape : ', W1.shape
    print 'b1 shape : ', b1.shape
    print 'W2 shape : ', W2.shape
    print 'b2 shape : ', b2.shape
    N, D = X.shape
    print 'N= %d, D = %d' % (N, D)

    #  forward compute
    scores = None
    #  Relu activation layer
    hidden_activation = np.maximum(X.dot(W1) + b1, 0)
    if verbose:
        print "Layer 1 result shape: " + str(hidden_activation.shape)
    #  softmax before scores
    scores = hidden_activation.dot(W2) + b2
    print 'scores shape: ', scores.shape
    print 'scores data: ', scores
    if verbose:
        print 'Layer 2 result shape: ' + str(scores.shape)
    #  if not give y, return the score
    if y is None:
        return scores
    #  compute the loss
    loss = 0
    #  some computing tricks ,to ensure the stability of computing(need to subtract the maximum score)
    scores = scores - np.expand_dims(np.amax(scores, axis=1), axis=1)  #amax(): Return the maximum of an array or maximum along an axis.

    print 'scores shape: ', scores.shape
    print 'scores data: %f' % scores

    #  axis=1: along the second axis(row-wise)
    #  axis=0: along the first axis (column-wise) ;Caution:python same as matlab, they are both column-prior
    #  expand_dims(): Expand the shape of an array.
    exp_scores = np.exp(scores)
    probability = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    #  cross-entropy loss
    loss = np.sum(- scores[range(len(y)), y] + np.log(np.sum(exp_scores, axis=1))) / N
    #  L2 regularization term
    loss += 0.5 * regularization * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

    #  compute the grads
    grads = {}
    #  compute the diff
    delta_scores = probability
    delta_scores[range(N), y] -= 1
    delta_scores /= N

    #  backward-propagation on softmax layer
    grads['W2'] = hidden_activation.T.dot(delta_scores)
    grads['b2'] = np.sum(delta_scores, axis=0)

    #  backward-propagation on Relu layer
    delta_hidden = delta_scores.dot(W2.T)
    #  derivation of segment function,so the input(< 0) is not back-propagated
    delta_hidden[hidden_activation <= 0] = 0
    grads['W1'] = X.T.dot(delta_hidden)
    grads['b1'] = np.sum(delta_hidden, axis=0)
    #  the grads in the regularization section
    grads['W2'] += regularization * W2
    grads['W1'] += regularization * W1
    return loss, grads





















































































