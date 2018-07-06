#!/usr/bin/env python
#coding=utf-8
#__author__ = 'eternity'


import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegressionCV

import matplotlib.pyplot as plt

#  using NN to realize the non-linear separation

#  produce an random distribution of planar point manually,and plot them.
np.random.seed(0)
x, y = make_moons(200, noise=0.20)
# plt.scatter(x[:, 0], x[:, 1], s=40, c=y, cmap=plt.cm.Spectral)

#  define a function as the decision boundary
def plot_decision_boundary(pred_func):

    #  set the max and min value , and fill the boundary
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = y[:, 1].min() - 0.5, y[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    #  predict using the predict func
    z = pred_func(np.c_[xx.ravel(), yy.ravel()])  #  ravel():Return a contiguous flattened array
    #  np.c_: concatenate alone the second axis

    z = z.reshape(xx.shape)

    #  use LR to classify
    #  Firstly,see the LR effect
    clf = LogisticRegressionCV()
    clf.fit(x, y)

    #  Obviously,the classification result is not satisfied,so we try to use NN to do it
    num_examples = len(x)  #  number of samples
    nn_input_dim = 2  #  dims of input
    nn_output_dim = 2  #  number of output classes

    #GD params
    lr = 0.01  #  learning rate
    reg = 0.01  #  regularization

    #  define loss func
    def calculate_loss(model):
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

        # forward computing
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)  #  softmax clssifier
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # compute loss
        correct_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(correct_logprobs)
        data_loss += reg/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1.0 / num_examples * data_loss

    def build_model(nn_hdim, num_passes=20000, print_loss=False):
        """
        Build model net.
        :param nn_hdim: number of hidden layer nodes
        :param num_passes: iteration num
        :param print_loss: if True, every 1000 iters,then output the current loss value
        :return:
        """

        #  randomly initialize weights
        np.random.seed(0)
        W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
        b1 = np.zeros((1, nn_hdim))
        W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_output_dim)
        b2 = np.zeros((1, nn_output_dim))

        # the model we finally to learn
        model = {}
        # starting to GD
        for i in xrange(0, num_passes):
            # forward-compute loss
            z1 = x.dot(W1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            #  BP
            delta3 = probs
            delta3[range(num_examples), y] -= 1
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(x.T, delta2)
            db1 = np.sum(delta2, axis=0)

            #  add the regularization item
            dW2 += reg * W2
            dW1 += reg * W1

            # renew params
            W1 += -lr * dW1
            b1 += -lr * db1
            W2 += -lr * dW2
            b2 += -lr * db2

            #  we get the model ,truly is to get this params above
            model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
            #  if already set print_loss,then we print the middle process
            if print_loss and i % 1000 == 0:
                print 'Loss after ieration %i: %f ' % (i, calculate_loss(model))
        return model

    #  func of determining the result
    def predict(model, x):
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        #  compute the class corresponding to max probability
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)  #  np.argmax(): Returns the indices of the maximum values along an axis.


    #  build a NN net with 3 nodes in hidden layer
    model = build_model(3, print_loss=True)
    plot_decision_boundary(lambda m: predict(model, x))
    plt.title("Dicision boundary for hidden layer size 3")
    plt.show()

    #  show the result under different hidden layer size
    plt.figure(figsize=(16, 32))
    hidden_layer_size = [1, 2, 3, 4, 5, 20, 50]
    for i, nn_hidden_num in enumerate(hidden_layer_size):
        plt.subplot(5, 2, i+1)
        plt.title("Hiden layer size %d" % nn_hidden_num)
        model = build_model(nn_hidden_num)
        plot_decision_boundary(lambda m: predict(model, x))
    plt.show()


























































































