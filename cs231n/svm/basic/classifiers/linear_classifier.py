import numpy as np
from svm.basic.classifiers.linear_svm import *
from svm.basic.classifiers.softmax import *

class LinearClassifier:
    def __init__(self):
        self.W = None

    def train(self, x, y, learning_rate=1e-3, regularization=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        use SGD to train classifiers,tune the wight W
        inputs:
        :param x: D x N array of training data ,each training point is a D-dimensional column.
        :param y: 1-dimensional array of length N with labels 0...k-1,for k classes.
        :param learning_rate: learning rate of optimization,float
        :param regularization: regularization strength,float.
        :param num_iters: number of steps to take when optimization,integer
        :param batch_size: number od training examples to use at each step,integer
        :param verbose: if true,print progress during optimization,boolean
        :outputs:
        a list containing the value of the loss function at each training iteration
        """
        dim, num_train = x.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            #lazily initialize W
            self.W = np.random.randn(num_classes, dim) * 0.001
        #run stochastic gradient desent to optimize W
        loss_history = []
        for it in xrange(num_iters):
            x_batch = None
            y_batch = None

            idx = np.random.choice(num_train, batch_size, replace=True)
            x_batch = x[:, idx]
            y_batch = y[idx]
            #evaluate loss and gradient
            loss, grad = self.loss(x_batch, y_batch, regularization)  #  compute the loss and grad
            loss_history.append(loss)
            self.W -= learning_rate * grad
            if verbose and it % 100 == 0:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)
            return loss_history

    def predict(self, x):
        """
        predict the network by using the well-trained classifier,compute the predict label y
        :param x:  D x N array of training data ,each training point is a D-dimensional column.
        :return:
        y_pred:predicted labels for the data in x.y_pred is a 1-dimensional array of length N,
        and each element is an integer fiving the predicted class
        """
        y_pred = np.zeros(x.shape[1])
        y_pred = np.argmax(self.W.dot(x), axis=0)
        return y_pred

    def loss(self, x_batch, y_batch, regularization):
        pass


class LinearSVM(LinearClassifier):
    """A subclass that uses the multiclass SVM loss function"""
    def loss(self, x_batch, y_batch, regularization):
        return svm_loss_vectorized(self.W, x_batch, y_batch,regularization)


class Softmax(LinearClassifier):
    """A subclass that uses NN + cross-entropy loss function"""
    def loss(self, x_batch, y_batch, regularization):
        return softmax_loss_vectorized(self.W, x_batch, y_batch, regularization)


