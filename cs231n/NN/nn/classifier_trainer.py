#!/usr/bin/env python
#  _*_ coding: utf-8 _*_
#__author__ = 'eternity'


import numpy as np


class ClassifierTrainer(object):
    """The trainer class performs SGD with momentum on a cost fucntion"""

    def __init__(self):
        self.step_cache = {}  #  for storing velocities(speed) in momentum update

    def train(self, X, y, X_val, y_val, model, loss_func,
              regularization=0.0,
              learning_rate=1e-2, momentum=0,
              learning_rate_decay=0.95,
              update='momentum', sample_batches=True,
              num_epoches=30, batch_size=100, accuracy_frequency=None,
              verbose=False):
        """
        Optimize the params of a model to minimize a loss func .
        We use training data X and y to compute the loss and grads , and periodically check the accuracy on the validation data.
        Inputs:
        :param X: Array of training data;each X[i] is a training sample.
        :param y: Vector of training labels;y[i] gives the label for X[i].
        :param X_val: Array of validation data.
        :param y_val: Vector of validation labels.
        :param model: Dict that maps params names to param values.Each param value is a numpy array.
        :param loss_func: A func that can be called in the following ways:
           scores = loss_func(X, model, reg=regularization)
           loss, grads = loss_func(X, model, y, reg=regularization)
        :param regularization: Regularization strength,which will be passed to the loss func.
        :param learning_rate: Initial learning rate .
        :param momentum: Parameter to use for momentum updates.
        :param learning_rate_decay: The Learning rate will be multiplied by it after each epoch.
        :param update: The update rule to use.One of ''sgd ,'momentum','rmsprop'.
        :param sample_batches: If True,use a mini-batch of data for each parameter update(Stochastic Gradient Descent);
           If False, use the entire training set for each parameter update(Gradient Descent).
        :param num_epochs: The number of epochs to take over the training data.
        :param batch_size: The number of training samples to use at each iteration.
        :param accuracy_frequency: If set to an integer, we compute the training and validation set error after every
              accuracy_frequency iteration.
        :param verbose: If True ,print status after each epoch.
        Returns a tuple of :
        --best_model: the model that got the highest validation accuracy during training.
        --loss_history: List containing the value of the loss func at each iteration.
        --train_accuracy_history: List storing the training set accuracy at each epoch.
        --val_accuracy_history: List storing the validation set accuracy at each epoch.

        """

        N = X.shape[0]
        if sample_batches:
            iterations_per_epoch = N / batch_size  #  using SGD
        else:
            iterations_per_epoch = 1  #  using GD
        num_iters = num_epoches * iterations_per_epoch
        epoch = 0
        best_val_accuracy = 0.0
        best_model = {}
        loss_history = []
        train_accuracy_history = []
        val_accuracy_history = []

        for it in xrange(num_iters):
            if it % 500 == 0:
                print 'starting iteration', it

            #get batch of data
            if sample_batches:
                batch_mask = np.random.choice(N, batch_size)
                X_batch = X[batch_mask]
                y_batch = y[batch_mask]
            else:
                #  No SGD used, full gradient descent
                X_batch = X
                y_batch = y

            #  evaluate cost and grad
            cost, grads = loss_func(X_batch, model, y_batch, regularization)
            loss_history.append(cost)

            #  perform a parameter update
            for p in model:
                #  compute the params step
                if update == 'sgd':
                    dx = -learning_rate * grads[p]
                elif update == 'momentum':
                    if not p in self.step_cache:
                        self.step_cache[p] = np.zeros(grads[p].shape)
                    self.step_cache[p] = momentum * self.step_cache[p] - learning_rate * grads[p]
                    dx = self.step_cache[p]
                elif update == 'rmsprop':
                    decay_rate = 0.99  #  you should also make this an option
                    if not p in self.step_cache:
                        self.step_cache[p] = np.zeros(grads[p].shape)
                    self.step_cache[p] = decay_rate * self.step_cache[p] + (1 - decay_rate) * (grads[p] ** 2)
                    dx = -learning_rate * grads[p] / np.sqrt(self.step_cache[p] + 1e-8)
                else:
                    raise ValueError('Unrecognized update type "%s"' % update)

                #  update the params
                model[p] += dx
            #  every epoch perform an evaluation on the validation set
            first_it = (it == 0)
            epoch_end = (it + 1) % iterations_per_epoch == 0
            accuracy_check = (accuracy_frequency is not None and it % accuracy_frequency == 0)
            if first_it or epoch_end or accuracy_check:
                if it > 0 and epoch_end:
                    #  decay the learning rate
                    learning_rate *= learning_rate_decay
                    epoch += 1
                # evaluate train accuracy
                if N > 1000:
                    train_mask = np.random.choice(N, 1000)
                    X_train_subset = X[train_mask]
                    y_train_subset = y[train_mask]
                else:
                    X_train_subset = X
                    y_train_subset = y
                scores_train = loss_func(X_train_subset, model)
                y_predict_train = np.argmax(scores_train, axis=1)
                train_accuracy = np.mean(y_predict_train == y_train_subset)
                train_accuracy_history.append(train_accuracy)
                #  evaluate validation accuracy
                scores_val = loss_func(X_val, model)
                y_predict_val = np.argmax(scores_val, axis=1)
                val_accuracy = np.mean(y_predict_val == y_val)
                val_accuracy_history.append(val_accuracy)

                #  keep track of the best model based on validation accuracy
                if val_accuracy > best_val_accuracy:
                    #  make a copy of the model
                    best_val_accuracy = val_accuracy
                    best_model = {}
                    for p in model:
                        best_model[p] = model[p].copy()
                #  print progress if needed
                if verbose:
                    print 'Finished epoch %d / %d : cost %f, train: %f, val %f, lr %e' % (epoch, num_epoches, cost,
                                                                                              train_accuracy,
                                                                                              val_accuracy,
                                                                                              learning_rate)
        if verbose:
            print 'finished optimization.Best validation accuracy : %f' % (best_val_accuracy, )

        #  return the best model and training history statistics
        return best_model, loss_history, train_accuracy_history, val_accuracy_history

































































































