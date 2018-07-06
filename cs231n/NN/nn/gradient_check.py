#!/usr/bin/env python
#  _*_ coding: utf-8 _*_
#__author__ = 'eternity'

import numpy as np
from random import randrange

def evaluate_numerical_grad(f, x, verbose=True, h=0.00001):
    """
    Compute the numerical grad.
    :param f: f function has only one param.
    :param x: the data point needs to compute grad
    :param verbose:
    :param h:
    :return:
    """
    fx = f(x)
    grad = np.zeros_like(x)  #  zeros_like():Return an array of zeros with the same shape and type as a given array.
    #  traverse x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        #  compute the value of x+h and x-h
        ix = it.multi_index
        val_old = x[ix]
        x[ix] = val_old + h
        fxph = f(x)
        x[ix] = val_old - h
        fxsh = f(x)
        x[ix] = val_old  #  must restore

        grad[ix] = (fxph - fxsh) / (2*h)
        if verbose:
            print ix, grad[ix]
        it.iternext()  #  next
    return grad

def evaluate_numerical_grad_array(f, x, df, h=1e-5):
    """To multi-dims x,solve the numerical grad"""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        val_old = x[ix]
        x[ix] = val_old + h
        pos = f(x)
        x[ix] = val_old - h
        neg = f(x)
        x[ix] = val_old
        grad[ix] = np.sum((pos - neg) * df ) / (2 * h)
        it.iternext()
    return grad

def evaluate_numerical_grad_blobs(f, inputs, output, h=1e-5):
    """
    Compute numerical grads for a function that operates on input and output blobs.
    We assume that f accepts several input blobs as args,followed by a blob into which outputs will be written .
    For example, f might be called like this:
    f(x, w, out)
    where x and w are input blobs, and the result of g will be written to out.
    Inputs:
    :param f: func
    :param inputs: tuple of input blobs
    :param output: output blob
    :param h: step size
    :return:
    """
    numerical_diffs = []
    for input_blob in inputs:
        diff = np.zeros_like(input_blob.diffs)
        it = np.nditer(input_blob.vals, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            origin = input_blob.vals[idx]
            input_blob.vals[idx] = origin + h
            f(*(inputs + (output, )))
            pos = np.copy(output.vals)
            input_blob.vals[idx] = origin - h
            f(*(inputs + (output, )))
            neg = np.copy(output.vals)
            input_blob.vals[idx] = origin
            diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)
            it.iternext()
        numerical_diffs.append(diff)
    return numerical_diffs

def evaluate_numerical_grad_net(net, inputs, output, h=1e-5):
    return evaluate_numerical_grad_blobs(lambda *args: net.forward(),
                                         inputs, output, h=h)

def grad_check_sparse(f, x, analytic_grad, num_checks):
    """Compare the numerical grad with analytic grad"""
    h = 1e-5
    x.shape
    for i in xrange(num_checks):
        ix = tuple([randrange(m) for m in x.shape])  # randrange():return the random number among the specified scope
        val_old = x[ix]
        x[ix] = val_old + h
        fxph = f(x)
        x[ix] = val_old - h
        fxsh = f(x)
        x[ix] = val_old

        grad_numerical = (fxph - fxsh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        relative_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print 'numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, relative_error)






































































































































