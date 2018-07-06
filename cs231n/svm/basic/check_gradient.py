#coding-utf-8

import numpy as np
from random import randrange

def evaluate_numerical_gradient(f, x):
    """
    given x,solve the gradient of f,this is numerical gradient
    the equation:(f(x+h) -f(x-h))/h
    but,in reality,the equation is : (f(x+h) -f(x-h))/2h
    :param f: should be a function that takes a single argument
    :param x: is the point (numpy array)to evaluate the gradient
    """
    fx = f(x)#evaluate the function value at original point
    gradient = np.zeros(x.shape)
    h = 0.00001
    #iterate over all indexes in x
    #compute each dimension on x
    iterate = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not iterate.finished:
        #evaluate function at x+h
        idx = iterate.multi_index
        x[idx] += h#increment by h
        fxh = f(x)#evaluate f(x+h)
        x[idx] -= h#restore to previous value (very important)

        #computr the partial derivative
        gradient[idx] = (fxh - fx) / h#the slope
        print idx, gradient[idx]
        iterate.iternext()#step to next dimension
    return gradient

def gradient_check_sparse(f, x, analytic_grad, num_checks):
    """
    sometimes do the gradinet check on the total dimensions is time cosuming,so we can randomly extract some elements of some dimension,
    to compare numerical gradient with analytic gradient.

    """
    h = 1e-5
    x.shape
    for i in xrange(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        x[ix] += h
        fxph = f(x)
        x[ix] -= 2*h
        fxmh = f(x)
        x[ix] += h
        grad_numerical = (fxph - fxmh) / (2*h)  #  compute the numerical grad
        grad_analytic = analytic_grad[ix]
        relative_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))  #  compare and adjust,if it is correct,
        # then begin to compute the analytic grad
        print 'numerical: %f analytic: %f,relative error: %.3f' % (grad_numerical, grad_analytic, relative_error)
    return grad_analytic















