import numpy as np
from random import shuffle

#non-vectorized
def softmax_loss_naive(W, x, y, reg):
    """
    compute the loss and gradient with the use of for loop
    inputs:
    :param w: C x D array of weights
    :param x: D x N array of data.Data are D-dimensional columns
    :param y: 1-dimensional array of length N with labels 0...k-1,for k classes
    :param reg: (float) regularization strength
    :return:
    a tuple of :
    - loss as single float
    - gradient with respect to weights w;an array of same shape as w
    """
    #Initialize the loss and gradient to zero
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = x.shape[1]
    num_classes = W.shape[0]
    fs = W.dot(x)
    for i in xrange(num_train):
        f = fs[:, i]
        #shift the f value to achieve numerical stability
        f -= np.max(f)
        #probability interpretation for each class
        p = np.exp(f) / np.sum(np.exp(f), axis=0)
        loss += -f[y[i]] + np.log(np.sum(np.exp(f), axis=0))
        for j in xrange(num_classes):
            dW[j, :] += p[j] * x[:, i]
        dW[y[i], :] -= x[:, i]
    #calcuate average batch gradient
    dW /= num_train
    #calculate average batch loss
    loss /= num_train
    #regularization
    loss += 0.5 * reg * np.sum(W ** 2)
    dW += reg *W
    return loss, dW

#vectorized
def softmax_loss_vectorized(W, x, y, reg):
    # Initialize the loss and gradient to zero
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = x.shape[1]
    num_classes = W.shape[0]
    fv = W.dot(x)
    fv -= np.amax(fv, axis=0)
    pro = np.exp(fv) / np.sum(np.exp(fv), axis=0)
    pro[y, range(num_train)] -= 1
    #data loss
    loss = np.sum(- fv[y, range(num_train)] + np.log10(np.sum(np.exp(fv), axis=0))) / num_train
    #weight sub-gradient
    dw = pro.dot(x.T) / num_train
    #regularization term
    loss += 0.5 * reg * np.sum(W ** 2)
    dw += reg * W
    return loss, dw

