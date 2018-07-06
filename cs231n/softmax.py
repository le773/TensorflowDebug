#coding=utf-8
#author='eternity'


import time

import matplotlib.pyplot as plt

from NN.nn.data_util import load_CTFAR10
from svm.basic.classifiers.softmax import *

plt.rcParams['figure.figsize'] = (10.0, 8.0) #set the plot params
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.camp'] = 'gray'


def get_cifar10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    load the data set,and turn the needed format
    :param num_training:
    :param num_validation:
    :param num_test:
    :return:
    """

    cifar10_dir = 'basic/datasets/cifar-10-batches-py'
    x_train, y_train, x_test, y_test = load_CTFAR10(cifar10_dir)
    #sample some data to use
    mask = range(num_training, num_training + num_validation)
    x_val = x_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    x_train = x_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    x_test = x_test[mask]
    y_test = y_test[mask]

    #preprocessing:turn to column vector
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_val = np.reshape(x_val, (x_val.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    #zero-mean processing
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_val -= mean_image
    x_test -= mean_image
    #add the bias item
    x_train = np.hstack([x_train, np.ones(x_train.shape[0], 1)]).T  #column wise
    x_val = np.hstack([x_val, np.ones(x_val.shape[0], 1)]).T
    x_test = np.hstack([x_test, np.ones(x_test.shape[0], 1)]).T
    return x_train, y_train,  x_val, y_val, x_test, y_test

    #softmax

    #initialize the weight randomly
    W = np.random.randn(10, 3073) * 0.0001
    loss, grad = softmax_loss_naive(W, x_train, y_train, 0.0)
    print 'loss: %f' % loss
    print 'sanity check: %f' % (-np.log10(0.1))

    # 用多层for循环实现一个梯度计算
    loss, grad = softmax_loss_naive(W, x_train, y_train, 0.0)

    # 同样要做一下梯度检查
    f = lambda w: softmax_loss_naive(w, x_train, y_train, 0.0)[0]
    grad_numerical = grad_check_sparse(f, W, grad, 10)

    # 实现一个向量化的损失函数和梯度计算方法
    tic = time.time()
    loss_naive, grad_naive = softmax_loss_naive(W, x_train, y_train, 0.00001)
    toc = time.time()
    print 'naive loss: %e computed in %fs' % (loss_naive, toc - tic)

    tic = time.time()
    loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, x_train, y_train, 0.00001)
    toc = time.time()
    print 'vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic)

    # 咱们对比一下用for循环实现的函数和用向量化实现的函数差别
    grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
    print 'Loss difference: %f' % np.abs(loss_naive - loss_vectorized)
    print 'Gradient difference: %f' % grad_difference

    # 咱们实现了一个softmax分类器
    results = {}
    best_val = -1
    best_softmax = None
    learning_rates = [5e-7, 1e-7, 5e-6, 1e-6]
    regularization_strengths = [5e4, 1e5]

    import sys
    verbose = True
    for lr in learning_rates:
        for reg in regularization_strengths:
            if verbose: sys.stdout.write("Training with hyper parameter learning rate: %e, regularization: %e\n"
                                         % (lr, reg))
            softmax = Softmax()
            loss_hist = softmax.train(X_train, y_train, learning_rate=lr, reg=reg,
                                      num_iters=1500, verbose=False)

            y_train_pred = softmax.predict(X_train)
            training_accuracy = np.mean(y_train == y_train_pred)

            y_val_pred = softmax.predict(X_val)
            val_accuracy = np.mean(y_val == y_val_pred)

            results[lr, reg] = (training_accuracy, val_accuracy)
            if val_accuracy > best_val:
                best_val = val_accuracy
                best_softmax = softmax

    # 输出结果
    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
            lr, reg, train_accuracy, val_accuracy)

    print 'best validation accuracy achieved during cross-validation: %f' % best_val
    # 在测试集上评估softmax分类器性能
    y_test_pred = best_softmax.predict(x_test)
    test_accuracy = np.mean(y_test == y_test_pred)
    print 'softmax on raw pixels final test set accuracy: %f' % (test_accuracy,)

    # 可视化一下学习到的权重
    w = best_softmax.W[:, :-1]  # strip out the bias
    w = w.reshape(10, 32, 32, 3)

    w_min, w_max = np.min(w), np.max(w)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in xrange(10):
        plt.subplot(2, 5, i + 1)

        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (w[i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])





























































































