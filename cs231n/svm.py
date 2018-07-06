#!/usr/bin/python
#coding=utf-8
#__author__ = 'eternity'

import math
import time

import matplotlib.pyplot as plt

from NN.nn.data_util import load_CTFAR10
from svm.basic.check_gradient import *
from svm.basic.classifiers.linear_classifier import LinearSVM
from svm.basic.classifiers.linear_svm import *

#初始化
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

"""CIFAR-10数据集的载入与预处理"""
#加载原始的CIFAR-10图片数据集
cifar10_dir = 'basic/datasets/cifar-10-batches-py'
X_train, Y_train, X_test, Y_test = load_CTFAR10(cifar10_dir)

#see 训练集和测试集维度
print 'Training data shape: ', X_train.shape
print 'Training labels shape: ', Y_train.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', Y_test.shape

#可视化一下图片集,每个类展示一些图片
# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog','horse', 'ship', 'truck']
# num_classes = len(classes)
# samples_per_class = 7
# for y, cls in enumerate(classes):
#     idxes = np.flatnonzero(Y_train == y)
#     idxes = np.random.choice(idxes, samples_per_class, replace=False)
#     for i, idx in enumerate(idxes):
#       plt_idx = i * num_classes + y + 1
#       plt.subplot(samples_per_class, num_classes, plt_idx)
#       plt.imshow(X_train[idx].astype('uint8'))
#       plt.axis('off')
#       if i == 0:
#           plt.title(cls)
# plt.show()

#extract the training set ,validation set ,and test set
num_training = 49000
num_validation = 1000
num_test = 1000

#fetch the image
mask = range(num_training, num_training + num_validation)
x_val = X_train[mask]
y_val = Y_train[mask]

mask = range(num_training)
x_train = X_train[mask]
y_train = Y_train[mask]

mask = range(num_test)
x_test = X_test[mask]
y_test = Y_test[mask]

print 'Train data shape: ', x_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', x_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape :', x_test.shape
print 'Test labels shape: ', y_test.shape

#preprocessing:change the data in view of column
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_val = np.reshape(x_val, (x_val.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))

print 'Training data shape: ', x_train.shape
print 'Validation data shape: ', x_val.shape
print 'Test data shape: ', x_test.shape

#preprocessing:substact the mean of image
mean_image = np.mean(x_train, axis=0)
print mean_image[:10]
# plt.figure(figsize=(4, 4))
# plt.imshow(mean_image.reshape((32, 32, 3)).astype('uint8'))
# time.sleep(1.0)
x_train = x_train - mean_image
x_val = x_val - mean_image
x_test = x_test - mean_image

#add the column of '1'
x_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))]).T
x_val = np.hstack([x_val, np.ones((x_val.shape[0], 1))]).T
x_test = np.hstack([x_test, np.ones((x_test.shape[0], 1))]).T
print x_train.shape, x_val.shape, x_test.shape

#svm
#evaluate the efficiency of svm_loss_naive

#produce the initial weights of svm
W = np.random.randn(10, 3073) * 0.0001  #  10x3073
#compute the gradient and loss under the wieght W
loss, gradient = svm_loss_naive(W, x_train, y_train, 0.00001)
print 'loss: %f' % (loss, )


loss, gradient = svm_loss_naive(W, x_train, y_train, 0.0)
#gradient check :check out weather the numerical gradient and analytic gradient is identical,because the latter is fast,
# but eary to error
f = lambda w: svm_loss_naive(w, x_train, y_train, 0.0)[0]  #  loss
grad_numerical = gradient_check_sparse(f, W, gradient, 10)  #对loss求导数

#two methods to cpmpute the loss of svm:generately,vectorize method is faster
#naive loss of non-vectorize svm ,loss computing
tic = time.time()
loss_naive, grad_naive = svm_loss_naive(W, x_train, y_train, 0.00001)
toc = time.time()
print 'non-vectorize:loss %e timeout %fs' % (loss_naive, toc - tic)
#vectorized
tic = time.time()
loss_vectorize, _ = svm_loss_vectorized(W, x_train, y_train, 0.00001)
toc = time.time()
print 'vectorize:loss %e timeout %fs' % (loss_vectorize, toc - tic)
#if your implementation is right ,the two value is same
print 'difference of two methods: %f ' % (loss_naive - loss_vectorize)

#sgd
# svm = LinearSVM()
# tic = time.time()
# loss_hist = svm.train(x_train, y_train, learning_rate=1e-7, regularization=5e4,
#                       num_iters=1500, verbose=True)
# toc = time.time()
# print 'it takes %fs' % (toc - tic)
# y_train_pred = svm.predict(x_train)
# print 'training accuracy: %f' % (np.mean(y_train == y_train_pred), )
# y_val_pred = svm.predict(x_val)
# print 'validation accuracy: %f' % (np.mean(y_val == y_val_pred), )

#cross-validation to test different learning rate
learning_rate = [5e-7, 1e-7, 5e-6, 1e-6, 1e-5]
regularization_strength = [5e4, 1e5]
results = {}
best_val = -1  #设定交叉验证最佳得分的初始值
best_svm = None  #设定交叉验证最佳SVM参数集的初始值
verbose = True
for lr in learning_rate:
    for reg in regularization_strength:
        if verbose: print "Training with hyper parameter learning rate: %e, regularization: %e" % (lr, reg)
        svm = LinearSVM()
        loss_hist = svm.train(x_train, y_train, learning_rate=lr, regularization=reg,
                              num_iters=1500, verbose=False)
        y_train_pred = svm.predict(x_train)
        training_accuracy = np.mean(y_train == y_train_pred)

        y_val_pred = svm.predict(x_val)
        val_accuracy = np.mean(y_val == y_val_pred)
        results[lr, reg] = (training_accuracy, val_accuracy)
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_svm = svm
#print all the results
for lr, reg in sorted(results):
    training_accuracy, val_accuracy = results[lr, reg]
    print 'lr %e reg %e train accuracy: %f val accuracy :%f' % (lr, reg, training_accuracy, val_accuracy)
print 'best validation accuarcy achieved during cross-validation: %f' % best_val

#可视化交叉验证结果
x_scatter = [math.log10(x[0]) for x in results]
y_scatter = [math.log10(x[1]) for x in results]
#training
sz = [results[x][0] * 1500 for x in results]
# plt.subplot(1, 2, 1)
# plt.scatter(x_scatter, y_scatter, sz)
# plt.xlabel('log learning rate')
# plt.ylabel('log regualrization strength')
# plt.title('CIFAR-10 training accuracy')
#validation
sz1 = [results[x][1]*1500 for x in results]
# plt.subplot(1, 2, 1)
# plt.scatter(x_scatter, y_scatter, sz1)
# plt.xlabel('log learning rate')
# plt.ylabel('log regualrization strength')
# plt.title('CIFAR-10 validation accuracy')

#在测试集上看效果
y_test_pred = best_svm.predict(x_test)
test_accuracy = np.mean(y_test == y_test_pred)
print 'linear svm on raw pixels final test set accuracy: %f' % test_accuracy
#可视化每个类对应的权重
#由于初始值和学习率的不同,结果可能会有一些差别
w = best_svm.W[:, :-1]#去掉bias项
w = w.reshape(10, 32, 32, 3)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat','deer','dog', 'frog', 'horse', 'ship', 'truck']
for i in xrange(10):
    # plt.subplot(2, 5, i + 1)
    #rescale th weights to 0-255
    w_img = 255.0 * (w[i].squeeze() - w_min) / (w_max - w_min)
    # plt.imshow(w_img.atype('uint8'))
    # plt.axis('off')
    # plt.title(classes[i])












































