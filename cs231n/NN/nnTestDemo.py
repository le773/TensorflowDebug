#coding=utf-8
#__author__='eternity'


import numpy as np





#  the main step as follow:
#    1.forward computing:compute the score(if not give the labels);
#    2.forward computing: compute the loss and grad(if give the labels)
#    3.back-propagation
#    4.train net
#    5.fine-tune

# from nn.data_util import load_CTFAR10
from nn.data_util import load_CTFAR10
from nn.classifier_trainer import ClassifierTrainer
from nn.classifiers.neural_net import init_two_layer_model
from nn.classifiers.neural_net import two_layer_net
from nn.gradient_check import evaluate_numerical_grad


#  This is a multi-layer NN on the dataset of CIFAR-10



#  we can ignore the initial set
# plt.rcParams['figure.figsize'] = (10.0, 8.0)  #  set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

#  for auto-reloading external modules
#  see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython


def relative_error(x, y):
    """
    return the relative error.
    """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
#  randomly initialize the model(acturally is the wights) and data set

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

#  build the initial model
def init_toy_model():
    model = {}
    model['W1'] = np.linspace(-0.2, 0.6, num=input_size * hidden_size).reshape(input_size, hidden_size)
    model['b1'] = np.linspace(-0.3, 0.7, num=hidden_size)
    model['W2'] = np.linspace(-0.4, 0.1, num=hidden_size * num_classes).reshape(hidden_size, num_classes)
    model['b2'] = np.linspace(-0.5, 0.9, num=num_classes)
    return model

#  contruct the used input data
def init_toy_data():
    x = np.linspace(-0.2, 0.5, num=num_inputs*input_size).reshape(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return x, y

model = init_toy_model()
x, y = init_toy_data()


#  forward-compute: obtain the score, loss, and the grads on the params
#  same as the svm and softmax

scores = two_layer_net(x, model, verbose=True)
print scores

correct_scores = [[-0.5328368, 0.20031504, 0.93346689],
 [-0.59412164, 0.15498488, 0.9040914],
 [-0.67658362, 0.08978957, 0.85616275],
 [-0.77092643, 0.01339997, 0.79772637],
 [-0.89110401, -0.08754544, 0.71601312]]

#  the diff between the computed score and the real score should be little
print 'the diff between the computed score and the real score is : %e' % np.sum(np.abs(scores - correct_scores))

#  forward compute: compute the loss(include the data loss and regularization)
regularization = 0.1
loss, _ = two_layer_net(x, model, y, regularization)
correct_loss = 1.38191946092
#  the diff also should be little
print 'the diff between the computed loss and the correct loss is : %e' % np.sum(np.abs(loss - correct_loss))

#  back-propagation
#  To loss,we need compute the grads on W1,b1,W2,b2,it also needs do the grad-checkout

#  use the numerical grad to checkout
loss, grads = two_layer_net(x, model, y, regularization)
#  it is save that each param should be less than 1e-8
for param_name in grads:
    param_grad_num = evaluate_numerical_grad(lambda w: two_layer_net(x, model, y, regularization)[0],
                                             model[param_name], verbose=False)  #  derivation of loss
    print '%s maximum relative error: %e' % (param_name, relative_error(param_grad_num, grads[param_name]))




#  train the NN
#  we use fixed-step SGD and SGD with Momentum to minimum loss function
#  fixed-step SGD
model = init_toy_model()
trainer = ClassifierTrainer()
#  Caution:here,the data is man-made,and small scale,so set 'sample_batched' to False;
best_model, loss_history, _, _ = trainer.train(x, y, x, y,
                                               model, two_layer_net, regularization=0.001,
                                               learning_rate=1e-1, momentum=0.0,
                                               learning_rate_decay=1, update='sgd',
                                               sample_batches=False, num_epoches=100,
                                               verbose=False)
print 'Final loss with vanilla SGD: %f' % (loss_history[-1], )

#  SGD with momentum,you will see that the loss is less than above
model1 = init_toy_model()
trainer1 = ClassifierTrainer()
#  call the trainer to optimize the loss
#  Notice that we are using sample_batches=False,so we are performing SGD(no sampled batches of data)
best_model1, loss_history1, _, _ = trainer1.train(x, y, x, y,
                                                  model1, two_layer_net,
                                                  regularization=0.001, learning_rate=1e-1,
                                                  momentum=0.9, learning_rate_decay=1,
                                                  update='momentum', sample_batches=False,
                                                  num_epoches=100, verbose=False)
correct_loss = 0.494394
print 'Final loss with momentum SGD: %f. We get : %f' % (loss_history1[-1],correct_loss)


#  try the other method: RMSProp
model2 = init_toy_model()
trainer2 = ClassifierTrainer()
best_model2, loss_history2, _, _ = trainer2.train(x, y, x, y,
                                                  model2,two_layer_net,
                                                  regularization=0.001,
                                                  learning_rate=1e-1, momentum=0.9, learning_rate_decay=1,
                                                  update='rmsprop', sample_batches=False,
                                                  num_epoches=100, verbose=False)
correct_loss2 = 0.439368
print 'Final loss with RMSProp: %f. We get : %f' % (loss_history2[-1], correct_loss2)

#  ##########################################
#load tha cifar10 data
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    cifar10_dir = 'nn/datasets/cifar-10-batches-py'
    x_train, y_train, x_test, y_test = load_CTFAR10(cifar10_dir)
    mask = range(num_training, num_training + num_validation)
    x_val = x_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    x_train = x_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    x_test = x_test[mask]
    y_test = y_test[mask]

    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_val -= mean_image
    x_test -= mean_image

    x_train = x_train.reshape(num_training, -1)
    x_val = x_val.reshape(num_validation, -1)
    x_test = x_test.reshape(num_test, -1)

    return x_train, y_train, x_val, y_val, x_test, y_test

#  train the NN
#  We use SGD with momentum to optimize.After each iteration,decrease the learning rate a bit

x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()
model0 = init_two_layer_model(32*32*3, 100, 10)  #  input_size, hidden size, number of classes
trainer0 = ClassifierTrainer()
best_model0, loss_history0, train_acc, val_acc = trainer0.train(x_train, y_train, x_val, y_val,
                                                                model0, two_layer_net,
                                                                num_epoches=5,regularization=1.0,
                                                                momentum=0.9,
                                                                learning_rate_decay=0.95,
                                                                learning_rate=1e-5,
                                                                verbose=True)
#  monitor the training process
#  First, we should ensure the training state is normal,so we can know the state by the means of below:
#   1.plot the loss variation curve,we expect it is decreasing
#   2.make the weights of the first layer visible
#  plot the loss function and train / validation accuracies
# plt.subplot(2, 1, 1)
# def show_net_weights(model):
#     plt.imshow(visualize_grid(model0['W1'].T.reshape(-1, 32, 32, 3), padding=3).astype('uint8'))
#     plt.gca().axis('off')
#     plt.show()



#  fine-tune
best_model3 = None  #  store the best result on the validation set
best_val_acc = -1
learning_rates = [1e-5, 5e-5, 1e-4]
model_capacity = [200, 300, 500, 1000]
regularization_strengths = [1e0, 1e1]
results = {}
verbose = True

for hidden_size in model_capacity:
    for lr in learning_rates:
        for reg in regularization_strengths:
            if verbose:
                print 'Training start: '
                print 'lr = %e, regularization = %e, hidden_size = %e' % (lr, reg, hidden_size)

            model3 = init_two_layer_model(32*32*3, hidden_size, 10)
            trainer3 = ClassifierTrainer()
            output_model3, loss_history3, train_acc3, val_acc3 = trainer3.train(x_train, y_train, x_val, y_val,
                                                                                model3, two_layer_net,
                                                                                num_epoches=5, regularization=1.0,
                                                                                momentum=0.9, learning_rate_decay=0.95,
                                                                                learning_rate=lr)
            results[hidden_size, lr, reg] = (loss_history3, train_acc3, val_acc3)
            if verbose:
                print 'Training Complete: '
                print 'Training accuracy = %f, validation accuracy = %f ' % (train_acc3[-1], val_acc3[-1])
            if val_acc3[-1] > best_val_acc:
                best_val_acc = val_acc3[-1]
                best_model3 = output_model3
print 'best validation accyracy achieved during cross-validation : %f' % best_val_acc

#  show the accuracy on the test set
scores_test = two_layer_net(x_test, best_model3)
print 'Test accuracy: ' % np.mean(np.argmax(scores_test, axis=1) == y_test)
































































































































































