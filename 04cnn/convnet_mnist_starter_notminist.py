
# coding: utf-8

# In[1]:


# 声明库依赖
import pickle
import numpy as np
import tensorflow as tf
import os
import time

image_size = 28
num_labels = 10

class NotMNIST:
    def __init__(self):
        class Train:
            def __init__(self):
                self.images = []
                self.labels = []
                self.batch_counter = 0

            @property
            def num_examples(self):
                return len(self.images)

            def next_batch(self, num):
                if self.batch_counter + num >= len(self.labels):
                    batch_images = self.images[self.batch_counter:]
                    batch_labels = self.labels[self.batch_counter:]
                    left = num - len(batch_labels)
                    self.batch_counter = left
                else:
                    batch_images = self.images[self.batch_counter:self.batch_counter + num]
                    batch_labels = self.labels[self.batch_counter:self.batch_counter + num]
                    self.batch_counter += num
                return batch_images, batch_labels

        class Test:
            def __init__(self):
                self.images = []
                self.labels = []

        self.train = Train()
        self.test = Test()

        pickle_file = os.getcwd() + '/../data/notMNIST.pickle'

        with open(pickle_file, 'rb') as f:
            save = pickle.load(f)
            train_dataset = save['train_dataset']
            train_labels = save['train_labels']
            valid_dataset = save['valid_dataset']
            valid_labels = save['valid_labels']
            test_dataset = save['test_dataset']
            test_labels = save['test_labels']
            del save  # 删除内存文件，等待gc回收释放内存

        def reformat(dataset, labels):
            dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
            labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
            return dataset, labels

        train_dataset, train_labels = reformat(train_dataset, train_labels)
        valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
        test_dataset, test_labels = reformat(test_dataset, test_labels)
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)
        self.train.images = train_dataset
        self.train.labels = train_labels
        self.test.images = test_dataset
        self.test.labels = test_labels


# In[2]:


# %matplotlib inline
""" Using convolutional net on MNIST dataset of handwritten digit
(http://yann.lecun.com/exdb/mnist/)
"""
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data


# In[3]:


# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
mnist = input_data.read_data_sets(os.getcwd() + "/MNIST_data/", one_hot=True)
# mnist = NotMNIST()
print("length mnist", len(mnist))


# In[4]:


# Step 2: Define paramaters for the model
N_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 128
SKIP_STEP = 10
DROPOUT = 0.75
N_EPOCHS = 50


# In[5]:


# Step 3: create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# We'll be doing dropout for hidden layer so we'll need a placeholder
# for the dropout probability too
# Use None for shape so we can change the batch_size once we've built the graph
with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, 784], name="X_placeholder")
    Y = tf.placeholder(tf.float32, [None, 10], name="Y_placeholder")

dropout = tf.placeholder(tf.float32, name='dropout')


# In[6]:


# Step 4 + 5: create weights + do inference
# the model is conv -> relu -> pool -> conv -> relu -> pool -> fully connected -> softmax
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

with tf.variable_scope('conv1') as scope:
    # first, reshape the image to [BATCH_SIZE, 28, 28, 1] to make it work with tf.nn.conv2d
    # use the dynamic dimension -1
    images = tf.reshape(X, shape=[-1, 28, 28, 1])
    # TO DO
    # create kernel variable of dimension [5, 5, 1, 32]
    # use tf.truncated_normal_initializer()

    kernel = tf.get_variable('kernel', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer())
    # TO DO

    # create biases variable of dimension [32]
    biases = tf.get_variable('biases', [32], initializer=tf.random_normal_initializer())
    # use tf.random_normal_initializer()
    # TO DO

    # apply tf.nn.conv2d. strides [1, 1, 1, 1], padding is 'SAME'
    conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')
    # TO DO

    conv1 = tf.nn.relu(conv + biases, name=scope.name)
    # apply relu on the sum of convolution output and biases
    # TO DO
    # output is of dimension BATCH_SIZE x 28 x 28 x 32


# In[7]:


with tf.variable_scope('pool1') as scope:
    # apply max pool with ksize [1, 2, 2, 1], and strides [1, 2, 2, 1], padding 'SAME'
    
    # TO DO
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2 ,2, 1], strides = [1, 2, 2, 1], padding='SAME')

    # output is of dimension BATCH_SIZE x 14 x 14 x 32


# In[8]:


with tf.variable_scope('conv2') as scope:
    # similar to conv1, except kernel now is of the size 5 x 5 x 32 x 64
    kernel = tf.get_variable('kernels', [5, 5, 32, 64], 
                        initializer=tf.truncated_normal_initializer())
    biases = tf.get_variable('biases', [64],
                        initializer=tf.random_normal_initializer())
    conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv + biases, name=scope.name)
    # output is of dimension BATCH_SIZE x 14 x 14 x 64


# In[9]:


with tf.variable_scope('pool2') as scope:
    # similar to pool1
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                            padding='SAME')

    # output is of dimension BATCH_SIZE x 7 x 7 x 64


# In[10]:


with tf.variable_scope('fc') as scope:
    # use weight of dimension 7 * 7 * 64 x 1024
    input_features = 7 * 7 * 64
    
    # create weights and biases

    w = tf.get_variable('weights',[input_features, 1024], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', [1024], initializer=tf.random_normal_initializer())
    # TO DO

    # reshape pool2 to 2 dimensional
    pool2 = tf.reshape(pool2, [-1, input_features])

    # apply relu on matmul of pool2 and w + b
    
    # TO DO
    fc = tf.nn.relu(tf.matmul(pool2, w) + b)

    # apply dropout
    fc = tf.nn.dropout(fc, dropout, name='relu_dropout')


# In[11]:


with tf.variable_scope('softmax_linear') as scope:
    # this you should know. get logits without softmax
    # you need to create weights and biases
    w = tf.get_variable('weights', [1024, N_CLASSES], initializer = tf.truncated_normal_initializer())
    b = tf.get_variable('biases', [N_CLASSES], initializer = tf.random_normal_initializer())
    logits = tf.matmul(fc, w) + b
    # TO DO


# In[ ]:


# Step 6: define loss function
# use softmax cross entropy with logits as the loss function
# compute mean cross entropy, softmax is applied internally
with tf.name_scope('loss'):
    # you should know how to do this too
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    loss = tf.reduce_mean(entropy, name='loss')
    # TO DO


# In[ ]:


# Step 7: define training op
# using gradient descent with learning rate of LEARNING_RATE to minimize cost
# don't forgot to pass in global_step
# TO DO

optimizer = tf.train.AdadeltaOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # to visualize using TensorBoard
    writer = tf.summary.FileWriter(os.getcwd() + '/my_graph/mnist', sess.graph)
    ##### You have to create folders to store checkpoints
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.getcwd() + '/checkpoints/convnet_mnist/checkpoint'))
    # if that checkpoint exists, restore from checkpoint
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    
    initial_step = global_step.eval()
    print('initial_step:',initial_step)

    start_time = time.time()
    n_batches = int(mnist.train.num_examples / BATCH_SIZE)

    total_loss = 0.0
    for index in range(initial_step, n_batches * N_EPOCHS): # train the model n_epochs times
        X_batch, Y_batch = mnist.train.next_batch(BATCH_SIZE)
        _, loss_batch = sess.run([optimizer, loss], 
                                feed_dict={X: X_batch, Y:Y_batch, dropout: DROPOUT}) 
        total_loss += loss_batch
        if (index + 1) % SKIP_STEP == 0:
            print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss / SKIP_STEP))
            total_loss = 0.0
            saver.save(sess, os.getcwd() + '/checkpoints/convnet_mnist/mnist-convnet', index)
    
    print("Optimization Finished!") # should be around 0.35 after 25 epochs
    print("Total time: {0} seconds".format(time.time() - start_time))


    # test the model
    n_batches = int(mnist.test.num_examples/BATCH_SIZE)
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(BATCH_SIZE)
        _, loss_batch, logits_batch = sess.run([optimizer, loss, logits], 
                                        feed_dict={X: X_batch, Y:Y_batch, dropout: DROPOUT}) 
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds += sess.run(accuracy)   
    
    print("Accuracy {0}".format(total_correct_preds/mnist.test.num_examples))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




