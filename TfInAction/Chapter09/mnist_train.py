import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import os
import numpy as np

BATCH = 100
TRAIN_STEPS = 6000
REGULARIZER_RATE = 0.0001
LEARNING_RATE = 0.015
LEARNING_DECAY = 0.99
#LEARNING_STEP = 50
MOVING_AVERGE_DECAY = 0.99


SAVE_PATH = "./saver_path/"
MODEL_NAME = "model.ckpt"
MNIST_DATA_PATH = "./mnist_data/"

def train(mnist):
    x = tf.placeholder(tf.float32, shape=[None,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.NUM_CHANNEL], name="input-x")
    y_ = tf.placeholder(tf.float32, shape=[None, mnist_inference.OUTPUT_SIZE], name="input-y")
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZER_RATE)
    y = mnist_inference.inference(x, train=False, regularizer=regularizer)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, mnist.train.num_examples / BATCH, LEARNING_DECAY)
    average_exp = tf.train.ExponentialMovingAverage(MOVING_AVERGE_DECAY, global_step)
    average_exp_op = average_exp.apply(tf.trainable_variables())

    cross_entrpy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    loss = tf.reduce_mean(cross_entrpy) + tf.add_n(tf.get_collection("losses"))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([average_exp_op, train_step]):
        train_op = tf.no_op("train")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAIN_STEPS):
            xs, ys = mnist.train.next_batch(BATCH)
            #xs_reshape = tf.reshape(xs, [BATCH, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNEL])
            xs_reshape = np.reshape(xs, [BATCH, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE,
                                         mnist_inference.NUM_CHANNEL])
            _, loss_value, steps = sess.run([train_op, loss, global_step], feed_dict={x:xs_reshape, y_:ys})

            if i % 10 == 0:
                print("After %d training step(s), loss on training batch is %g." % (steps, loss_value))
                saver.save(sess, os.path.join(SAVE_PATH, MODEL_NAME), global_step)

def main(argv=None):
    mnist = input_data.read_data_sets(MNIST_DATA_PATH, one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()


