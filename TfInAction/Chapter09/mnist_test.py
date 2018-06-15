import tensorflow as tf
import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.NUM_CHANNEL], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_SIZE], name="y-input")
        xs, ys = mnist.test.next_batch(1000)


        xs_reshape = np.reshape(xs, [-1, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE,
                                     mnist_inference.NUM_CHANNEL])

        validate_feed = {x: xs_reshape, y_: ys}

        y = mnist_inference.inference(x)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)


        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    gloval_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)

                    print("After %s training step(s), validarion accuracy = %g" % (gloval_step, accuracy_score))

                else:
                    print("No checkpoint file found")
                    return
                time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets(mnist_train.MNIST_DATA_PATH, one_hot=True)
    evaluate(mnist)

if __name__ == "__main__":
    tf.app.run()

