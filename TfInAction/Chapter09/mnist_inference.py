import tensorflow as tf

INPUT_SIZE = 784
OUTPUT_SIZE = 10
IMAGE_SIZE = 28
NUM_CHANNEL = 1
CONV1_SIZE = 5
CONV1_DEEP = 32
CONV2_SIZE = 5
CONV2_DEEP = 64
FC1_SIZE = 512


def inference(input_tensor, train=None, regularizer=None):
    with tf.variable_scope("layers1-conv1"):
        conv1_weights = tf.get_variable("weights", shape=[CONV1_SIZE, CONV1_SIZE, NUM_CHANNEL, CONV1_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("biases", shape=[CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, [1, 1, 1, 1], padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))


    with tf.variable_scope("layers2-pool1"):
        #pool1 = tf.nn.pool(relu1, [1, 2, 2, 1], pooling_type="MAX", strides=[1, 1, 1, 1], padding="SAME")
        pool1 = tf.nn.max_pool(relu1, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")


    with tf.variable_scope("layers3-conv2"):
        conv2_weights = tf.get_variable("weights", shape=[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("biases", shape=[CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(relu1, conv2_weights, [1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))


    with tf.variable_scope("layers4-pool2"):
        pool2 = tf.nn.max_pool(relu2, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")


    shape_pool2 = pool2.get_shape().as_list()
    nodes = shape_pool2[1] * shape_pool2[2] * shape_pool2[3]
    pool_reshape = tf.reshape(tensor=pool2, shape=[-1, nodes], name="pool_reshape")

    with tf.variable_scope("layers5-fc1"):
        fc1_weights = tf.get_variable("weights", shape=[nodes, FC1_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc1_weights))
        fc1_biases = tf.get_variable("biases", shape=[FC1_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(pool_reshape, fc1_weights) + fc1_biases)
        if train != None:
            fc1 = tf.nn.dropout(fc1, 0.5)


    with tf.variable_scope("layers6-fc2"):
        fc2_weights = tf.get_variable("weights", [FC1_SIZE, OUTPUT_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc2_weights))
        fc2_biases = tf.get_variable("biases", shape=[OUTPUT_SIZE], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
        #if train != None:
        #    logit = tf.nn.dropout(logit, 0.5)


    return logit

