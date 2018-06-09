### Saver使用
1. 计算图
保存所有神经网络的计算逻辑
tf.Graph
2. 参数
记录神经元的参数
tf.Variable
```
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    ...
    ##### You have to create folders to store checkpoints
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.getcwd() + '/checkpoints/convnet_mnist/checkpoint'))
    # if that checkpoint exists, restore from checkpoint
    if ckpt and ckpt.model_checkpoint_path:
        print('saver restore')
        print('ckpt.model_checkpoint_path:',ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    ...
    saver.save(sess, os.getcwd() + '/checkpoints/convnet_mnist/mnist-convnet', global_step=index)
```