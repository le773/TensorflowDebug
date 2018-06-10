Saver是Tensorflow提供的用来保存和还原神经网络的api
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

- model.ckpt.meta:保存了Tensorflow计算图的结构；
- model.ckpt:保存了Tensorflow程序中的每一个变量的取值；
- model.ckpt.index:保存了一个目录下所有模型文件列表。

### 持久化原理及数据格式
Tensorflow通过元图(MetaGraph)来记录计算图中节点的信息以及运行计算图中节点所需要的元数据。

Tensorflow中的元图由MetaGraphDef Protocol Buffer定义。MetaGraphDef的内容就构成了Tensorflow持久化的第一个文件。