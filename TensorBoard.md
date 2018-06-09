### 1.0 TensorBoard 使用流程
1. 添加记录节点：tf.summary.scalar/image/histogram()等
2. 汇总记录节点：merged = tf.summary.merge_all()
3. 运行汇总节点：summary = sess.run(merged)，得到汇总结果
4. 日志书写器实例化：summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)，实例化的同时传入 graph 将当前计算图写入日志
5. 调用日志书写器实例对象summary_writer的add_summary(summary, global_step=i)方法将所有汇总日志写入文件
6. 调用日志书写器实例对象summary_writer的close()方法写入内存，否则它每隔120s写入一次

### 2.0 词向量可视化
chrome
```
http://projector.tensorflow.org/
```

T-SNE:词向量可视化