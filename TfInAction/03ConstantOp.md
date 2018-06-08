### 1.0 Constant Value Tensors

### 2.0 Sequences
- linspace
- range

### 3.0 Random Tensors
- random_normal
从正态分布输出随机值。

- truncated_normal
截断的正态分布函数。生成的值遵循一个正态分布，但不会大于平均值2个标准差。

- random_uniform
从均匀分布中返回随机值。

- random_shuffle
沿着要被洗牌的张量的第一个维度，随机打乱。

- set_random_seed

#### 4.0 other
#### 4.1 global_step
global_step经常在滑动平均，学习速率变化的时候需要用到，这个参数在tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_steps)里面有，系统会自动更新这个参数的值，从1开始。