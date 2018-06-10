### 1.0 tf.get_variable tf.Variable
tf.get_variable 和tf.Variable不同的一点是，前者拥有一个变量检查机制，会检测已经存在的变量是否设置为共享变量，如果已经存在的变量没有设置为共享变量，TensorFlow 运行到第二个拥有相同名字的变量的时候，就会报错。

#### 1.2 name_scope variable_scope
1. 在不同的name_scope下同一变量不能定义两次。
2. 在相同的variable_scope下同一变量不能定义两次，但是定义共享标志后可以突破此限制。
3. 将参数reuse设置为TRUE时，tf.variable_scope将只能获取已经创建过的变量。

### 2.0 Tensorflow中变量初始化函数
![init_var_1.png](https://i.imgur.com/9V0HAYx.png)

参考：
1. [tensorflow里面name_scope, variable_scope等如何理解？](https://www.zhihu.com/question/54513728)