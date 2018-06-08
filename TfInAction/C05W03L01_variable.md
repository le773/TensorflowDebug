### 1.0 tf.get_variable tf.Variable
tf.get_variable 和tf.Variable不同的一点是，前者拥有一个变量检查机制，会检测已经存在的变量是否设置为共享变量，如果已经存在的变量没有设置为共享变量，TensorFlow 运行到第二个拥有相同名字的变量的时候，就会报错。

在不同的name_scope下同一变量不能定义两次。
在相同的variable_scope下同一变量不能定义两次，但是定义共享标志后可以突破此限制。

[tensorflow里面name_scope, variable_scope等如何理解？](https://www.zhihu.com/question/54513728)