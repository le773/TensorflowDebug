### 1.0 tf.get_variable tf.Variable
tf.get_variable 和tf.Variable不同的一点是，前者拥有一个变量检查机制，会检测已经存在的变量是否设置为共享变量，如果已经存在的变量没有设置为共享变量，TensorFlow 运行到第二个拥有相同名字的变量的时候，就会报错。

#### 1.2 name_scope variable_scope
1. 在不同的name_scope下同一变量不能定义两次。
2. 在相同的variable_scope下同一变量不能定义两次，但是定义共享标志后可以突破此限制。
3. 将参数reuse设置为TRUE时，tf.variable_scope将只能获取已经创建过的变量。

### 2.0 Tensorflow中变量初始化函数
![init_var_1.png](https://i.imgur.com/9V0HAYx.png)


### 3.0 tf.split
API原型（TensorFlow 1.8.0）：
```
tf.split(
    value,
    num_or_size_splits,
    axis=0,
    num=None,
    name='split'
)
```
这个函数是用来切割张量的。输入切割的张量和参数，返回切割的结果。 value传入的就是需要切割的张量。 这个函数有两种切割的方式：

以三个维度的张量为例，比如说一个20 * 30 * 40的张量my_tensor，就如同一个长20厘米宽30厘米高40厘米的蛋糕，每立方厘米都是一个分量。有两种切割方式： 
1. 如果num_or_size_splits传入的是一个整数，这个整数代表这个张量最后会被切成几个小张量。此时，传入axis的数值就代表切割哪个维度（从0开始计数）。调用tf.split(my_tensor, 2，0)返回两个10 * 30 * 40的小张量。 
2. 如果num_or_size_splits传入的是一个向量，那么向量有几个分量就分成几份，切割的维度还是由axis决定。比如调用tf.split(my_tensor, [10, 5, 25], 2)，则返回三个张量分别大小为 20 * 30 * 10、20 * 30 * 5、20 * 30 * 25。很显然，传入的这个向量各个分量加和必须等于axis所指示原张量维度的大小 (10 + 5 + 25 = 40)。


参考：
1. [tensorflow里面name_scope, variable_scope等如何理解？](https://www.zhihu.com/question/54513728)
2. [用人话讲解tf.split](https://blog.csdn.net/SangrealLilith/article/details/80272346)