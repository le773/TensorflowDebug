### 滑动平均模型
滑动平均模型，它可以使得模型在测试数据上更健壮，在使用随机梯度下降算法训练神经网络时，通过滑动平均模型可以在很多的应用中在一定程度上提高最终模型在测试数据上的表现。其实滑动平均模型，主要是通过控制衰减率来控制参数更新前后之间的差距，从而达到减缓参数的变化值（如，参数更新前是5，更新后的值是4，通过滑动平均模型之后，参数的值会在4到5之间），如果参数更新前后的值保持不变，通过滑动平均模型之后，参数的值仍然保持不变。

### 滑动平均模型原理
ExponentialMovingAverage对每一个变量维护一个影子变量(shadow variable)，影子变量的初始值就是相应变量的初始值，而每次运行变量更新时，影子变量会更新为：
```
shadow_variable = decay * shadow + (1-decay) * variable
```
其中，shadow_variable为影子变量，variable为待更新的变量，decay为衰减率。decay决定了模型更新的速度，decay越大模型越稳定。

decay的更新：

![emoveavg_1.png](https://i.imgur.com/vS5SIq5.png)

参考：
1. [TensorFlow优化之滑动平均模型](https://blog.csdn.net/sinat_29957455/article/details/78409049)