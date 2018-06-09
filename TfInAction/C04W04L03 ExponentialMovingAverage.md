ExponentialMovingAverage对每一个变量维护一个影子变量(shadow variable)，影子变量的初始值就是相应变量的初始值，而每次运行变量更新时，影子变量会更新为：
```
shadow_variable = decay * shadow + (1-decay) * variable
```
其中，shadow_variable为影子变量，variable为待更新的变量，decay为衰减率。decay决定了模型更新的速度，decay越大模型越稳定。

decay的更新：

![emoveavg_1.png](https://i.imgur.com/vS5SIq5.png)