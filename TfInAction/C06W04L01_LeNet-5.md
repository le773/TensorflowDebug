### 1.0 LeNet-5模型
![LeNet-5模型](https://img-blog.csdn.net/20171012155122043?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWEpZMTA0MTY1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### 2.0 LeNet-5模型解释
1. 第一层，卷积层
这一层的输入就是原始的图像像素32\*32\*1。第一个卷积层过滤器尺寸为5\*5，深度为6，不使用全0填充，步长为1。所以这一层的输出：28\*28\*6，卷积层共有5\*5\*1\*6+6=156个参数</br>
2. 第二层，池化层
这一层的输入为第一层的输出，是一个28\*28\*6的节点矩阵。本层采用的过滤器大小为2\*2，长和宽的步长均为2，所以本层的输出矩阵大小为14\*14\*6。</br>
3. 第三层，卷积层
本层的输入矩阵大小为14\*14\*6，使用的过滤器大小为5\*5，深度为16.本层不使用全0填充，步长为1。本层的输出矩阵大小为10\*10\*16。本层有5\*5\*6\*16+16=2416个参数。</br>
4. 第四层，池化层
本层的输入矩阵大小10\*10\*16。本层采用的过滤器大小为2\*2，长和宽的步长均为2，所以本层的输出矩阵大小为5\*5\*16。</br>
5. 第五层，全连接层
本层的输入矩阵大小为5\*5\*16，在LeNet-5论文中将这一层成为卷积层，但是因为过滤器的大小就是5\*5，所以和全连接层没有区别。如果将5\*5\*16矩阵中的节点拉成一个向量，那么这一层和全连接层就一样了。本层的输出节点个数为120，总共有5\*5\*16\*120+120=48120个参数。</br>
6. 第六层，全连接层
本层的输入节点个数为120个，输出节点个数为84个，总共参数为120\*84+84=10164个。</br>
7. 第七层，全连接层
本层的输入节点个数为84个，输出节点个数为10个，总共参数为84\*10+10=850</br>