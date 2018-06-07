### 1.0 np.r_ np.c_
在numpy中，一个列表虽然是横着表示的，但它是列向量。

np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等，类似于pandas中的concat()

np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()

[Numpy模块函数np.c_和np.r_学习使用](https://blog.csdn.net/together_cz/article/details/79548217)