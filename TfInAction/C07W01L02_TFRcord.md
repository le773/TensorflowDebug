### tfrecord
tfrecord数据文件是一种将图像数据和标签统一存储的二进制文件，能更好的利用内存，在tensorflow中快速的复制，移动，读取，存储等。

tfrecord文件包含了tf.train.Example协议缓冲区(protocol buffer，协议缓冲区包含了特征 Features)。你可以写一段代码获取你的数据， 将数据填入到Example协议缓冲区(protocol buffer)，将协议缓冲区序列化为一个字符串， 并且通过tf.python_io.TFRecordWriter class写入到TFRecords文件。tensorflow/g3doc/how_tos/reading_data/convert_to_records.py就是这样的一个例子。

tf.train.Example的定义如下：
```
message Example {
 Features features = 1;
};

message Features{
 map<string,Feature> featrue = 1;
};

message Feature{
    oneof kind{
        BytesList bytes_list = 1;
        FloatList float_list = 2;
        Int64List int64_list = 3;
    }
};
```

从上述代码可以看出，tf.train.Example中包含了属性名称到取值的字典，其中属性名称为字符串，属性的取值可以为字符串（BytesList）、实数列表（FloatList）或者整数列表（Int64List）。