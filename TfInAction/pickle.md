### 1.0 pickle dump
序列化对象，并将结果数据流写入到文件对象中。
参数protocol是序列化模式，默认值为0，表示以文本的形式序列化。protocol的值还可以是1或2，表示以二进制的形式序列化。
```
pickle_file = 'notMNIST.pickle'

try:
    with open(pickle_file, 'wb') as f:
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
```
### 2.0 pickle load
```
try:
    print(pickle_file)
    with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        ...
except Exception as e:
    print('Unable to process data from', pickle_file, ':', e)
    raise
```