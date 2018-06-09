from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import random
import os
import zipfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

# Parameters for downloading data
DOWNLOAD_URL = 'http://mattmahoney.net/dc/'
EXPECTED_BYTES = 31344016
DATA_FOLDER = './data/'
FILE_NAME = 'text8.zip'

def download(file_name, expected_bytes):
    """ Download the dataset text8 if it's not already downloaded """
    file_path = DATA_FOLDER + file_name
    print('DATA_FOLDER: ', DATA_FOLDER, ' file_name: ', file_name)
    if os.path.exists(file_path):
        print("Dataset ready")
        return file_path
    file_name, _ = urllib.request.urlretrieve(DOWNLOAD_URL + file_name, file_path)
    file_stat = os.stat(file_path)
    if file_stat.st_size == expected_bytes:
        print('Successfully downloaded the file', file_name)
    else:
        raise Exception('File ' + file_name +
                        ' might be corrupted. You should try downloading it with a browser.')
    return file_path

def read_data(file_path):
    """ Read data into a list of tokens 
    There should be 17,005,207 tokens
    """
    with zipfile.ZipFile(file_path) as f:
        words = tf.compat.as_str(f.read(f.namelist()[0])).split() 
        # tf.compat.as_str() converts the input into the string
    return words

def build_vocab(words, vocab_size):
    """ Build vocabulary of VOCAB_SIZE most frequent words """
    dictionary = dict()
    count = [('UNK', -1)]
    # 出现次数最多的vocab_size - 1个数
    count.extend(Counter(words).most_common(vocab_size - 1))
    print('top 4 count')
    for i in range(1,4):
        print(count[i])
    index = 0
    with open('processed/vocab_1000.tsv', "w") as f:
        # f.write("Name\n")
        for word, _ in count:
            dictionary[word] = index
            if index < 1000:
                f.write(word + "\n")
            index += 1
    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
#     print('top 4 dictionary')
#     for i in range(1,4):
#         print(dictionary[i])
#     print('top 4 index_dictionary')
#     for i in range(1,4):
#         print(index_dictionary[i])
    return dictionary, index_dictionary

def convert_words_to_index(words, dictionary):
    """ Replace each word in the dataset with its index in the dictionary """
    return [dictionary[word] if word in dictionary else 0 for word in words]

def generate_sample(index_words, context_window_size):
    """ Form training pairs according to the skip-gram model. """
    for index, center in enumerate(index_words):
        context = random.randint(1, context_window_size)
        # get a random target before the center word
        for target in index_words[max(0, index - context): index]:
            yield center, target
        # get a random target after the center wrod
        for target in index_words[index + 1: index + context + 1]:
            yield center, target

def get_batch(iterator, batch_size):
    """ Group a numerical stream into batches and yield them as Numpy arrays. """
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(iterator)
            if index == 10:
                pass
                # print('center_batch[index]={}, target_batch[index]={}'.format(center_batch[index], target_batch[index]))
        yield center_batch, target_batch

def process_data(vocab_size, batch_size, skip_window):
    print('vocab_size={}, batch_size={}, skip_window={}'.format(vocab_size, batch_size, skip_window))
    file_path = download(FILE_NAME, EXPECTED_BYTES)
    print('file_path:',file_path)
    words = read_data(file_path)
    print('words length:{}'.format(len(words)))
    for i in range(1,5):
        print('words={}'.format(words[i]))
    dictionary, _ = build_vocab(words, vocab_size)
    print('after build_vocab')
    for i in range(1,5):
        print('words={}'.format(words[i]))
    index_words = convert_words_to_index(words, dictionary)
    print('type of index_words:',type(index_words))
    print('after convert_words_to_index')
    for i in range(1,5):
        print('words={}'.format(index_words[i]))
    del words # to save memory
    single_gen = generate_sample(index_words, skip_window)
    print('single_gen',single_gen)
    return get_batch(single_gen, batch_size)

def get_index_vocab(vocab_size):
    file_path = download(FILE_NAME, EXPECTED_BYTES)
    words = read_data(file_path)
    return build_vocab(words, vocab_size)
