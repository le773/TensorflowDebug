
# coding: utf-8

# In[8]:


import numpy as np
import threading
import time
import tensorflow as tf

def MyLoop(coord, work_id):
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print('Stop from id: {}\n'.format(work_id))
            coord.request_stop()
        else:
            print('Working from id: {}\n'.format(work_id))
        time.sleep(1)


coord = tf.train.Coordinator()
threads = [threading.Thread(target=MyLoop, args=(coord, i,)) for i in range(5)]
for t in threads:
    t.start()
coord.join(threads)

