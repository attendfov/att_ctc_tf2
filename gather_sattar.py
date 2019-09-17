
import os
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

'''
batch = 6
cls_num = 4

label = [np.random.randint(0, cls_num) for i in range(batch)]
label = tf.convert_to_tensor(label, dtype=tf.int32)
logits = np.random.random((batch, cls_num))
logits = tf.convert_to_tensor(logits, dtype=tf.float32)

print(label)
print(logits)

range = tf.range(batch, dtype=label.dtype)
indices = tf.concat([tf.expand_dims(range, axis=1), tf.expand_dims(label, axis=1)], axis=1)
print('indices:', indices)
gather = tf.gather_nd(logits, indices)
print('gather:', gather)
gather = gather + 10.
print('gather:', gather)

shape = tf.shape(logits)
logits = tf.scatter_nd(indices, gather, shape)
print(logits)
'''















