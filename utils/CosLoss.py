#-*- encode:utf-8 -*-
import os
import numpy as np
import tensorflow as tf

abspath = os.path.dirname(os.path.realpath(__file__))
abspath = os.path.abspath(abspath)
from Logger import logger

tf.enable_eager_execution()

def cos_loss(x, y, w, alpha=0.25, scale=64):
    '''
    x: (B x D) - features, B-->batch, D-->units
    y: (B,)    - labels, B-->batch,
    w: (D x C) - weigth, D-->units, C-->cls num
    alpah: 1   - margin
    scale: 1   - scaling paramter
    '''

    print("y")
    print(y)


    batch = int(tf.shape(x)[0])
    units = int(tf.shape(x)[1])

    # normalize the feature and weight
    # (N,D)
    x_feat_norm = tf.nn.l2_normalize(x, 1, 1e-10)
    # (D,C)
    w_feat_norm = tf.nn.l2_normalize(w, 0, 1e-10)

    # get the scores after normalization
    # (N,C)
    xw_norm = tf.matmul(x_feat_norm, w_feat_norm)
    logger.debug("xw_norm")
    logger.debug(xw_norm)
    indices = tf.concat([tf.expand_dims(tf.range(batch, dtype=tf.int32), axis=1),
                         tf.expand_dims(y, axis=1)], axis=1)
    updates = tf.convert_to_tensor([-alpha] * batch, dtype=x.dtype)
    updates = tf.scatter_nd(indices, updates, tf.shape(xw_norm))
    logger.debug("updates")
    logger.debug(updates)
    margins = tf.add(xw_norm, updates)*scale
    logger.debug('margins')
    logger.debug(margins)
    return margins


def test_cos_loss():
    batch = 5
    units = 6
    cls_num = 4

    x = np.random.random((batch, units))
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    w = np.random.random((units, cls_num))
    w = tf.convert_to_tensor(w, dtype=tf.float32)

    y = [np.random.randint(0, cls_num) for i in range(batch)]
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    margins = cos_loss(x, y, w, alpha=-10, scale=64)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=margins))
    logger.debug('loss:{}'.format(loss))


if __name__=='__main__':
    test_cos_loss()