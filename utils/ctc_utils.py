# _*_ coding:utf-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
abspath = os.path.dirname(os.path.realpath(__file__))

from tensor_utils import dense_to_sparse


def ctc_metrics(inf_tensor, grt_tensor, label_lens, sparse_val=0, inf_sparse=True, grt_sparse=True):
    if not inf_sparse:
        inf_tensor = dense_to_sparse(inf_tensor, sparse_val)
    if not grt_sparse:
        grt_tensor = dense_to_sparse(grt_tensor, sparse_val)

    edit_dist = tf.edit_distance(inf_tensor, grt_tensor)
    seq_count = tf.cast(tf.reduce_max(tf.shape(edit_dist)), tf.float32)
    seq_error = tf.cast(tf.math.count_nonzero(edit_dist, axis=0), tf.float32)

    char_count = tf.cast(tf.reduce_sum(label_lens), tf.float32)
    char_error = tf.cast(tf.reduce_sum(edit_dist), tf.float32)

    return seq_error, seq_count, char_error, char_count


def sparse_ctc_loss(ctc_logits, ctc_labels, seq_lens, blank_index):
    loss = tf.nn.ctc_loss(ctc_labels, ctc_logits,
                          label_length=None,
                          logit_length=seq_lens,
                          logits_time_major=False,
                          blank_index=blank_index)
    return loss


if __name__ == '__main__':
    a = [[1,2,3,4,5,1,1],
        [1,2,0,4,0,2,0],
        [5,2,1,4,1,0,0]]

    b = [[1,2,3,4,5],
        [1,2,0,4,0],
        [5,2,1,4,1]]

    inf_tensor = dense_to_sparse(tf.convert_to_tensor(a))
    grt_tensor = dense_to_sparse(tf.convert_to_tensor(b))

    time0 = time.time() * 1000
    edit_dist = tf.edit_distance(inf_tensor, grt_tensor, normalize=False)
    sequence_errors = tf.math.count_nonzero(edit_dist, axis=0)
    print(edit_dist)
    print(sequence_errors)
    time1 = time.time() * 1000
    print(time1 - time0)

    label_lens = [5, 5, 5]
    seq_error, seq_count, char_error, char_count = ctc_metrics(inf_tensor, grt_tensor, label_lens)
    print(seq_error, seq_count, char_error, char_count)




