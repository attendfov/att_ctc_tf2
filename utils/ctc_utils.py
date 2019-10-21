# _*_ coding:utf-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
abspath = os.path.dirname(os.path.realpath(__file__))


def dense_to_sparse(dense_tensor, sparse_val=0):
    """Inverse of tf.sparse_to_dense.
    Parameters:
        dense_tensor: The dense tensor. Duh.
        sparse_val: The value to "ignore": Occurrences of this value in the
                    dense tensor will not be represented in the sparse tensor.
                    NOTE: When/if later restoring this to a dense tensor, you
                    will probably want to choose this as the default value.
    Returns:
        SparseTensor equivalent to the dense input.
    """
    with tf.name_scope("dense_to_sparse"):
        sparse_inds = tf.where(tf.not_equal(dense_tensor, sparse_val),
                               name="sparse_inds")
        sparse_vals = tf.gather_nd(dense_tensor, sparse_inds,
                                   name="sparse_vals")
        dense_shape = tf.shape(dense_tensor, name="dense_shape",
                               out_type=tf.int64)
        return tf.SparseTensor(sparse_inds, sparse_vals, dense_shape)


def ctc_metrics(inf_tensor, grt_tensor, label_lens, sparse_val=0, inf_sparse=True, grt_sparse=True):
    if not inf_sparse:
        inf_tensor = dense_to_sparse(inf_tensor, sparse_val)
    if not grt_sparse:
        grt_tensor = dense_to_sparse(grt_tensor, sparse_val)

    inf_tensor = tf.cast(inf_tensor, tf.int32)
    grt_tensor = tf.cast(grt_tensor, tf.int32)

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


def get_label_length(target_input, eos_id, algo_name='ctc'):
    batch = target_input.shape[0]
    steps = target_input.shape[1]

    target_len = tf.zeros([batch], dtype=tf.int32)
    target_input = tf.cast(target_input, dtype=tf.int32)
    batch_eos_id = tf.fill([batch], value=eos_id)
    if algo_name == 'att':
        step_lens = tf.ones([batch], dtype=tf.int32)
    elif algo_name == 'ctc':
        step_lens = tf.zeros([batch], dtype=tf.int32)

    for step in range(steps):
        step_target = target_input[:, step]
        target_mask = tf.equal(step_target, batch_eos_id)
        update_mask = tf.logical_and(target_mask, target_len <= 0)
        target_len = tf.where(update_mask, step_lens, target_len)
        step_lens = step_lens + 1
    return target_len


if __name__ == '__main__':
    a = [[1, 2, 3, 4, 5, 1, 1],
         [1, 2, 0, 4, 0, 2, 0],
         [5, 2, 1, 4, 1, 0, 0]]

    b = [[1, 2, 3, 4, 5],
         [1, 2, 0, 4, 0],
         [5, 2, 1, 4, 1]]

    c = [[1, 2, 3, 4, 9, 9, 9, 9],
         [1, 2, 3, 9, 9, 9, 9, 9],
         [1, 1, 8, 9, 9, 9, 9, 9],
         [1, 9, 9, 9, 9, 9, 9, 9]]

    inf_tensor = dense_to_sparse(tf.convert_to_tensor(a))
    grt_tensor = dense_to_sparse(tf.convert_to_tensor(b))

    time0 = time.time() * 1000
    edit_dist = tf.edit_distance(inf_tensor, grt_tensor, normalize=False)
    sequence_errors = tf.math.count_nonzero(edit_dist, axis=0)
    print('edit_dist:', edit_dist)
    print('sequence_errors:', sequence_errors)
    time1 = time.time() * 1000
    print('time lost:', time1 - time0)

    label_lens = [5, 5, 5]
    seq_error, seq_count, char_error, char_count = ctc_metrics(inf_tensor, grt_tensor, label_lens)
    print(seq_error, seq_count, char_error, char_count)

    label = get_label_length(tf.convert_to_tensor(c), 9)
    print(label)




