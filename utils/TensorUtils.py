# _*_ coding:utf-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf


def clip_gradients(max_grad_norm, grad_and_vars):
    # clip gradient to prevent inf loss
    if max_grad_norm > 0:
        clipped_grads_and_vars = []
        for grad, var in grad_and_vars:
            if grad is not None:
                if isinstance(grad, tf.IndexedSlices):
                    tmp = tf.clip_by_norm(grad.values, max_grad_norm)
                    grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
                else:
                    grad = tf.clip_by_norm(grad, max_grad_norm)
            clipped_grads_and_vars.append((grad, var))
        return clipped_grads_and_vars
    return grad_and_vars


def read_checkpoint(checkpoint_prefix):
    reader = tf.train.NewCheckpointReader(checkpoint_prefix)
    shape_dict = reader.get_variable_to_shape_map()
    dtype_dict = reader.get_variable_to_dtype_map()
    numpy_dict = {}

    for tensor_key in shape_dict:
        tensor = reader.get_tensor(tensor_key)
        numpy_dict[tensor_key] = tensor
    return shape_dict, dtype_dict, numpy_dict


def string2sparse(tf_strings, output_shape, dtype=tf.int32):
    str_sparse = tf.string_split(tf_strings, ',')
    incides = str_sparse.indices
    values = str_sparse.values
    values = tf.string_to_number(values, out_type=dtype)
    sparse_tensor = tf.SparseTensor(indices=incides, values=values, dense_shape=output_shape)
    return sparse_tensor


def string2dense(tf_strings, output_shape, default_value, dtype=tf.int32):
    str_sparse = tf.string_split(tf_strings, ',')
    incides = str_sparse.indices
    values = str_sparse.values
    values = tf.string_to_number(values, out_type=dtype)
    dense_tensor = tf.sparse_to_dense(incides, output_shape, values, default_value)
    return dense_tensor


def single_cell_class(cell_type):
    assert(cell_type in ('gru', 'rnn', 'lstm'))
    if cell_type == 'rnn':
        single_cell = tf.nn.rnn_cell.RNNCell
    elif cell_type == "gru":
        single_cell = tf.nn.rnn_cell.GRUCell
    elif cell_type == "lstm":
        single_cell = tf.nn.rnn_cell.LSTMCell
    return single_cell


def bidirectional_rnn_foreward(inputs, rnn_fw_cell, rnn_bw_cell, dtype=tf.float32, time_major=False):
    outputs_fb, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn_fw_cell,
                                                                cell_bw=rnn_bw_cell,
                                                                inputs=inputs,
                                                                dtype=dtype,
                                                                time_major=time_major)

    outputs_fb = tf.concat(outputs_fb, axis=2)
    return outputs_fb, output_states


def convert_bhwc2hwc(tensor):
    x_shape = tf.shape(tensor)
    b = x_shape[0]
    h = x_shape[1]
    w = x_shape[2]
    c = x_shape[3]

    # BHWC-->BWHC
    tensor = tf.transpose(tensor, perm=(0, 2, 1, 3))
    tensor = tf.reshape(tensor, [b, w, h * c])
    return tensor


def dense2sparse(tensor):
    tensor_idx = tf.where(tf.not_equal(tensor, 0))
    tensor_sparse = tf.SparseTensor(tensor_idx, tf.gather_nd(tensor, tensor_idx), tf.shape(tensor))
    return tensor_sparse


def get_target_length(target_input, eos_id):
    # target_input:batch_size*steps
    batch = int(target_input.shape[0])
    steps = int(target_input.shape[1])

    target_len = tf.convert_to_tensor([0] * batch, dtype=tf.int32)
    target_input = tf.cast(target_input, dtype=tf.int32)
    batch_eos_id = tf.convert_to_tensor([eos_id] * batch, dtype=tf.int32)

    step_lens = tf.convert_to_tensor([1] * batch, dtype=tf.int32)
    for step in range(steps):
        step_target = target_input[:, step]
        target_mask = tf.equal(step_target, batch_eos_id)
        update_mask = tf.logical_and(target_mask, target_len <= 0)
        target_len = tf.where(update_mask, step_lens, target_len)
        step_lens = step_lens + 1
    return target_len


def get_target_length_unit():
    import random
    sos_id = 0
    eos_id = 1
    vocab_size = 8
    batch_size = 12
    padding_len = 10
    target_inputs = []
    target_lengths = []
    for b in range(batch_size):
        target = [sos_id]
        valid_len = random.choice(range(2, padding_len - 2))
        target_lengths.append(valid_len + 2)
        for i in range(valid_len):
            target.append(random.choice(range(2, vocab_size)))
        for i in range(len(target), padding_len):
            target.append(eos_id)
        target_inputs.append(target)

    print(target_inputs)
    print(target_lengths)

    target_inputs = tf.convert_to_tensor(target_inputs)
    target_lengths = get_target_length(target_inputs, eos_id)
    print(target_lengths)


if __name__ == '__main__':
    tf.enable_eager_execution()
    tf_strings = ['1,2,3,4,5,6', '2,4,6,8']
    b = string2sparse(tf_strings, dtype=tf.int32)
    print(type(b), b)
    restore_path = '/Users/junhuang.hj/Desktop/code_paper/code/attention_eager/transformer/training_checkpoints/ckpt-1'
    reader = tf.train.NewCheckpointReader(restore_path)
    shape_dict = reader.get_variable_to_shape_map()
    dtype_dict = reader.get_variable_to_dtype_map()

    print(type(shape_dict), len(shape_dict), shape_dict.items())
    print(type(dtype_dict), len(dtype_dict), dtype_dict.items())

    for key in shape_dict:
        tensor = reader.get_tensor(key)
        if isinstance(tensor, numpy.ndarray):
            print(key, type(tensor), tensor.shape, tensor.dtype)
        else:
            print(key, type(tensor), tensor)









