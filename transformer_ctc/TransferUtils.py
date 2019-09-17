# _*_ coding:utf-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()
if sys.version.startswith('2'):
    reload(sys)
    sys.setdefaultencoding('utf-8')

abspath = os.path.dirname(os.path.realpath(__file__))
abspath = os.path.abspath(abspath)

sys.path.append(abspath)
sys.path.append(os.path.join(abspath, '.'))
sys.path.append(os.path.join(abspath, '../utils'))


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :], d_model)

    # apply sin to even indices in the array; 2i
    sines = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def positional_encoding_unit():
    pos_encoding = positional_encoding(50, 512)
    print('pos_encoding shape {}'.format(pos_encoding.shape))


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_padding_mask_unit():
    x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    mask = create_padding_mask(x)
    print('create_padding_mask input:', x)
    print('create_padding_mask otput:', mask)


def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


def create_look_ahead_mask_unit():
    temp = create_look_ahead_mask(6)
    print("create_look_ahead_mask size:", temp)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    #print("scaled_attention_logits shape: {}".format(scaled_attention_logits.shape))
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_v, depth)

    return output, attention_weights


def print_out(q, k, v):
  print('query shape: {}'.format(q.shape))
  print('keys  shape: {}'.format(k.shape))
  print('value shape: {}'.format(v.shape))
  temp_out, temp_attn = scaled_dot_product_attention(q, k, v, None)
  print('Attention weights shape: {}'.format(temp_attn.shape))
  print('Attention outputs shape: {}'.format(temp_out.shape))


def scaled_dot_product_attention_unit():
    np.set_printoptions(suppress=True)

    temp_k = tf.constant([[10,0,0],
                          [0,10,0],
                          [0,0,10],
                          [0,0,10]], dtype=tf.float32)  # (4, 3)

    temp_v = tf.constant([[   1,0],
                          [  10,0],
                          [ 100,5],
                          [1000,6]], dtype=tf.float32)  # (4, 3)

    # This `query` aligns with the second `key`,
    # so the second `value` is returned.
    temp_q = tf.constant([[0, 10, 0], [0, 2, 0]], dtype=tf.float32)  # (1, 3)
    print_out(temp_q, temp_k, temp_v)


class LayerNormalization(tf.keras.Model):
    def __init__(self, depth, scope='LayerNormalization'):
        super(LayerNormalization, self).__init__()
        self.scope = scope
        self.depth = depth
        self.epsilon = 0.000001
        with tf.variable_scope(scope):
            self.beta = tf.get_variable("beta", [depth], initializer=tf.zeros_initializer())
            self.gamma = tf.get_variable("gamma", [depth], initializer=tf.ones_initializer())

    def call(self, inputs):
        '''
        :param inputs: batch * ... * depth
        :return: LayerNormalization output
        '''
        inputs_shape = inputs.shape
        print("inputs_length: {}, {}".format(type(inputs_shape), inputs_shape))
        inputs_length = int(len(inputs_shape))
        axis = [1]
        if inputs_length > 3:
            axis = list(range(1, inputs_length-1))

        tf.logging.debug("inputs shape: {}".format(inputs.shape))
        mean, variance = tf.nn.moments(inputs, axes=axis, keep_dims=True)
        tf.logging.debug("mean shape: {}".format(mean.shape))
        tf.logging.debug("variance shape: {}".format(variance.shape))
        normalized = (inputs - mean) / ((variance + self.epsilon)**(0.5))
        outputs = self.gamma * normalized + self.beta
        return outputs


class MultiHeadAttention(tf.keras.Model):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # (batch_size, seq_len_v, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_v, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)  # (batch_size, seq_len_v, d_model)

        return output, attention_weights


def MultiHeadAttentionUnit():
    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    print('multiHead  outputs  shape {}'.format(out.shape))
    print('multiHead attention shape {}'.format(attn.shape))


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

