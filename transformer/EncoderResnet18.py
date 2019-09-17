# _*_ coding:utf-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf

abspath = os.path.dirname(os.path.realpath(__file__))
abspath = os.path.abspath(abspath)
sys.path.append(abspath)
sys.path.append(os.path.join(abspath, '../utils'))

from Logger import logger
from TransferUtils import *


layers = tf.keras.layers


class EncoderLayer(tf.keras.Model):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.BatchNormalization()
        self.layernorm2 = tf.keras.layers.BatchNormalization()

        #self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        #self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class _IdentityBlock(tf.keras.Model):
  """_IdentityBlock is the block that has no conv layer at shortcut.
  Args:
    kernel_size: the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    data_format: data_format for the input ('channels_first' or
      'channels_last').
  """

  def __init__(self, kernel_size, filters, stage, block, data_format):
    super(_IdentityBlock, self).__init__(name='')
    filters1, filters2 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    bn_axis = 1 if data_format == 'channels_first' else 3

    self.conv2a = layers.Conv2D(filters1,
                                (3, 3),
                                padding='same',
                                name=conv_name_base + '2a',
                                data_format=data_format)
    self.bn2a = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')

    self.conv2b = layers.Conv2D(filters2,
                                kernel_size,
                                padding='same',
                                data_format=data_format,
                                name=conv_name_base + '2b')
    self.bn2b = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)


class _ConvBlock(tf.keras.Model):
  """_ConvBlock is the block that has a conv layer at shortcut.
  Args:
      kernel_size: the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      data_format: data_format for the input ('channels_first' or
        'channels_last').
      strides: strides for the convolution. Note that from stage 3, the first
       conv layer at main path is with strides=(2,2), and the shortcut should
       have strides=(2,2) as well.
  """

  def __init__(self,
               kernel_size,
               filters,
               stage,
               block,
               data_format,
               strides=(2, 2)):
    super(_ConvBlock, self).__init__(name='')
    filters1, filters2 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    bn_axis = 1 if data_format == 'channels_first' else 3

    self.conv2a = layers.Conv2D(filters1, (3, 3),
                                strides=strides,
                                padding='same',
                                name=conv_name_base + '2a',
                                data_format=data_format)
    self.bn2a = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')

    self.conv2b = layers.Conv2D(filters2,
                                kernel_size,
                                padding='same',
                                name=conv_name_base + '2b',
                                data_format=data_format)
    self.bn2b = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')

    self.conv_shortcut = layers.Conv2D(filters2,
                                       (1, 1),
                                       padding='same',
                                       strides=strides,
                                       name=conv_name_base + '1',
                                       data_format=data_format)
    self.bn_shortcut = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1')

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)

    shortcut = self.conv_shortcut(input_tensor)
    shortcut = self.bn_shortcut(shortcut, training=training)

    x += shortcut
    return tf.nn.relu(x)


# pylint: disable=not-callable
class ResNet18(tf.keras.Model):
  """Instantiates the ResNet50 architecture.
  Args:
    data_format: format for the image. Either 'channels_first' or
      'channels_last'.  'channels_first' is typically faster on GPUs while
      'channels_last' is typically faster on CPUs. See
      https://www.tensorflow.org/performance/performance_guide#data_formats
    name: Prefix applied to names of variables created in the model.
    trainable: Is the model trainable? If true, performs backward
        and optimization after call() method.
    include_top: whether to include the fully-connected layer at the top of the
      network.
    pooling: Optional pooling mode for feature extraction when `include_top`
      is `False`.
      - `None` means that the output of the model will be the 4D tensor
          output of the last convolutional layer.
      - `avg` means that global average pooling will be applied to the output of
          the last convolutional layer, and thus the output of the model will be
          a 2D tensor.
      - `max` means that global max pooling will be applied.
    classes: optional number of classes to classify images into, only to be
      specified if `include_top` is True.
  Raises:
      ValueError: in case of invalid argument for data_format.
  """

  def __init__(self,
               num_layers=2,
               d_model=512,
               num_heads=4,
               dff=1024,
               rate=0.1,
               used_rnn=False,
               max_width=1600,
               name='ResNet18'):
    super(ResNet18, self).__init__(name=name)
    self.used_rnn = used_rnn
    data_format = 'channels_last'
    valid_channel_values = ('channels_first', 'channels_last')
    if data_format not in valid_channel_values:
      raise ValueError('Unknown data_format: %s. Valid values: %s' %
                       (data_format, valid_channel_values))
    self.name_prefix = name

    def conv_block(filters, stage, block, strides=(2, 2)):
      return _ConvBlock(
          3,
          filters,
          stage=stage,
          block=block,
          data_format=data_format,
          strides=strides)

    def id_block(filters, stage, block):
      return _IdentityBlock(
          3, filters, stage=stage, block=block, data_format=data_format)

    self.conv1 = layers.Conv2D(
        64, (3, 3),
        data_format=data_format,
        padding='same',
        name='conv1')
    bn_axis = 1 if data_format == 'channels_first' else 3
    self.bn_conv1 = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')

    self.l2a = conv_block([48, 48], stage=2, block='a', strides=(2, 2))
    self.l2b = id_block([48, 48], stage=2, block='b')
    self.l2_dropout = layers.Dropout(0.1)

    self.l3a = conv_block([96, 96], stage=3, block='a', strides=(2, 2))
    self.l3b = id_block([96, 96], stage=3, block='b')
    self.l3_dropout = layers.Dropout(0.1)

    self.l4a = conv_block([192, 192], stage=4, block='a', strides=(2, 2))
    self.l4b = id_block([192, 192], stage=4, block='b')
    self.l4_dropout = layers.Dropout(0.1)

    self.l5a = conv_block([384, 384], stage=5, block='a', strides=(2, 1))
    self.l5b = id_block([384, 384], stage=5, block='b')
    self.l5b = id_block([384, 384], stage=5, block='b')
    self.l5_dropout = layers.Dropout(0.1)

    self.dff = dff
    self.rate = rate
    self.d_model = d_model
    self.num_heads = num_heads
    self.max_width = max_width
    self.num_layers = num_layers
    self.transfer_enc_layers = []
    self.dropout = tf.keras.layers.Dropout(rate)
    self.pos_encoding = positional_encoding(self.max_width, self.d_model)

    if self.num_layers > 0:
        assert (self.d_model % self.num_heads == 0)

    units = self.d_model
    rnn_type = 'gru'
    assert (rnn_type in ('rnn', 'lstm', 'gru'))
    if rnn_type == 'rnn':
        rnn_class = tf.nn.rnn_cell.RNNCell
    elif rnn_type == 'lstm':
        rnn_class = tf.nn.rnn_cell.LSTMCell
    elif rnn_type == 'gru':
        rnn_class = tf.nn.rnn_cell.GRUCell

    if used_rnn:
        rnn_fw_name = 'encode_rnn_fw0'
        rnn_bw_name = 'encode_rnn_bw0'
        self.rnn_fw0 = rnn_class(num_units=units, dtype=tf.float32, name=rnn_fw_name)
        self.rnn_bw0 = rnn_class(num_units=units, dtype=tf.float32, name=rnn_bw_name)

        rnn_fw_name = 'encode_rnn_fw1'
        rnn_bw_name = 'encode_rnn_bw1'
        self.rnn_fw1 = rnn_class(num_units=units, dtype=tf.float32, name=rnn_fw_name)
        self.rnn_bw1 = rnn_class(num_units=units, dtype=tf.float32, name=rnn_bw_name)
        self.rnn_multi_fw = tf.nn.rnn_cell.MultiRNNCell([self.rnn_fw0, self.rnn_fw1])
        self.rnn_multi_bw = tf.nn.rnn_cell.MultiRNNCell([self.rnn_bw0, self.rnn_bw1])
        # self.rnn_multi_fw = tf.nn.rnn_cell.DropoutWrapper(self.rnn_multi_fw, 0.9, 0.95, 0.95)
        # self.rnn_multi_bw = tf.nn.rnn_cell.DropoutWrapper(self.rnn_multi_bw, 0.9, 0.95, 0.95)

    self.dense_layer = tf.keras.layers.Dense(self.d_model)
    for layer_id in range(num_layers):
        self.transfer_enc_layers.append(EncoderLayer(self.d_model, self.num_heads, self.dff, self.rate))

  def get_sequence_lengths(self, width):
    width = tf.cast(width, dtype=tf.float32)
    after_conv1 = width-0.0
    after_block1 = tf.cast(tf.ceil(after_conv1 / 2.0), dtype=tf.float32)
    after_block2 = tf.cast(tf.ceil(after_block1 / 2.0), dtype=tf.float32)
    after_block4 = tf.cast(tf.ceil(after_block2 / 2.0), dtype=tf.float32)
    after_maxpool = after_block4 - 0.0
    return tf.cast(after_maxpool, tf.int32)

  def get_reverse_points(self, points):
    ret_points = []
    for point in points:
      pointx, pointy = point
      anchor_xcent = 8 * pointx + 4
      anchor_xsoff = max(anchor_xcent - 8, 0)
      anchor_xeoff = max(anchor_xcent + 8, 0)

      anchor_ycent = 16 * pointy + 8
      anchor_ysoff = max(anchor_ycent - 16, 0)
      anchor_yeoff = max(anchor_ycent + 16, 0)

      positions = []
      for anchory_off in range(anchor_ysoff, anchor_yeoff):
        if int(anchory_off / 16.0) != pointy:
          continue
        for anchorx_off in range(anchor_xsoff, anchor_xeoff):
          if int(anchorx_off / 8.0) == pointx:
            positions.append([int(anchorx_off / 8.0), int(anchory_off / 16.0)])
      ret_points.append(positions)
    return ret_points

  def bidirectional_rnn_foreward(self, inputs, rnn_fw_name, rnn_bw_name):
    outputs_fb, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.op_map[rnn_fw_name],
                                                                cell_bw=self.op_map[rnn_bw_name],
                                                                inputs=inputs,
                                                                dtype=tf.float32,
                                                                time_major=False)
    result = tf.concat(outputs_fb, axis=2)
    return result, output_states

  def call(self, inputs, widths, training=True):
    x = self.conv1(inputs)
    x = self.bn_conv1(x, training=training)
    x = tf.nn.relu(x)

    x = self.l2a(x, training=training)
    x = self.l2b(x, training=training)
    x = self.l2_dropout(x, training=training)

    x = self.l3a(x, training=training)
    x = self.l3b(x, training=training)
    x = self.l3_dropout(x, training=training)

    x = self.l4a(x, training=training)
    x = self.l4b(x, training=training)
    x = self.l4_dropout(x, training=training)

    x = self.l5a(x, training=training)
    x = self.l5b(x, training=training)
    features = self.l5_dropout(x, training=training)

    logger.info("B*H*W*C features shape: {}".format(features.shape))

    features_s = tf.shape(features)
    features_b = features_s[0]
    features_h = features_s[1]
    features_w = features_s[2]
    features_c = features_s[3]
    # B*H*W*C-->BH*W*C
    features = tf.reshape(features, [features_b * features_h, features_w, features_c])
    logger.debug("B*H*W*C-->BH*W*C shape: {}".format(features.shape))
    if self.used_rnn:
        features, fw_status = self.bidirectional_rnn_foreward(features, self.rnn_multi_fw, self.rnn_multi_bw)

    features = self.dense_layer(features)
    logger.debug("BH*W*C-->BH*W*D shape: {}".format(features.shape))

    weight_mask = None
    if widths is not None:
        widths = self.get_sequence_lengths(widths)
        logger.debug('widths shape {}, {}'.format(widths.shape, widths))
        widths = tf.reshape(widths, [-1, 1], name='seq_len')
        widths = tf.concat([widths for h in range(features_h)], axis=-1)
        widths = tf.reshape(widths, [-1])
        weight_mask = tf.sequence_mask(widths, features_w, dtype=tf.float32)
        weight_mask = tf.reshape(weight_mask, [features_b, features_h * features_w])
        logger.debug("weight mask shape: {}".format(weight_mask.shape))

    features += self.pos_encoding[:, :features_w, :]

    # BH*W*D --> B*HW*D
    features = tf.reshape(features, [features_b, features_h * features_w, -1])

    logger.debug("features shape: {}".format(features.shape))
    if self.num_layers > 0:
        mask = None
        if weight_mask is not None:
            mask = tf.matmul(tf.expand_dims(weight_mask, -1),
                             tf.expand_dims(weight_mask, -1), transpose_b=True)
            mask = tf.expand_dims(1.0 - mask, axis=1)
        features = self.dropout(features, training=training)
        for i in range(self.num_layers):
            features = self.transfer_enc_layers[i](features, training, mask)

    return features, weight_mask


if __name__ == '__main__':
    import numpy as np
    data = np.random.random((4, 48, 128, 3))
    widths = [48, 72, 96, 128]
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    ecnoder = ResNet18(1)
    features, weight_mask = ecnoder(data, widths, True)
    print("features shape: {}".format(features.shape))
    print("weight   shape: {}".format(weight_mask.shape))
    print(weight_mask)