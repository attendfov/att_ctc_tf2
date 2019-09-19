# _*_ coding:utf-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

layers = tf.keras.layers


class IdentityBlock(tf.keras.Model):
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
    super(IdentityBlock, self).__init__(name='')
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


class ConvBlock(tf.keras.Model):
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
    super(ConvBlock, self).__init__(name='')
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
               lstm_unit,
               data_format='channels_last',
               name='ResNet18'):
    super(ResNet18, self).__init__(name=name)

    valid_channel_values = ('channels_first', 'channels_last')
    if data_format not in valid_channel_values:
      raise ValueError('Unknown data_format: %s. Valid values: %s' %
                       (data_format, valid_channel_values))
    self.name_prefix = name
    self.data_format = data_format

    self.conv1 = layers.Conv2D(
        64, (3, 3),
        data_format=data_format,
        padding='same',
        name='conv1')
    bn_axis = 1 if data_format == 'channels_first' else 3
    self.bn_conv1 = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')

    self.l2a = ConvBlock(3, [48, 48], stage=2, block='a', strides=(2, 2), data_format=self.data_format)
    self.l2b = IdentityBlock(3, [48, 48], stage=2, block='b', data_format=self.data_format)
    self.l2_dropout = layers.Dropout(0.1)

    self.l3a = ConvBlock(3, [96, 96], stage=3, block='a', strides=(2, 2), data_format=self.data_format)
    self.l3b = IdentityBlock(3, [96, 96], stage=3, block='b', data_format=self.data_format)
    self.l3_dropout = layers.Dropout(0.1)

    self.l4a = ConvBlock(3, [192, 192], stage=4, block='a', strides=(2, 2), data_format=self.data_format)
    self.l4b = IdentityBlock(3, [192, 192], stage=4, block='b', data_format=self.data_format)
    self.l4_dropout = layers.Dropout(0.1)

    self.l5a = ConvBlock(3, [384, 384], stage=5, block='a', strides=(2, 1), data_format=self.data_format)
    self.l5b = IdentityBlock(3, [384, 384], stage=5, block='b', data_format=self.data_format)
    self.l5b = IdentityBlock(3, [384, 384], stage=5, block='b', data_format=self.data_format)
    self.l5_dropout = layers.Dropout(0.1)

    rnn_type = 'lstm'
    assert (rnn_type in ('rnn', 'lstm', 'gru'))
    if rnn_type == 'lstm':
        self.bilstm0 = tf.keras.layers.Bidirectional(
            layers.LSTM(units=lstm_unit, return_sequences=True, name='encode_lstm0'),
            name='bilstm0')
        self.bilstm1 = tf.keras.layers.Bidirectional(
            layers.LSTM(units=lstm_unit, return_sequences=True, name='encode_lstm1'),
            name='bilstm1')
    elif rnn_type == 'gru':
        self.bilstm0 = tf.keras.layers.Bidirectional(
            layers.GRU(units=lstm_unit, return_sequences=True, name='encode_gru0'),
            name='bilstm0')
        self.bilstm1 = tf.keras.layers.Bidirectional(
            layers.GRU(units=lstm_unit, return_sequences=True, name='encode_gru1'),
            name='bilstm1')

  def get_sequence_lengths(self, width):
    width = tf.cast(width, dtype=tf.float32)
    after_conv1 = width-0.0
    after_block1 = tf.cast(tf.math.ceil(after_conv1 / 2.0), dtype=tf.float32)
    after_block2 = tf.cast(tf.math.ceil(after_block1 / 2.0), dtype=tf.float32)
    after_block4 = tf.cast(tf.math.ceil(after_block2 / 2.0), dtype=tf.float32)
    after_maxpool = after_block4 - 0.0
    return tf.cast(after_maxpool, tf.int32)

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
    x = self.l5_dropout(x, training=training)
    print("features shape:", x.shape)
    # BHWC-->BWHC
    features = tf.transpose(x, perm=(0, 2, 1, 3))
    features_s = tf.shape(features)
    features_b = features_s[0]
    features_w = features_s[1]
    features_h = features_s[2]
    features_c = features_s[3]
    features = tf.reshape(features, [features_b, features_w, features_h * features_c])

    print("features reshape:", features.shape)

    features = self.bilstm0(features)
    features = self.bilstm1(features)

    widths = self.get_sequence_lengths(widths)
    widths = tf.reshape(widths, [-1], name='seq_len')
    return features, features, widths


def check_seqlen():
  import numpy as np
  imgh = 48
  imgc = 3
  lstm_units=12
  model = ResNet18(lstm_units)
  for imgw in range(imgh, 3000):
    image = np.random.random((1, imgh, imgw, imgc))
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    print('img shape:', image.shape)
    features, state, width = model(image, [imgw])
    print(type(state))
    b, w, c = features.numpy().shape
    assert(w == width.numpy()[0])
    print(imgw, features.numpy().shape, width.numpy())
    print(model.variables)


if __name__ == '__main__':
    check_seqlen()
