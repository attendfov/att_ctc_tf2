# _*_ coding:utf-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
layers = tf.keras.layers
abspath = os.path.dirname(os.path.realpath(__file__))
abspath = os.path.abspath(abspath)
sys.path.append(abspath)
sys.path.append(os.path.join(abspath, '../utils'))

from Logger import logger
from resnet_utils import ResBlockASTE


class BackboneAste(tf.keras.Model):
    def __init__(self, name=''):
        super(BackboneAste, self).__init__()
        data_format = 'channels_last'
        valid_channel_values = ('channels_first', 'channels_last')
        if data_format not in valid_channel_values:
            raise ValueError('Unknown data_format: %s. Valid values: %s' %
                             (data_format, valid_channel_values))
        self.name_prefix = name + '/BackBoneASTE' if len(name) > 0 else 'BackBoneASTE'
        self.bn_axis = 1 if data_format == 'channels_first' else 3

        self.conv1 = layers.Conv2D(32, (3, 3),
                                   data_format=data_format,
                                   padding='same',
                                   name=self.name_prefix + '/conv1/backbone')

        self.pool1 = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same')
        self.pool2 = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same')
        self.pool3 = tf.layers.MaxPooling2D((2, 2), (2, 1), padding='same')
        self.pool4 = tf.layers.MaxPooling2D((2, 2), (2, 1), padding='same')
        self.pool5 = tf.layers.MaxPooling2D((2, 2), (2, 1), padding='valid')

        self.ResBlock1_1 = ResBlockASTE([32, 32], 1, 'a', self.name_prefix)
        self.ResBlock1_2 = ResBlockASTE([32, 32], 1, 'b', self.name_prefix)
        self.ResBlock1_3 = ResBlockASTE([32, 32], 1, 'c', self.name_prefix)

        self.ResBlock2_1 = ResBlockASTE([64, 64], 2, 'a', self.name_prefix)
        self.ResBlock2_2 = ResBlockASTE([64, 64], 2, 'b', self.name_prefix)
        self.ResBlock2_3 = ResBlockASTE([64, 64], 2, 'c', self.name_prefix)
        self.ResBlock2_4 = ResBlockASTE([64, 64], 2, 'd', self.name_prefix)

        self.ResBlock3_1 = ResBlockASTE([128, 128], 3, 'a', self.name_prefix)
        self.ResBlock3_2 = ResBlockASTE([128, 128], 3, 'b', self.name_prefix)
        self.ResBlock3_3 = ResBlockASTE([128, 128], 3, 'c', self.name_prefix)
        self.ResBlock3_4 = ResBlockASTE([128, 128], 3, 'd', self.name_prefix)
        self.ResBlock3_5 = ResBlockASTE([128, 128], 3, 'e', self.name_prefix)
        self.ResBlock3_6 = ResBlockASTE([128, 128], 3, 'f', self.name_prefix)

        self.ResBlock4_1 = ResBlockASTE([256, 256], 4, 'a', self.name_prefix)
        self.ResBlock4_2 = ResBlockASTE([256, 256], 4, 'b', self.name_prefix)
        self.ResBlock4_3 = ResBlockASTE([256, 256], 4, 'c', self.name_prefix)
        self.ResBlock4_4 = ResBlockASTE([256, 256], 4, 'd', self.name_prefix)
        self.ResBlock4_5 = ResBlockASTE([256, 256], 4, 'e', self.name_prefix)
        self.ResBlock4_6 = ResBlockASTE([256, 256], 4, 'f', self.name_prefix)

        self.ResBlock5_1 = ResBlockASTE([512, 512], 5, 'a', self.name_prefix)
        self.ResBlock5_2 = ResBlockASTE([512, 512], 5, 'b', self.name_prefix)
        self.ResBlock5_3 = ResBlockASTE([512, 512], 5, 'c', self.name_prefix)

    def get_sequence_length(self, width):
        width = tf.cast(width, dtype=tf.float32)
        after_block1 = tf.cast(tf.ceil(width / 2.0), dtype=tf.float32)
        after_block2 = tf.cast(tf.ceil(after_block1 / 2.0), dtype=tf.float32)
        after_pool5 = after_block2 - 1
        return tf.cast(after_pool5, tf.int32)

    def call(self, inputs, widths, training=True):
        x = self.conv1(inputs)

        x = self.ResBlock1_1(x, training)
        x = self.ResBlock1_2(x, training)
        x = self.ResBlock1_3(x, training)
        x = self.pool1(x)

        x = self.ResBlock2_1(x, training)
        x = self.ResBlock2_2(x, training)
        x = self.ResBlock2_3(x, training)
        x = self.ResBlock2_4(x, training)
        x = self.pool2(x)

        x = self.ResBlock3_1(x, training)
        x = self.ResBlock3_2(x, training)
        x = self.ResBlock3_3(x, training)
        x = self.ResBlock3_4(x, training)
        x = self.ResBlock3_5(x, training)
        x = self.ResBlock3_6(x, training)
        x = self.pool3(x)

        x = self.ResBlock4_1(x, training)
        x = self.ResBlock4_2(x, training)
        x = self.ResBlock4_3(x, training)
        x = self.ResBlock4_4(x, training)
        x = self.ResBlock4_5(x, training)
        x = self.ResBlock4_6(x, training)
        x = self.pool4(x)

        x = self.ResBlock5_1(x, training)
        x = self.ResBlock5_2(x, training)
        x = self.ResBlock5_3(x, training)
        x = self.pool5(x)

        widths = self.get_sequence_length(widths)
        logger.debug("BackboneAste:B*H*W*C shape".format(x.shape))
        return x, widths


if __name__ == '__main__':
    import numpy as np
    tf.enable_eager_execution()
    data = np.random.random((1, 32, 24, 3))
    widths0 = [24, 32, 48, 64, 96, 128]
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    ecnoder = BackboneAste()
    features, widths1 = ecnoder(data, widths0, True)

    for var in ecnoder.trainable_variables:
        print(type(var), var.name, var.shape)

    print('data shape:', data.shape)
    print('features shape:', features.shape)

    print('widths0:', widths0[0])
    print('widths1:', widths1.numpy()[0])



