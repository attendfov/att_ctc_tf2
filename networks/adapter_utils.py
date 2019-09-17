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

layers = tf.keras.layers
l2_regularizers = tf.keras.regularizers.l2
l1_regularizers = tf.keras.regularizers.l1
l1_l2_regularizers = tf.keras.regularizers.l1_l2


def adapter_structure(adap_type, filters, name, data_format):
    if adap_type == 'easy':
        conv_ops1x1 = layers.Conv2D(filters,
                                    (1, 1),
                                    padding='same',
                                    name=name + '_' + adap_type + '_1x1',
                                    kernel_regularizer=l2_regularizers(1e-5),
                                    kernel_initializer="he_normal",
                                    data_format=data_format)

        return [conv_ops1x1]
    elif adap_type == 'medium':
        conv_ops2x2 = layers.Conv2D(filters,
                                    (2, 2),
                                    padding='same',
                                    name=name + '_' + adap_type + '_2x2',
                                    kernel_regularizer=l2_regularizers(1e-5),
                                    kernel_initializer="he_normal",
                                    data_format=data_format)

        return [conv_ops2x2]

    elif adap_type == 'hard':
        conv_ops3x1 = layers.Conv2D(filters,
                                    (3, 1),
                                    padding='same',
                                    name=name + '_' + adap_type + '_3x1',
                                    kernel_regularizer=l2_regularizers(1e-5),
                                    kernel_initializer="he_normal",
                                    data_format=data_format)

        conv_ops1x3 = layers.Conv2D(filters,
                                    (1, 3),
                                    padding='same',
                                    name=name + '_' + adap_type + '_1x3',
                                    kernel_regularizer=l2_regularizers(1e-5),
                                    kernel_initializer="he_normal",
                                    data_format=data_format)

        return [conv_ops3x1, conv_ops1x3]
