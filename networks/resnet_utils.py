# _*_ coding:utf-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy
import tensorflow as tf
import string

abspath = os.path.dirname(os.path.realpath(__file__))
abspath = os.path.abspath(abspath)
sys.path.append(abspath)
sys.path.append(os.path.join(abspath, '../utils'))

from Logger import logger
from adapter_utils import adapter_structure

layers = tf.keras.layers
l2_regularizers = tf.keras.regularizers.l2
l1_regularizers = tf.keras.regularizers.l1
l1_l2_regularizers = tf.keras.regularizers.l1_l2


class ResIdentityBlock(tf.keras.Model):
    def __init__(self,
                 kernel_size,
                 filters,
                 stage,
                 block,
                 data_format):
        super(ResIdentityBlock, self).__init__(name='')
        filters1, filters2 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.conv2a = layers.Conv2D(filters1,
                                    (3, 3),
                                    padding='same',
                                    name=conv_name_base + '2a' + '/backbone',
                                    kernel_regularizer=l2_regularizers(1e-5),
                                    kernel_initializer="he_normal",
                                    data_format=data_format)
        self.bn2a = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a' + '/backbone')

        self.conv2b = layers.Conv2D(filters2,
                                    kernel_size,
                                    padding='same',
                                    data_format=data_format,
                                    name=conv_name_base + '2b' + '/backbone',
                                    kernel_regularizer=l2_regularizers(1e-5),
                                    kernel_initializer="he_normal",
                                    )
        self.bn2b = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b' + '/backbone')

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


class ResConvBlock(tf.keras.Model):
    def __init__(self,
                 kernel_size,
                 filters,
                 stage,
                 block,
                 data_format,
                 strides=(2, 2)):
        super(ResConvBlock, self).__init__(name='')
        filters1, filters2 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.conv2a = layers.Conv2D(filters1, (3, 3),
                                    strides=strides,
                                    padding='same',
                                    name=conv_name_base + '2a' + '/backbone',
                                    kernel_regularizer=l2_regularizers(1e-5),
                                    kernel_initializer="he_normal",
                                    data_format=data_format)
        self.bn2a = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a' + '/backbone')

        self.conv2b = layers.Conv2D(filters2,
                                    kernel_size,
                                    padding='same',
                                    name=conv_name_base + '2b' + '/backbone',
                                    kernel_regularizer=l2_regularizers(1e-5),
                                    kernel_initializer="he_normal",
                                    data_format=data_format)
        self.bn2b = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b' + '/backbone')

        self.conv_shortcut = layers.Conv2D(filters2,
                                           (1, 1),
                                           padding='same',
                                           strides=strides,
                                           name=conv_name_base + '1' + '/backbone',
                                           kernel_regularizer=l2_regularizers(1e-5),
                                           kernel_initializer="he_normal",
                                           data_format=data_format)
        self.bn_shortcut = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1' + '/backbone')

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


class ResBlock(tf.keras.Model):
    def __init__(self,
                 filters,
                 stage,
                 block,
                 data_format=None):
        super(ResBlock, self).__init__(name='')
        self.filters = filters
        self.branch_names = ['2'+str(abc) for abc in string.ascii_lowercase]
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.bn_dict = {}
        self.cnn_dict = {}
        for index, filter in enumerate(self.filters):
            cnn_var_name = self.branch_names[index]
            kernel_size = (1, 1) if index == 0 else (3, 3)
            self.cnn_dict[cnn_var_name] = layers.Conv2D(filter,
                                                        kernel_size,
                                                        padding='same',
                                                        name=conv_name_base + cnn_var_name + '/backbone',
                                                        kernel_regularizer=l2_regularizers(1e-5),
                                                        kernel_initializer="he_normal",
                                                        data_format=data_format)

            self.bn_dict[cnn_var_name] = layers.BatchNormalization(axis=bn_axis,
                                                                   name=bn_name_base + cnn_var_name + '/backbone')

    def call(self, input_tensor, training=False):
        x = input_tensor
        cnn_var_name = self.branch_names[0]
        x = self.cnn_dict[cnn_var_name](x)
        x = self.bn_dict[cnn_var_name](x)
        x = tf.nn.relu(x)
        res_input = x
        for index, filter in enumerate(self.filters[1:-1]):
            cnn_var_name = self.branch_names[index]
            x = self.cnn_dict[cnn_var_name](x)
            x = self.bn_dict[cnn_var_name](x)
            x = tf.nn.relu(x)

        cnn_var_name = self.branch_names[len(self.filters)-1]
        x = self.cnn_dict[cnn_var_name](x)
        x = self.bn_dict[cnn_var_name](x)
        x += res_input
        return tf.nn.relu(x)


class ResBlockASTE(tf.keras.Model):
    def __init__(self,
                 filters,
                 stage,
                 block,
                 name_prefix='',
                 data_format=None):
        super(ResBlockASTE, self).__init__(name='')
        self.filters = filters
        self.name_prefix = name_prefix
        self.branch_names = ['2'+str(abc) for abc in string.ascii_lowercase]
        conv_name_base = self.name_prefix + '/' + 'res' + str(stage) + block + '_branch'
        bn_name_base = self.name_prefix + '/' + 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.bn_dict = {}
        self.cnn_dict = {}
        for index, filter in enumerate(self.filters):
            cnn_var_name = self.branch_names[index]
            kernel_size = (1, 1) if index == 0 else (3, 3)
            self.cnn_dict[cnn_var_name] = layers.Conv2D(filter,
                                                        kernel_size,
                                                        padding='same',
                                                        name=conv_name_base + cnn_var_name + '/backbone',
                                                        kernel_regularizer=l2_regularizers(1e-5),
                                                        kernel_initializer="he_normal",
                                                        data_format=data_format)

            self.bn_dict[cnn_var_name] = layers.BatchNormalization(axis=bn_axis,
                                                                   name=bn_name_base + cnn_var_name + '/backbone')

    def call(self, input_tensor, training=False):
        outnums = int(self.filters[-1])
        channel = int(tf.shape(input_tensor)[-1])
        shortcut = input_tensor
        x = input_tensor
        for index, filter in enumerate(self.filters[:-1]):
            cnn_var_name = self.branch_names[index]
            x = self.cnn_dict[cnn_var_name](x)
            x = self.bn_dict[cnn_var_name](x)
            if index == 0 and outnums != channel:
                shortcut = x
            x = tf.nn.relu(x)

        cnn_var_name = self.branch_names[len(self.filters)-1]
        x = self.cnn_dict[cnn_var_name](x)
        x = self.bn_dict[cnn_var_name](x)
        x += shortcut
        return tf.nn.relu(x)


class ResIdentityBlockAda(tf.keras.Model):
    def __init__(self,
                 kernel_size,
                 filters,
                 stage,
                 block,
                 adap_names,
                 adap_types,
                 data_format):
        super(ResIdentityBlockAda, self).__init__(name='')
        filters1, filters2 = filters
        assert (len(adap_names) == len(adap_types))
        conv_name_base = 'res' + str(stage) + block + '_branch'
        adap_name_base = 'ada' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.conv2a = layers.Conv2D(filters1,
                                    (3, 3),
                                    padding='same',
                                    name=conv_name_base + '2a' + '/backbone',
                                    kernel_regularizer=l2_regularizers(1e-5),
                                    kernel_initializer="he_normal",
                                    data_format=data_format)
        self.bn2a = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a' + '/backbone')

        self.conv2b = layers.Conv2D(filters2,
                                    kernel_size,
                                    padding='same',
                                    data_format=data_format,
                                    name=conv_name_base + '2b',
                                    kernel_regularizer=l2_regularizers(1e-5),
                                    kernel_initializer="he_normal",
                                    )

        self.bn2b = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b' + '/backbone')

        self.ada_convops = {}
        for adap_name, adap_type in zip(adap_names, adap_types):
            convops_var_name = adap_name_base
            convops_key_name = adap_name + '_' + adap_type
            self.ada_convops[convops_key_name] = adapter_structure(adap_type, filters2, convops_var_name, data_format)

    def call(self, input_tensor, ada_name, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


class ResConvBlockAda(tf.keras.Model):
    def __init__(self,
                 kernel_size,
                 filters,
                 stage,
                 block,
                 adap_names,
                 adap_types,
                 data_format,
                 strides=(2, 2)):
        super(ResConvBlockAda, self).__init__(name='')
        filters1, filters2 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        adap_name_base = 'ada' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.conv2a = layers.Conv2D(filters1, (3, 3),
                                    strides=strides,
                                    padding='same',
                                    name=conv_name_base + '2a' + '/backbone',
                                    kernel_regularizer=l2_regularizers(1e-5),
                                    kernel_initializer="he_normal",
                                    data_format=data_format)
        self.bn2a = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a' + '/backbone')

        self.conv2b = layers.Conv2D(filters2,
                                    kernel_size,
                                    padding='same',
                                    name=conv_name_base + '2b' + '/backbone',
                                    kernel_regularizer=l2_regularizers(1e-5),
                                    kernel_initializer="he_normal",
                                    data_format=data_format)
        self.bn2b = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b' + '/backbone')

        self.conv_shortcut = layers.Conv2D(filters2,
                                           (1, 1),
                                           padding='same',
                                           strides=strides,
                                           name=conv_name_base + '1' + '/backbone',
                                           kernel_regularizer=l2_regularizers(1e-5),
                                           kernel_initializer="he_normal",
                                           data_format=data_format)
        self.bn_shortcut = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1' + '/backbone')

        self.ada_convops = {}
        for adap_name, adap_type in zip(adap_names, adap_types):
            convops_var_name = adap_name_base
            convops_key_name = adap_name + '_' + adap_type
            self.ada_convops[convops_key_name] = adapter_structure(adap_type, filters2, convops_var_name, data_format)

    def call(self, input_tensor, ada_name, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)

        shortcut = self.conv_shortcut(input_tensor)
        shortcut = self.bn_shortcut(shortcut, training=training)

        x += shortcut
        return tf.nn.relu(x)


class AdapterResBlockV0(tf.keras.Model):
    def __init__(self,
                 filters,
                 stage,
                 block,
                 adap_names,
                 adap_types,
                 data_format,
                 strides=(1, 1)):
        super(ResConvBlockAda, self).__init__(name='')
        filters1, filters2, filter3 = filters
        assert (len(adap_names) == len(adap_types))

        self.names_types_dict = {}
        for i in range(len(adap_names)):
            self.names_types_dict[adap_names[i]] = adap_types[i]

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        adap_cnn_name_base = 'ada' + str(stage) + block + '_branch'
        adap_bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3
        self.conv_name_base = conv_name_base
        self.bn_name_base = bn_name_base
        self.adap_cnn_name_base = adap_cnn_name_base
        self.adap_bn_name_base = adap_bn_name_base

        self.conv2a = layers.Conv2D(filters1, (3, 3),
                                    strides=strides,
                                    padding='same',
                                    name=conv_name_base + '2a' + '/backbone',
                                    kernel_regularizer=l2_regularizers(1e-5),
                                    kernel_initializer="he_normal",
                                    data_format=data_format)
        self.bn2a = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a' + '/backbone')

        self.conv2b = layers.Conv2D(filters2,
                                    (3, 3),
                                    padding='same',
                                    name=conv_name_base + '2b' + '/backbone',
                                    kernel_regularizer=l2_regularizers(1e-5),
                                    kernel_initializer="he_normal",
                                    data_format=data_format)
        self.bn2b = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b' + '/backbone')

        self.conv2c = layers.Conv2D(filters2,
                                    (3, 3),
                                    padding='same',
                                    name=conv_name_base + '2c' + '/backbone',
                                    kernel_regularizer=l2_regularizers(1e-5),
                                    kernel_initializer="he_normal",
                                    data_format=data_format)
        self.bn2c = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c' + '/backbone')

        self.bn2a_adapter = {}
        for adap_name, adap_type in zip(adap_names, adap_types):
            convops_var_name = self.adap_bn_name_base + '_' + adap_name + '_' + adap_type + '_2a'
            self.bn2a_adapter[convops_var_name] = layers.BatchNormalization(axis=bn_axis, name=convops_var_name)

        self.bn2b_adapter = {}
        for adap_name, adap_type in zip(adap_names, adap_types):
            convops_var_name = self.adap_bn_name_base + '_' + adap_name + '_' + adap_type + '_2b'
            self.bn2b_adapter[convops_var_name] = layers.BatchNormalization(axis=bn_axis, name=convops_var_name)

        self.bn2c_adapter = {}
        for adap_name, adap_type in zip(adap_names, adap_types):
            convops_var_name = self.adap_bn_name_base + '_' + adap_name + '_' + adap_type + '_2c'
            self.bn2b_adapter[convops_var_name] = layers.BatchNormalization(axis=bn_axis, name=convops_var_name)

        self.cnn2b_adapter = {}
        for adap_name, adap_type in zip(adap_names, adap_types):
            convops_var_name = self.adap_cnn_name_base + '_' + adap_name + '_' + adap_type + '_2b'
            self.cnn2b_adapter[convops_var_name] = adapter_structure(adap_type, filters2, convops_var_name, data_format)

        self.cnn2c_adapter = {}
        for adap_name, adap_type in zip(adap_names, adap_types):
            convops_var_name = self.adap_cnn_name_base + '_' + adap_name + '_' + adap_type + '_2c'
            self.cnn2c_adapter[convops_var_name] = adapter_structure(adap_type, filters2, convops_var_name, data_format)

    def call(self, input_tensor, adap_name=None, training=False):
        if adap_name is not None and adap_name not in self.names_types_dict:
            logger.error("adap_name:{} not in names_types_dict:{}".format(adap_name, self.names_types_dict))

        adap_type = ''
        if adap_name is not None:
            adap_type = self.names_types_dict[adap_name]

        x0 = self.conv2a(input_tensor)
        if adap_name is not None:
            convops_var_name = self.adap_bn_name_base + '_' + adap_name + '_' + adap_type + '_2a'
            x0 = self.bn2a_adapter[convops_var_name](x0, training=training)
        else:
            x0 = self.bn2a(x0, training=training)

        x = self.conv2b(x0)
        if adap_name is not None:
            convops_var_name = self.adap_cnn_name_base + '_' + adap_name + '_' + adap_type + '_2b'
            x = x + self.cnn2b_adapter[convops_var_name](x)
            convops_var_name = self.adap_bn_name_base + '_' + adap_name + '_' + adap_type + '_2b'
            x = self.bn2b_adapter[convops_var_name](x, training=training)
        else:
            x = self.bn2b(x, training=training)

        x = tf.nn.relu(x)

        x = self.conv2c(x)
        if adap_name is not None:
            convops_var_name = self.adap_cnn_name_base + '_' + adap_name + '_' + adap_type + '_2b'
            x = x + self.cnn2c_adapter[convops_var_name](x)
            convops_var_name = self.adap_bn_name_base + '_' + adap_name + '_' + adap_type + '_2b'
            x = self.bn2c_adapter[convops_var_name](x, training=training)
        else:
            x = self.bn2c(x, training=training)

        return tf.nn.relu(x+x0)


class AdapterResBlockV1(tf.keras.Model):
    def __init__(self,
                 filters,
                 stage,
                 block,
                 adap_names,
                 adap_types,
                 data_format,
                 strides=(1, 1)):
        super(ResConvBlockAda, self).__init__(name='')
        filters1, filters2, filter3 = filters
        assert (len(adap_names) == len(adap_types))

        self.names_types_dict = {}
        for i in range(len(adap_names)):
            self.names_types_dict[adap_names[i]] = adap_types[i]

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        adap_cnn_name_base = 'ada' + str(stage) + block + '_branch'
        adap_bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3
        self.conv_name_base = conv_name_base
        self.bn_name_base = bn_name_base
        self.adap_cnn_name_base = adap_cnn_name_base
        self.adap_bn_name_base = adap_bn_name_base

        self.conv2a = layers.Conv2D(filters1, (3, 3),
                                    strides=strides,
                                    padding='same',
                                    name=conv_name_base + '2a' + '/backbone',
                                    kernel_regularizer=l2_regularizers(1e-5),
                                    kernel_initializer="he_normal",
                                    data_format=data_format)
        self.bn2a = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a' + '/backbone')

        self.conv2b = layers.Conv2D(filters2,
                                    (3, 3),
                                    padding='same',
                                    name=conv_name_base + '2b' + '/backbone',
                                    kernel_regularizer=l2_regularizers(1e-5),
                                    kernel_initializer="he_normal",
                                    data_format=data_format)
        self.bn2b = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b' + '/backbone')

        self.conv2c = layers.Conv2D(filters2,
                                    (3, 3),
                                    padding='same',
                                    name=conv_name_base + '2c' + '/backbone',
                                    kernel_regularizer=l2_regularizers(1e-5),
                                    kernel_initializer="he_normal",
                                    data_format=data_format)
        self.bn2c = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c' + '/backbone')

        self.bn2a_adapter = {}
        for adap_name, adap_type in zip(adap_names, adap_types):
            convops_var_name = self.adap_bn_name_base + '_' + adap_name + '_' + adap_type + '_2a'
            self.bn2a_adapter[convops_var_name] = layers.BatchNormalization(axis=bn_axis, name=convops_var_name)

        self.bn2b_adapter = {}
        for adap_name, adap_type in zip(adap_names, adap_types):
            convops_var_name = self.adap_bn_name_base + '_' + adap_name + '_' + adap_type + '_2b'
            self.bn2b_adapter[convops_var_name] = layers.BatchNormalization(axis=bn_axis, name=convops_var_name)

        self.bn2c_adapter = {}
        for adap_name, adap_type in zip(adap_names, adap_types):
            convops_var_name = self.adap_bn_name_base + '_' + adap_name + '_' + adap_type + '_2c'
            self.bn2b_adapter[convops_var_name] = layers.BatchNormalization(axis=bn_axis, name=convops_var_name)

        self.cnn2a_adapter = {}
        for adap_name, adap_type in zip(adap_names, adap_types):
            convops_var_name = self.adap_cnn_name_base + '_' + adap_name + '_' + adap_type + '_2a'
            self.cnn2b_adapter[convops_var_name] = adapter_structure(adap_type, filters2, convops_var_name, data_format)

        self.cnn2b_adapter = {}
        for adap_name, adap_type in zip(adap_names, adap_types):
            convops_var_name = self.adap_cnn_name_base + '_' + adap_name + '_' + adap_type + '_2b'
            self.cnn2b_adapter[convops_var_name] = adapter_structure(adap_type, filters2, convops_var_name, data_format)

        self.cnn2c_adapter = {}
        for adap_name, adap_type in zip(adap_names, adap_types):
            convops_var_name = self.adap_cnn_name_base + '_' + adap_name + '_' + adap_type + '_2c'
            self.cnn2c_adapter[convops_var_name] = adapter_structure(adap_type, filters2, convops_var_name, data_format)

    def call(self, input_tensor, adap_name=None, training=False):
        if adap_name is not None and adap_name not in self.names_types_dict:
            logger.error("adap_name:{} not in names_types_dict:{}".format(adap_name, self.names_types_dict))

        adap_type = ''
        if adap_name is not None:
            adap_type = self.names_types_dict[adap_name]

        x0 = self.conv2a(input_tensor)
        if adap_name is not None:
            convops_var_name = self.adap_bn_name_base + '_' + adap_name + '_' + adap_type + '_2a'
            x0 = self.bn2a_adapter[convops_var_name](x0, training=training)
        else:
            x0 = self.bn2a(x0, training=training)

        x = self.conv2b(x0)
        if adap_name is not None:
            convops_var_name = self.adap_cnn_name_base + '_' + adap_name + '_' + adap_type + '2b'
            x = x + self.cnn2b_adapter[convops_var_name](x)
            convops_var_name = self.adap_bn_name_base + '_' + adap_name + '_' + adap_type + '_2b'
            x = self.bn2b_adapter[convops_var_name](x, training=training)
        else:
            x = self.bn2b(x, training=training)

        x = tf.nn.relu(x)

        x = self.conv2c(x)
        if adap_name is not None:
            convops_var_name = self.adap_cnn_name_base + '_' + adap_name + '_' + adap_type + '_2b'
            x = x + self.cnn2c_adapter[convops_var_name](x)
            convops_var_name = self.adap_bn_name_base + '_' + adap_name + '_' + adap_type + '_2b'
            x = self.bn2c_adapter[convops_var_name](x, training=training)
        else:
            x = self.bn2c(x, training=training)

        if adap_name is not None:
            convops_var_name = self.adap_cnn_name_base + '_' + adap_name + '_' + adap_type + '_2a'
            x = x + self.cnn2b_adapter[convops_var_name](x)

        return tf.nn.relu(x + x0)





