# _*_ coding:utf-8 _*_
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import TensorFlow >= 1.10 and enable eager execution
import re
import os
import sys
import time
import logging
import numpy as np
import tensorflow as tf
layers = tf.keras.layers
abspath = os.path.dirname(os.path.realpath(__file__))
abspath = os.path.abspath(abspath)
sys.path.append(abspath)
sys.path.append(os.path.join(abspath, '../utils'))
sys.path.append(os.path.join(abspath, '../networks'))

from Logger import logger
from backbone_aste import BackboneAste


class EncoderAste(tf.keras.Model):
    def __init__(self,
                 rnn_type='gru',
                 rnn_unit=256,
                 name_prefix='EncoderAste'):

        super(EncoderAste, self).__init__(name=name_prefix)
        self.rnn_type = rnn_type
        self.rnn_unit = rnn_unit
        self.backbone = BackboneAste(name_prefix)
        rnn_class = single_cell_class(rnn_type)

        rnn_fw_name = 'encode_rnn_fw0/backbone'
        rnn_bw_name = 'encode_rnn_bw0/backbone'
        self.rnn_fw0 = rnn_class(num_units=rnn_unit, dtype=tf.float32, name=rnn_fw_name)
        self.rnn_bw0 = rnn_class(num_units=rnn_unit, dtype=tf.float32, name=rnn_bw_name)

        rnn_fw_name = 'encode_rnn_fw1/backbone'
        rnn_bw_name = 'encode_rnn_bw1/backbone'
        self.rnn_fw1 = rnn_class(num_units=rnn_unit, dtype=tf.float32, name=rnn_fw_name)
        self.rnn_bw1 = rnn_class(num_units=rnn_unit, dtype=tf.float32, name=rnn_bw_name)

    def call(self, inputs, widths, training):
        bone_features, valid_widths = self.backbone(inputs, widths, training)
        lstm_features = tf.transpose(bone_features, perm=(0, 2, 1, 3))
        logger.info("bone feature output B*H*W*C shape: {}".format(bone_features.shape))
        logger.info("bone feature output B*W*H*C shape: {}".format(lstm_features.shape))
        features_s = tf.shape(lstm_features)
        features_b = features_s[0]
        features_w = features_s[1]
        features_h = features_s[2]
        features_c = features_s[3]
        lstm_features = tf.reshape(lstm_features, [features_b, features_w, features_h*features_c])
        logger.info("lstm feature input B*W*C shape: {}".format(lstm_features.shape))
        lstm_features, status = bidirectional_rnn_foreward(lstm_features, self.rnn_fw0, self.rnn_bw0)
        lstm_features, status = bidirectional_rnn_foreward(lstm_features, self.rnn_fw1, self.rnn_bw1)
        logger.info("lstm output B*W*C shape: {}".format(lstm_features.shape))

        '''
        weight_mask = None
        if widths is not None:
            widths = tf.reshape(widths, [-1, 1], name='seq_len')
            widths = tf.concat([widths for h in range(features_h)], axis=-1)
            widths = tf.reshape(widths, [-1])
            weight_mask = tf.sequence_mask(widths, features_w, dtype=tf.float32)
            weight_mask = tf.reshape(weight_mask, [features_b, features_h * features_w])
            logger.debug("weight mask shape: {}".format(weight_mask.shape))
        '''

        return bone_features, lstm_features, valid_widths


if __name__ == '__main__':
    tf.enable_eager_execution()
    import numpy as np
    data = np.random.random((4, 48, 64, 3))
    widths = [16, 32, 48, 64]
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    ecnoder = EncoderAste()
    bone_features, lstm_features, valid_widths = ecnoder(data, widths, True)
    variables = ecnoder.variables
    print("bone_features shape: {}".format(bone_features.shape))
    print("lstm_features shape: {}".format(lstm_features.shape))
    print("valid_widths  shape: {}".format(valid_widths.shape))
    print("orig_width:", widths)
    print("vald_width:", valid_widths.numpy())

    for var in variables:
        print(var.name, type(var), var.shape)

