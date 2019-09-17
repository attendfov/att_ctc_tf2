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
tf.enable_eager_execution()
abspath = os.path.dirname(os.path.realpath(__file__))
abspath = os.path.abspath(abspath)
sys.path.append(abspath)
sys.path.append(os.path.join(abspath, '../utils'))
from Logger import logger

class Encoder(tf.keras.Model):
    def __init__(self, enc_units):
        super(Encoder, self).__init__()

        self.conv1 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv1')
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')
        self.conv2 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv2')
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')
        self.conv3 = layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv3')
        self.conv4 = layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv4')

        self.padd4 = layers.ZeroPadding2D(padding=(0, 1))
        self.pool4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool3')
        self.conv5 = layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv5')
        self.bncv5 = layers.BatchNormalization(axis=-1, name='bnconv5')
        self.conv6 = layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv6')
        self.bncv6 = layers.BatchNormalization(axis=-1, name='bnconv6')
        self.pddd6 = layers.ZeroPadding2D(padding=(0, 1))
        self.pool6 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool4')
        self.conv7 = layers.Conv2D(512, kernel_size=(2, 2), activation='relu', padding='valid', name='conv7')

        units = enc_units
        rnn_type = 'gru'
        assert (rnn_type in ('rnn', 'lstm', 'gru'))

        if rnn_type == 'rnn':
            rnn_class = tf.nn.rnn_cell.RNNCell
        elif rnn_type == 'lstm':
            rnn_class = tf.nn.rnn_cell.LSTMCell
        elif rnn_type == 'gru':
            rnn_class = tf.nn.rnn_cell.GRUCell

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

        #self.rnn_multi_fw = tf.nn.rnn_cell.DropoutWrapper(self.rnn_multi_fw, 0.8, 0.8, 0.8)
        #self.rnn_multi_bw = tf.nn.rnn_cell.DropoutWrapper(self.rnn_multi_bw, 0.8, 0.8, 0.8)

    def bidirectional_rnn_foreward(self, inputs, rnn_fw_inst, rnn_bw_inst):
        outputs_fb, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn_fw_inst,
                                                                    cell_bw=rnn_bw_inst,
                                                                    inputs=inputs,
                                                                    dtype=tf.float32,
                                                                    time_major=False)

        logger.debug("outputs_fb size: {}".format(len(outputs_fb)))
        logger.debug("outputs_fb[0] shape:{}".format(outputs_fb[0].shape))
        logger.debug("outputs_fb[1] shape:{}".format(outputs_fb[1].shape))
        logger.debug("output_states size: {}".format(len(output_states)))
        logger.debug("output_states 0 0 shape: {}".format(output_states[0][0].shape))
        logger.debug("output_states 0 1 shape: {}".format(output_states[0][1].shape))
        result = tf.concat(outputs_fb, axis=2)
        fw_status, bw_status = output_states
        return result, fw_status


    def dynamic_rnn_foreward(self, inputs, rnn_cell_inst):
        outputs, state = tf.nn.dynamic_rnn(cell=rnn_cell_inst,
                                           inputs=inputs,
                                           time_major=False)
        return outputs, state

    def get_sequence_lengths(self, widths):
        return tf.cast(tf.div(widths, 4) + 1, dtype=tf.int32)

    def get_reverse_lengths(self, widths):
        lengths = []
        for width in widths:
            anchor_cent = 4*width
            anchor_satrt = max(anchor_cent-10, 0)
            anchor_end = max(anchor_cent+10, 0)
            positions = []
            for anchor_off in range(anchor_satrt, anchor_end):
                if int(anchor_off/4.0+1) == width:
                    positions.append(anchor_off)
                elif int(anchor_off/4.0+1) > width:
                    break
            lengths.append(positions)
        return lengths

    def call(self, inputs, widths, training):

        features = self.conv1(inputs)
        features = self.pool1(features)
        features = self.conv2(features)
        features = self.pool2(features)
        features = self.conv3(features)
        features = self.conv4(features)
        features = self.padd4(features)
        features = self.pool4(features)
        features = self.conv5(features)
        features = self.bncv5(features, training=training)
        features = self.conv6(features)
        features = self.bncv6(features, training=training)
        features = self.pddd6(features)
        features = self.pool6(features)
        features = self.conv7(features)

        logger.debug("origin features shape: {}".format(features.shape))

        # BHWC-->BWHC
        features = tf.transpose(features, perm=(0, 2, 1, 3))
        features_s = tf.shape(features)
        features_b = features_s[0]
        features_w = features_s[1]
        features_h = features_s[2]
        features_c = features_s[3]
        features = tf.reshape(features, [features_b, features_w, features_h * features_c])
        logger.debug("wFirst features shape: {}".format(features.shape))
        features, fw_status = self.bidirectional_rnn_foreward(features, self.rnn_multi_fw, self.rnn_multi_bw)

        widths = self.get_sequence_lengths(widths)
        widths = tf.reshape(widths, [-1], name='seq_len')
        return features, fw_status, widths


if __name__=='__main__':
    import numpy as np

    data = np.random.random((6, 32, 128, 3))
    widths = [16, 32, 48, 64, 96, 128]
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    ecnoder = Encoder(76)
    ecnoder(data, widths, True)


