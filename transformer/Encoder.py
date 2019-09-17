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
from TransferUtils import *


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


class Encoder(tf.keras.Model):
    def __init__(self,
                 num_layers=2,
                 d_model=512,
                 num_heads=4,
                 dff=1024,
                 rate=0.1,
                 used_rnn=False,
                 max_width=1600):
        super(Encoder, self).__init__()

        self.used_rnn = used_rnn
        self.conv1 = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same', name='conv1')
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')
        self.conv2 = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same', name='conv2')
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')
        self.conv3 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv3')
        self.conv4 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv4')

        self.padd4 = layers.ZeroPadding2D(padding=(0, 1))
        self.pool4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool3')
        self.conv5 = layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv5')
        self.bncv5 = layers.BatchNormalization(axis=-1, name='bnconv5')
        self.conv6 = layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv6')
        self.bncv6 = layers.BatchNormalization(axis=-1, name='bnconv6')
        self.pddd6 = layers.ZeroPadding2D(padding=(0, 1))
        self.pool6 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool4')
        self.conv7 = layers.Conv2D(512, kernel_size=(2, 2), activation='relu', padding='valid', name='conv7')

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
            assert(self.d_model % self.num_heads == 0)

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
            #self.rnn_multi_fw = tf.nn.rnn_cell.DropoutWrapper(self.rnn_multi_fw, 0.9, 0.95, 0.95)
            #self.rnn_multi_bw = tf.nn.rnn_cell.DropoutWrapper(self.rnn_multi_bw, 0.9, 0.95, 0.95)

        self.dense_layer = tf.keras.layers.Dense(self.d_model)
        for layer_id in range(num_layers):
            self.transfer_enc_layers.append(EncoderLayer(self.d_model, self.num_heads, self.dff, self.rate))

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

    def get_reverse_points(self, points):
        ret_points = []
        for point in points:
            pointx, pointy = point
            anchor_xcent = 4 * pointx + 2
            anchor_xsoff = max(anchor_xcent - 4, 0)
            anchor_xeoff = max(anchor_xcent + 4, 0)

            anchor_ycent = 16 * pointy + 8
            anchor_ysoff = max(anchor_ycent - 16, 0)
            anchor_yeoff = max(anchor_ycent + 16, 0)

            positions = []
            for anchory_off in range(anchor_ysoff, anchor_yeoff):
                if int(anchory_off / 16.0) != pointy:
                    continue
                for anchorx_off in range(anchor_xsoff, anchor_xeoff):
                    if int(anchorx_off / 4.0) == pointx:
                        positions.append([int(anchorx_off), int(anchory_off)])
            ret_points.append(positions)
        return ret_points


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

        logger.debug("B*H*W*C features shape: {}".format(features.shape))

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

        #this code cause loss degrade failed
        #if self.num_layers > 0:
        #    features *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        #    features += self.pos_encoding[:, :features_w, :]

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
    data = np.random.random((4, 48, 64, 3))
    widths = [2, 4, 6, 8]
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    ecnoder = Encoder(1)
    features, weight_mask = ecnoder(data, widths, True)
    print("features shape: {}".format(features.shape))
    print("weight   shape: {}".format(weight_mask.shape))
    print(weight_mask)


