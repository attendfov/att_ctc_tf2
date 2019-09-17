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

import numpy as np
from Logger import logger
from TensorUtils import single_cell_class
from TensorUtils import bidirectional_rnn_foreward


class CTCDecoder(tf.keras.Model):
    def __init__(self,
                 vocab_size,
                 num_layers=2,
                 units=256,
                 rnn_type='gru',
                 rate=0.1,
                 name=''
                 ):
        super(CTCDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.prefix_name = name
        self.rate = rate
        self.units = units
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = tf.keras.layers.Dropout(self.rate)
        self.final_layer = tf.keras.layers.Dense(self.vocab_size)

        rnn_class = single_cell_class(rnn_type)

        self.rnn_fw_cells = []
        for i in range(self.num_layers):
            rnn_fw_name = 'CTCDecoder/rnn_fw{}'.format(i)
            if len(self.prefix_name) > 0:
                rnn_fw_name = self.prefix_name + '/' + 'CTCDecoder/rnn_fw{}'.format(i)
            self.rnn_fw_cells.append(rnn_class(num_units=units, dtype=tf.float32, name=rnn_fw_name))

        self.rnn_bw_cells = []
        for i in range(self.num_layers):
            rnn_bw_name = 'CTCDecoder/rnn_bw{}'.format(i)
            if len(self.prefix_name) > 0:
                rnn_bw_name = self.prefix_name + '/' + 'CTCDecoder/rnn_bw{}'.format(i)
            self.rnn_bw_cells.append(rnn_class(num_units=units, dtype=tf.float32, name=rnn_bw_name))

    def call(self, x, training):
        features = self.dropout(x, training=training)
        x_shape = list(tf.shape(x))
        assert(len(x_shape) >= 3)
        if len(x_shape) == 4:
            b, h, w, c = x_shape[:4]
            # BHWC-->BWHC
            features = tf.transpose(features, perm=(0, 2, 1, 3))
            features = tf.reshape(features, [b, w, h*c])

        for i in range(self.num_layers):
            features, fw_status = bidirectional_rnn_foreward(features, self.rnn_fw_cells[i], self.rnn_bw_cells[i])

        final_output = self.final_layer(features)
        return features, final_output


def ctc_decoder_test():
    from Encoder import EncoderAste
    batch_size = 6
    image_width = 128
    image_height = 32
    image_channel = 3
    data = np.random.random((batch_size, image_height, image_width, image_channel))
    widths = [16, 32, 64, 96, 128]
    data = tf.convert_to_tensor(data, dtype=tf.float32)

    encoder = EncoderAste()
    bone_features, lstm_features, valid_widths = encoder(data, widths, True)

    print("bone_features shape: {}".format(bone_features.shape))
    print("lstm_featmask shape: {}".format(lstm_features.shape))

    decoder = CTCDecoder(vocab_size=12)
    features, final_output = decoder(bone_features, True)
    print("ctc bone_features:", features.shape, final_output.shape)
    features, final_output = decoder(lstm_features, True)
    print("ctc lstm_features:", features.shape, final_output.shape)

    variables = decoder.variables
    for var in variables:
        print("decoder:", var.name, type(var), var.shape)


if __name__ == '__main__':
    tf.enable_eager_execution()
    ctc_decoder_test()










