# _*_ coding:utf-8 _*_
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
layers = tf.keras.layers

abspath = os.path.dirname(os.path.realpath(__file__))
abspath = os.path.abspath(abspath)

sys.path.append(abspath)
sys.path.append(os.path.join(abspath, '../utils'))


class Decoder(tf.keras.Model):
    def __init__(self,
                 vocab_size,
                 num_layers=0,
                 units=256,
                 rnn_type='gru',
                 rate=0.1,
                 name='Decoder'):
        super(Decoder, self).__init__(name=name)
        self.rate = rate
        self.units = units
        self.rnn_type = rnn_type
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout = tf.keras.layers.Dropout(self.rate)
        self.final_layer = tf.keras.layers.Dense(self.vocab_size)

        self.rnn_layer_map = {}
        assert (self.rnn_type in ('rnn', 'lstm', 'gru'))
        if rnn_type == 'rnn':
            rnn_class = layers.SimpleRNN
        elif rnn_type == 'gru':
            rnn_class = layers.GRU
        elif rnn_type == 'lstm':
            rnn_class = layers.LSTM

        for layerid in range(self.num_layers):
            backward = rnn_class(units=units, return_sequences=True, go_backwards=True,
                                 name='encode_fore{}'.format(layerid))
            foreward = rnn_class(units=units, return_sequences=True, go_backwards=False,
                                 name='encode_back{}'.format(layerid))

            layer_name = "rnn_{}".format(layerid)
            self.rnn_layer_map[layer_name] = tf.keras.layers.Bidirectional(layer=foreward,
                                                                           backward_layer=backward,
                                                                           name='bilstm{}'.format(layerid))

    def call(self, x, training):
        features = self.dropout(x, training=training)
        x_shape = list(tf.shape(x))
        assert (len(x_shape) >= 3)
        if len(x_shape) == 3:
            b, w, c = x_shape[:3]
        if len(x_shape) == 4:
            # BHWC-->BWHC
            b, h, w, c = x_shape[:4]
            features = tf.transpose(features, perm=(0, 2, 1, 3))
            features = tf.reshape(features, [b, w, h*c])

        for layerid in range(self.num_layers):
            layer_name = "rnn_{}".format(layerid)
            features = self.rnn_layer_map[layer_name](features)

        final_output = self.final_layer(features)
        return features, final_output


def decoder_test():
    from Encoder import ResNet18
    batch_size = 6
    image_width = 128
    image_height = 32
    image_channel = 3
    data = np.random.random((batch_size, image_height, image_width, image_channel))
    widths = [16, 32, 64, 96, 128]
    data = tf.convert_to_tensor(data, dtype=tf.float32)

    encoder = ResNet18(76)
    cnn_features, rnn_features, step_widths = encoder(data, widths, True)

    print("cnn_features shape: {}".format(cnn_features.shape))
    print("rnn_features shape: {}".format(rnn_features.shape))

    decoder_cnn = Decoder(vocab_size=12)
    features, final_output = decoder_cnn(cnn_features, True)
    print("ctc bone_features:", features.shape, final_output.shape)
    decoder_rnn = Decoder(vocab_size=12)
    features, final_output = decoder_rnn(rnn_features, True)
    print("ctc lstm_features:", features.shape, final_output.shape)

    cnn_map = {}
    variables = decoder_cnn.trainable_variables
    for var in variables:
        print("decoder cnn:", var.name, type(var), var.shape)
        cnn_map[var.name] = var

    rnn_map = {}
    variables = decoder_rnn.trainable_variables
    for var in variables:
        print("decoder rnn:", var.name, type(var), var.shape)
        rnn_map[var.name] = var

    key_sets = cnn_map.keys() & rnn_map.keys()
    for key in key_sets:
        var1 = cnn_map[key].numpy()
        var2 = rnn_map[key].numpy()
        if var1.shape == var2.shape:
            print(key, (var1 == var2).all())


if __name__ == '__main__':
    decoder_test()










