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


class CRNNEncoder(tf.keras.Model):
    def __init__(self, lstm_unit=128, rnn_type='lstm'):
        super(CRNNEncoder, self).__init__()

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

        self.rnn_type = rnn_type
        assert (rnn_type in ('rnn', 'lstm', 'gru'))
        if rnn_type == 'rnn':
            backward0 = layers.SimpleRNN(units=lstm_unit, return_sequences=True, go_backwards=True, name='encode_fore0')
            foreward0 = layers.SimpleRNN(units=lstm_unit, return_sequences=True, go_backwards=False,
                                         name='encode_back0')
            backward1 = layers.SimpleRNN(units=lstm_unit, return_sequences=True, go_backwards=True, name='encode_fore1')
            foreward1 = layers.SimpleRNN(units=lstm_unit, return_sequences=True, go_backwards=False,
                                         name='encode_back1')
            self.bilstm0 = tf.keras.layers.Bidirectional(layer=foreward0, backward_layer=backward0, name='bilstm0')
            self.bilstm1 = tf.keras.layers.Bidirectional(layer=foreward1, backward_layer=backward1, name='bilstm1')

        elif rnn_type == 'lstm':
            backward0 = layers.LSTM(units=lstm_unit, return_sequences=True, go_backwards=True, name='encode_fore0')
            foreward0 = layers.LSTM(units=lstm_unit, return_sequences=True, go_backwards=False, name='encode_back0')
            backward1 = layers.LSTM(units=lstm_unit, return_sequences=True, go_backwards=True, name='encode_fore1')
            foreward1 = layers.LSTM(units=lstm_unit, return_sequences=True, go_backwards=False, name='encode_back1')
            self.bilstm0 = tf.keras.layers.Bidirectional(layer=foreward0, backward_layer=backward0, name='bilstm0')
            self.bilstm1 = tf.keras.layers.Bidirectional(layer=foreward1, backward_layer=backward1, name='bilstm1')
        elif rnn_type == 'gru':
            backward0 = layers.GRU(units=lstm_unit, return_sequences=True, go_backwards=True, name='encode_fore0')
            foreward0 = layers.GRU(units=lstm_unit, return_sequences=True, go_backwards=False, name='encode_back0')
            backward1 = layers.GRU(units=lstm_unit, return_sequences=True, go_backwards=True, name='encode_fore1')
            foreward1 = layers.GRU(units=lstm_unit, return_sequences=True, go_backwards=False, name='encode_back1')
            self.bilstm0 = tf.keras.layers.Bidirectional(layer=foreward0, backward_layer=backward0, name='bilstm0')
            self.bilstm1 = tf.keras.layers.Bidirectional(layer=foreward1, backward_layer=backward1, name='bilstm1')

    def get_feature_step(self, widths):
        return tf.cast((tf.cast(widths, tf.float32)/4.0) + 1, dtype=tf.int32)

    tf.math.ceil

    def call(self, inputs, widths, training=True):

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
        cnn_features = features
        features = tf.transpose(features, perm=(0, 2, 1, 3))
        features_s = tf.shape(features)
        features_b = features_s[0]
        features_w = features_s[1]
        features_h = features_s[2]
        features_c = features_s[3]
        features = tf.reshape(features, [features_b, features_w, features_h * features_c])
        features = self.bilstm0(features)
        rnn_features = self.bilstm1(features)
        widths = self.get_feature_step(widths)
        return cnn_features, rnn_features, widths


def check_seqlen():
  import numpy as np
  imgh = 48
  imgc = 3
  lstm_units=12
  model = CRNNEncoder(lstm_units)
  for imgw in range(imgh, 3000):
    image = np.random.random((1, imgh, imgw, imgc))
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    print('img shape:', image.shape)
    cnn_features, rnn_features, widths = model(image, tf.convert_to_tensor([imgw]))
    b, w, c = rnn_features.numpy().shape
    assert(w == widths.numpy()[0])
    #print(model.variables)


if __name__ == '__main__':
    check_seqlen()
