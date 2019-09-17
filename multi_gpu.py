# _*_ coding:utf-8 _*_


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import sys
import string
import numpy as np
import multiprocessing
import tensorflow as tf


layers = tf.keras.layers
l2_regularizers = tf.keras.regularizers.l2
l1_regularizers = tf.keras.regularizers.l1
l1_l2_regularizers = tf.keras.regularizers.l1_l2


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads




class Encoder(tf.keras.Model):
    def __init__(self, enc_units, name='ttttt'):
        super(Encoder, self).__init__(name=name)
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
        self.bncv6 = layers.BatchNormalization(axis=-1, name='bnconv5')
        self.pddd6 = layers.ZeroPadding2D(padding=(0, 1))
        self.pool6 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool4')
        self.conv7 = layers.Conv2D(512, kernel_size=(2, 2), activation='relu', padding='valid', name='conv7')
        self.flatn8 = layers.Flatten(name='flatten8')
        self.dense9 = layers.Dense(units=4, name='dense9')

        units = enc_units
        rnn_type = 'lstm'
        assert (rnn_type in ('rnn', 'gru', 'lstm'))

        if rnn_type == 'lstm':
            self.bilstm0 = tf.keras.layers.Bidirectional(
                layers.LSTM(units=units, return_sequences=True, name='encode_lstm0'),
                name='bilstm0')
            self.bilstm1 = tf.keras.layers.Bidirectional(
                layers.LSTM(units=units, return_sequences=True, name='encode_lstm1'),
                name='bilstm1')
        elif rnn_type == 'gru':
            self.bilstm0 = tf.keras.layers.Bidirectional(
                layers.GRU(units=units, return_sequences=True, name='encode_gru0'),
                name='bilstm0')
            self.bilstm1 = tf.keras.layers.Bidirectional(
                layers.GRU(units=units, return_sequences=True, name='encode_gru1'),
                name='bilstm1')

        # self.bilstm1 = tf.keras.layers.Bidirectional(layers.GRU(units=units, name='encode_rnn1'), name='bilist1')
        # self.bilstm1 = tf.keras.layers.Bidirectional(layers.GRU(units=units, name='encode_rnn1'), name='bilist1')

    def get_sequence_lengths(self, widths):
        return tf.cast(tf.floor_div(widths, 4) + 1, dtype=tf.int32)

    tf.math.floordiv

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

        # BHWC-->BWHC
        features = tf.transpose(features, perm=(0, 2, 1, 3))
        features_s = tf.shape(features)
        features_b = features_s[0]
        features_w = features_s[1]
        features_h = features_s[2]
        features_c = features_s[3]
        features = tf.reshape(features, [features_b, features_w, features_h * features_c])
        widths = self.get_sequence_lengths(widths)
        widths = tf.reshape(widths, [-1], name='seq_len')
        #features = self.bilstm0(features)
        #features = self.bilstm1(features)

        #features = tf.reshape(features, [features_b, features_w, features_h, features_c])
        features_f = self.flatn8(features)

        print("features shape:{}, features_f shape:{}".format(features.shape, features_f.shape))
        logits = self.dense9(features_f)

        return features, widths, logits


def test_sigcpu():
    import numpy as np
    learning_rate = 0.001
    tf.config.experimental_list_devices()
    tf.distribute.MirroredStrategy
    tf.config.experimental_list_devices

    optimizer = tf.keras.optimizers.SGD(learning_rate)

    tower_grads = []
    data = np.random.random((6, 32, 128, 3))
    label = [0, 1, 2, 3, 3, 1]
    widths = [16, 32, 48, 64, 96, 128]
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    label = tf.convert_to_tensor(label, dtype=tf.int64)
    ecnoder = Encoder(76)

    with tf.GradientTape() as tape:
        features, widths1, logits = ecnoder(data, widths, True)
        print('data shape:', data.shape)
        print('logits shape:', logits.shape)
        print('features shape:', features.shape)

        total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
        total_loss = tf.reduce_mean(total_loss)
        gradients = tape.gradient(total_loss, ecnoder.variables)
        tower_grads.append(gradients)
        print("gradients:", gradients)
        for var in ecnoder.variables:
            if 'ttttt/conv1/kernel' in var.name:
                print(var.name)
                #print(var.name, max(var.numpy()), min(var.numpy()))

    #grads = average_gradients(tower_grads)
    #apply_gradient_op = optimizer.apply_gradients(grads, global_step=11)


def test_multigpu():
    #tf.enable_eager_execution()
    import numpy as np
    learning_rate = 0.001
    cpu_cnt = multiprocessing.cpu_count()
    optimizer = tf.train.AdamOptimizer(learning_rate)

    ecnoder = Encoder(76)
    tower_grads = []

    for i, cpu_id in enumerate(range(cpu_cnt)):
        with tf.device('/cpu:%d' % cpu_id):
            data = np.random.random((6, 32, 128, 3))
            label = [0, 1, 2, 3, 3, 1]
            widths = [16, 32, 48, 64, 96, 128]
            data = tf.convert_to_tensor(data, dtype=tf.float32)
            label = tf.convert_to_tensor(label, dtype=tf.int64)

            with tf.GradientTape() as tape:
                features, widths1, logits = ecnoder(data, widths, True)
                print('data shape:', data.shape)
                print('logits shape:', logits.shape)
                print('features shape:', features.shape)
                total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)

            variables = ecnoder.variables
            gradients = tape.gradient(total_loss, variables)
            tower_grads.append(gradients)
            for var in ecnoder.variables:
                if 'ttttt/conv1/kernel' in var.name:
                    print(var.name)
                    #print(var.name, max(var.numpy()), min(var.numpy()))

    grads = average_gradients(tower_grads)
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=11)


if __name__ == '__main__':
    #cpus = tf.config.experimental.get_device_policy()
    #gpus = tf.config.experimental.get_synchronous_execution()

    #print(cpus)
    #print(gpus)
    test_sigcpu()
