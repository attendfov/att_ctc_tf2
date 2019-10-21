# _*_ coding:utf-8 _*_
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import TensorFlow >= 1.10 and enable eager execution
import os
import sys
import numpy as np
import tensorflow as tf
layers = tf.keras.layers

abspath = os.path.dirname(os.path.realpath(__file__))
abspath = os.path.abspath(abspath)
sys.path.append(abspath)
sys.path.append(os.path.join(abspath, '../utils'))

import random
from Encoder import ResNet18
from Decoder import Decoder
from Logger import logger


class Seq2Seq(tf.keras.Model):
    def __init__(self, vocab_size, eos_id=1):
        super(Seq2Seq, self).__init__()
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self.encoder = ResNet18(128)
        self.decoder = Decoder(vocab_size=self.vocab_size, num_layers=0, name='ctc_layer2')

    def call(self, input_tensor, input_widths, training):
        cnn_features, rnn_features, step_widths = self.encoder(input_tensor, input_widths, training)
        ctc_features, ctc_output = self.decoder(rnn_features, training)
        return ctc_features, ctc_output, step_widths


if __name__ == '__main__':

    eos_id = 1
    vocab_size = 8
    seq2seq = Seq2Seq(vocab_size=vocab_size, eos_id=eos_id)
    training = True
    batch_size = 4
    image_width = 256
    image_height = 32
    image_channel = 3
    data = np.random.random((batch_size, image_height, image_width, image_channel))
    widths = [128, 144, 160, 252]
    data = tf.convert_to_tensor(data, dtype=tf.float32)

    padding_len = 5
    ctc_indices = []
    ctc_values = []
    ctc_lengs = []

    for b in range(batch_size):
        ctc_target = []
        valid_len = random.choice(range(2, padding_len - 2))
        ctc_lengs.append(valid_len)
        for i in range(valid_len):
            index_id = random.choice(range(2, vocab_size-2))
            ctc_values.append(index_id)
            ctc_indices.append([b, i])

    traget_sparse = tf.SparseTensor(indices=ctc_indices, values=ctc_values, dense_shape=[batch_size, min(ctc_lengs)])

    ctc_features, ctc_output, step_widths = seq2seq(data, widths, training)
    print("ctc_final_output shape {}, {}, {}".format(ctc_output.shape, ctc_features.shape, step_widths))


