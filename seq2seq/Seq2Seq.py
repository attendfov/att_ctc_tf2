# _*_ coding:utf-8 _*_
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import TensorFlow >= 1.10 and enable eager execution
import re
import os
import sys
import time
import random
import numpy as np
import tensorflow as tf
layers = tf.keras.layers
tf.enable_eager_execution()

print(tf.__version__)

abspath = os.path.dirname(os.path.realpath(__file__))
abspath = os.path.abspath(abspath)
sys.path.append(abspath)

from Encoder import *
from Decoder import *
from EncoderResnet18 import *


class Seq2Seq(tf.keras.Model):
    def __init__(self,
                 vocab_size=1000,
                 embedding_dim=96,
                 SOS_ID = 0,
                 EOS_ID = 1,
                 dec_units=128,
                 enc_units=128,
                 attention_name='luong',
                 attention_type=0,
                 rnn_type='gru',
                 max_length=10,
                 teacher_forcing_ratio=0.5):
        super(Seq2Seq, self).__init__()

        self.encoder = ResNet18(enc_units)
        #self.encoder = Encoder(enc_units)
        self.decoder = Decoder(vocab_size,
                               embedding_dim,
                               SOS_ID,
                               EOS_ID,
                               dec_units,
                               enc_units,
                               attention_name,
                               attention_type,
                               rnn_type,
                               max_length
                               )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.SOS_ID = SOS_ID
        self.EOS_ID = EOS_ID
        self.dec_units = dec_units
        self.enc_units = enc_units
        self.attention_name = attention_name
        self.attention_type = attention_type
        self.rnn_type = rnn_type
        self.max_length = max_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.loss_value = 0.0

    def loss(self, input_label, decoder_outputs):
        batch = int(input_label.shape[0])
        width = int(input_label.shape[1])

        #slice to excluding the begining of SOE
        input_label = tf.slice(input_label, [0, 1], [batch, width-1])
        input_length = np.array([width-1]*batch)
        for step in range(width-1):
            step_target = input_label[:, step]
            step_eosidx = step_target.numpy() == self.EOS_ID
            update_index = (step_eosidx & (input_length > step + 1))
            input_length[update_index] = step+1

        input_mask = tf.cast(tf.sequence_mask(input_length, width-1), dtype=tf.float32)

        crop_outputs = []
        for index in range(min(width-1, len(decoder_outputs))):
            crop_outputs.append(tf.expand_dims(decoder_outputs[index], axis=1))
        crop_outputs = tf.concat(crop_outputs, axis=1)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_label, logits=crop_outputs) * input_mask
        loss = tf.reduce_sum(loss) / tf.reduce_sum(input_mask)

        return loss

    def calc_acc(self, input_data, widths, input_labe):
        pass

    def call(self, input_data, widths, input_label, training):
        enc_output, fw_status, widths = self.encoder(input_data, widths, training)
        weight_mask = tf.sequence_mask(lengths=widths, maxlen=enc_output.shape[1])
        logger.debug("weight_mask: shape:{}".format(weight_mask.shape))
        decoder_outputs, decoder_hidden, decoder_dict = self.decoder(input_label,
                                                                     fw_status,
                                                                     enc_output,
                                                                     self.teacher_forcing_ratio,
                                                                     weight_mask,
                                                                     training)
        self.decoder_dict = decoder_dict
        self.decoder_outputs = decoder_outputs
        if training:
            self.loss_value = self.loss(input_label, decoder_outputs)
            logger.info(self.loss_value)
        return self.decoder_dict, self.loss_value


if __name__=='__main__':
    data = np.random.random((6, 32, 128, 3))
    widths = [16, 32, 48, 64, 96, 128]
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    seq2seq_model = Seq2Seq()

    input_data = data
    input_label = np.array([[3,1,1,1,1,1],[6,1,1,1,1,1],[5,9,1,1,1,1],[7,6,1,1,1,1],[4,8,7,1,1,1],[5,4,7,8,1,1]])
    input_label = tf.convert_to_tensor(input_label, dtype=tf.int32)
    training = True

    seq2seq_model(input_data, widths, input_label, training)









