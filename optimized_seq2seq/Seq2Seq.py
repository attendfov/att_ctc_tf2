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

        self.encoder = Encoder(enc_units)
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
        self.loss_value = 0.0
        self.dec_units = dec_units
        self.enc_units = enc_units
        self.attention_name = attention_name
        self.attention_type = attention_type
        self.rnn_type = rnn_type
        self.max_length = max_length
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def get_target_length(self, target_input, eos_id):
        # target_input:batch_size*steps
        batch = int(target_input.shape[0])
        steps = int(target_input.shape[1])

        target_len = tf.convert_to_tensor([0] * batch, dtype=tf.int32)
        target_input = tf.cast(target_input, dtype=tf.int32)
        batch_eos_id = tf.convert_to_tensor([eos_id] * batch, dtype=tf.int32)

        step_lens = tf.convert_to_tensor([1] * batch, dtype=tf.int32)
        for step in range(steps):
            step_target = target_input[:, step]
            target_mask = tf.equal(step_target, batch_eos_id)
            update_mask = tf.logical_and(target_mask, target_len <= 0)
            target_len = tf.where(update_mask, step_lens, target_len)
            step_lens = step_lens + 1
        return target_len


    # input_label shape == (batch x padding_steps)
    # decoder_outputs shape == (batch x padding_steps x vocab_size)
    def loss(self, input_label, decoder_outputs):

        label_batch, label_steps = [int(x) for x in input_label.shape[:2]]
        otput_batch, otput_steps = [int(x) for x in decoder_outputs.shape[:2]]

        assert (label_batch == otput_batch)
        assert (label_steps == otput_steps)

        batch_eos = tf.expand_dims([self.EOS_ID] * label_batch, axis=1)
        input_label = tf.slice(input_label, [0, 1], [label_batch, label_steps - 1])
        input_label = tf.concat([input_label, batch_eos], axis=1)

        target_len = self.get_target_length(input_label, self.EOS_ID)
        input_mask = tf.cast(tf.sequence_mask(target_len, label_steps), dtype=tf.float32)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_label, logits=decoder_outputs) * input_mask
        loss = tf.reduce_sum(loss) / tf.reduce_sum(input_mask)

        return loss

    def call(self, input_data, widths, input_label, training):
        enc_output, enc_status, widths = self.encoder(input_data, widths, training)
        weight_mask = tf.sequence_mask(lengths=widths, maxlen=enc_output.shape[1])
        decoder_outputs, decoder_attentions = self.decoder(input_label,
                                                           enc_status,
                                                           enc_output,
                                                           self.teacher_forcing_ratio,
                                                           weight_mask,
                                                           training)
        if training:
            self.loss_value = self.loss(input_label, decoder_outputs)
            logger.info(self.loss_value)
        return decoder_outputs, decoder_attentions, self.loss_value

    def evaluate(self, input_data, widths):
        enc_output, enc_status, widths = self.encoder(input_data, widths, training)
        weight_mask = tf.sequence_mask(lengths=widths, maxlen=enc_output.shape[1])
        decoder_outputs, decoder_attentions = self.decoder.evaluate(enc_status, enc_output, weight_mask)
        return decoder_outputs, decoder_attentions


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









