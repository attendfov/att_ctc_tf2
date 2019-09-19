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
from EncoderResnet18 import ResNet18
from CtcDecoder import CTCDecoder
from tensor_utils import *
from Logger import logger
from Logger import time_func


class Seq2Seq(tf.keras.Model):
    def __init__(self,
                 vocab_size,
                 eos_id=1
                 ):

        super(Seq2Seq, self).__init__()
        self.vocab_size = vocab_size
        self.eos_id = eos_id

        self.encoder = ResNet18(128)
        self.ctc_decoder = CTCDecoder(vocab_size=self.vocab_size,
                                      num_layers=0,
                                      name='ctc_layer2')

    def call(self, input_tensor, input_widths, ctc_sparse_label, training):
        # (batch_size, inp_seq_len, d_model), (batch_size, inp_seq_len)
        bone_features, lstm_features, valid_widths = self.encoder(input_tensor, input_widths, training)
        print("valid_widths:", valid_widths)
        ctc_features, ctc_final_output = self.ctc_decoder(lstm_features, training)

        loss_value = 0.0
        if training:
            loss_value = self.ctc_loss(ctc_final_output, ctc_sparse_label, valid_widths)
            logger.info("loss_value:{}".format(loss_value))
        return ctc_final_output, loss_value

    def ctc_loss(self, ctc_logits, ctc_labels, seq_lens):
        loss = tf.nn.ctc_loss(ctc_labels, ctc_logits,
                              label_length=None,
                              logit_length=seq_lens,
                              logits_time_major=False,
                              blank_index=self.eos_id)
        loss = tf.reduce_mean(loss)
        return loss

    def ctc_evaluate(self, input_image, input_widths=None):
        training = False
        input_shape = input_image.shape
        assert(len(input_shape) in (3, 4))
        #convert to B*H*W*C
        if len(input_shape) == 3:
            input_image = tf.expand_dims(input_image, axis=0)

        batch = int(input_shape[0])
        bone_features, lstm_features, valid_widths = self.encoder(input_image, input_widths, training)
        ctc_features, ctc_final_output = self.ctc_decoder(lstm_features, False)

        #final_output shape: B*W*C
        final_output_argmax = tf.argmax(ctc_final_output, axis=-1)
        final_output_softmax = tf.nn.softmax(ctc_final_output, axis=-1)

        return final_output_softmax, final_output_argmax


if __name__ == '__main__':
    tf.enable_eager_execution()
    dec_num_layers = 1
    d_model = 512
    vocab_size = 8
    dec_num_heads = 8
    dec_dff = 1024
    sos_id = 0
    eos_id = 1
    max_dec_length = 48
    dec_rate = 0.1

    seq2seq = Seq2Seq(dec_num_layers,
                      d_model,
                      vocab_size,
                      dec_num_heads,
                      dec_dff,
                      sos_id,
                      eos_id,
                      max_dec_length,
                      dec_rate)

    training = True
    batch_size = 4
    image_width = 256
    image_height = 32
    image_channel = 3
    data = np.random.random((batch_size, image_height, image_width, image_channel))
    widths = [128, 144, 160, 252]
    data = tf.convert_to_tensor(data, dtype=tf.float32)

    padding_len = 5
    att_target_inputs = []
    att_target_lengths = []
    ctc_indices = []
    ctc_values = []
    ctc_lengs = []

    for b in range(batch_size):
        att_target = [sos_id]
        ctc_target = []
        valid_len = random.choice(range(2, padding_len - 2))
        ctc_lengs.append(valid_len)
        att_target_lengths.append(valid_len + 2)
        for i in range(valid_len):
            index_id = random.choice(range(2, vocab_size-2))
            att_target.append(index_id)
            ctc_values.append(index_id)
            ctc_indices.append([b, i])
        for i in range(len(att_target), padding_len):
            att_target.append(eos_id)
        att_target_inputs.append(att_target)

    target_dense = tf.convert_to_tensor(att_target_inputs, dtype=tf.int32)
    traget_sparse = tf.SparseTensor(indices=ctc_indices, values=ctc_values, dense_shape=[batch_size, min(ctc_lengs)])

    att_final_output, ctc_final_output, attention_weights, loss_value = seq2seq(data, widths, target_dense, traget_sparse, training)
    print("att_final_output shape {}, {}".format(att_final_output.shape, loss_value))
    print("ctc_final_output shape {}, {}".format(ctc_final_output.shape, loss_value))

    output, probility, atten_weight = seq2seq.att_evaluate(data, widths, 20)
    print(output.shape)
    print(probility.shape)
    print("decoder_layer1_block1 shape:", atten_weight['decoder_layer1_block1'].shape)
    print("decoder_layer1_block2 shape:", atten_weight['decoder_layer1_block2'].shape)

    output_sfmx, output_argmx = seq2seq.ctc_evaluate(data, widths)
    print(output_sfmx.shape)
    print(output_argmx.shape)

