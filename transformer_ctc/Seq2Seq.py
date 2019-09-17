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
tf.enable_eager_execution()

print(tf.__version__)

abspath = os.path.dirname(os.path.realpath(__file__))
abspath = os.path.abspath(abspath)
sys.path.append(abspath)
sys.path.append(os.path.join(abspath, '../utils'))

import random
from Encoder import *
from Decoder import *
from EncoderRes import *
from Logger import time_func

class Seq2Seq(tf.keras.Model):
    def __init__(self,
                 enc_num_layers,
                 dec_num_layers,
                 d_model,
                 vocab_size,
                 enc_num_heads,
                 dec_num_heads,
                 enc_dff,
                 dec_dff,
                 enc_used_rnn = False,
                 sos_id=0,
                 eos_id=1,
                 max_enc_length=1200,
                 max_dec_length=48,
                 enc_rate=0.1,
                 dec_rate=0.1):

        super(Seq2Seq, self).__init__()

        self.enc_num_layers = enc_num_layers
        self.dec_num_layers = dec_num_layers
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.enc_num_heads = enc_num_heads
        self.dec_num_heads = dec_num_heads
        self.enc_dff = enc_dff
        self.dec_dff = dec_dff
        self.enc_used_rnn = enc_used_rnn
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.max_enc_length = max_enc_length
        self.max_dec_length = max_dec_length
        self.enc_rate = enc_rate
        self.dec_rate = dec_rate

        self.encoder = ResNet18(num_layers=self.enc_num_layers,
                                d_model=self.d_model,
                                num_heads=self.enc_num_heads,
                                dff=self.enc_dff,
                                rate=self.enc_rate,
                                used_rnn=self.enc_used_rnn,
                                max_width=self.max_enc_length)

        self.att_decoder = ATTDecoder(num_layers=self.dec_num_layers,
                                      vocab_size=self.vocab_size,
                                      d_model=self.d_model,
                                      num_heads=self.dec_num_heads,
                                      dff=self.dec_dff,
                                      sos_id=self.sos_id,
                                      eos_id=self.eos_id,
                                      max_length=self.max_dec_length,
                                      rate=self.dec_rate)

        self.ctc_decoder = CTCDecoder(num_layers=1)
        self.ctc_final_layer = tf.keras.layers.Dense(self.vocab_size-1)
        self.att_final_layer = tf.keras.layers.Dense(self.vocab_size)

    def call(self, input, input_widths, att_label, att_lens, ctc_label, ctc_lens, training):
        # (batch_size, inp_seq_len, d_model), (batch_size, inp_seq_len)
        enc_output, weight_mask, ctc_feature, ctc_lens = self.encoder(input, input_widths, training)

        batch = att_label.shape[0]
        steps = att_label.shape[1]
        logger.debug("batch: {}, steps: {}".format(batch, steps))
        look_ahead_mask = tf.sequence_mask(tf.range(1, steps + 1), steps, dtype=tf.float32)

        dec_padding_mask = None
        if att_lens is None:
            #通过sos_id和eos_id计算有效长度
            att_lens = self.get_target_length(att_label, self.eos_id)
            target_mask = tf.sequence_mask(att_lens, steps, dtype=tf.float32)
        else:
            target_mask = tf.sequence_mask(att_lens, steps, dtype=tf.float32)

        if weight_mask is not None and target_mask is not None:
            dec_padding_mask = tf.matmul(tf.expand_dims(target_mask, -1),
                                         tf.expand_dims(weight_mask, -1), transpose_b=True)
            dec_padding_mask = tf.expand_dims(1.0 - dec_padding_mask, axis=1)
            logger.debug("padding_mask.shape {}".format(dec_padding_mask.shape))

        if ctc_lens is None:
            # 通过eos_id计算有效长度
            ctc_lens = self.get_target_length(ctc_label, self.eos_id)

        look_ahead_mask = 1.0 - look_ahead_mask
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        att_dec_output, attention_weights = self.att_decoder(att_label,
                                                             enc_output,
                                                             training,
                                                             look_ahead_mask,
                                                             dec_padding_mask)

        ctc_dec_output = self.ctc_decoder(ctc_feature, training)
        att_final_output = self.att_final_layer(att_dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        ctc_final_output = self.ctc_final_layer(ctc_dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        loss_value = 0.0
        if training:
            att_loss = self.att_loss(att_final_output, att_label, att_lens)
            ctc_loss = self.ctc_loss(ctc_final_output, ctc_label, ctc_lens)
            loss_value = att_loss + ctc_loss
            print("att_loss:{}, ctc_loss:{}".format(att_loss, ctc_loss))
        return att_final_output, ctc_final_output, attention_weights, loss_value

    def att_loss(self, att_logit, att_label, att_lens=None):
        batch = int(att_label.shape[0])
        steps = int(att_label.shape[1])
        batch_eos = tf.expand_dims([self.eos_id] * batch, axis=1)
        att_label = tf.slice(att_label, [0, 1], [batch, steps-1])
        att_label = tf.concat([att_label, batch_eos], axis=1)

        if att_lens is None:
            att_lens = self.get_target_length(att_label, self.eos_id)

        input_mask = tf.cast(tf.sequence_mask(att_lens, steps), dtype=tf.float32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=att_label,
                                                              logits=att_logit) * input_mask
        loss = tf.reduce_sum(loss) / tf.reduce_sum(input_mask)
        return loss

    def ctc_loss(self, ctc_logits, ctc_labels, seq_lens):
        loss = tf.nn.ctc_loss(ctc_labels, ctc_logits, seq_lens,
                              time_major=False, ignore_longer_outputs_than_inputs=True)
        loss = tf.reduce_mean(loss)
        return loss


    @staticmethod
    def get_target_length(target_input, eos_id):
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

    def att_evaluate(self, input_image, input_widths=None, max_length=40):
        training = False
        input_shape = input_image.shape
        assert(len(input_shape) in (3, 4))
        if len(input_shape) == 3:
            # batch x H x W x C
            input_image = tf.expand_dims(input_image, axis=0)

        batch = int(input_shape[0])
        enc_output, weight_mask, _, _ = self.encoder(input_image, input_widths, training)
        decoder_input = tf.expand_dims([self.sos_id] * batch, 1)
        decoder_prob = tf.cast(tf.expand_dims([1.0] * batch, 1), dtype=tf.float32)
        decoder_finish = tf.expand_dims([False] * batch, 1)
        finish_constant = tf.expand_dims([True] * batch, 1)
        eosid_constant = tf.expand_dims([self.eos_id] * batch, 1)

        for step in range(1, max_length):
            decoder_steps = int(decoder_input.shape[1])
            look_ahead_mask = tf.sequence_mask(tf.range(1, decoder_steps + 1), decoder_steps, dtype=tf.float32)
            look_ahead_mask = 1.0 - look_ahead_mask

            dec_padding_mask = None
            decode_mask = tf.sequence_mask([decoder_steps]*batch, decoder_steps, dtype=tf.float32)

            if weight_mask is not None:
                dec_padding_mask = tf.matmul(tf.expand_dims(decode_mask, -1),
                                             tf.expand_dims(weight_mask, -1), transpose_b=True)

                dec_padding_mask = tf.expand_dims(1.0 - dec_padding_mask, axis=1)
                logger.debug("padding_mask.shape {}".format(dec_padding_mask.shape))

            # dec_output.shape == (batch_size, tar_seq_len, d_model)
            dec_output, attention_weights = self.att_decoder(decoder_input,
                                                             enc_output,
                                                             training,
                                                             look_ahead_mask,
                                                             dec_padding_mask)

            # final_output.shape == (batch_size, tar_seq_len, vocab_size)
            final_output = self.att_final_layer(dec_output)
            final_output = tf.nn.softmax(final_output, axis=-1)
            # predictions.shape == (batch_size, 1, vocab_size)
            predictions = final_output[:, -1:, :]
            # predictions.shape == (batch_size, vocab_size)
            #predictions = final_output[:, -1, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            predicted_pb = tf.cast(tf.reduce_max(predictions, axis=-1), tf.float32)

            predicted_id = tf.where(decoder_finish, eosid_constant, predicted_id)
            update_masks = tf.logical_and(tf.equal(predicted_id, self.eos_id),
                                          tf.equal(decoder_finish, False))
            decoder_finish = tf.where(update_masks, finish_constant, decoder_finish)
            decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)
            decoder_prob = tf.concat([decoder_prob, predicted_pb], axis=-1)
            if tf.reduce_all(decoder_finish):
                return decoder_input, decoder_prob, attention_weights

        return decoder_input, decoder_prob, attention_weights

    def ctc_evaluate(self, input_image, input_widths=None):
        training = False
        input_shape = input_image.shape
        assert(len(input_shape) in (3, 4))
        #convert to B*H*W*C
        if len(input_shape) == 3:
            input_image = tf.expand_dims(input_image, axis=0)

        batch = int(input_shape[0])
        _, _, ctc_feature, ctc_lens = self.encoder(input_image, input_widths, training)
        ctc_output = self.ctc_decoder(ctc_feature, False)

        #final_output shape: B*W*C
        final_output = self.ctc_final_layer(ctc_output)
        final_output_argmax = tf.argmax(final_output, axis=-1)
        final_output_softmax = tf.nn.softmax(final_output, axis=-1)

        return final_output_softmax, final_output_argmax


def get_target_length_unit():
    sos_id = 0
    eos_id = 1
    vocab_size = 8
    batch_size = 12
    padding_len = 10
    target_inputs = []
    target_lengths = []
    for b in range(batch_size):
        target = [sos_id]
        valid_len = random.choice(range(2, padding_len - 2))
        target_lengths.append(valid_len + 2)
        for i in range(valid_len):
            target.append(random.choice(range(2, vocab_size)))
        for i in range(len(target), padding_len):
            target.append(eos_id)
        target_inputs.append(target)

    print(target_inputs)
    print(target_lengths)

    target_inputs = tf.convert_to_tensor(target_inputs)
    target_lengths = Seq2Seq.get_target_length(target_inputs, eos_id)
    print(target_lengths)


if __name__ == '__main__':
    get_target_length_unit()
    enc_num_layers = 0
    dec_num_layers = 1
    d_model = 512
    vocab_size = 8
    enc_num_heads = 2
    dec_num_heads = 8
    enc_dff = 1024
    dec_dff = 1024
    enc_used_rnn = False
    sos_id = 0
    eos_id = 1
    max_enc_length = 256
    max_dec_length = 48
    enc_rate = 0.1
    dec_rate = 0.1

    seq2seq = Seq2Seq(enc_num_layers,
                      dec_num_layers,
                      d_model,
                      vocab_size,
                      enc_num_heads,
                      dec_num_heads,
                      enc_dff,
                      dec_dff,
                      enc_used_rnn,
                      sos_id,
                      eos_id,
                      max_enc_length,
                      max_dec_length,
                      enc_rate,
                      dec_rate)

    training = True
    batch_size = 6
    image_width = 32
    image_height = 64
    image_channel = 3
    data = np.random.random((batch_size, image_height, image_width, image_channel))
    widths = [2, 4, 6, 8, 10, 12]
    data = tf.convert_to_tensor(data, dtype=tf.float32)

    padding_len = 10
    target_inputs = []
    target_lengths = []
    for b in range(batch_size):
        target = [sos_id]
        valid_len = random.choice(range(2, padding_len - 2))
        target_lengths.append(valid_len + 2)
        for i in range(valid_len):
            target.append(random.choice(range(2, vocab_size)))
        for i in range(len(target), padding_len):
            target.append(eos_id)
        target_inputs.append(target)

    target_inputs = tf.convert_to_tensor(target_inputs, dtype=tf.int32)
    target_lengths = tf.convert_to_tensor(target_lengths, dtype=tf.int32)

    final_output, attention_weights, loss_value = seq2seq(data, widths, target_inputs, target_lengths, training)
    print("final_output shape {}, {}".format(final_output.shape, loss_value))

    output, probility, atten_weight = seq2seq.evaluate(data, widths, 20)
    print(output.shape)
    print(probility.shape)
    print("decoder_layer1_block1 shape:", atten_weight['decoder_layer1_block1'].shape)
    print("decoder_layer1_block2 shape:", atten_weight['decoder_layer1_block2'].shape)










