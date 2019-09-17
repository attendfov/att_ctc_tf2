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
from Encoder import EncoderAste
from CtcDecoder import CTCDecoder
from AttDecoder import ATTDecoder
from TensorUtils import get_target_length
from Logger import logger
from Logger import time_func


class Seq2Seq(tf.keras.Model):
    def __init__(self,
                 dec_num_layers,
                 d_model,
                 vocab_size,
                 dec_num_heads,
                 dec_dff,
                 sos_id=0,
                 eos_id=1,
                 max_dec_length=64,
                 dec_rate=0.1):

        super(Seq2Seq, self).__init__()

        self.dec_num_layers = dec_num_layers
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.dec_num_heads = dec_num_heads
        self.dec_dff = dec_dff
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.max_dec_len = max_dec_length
        self.dec_rate = dec_rate

        self.encoder = EncoderAste()
        self.ctc_decoder = CTCDecoder(vocab_size=self.vocab_size-1,
                                      num_layers=0,
                                      name='ctc_layer2')
        self.att_decoder = ATTDecoder(num_layers=self.dec_num_layers,
                                      vocab_size=self.vocab_size,
                                      num_heads=self.dec_num_heads,
                                      dff=self.dec_dff, d_model=self.d_model,
                                      sos_id=self.eos_id, eos_id=self.sos_id,
                                      max_length=self.max_dec_len, rate=0.1)

    def call(self, input_tensor, input_widths, att_dense_label, ctc_sparse_label, training):
        # (batch_size, inp_seq_len, d_model), (batch_size, inp_seq_len)
        bone_features, lstm_features, valid_widths = self.encoder(input_tensor, input_widths, training)
        steps = att_dense_label.shape[1]

        features_b, features_w, features_c = lstm_features.shape[:3]
        widths = tf.reshape(valid_widths, [-1])
        weight_mask = tf.sequence_mask(widths, features_w, dtype=tf.float32)
        weight_mask = tf.reshape(weight_mask, [features_b, features_w])
        att_lens = get_target_length(att_dense_label, self.eos_id)
        target_mask = tf.sequence_mask(att_lens, steps, dtype=tf.float32)

        padding_mask = tf.matmul(tf.expand_dims(target_mask, -1),
                                 tf.expand_dims(weight_mask, -1), transpose_b=True)
        padding_mask = tf.expand_dims(1.0 - padding_mask, axis=1)
        logger.debug("padding_mask.shape {}".format(padding_mask.shape))

        look_ahead_mask = tf.sequence_mask(tf.range(1, steps + 1), steps, dtype=tf.float32)
        look_ahead_mask = 1.0 - look_ahead_mask
        att_features, att_final_output, attention_weights = self.att_decoder(att_dense_label,
                                                                             lstm_features,
                                                                             training,
                                                                             look_ahead_mask,
                                                                             padding_mask)
        ctc_features, ctc_final_output = self.ctc_decoder(lstm_features, training)

        loss_value = 0.0
        if training:
            att_loss = self.att_loss(att_final_output, att_dense_label, att_lens)
            ctc_loss = self.ctc_loss(ctc_final_output, ctc_sparse_label, valid_widths)
            loss_value = att_loss + ctc_loss
            logger.info("att_loss:{}, ctc_loss:{}".format(att_loss, ctc_loss))
        return att_final_output, ctc_final_output, attention_weights, loss_value

    def att_loss(self, att_logit, att_label, att_lens=None):
        batch = int(att_label.shape[0])
        steps = int(att_label.shape[1])
        batch_eos = tf.expand_dims([self.eos_id] * batch, axis=1)
        att_label = tf.slice(att_label, [0, 1], [batch, steps-1])
        att_label = tf.concat([att_label, batch_eos], axis=1)

        if att_lens is None:
            att_lens = get_target_length(att_label, self.eos_id)

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

    def att_evaluate(self, input_image, input_widths=None, max_length=40):
        training = False
        input_shape = list(input_image.shape)
        assert(len(input_shape) in (3, 4))
        if len(input_shape) == 3:
            # batch x H x W x C
            input_image = tf.expand_dims(input_image, axis=0)

        batch = int(input_shape[0])
        bone_features, lstm_features, valid_widths = self.encoder(input_image, input_widths, training)
        features_b, features_w, features_c = lstm_features.shape[:3]
        enc_output = lstm_features
        widths = tf.reshape(valid_widths, [-1])
        weight_mask = tf.sequence_mask(widths, features_w, dtype=tf.float32)
        weight_mask = tf.reshape(weight_mask, [features_b, features_w])

        decoder_input = tf.expand_dims([self.sos_id] * batch, 1)
        decoder_prob = tf.cast(tf.expand_dims([1.0] * batch, 1), dtype=tf.float32)
        decoder_finish = tf.expand_dims([False] * batch, 1)
        finish_constant = tf.expand_dims([True] * batch, 1)
        eosid_constant = tf.expand_dims([self.eos_id] * batch, 1)

        for step in range(1, max_length):
            decoder_steps = int(decoder_input.shape[1])
            look_ahead_mask = tf.sequence_mask(tf.range(1, decoder_steps + 1), decoder_steps, dtype=tf.float32)
            look_ahead_mask = 1.0 - look_ahead_mask

            decode_mask = tf.sequence_mask([decoder_steps]*batch, decoder_steps, dtype=tf.float32)
            padding_mask = tf.matmul(tf.expand_dims(decode_mask, -1),
                                     tf.expand_dims(weight_mask, -1), transpose_b=True)

            padding_mask = tf.expand_dims(1.0 - padding_mask, axis=1)
            logger.debug("padding_mask.shape {}".format(padding_mask.shape))

            # dec_output.shape == (batch_size, tar_seq_len, d_model)
            att_features, att_final_output, attention_weights = self.att_decoder(decoder_input,
                                                                                 enc_output,
                                                                                 training,
                                                                                 look_ahead_mask,
                                                                                 padding_mask)

            # final_output.shape == (batch_size, tar_seq_len, vocab_size)
            final_output = tf.nn.softmax(att_final_output, axis=-1)
            # predictions.shape == (batch_size, 1, vocab_size)
            predictions = final_output[:, -1:, :]
            # predictions.shape == (batch_size, vocab_size)
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

