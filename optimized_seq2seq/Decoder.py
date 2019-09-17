# _*_ coding:utf-8 _*_
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import TensorFlow >= 1.10 and enable eager execution
import os
import sys
import random
import numpy as np
import tensorflow as tf
layers = tf.keras.layers
tf.enable_eager_execution()

print(tf.__version__)

abspath = os.path.dirname(os.path.realpath(__file__))
abspath = os.path.abspath(abspath)
sys.path.append(abspath)
sys.path.append(os.path.join(abspath, '../utils'))

from Logger import logger
from Logger import time_func

from Encoder import *

class Decoder(tf.keras.Model):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 SOS_ID=0,
                 EOS_ID=1,
                 dec_units=128,
                 enc_units=128,
                 attention_name='luong',
                 attention_type=0,
                 rnn_type='gru',
                 max_length=10,
                 **kwargs
                 ):
        super(Decoder, self).__init__()

        self.SOS_ID = SOS_ID
        self.EOS_ID = EOS_ID
        self.DEC_OUTPUTS = 'decoder_outputs'
        self.DEC_ARGMAXS = 'decoder_argmaxs'
        self.DEC_LENGTHS = 'decoder_lengths'
        self.DEC_ATTENTIONS = 'decoder_attentions'

        self.attention_show = True
        self.vocab_size = vocab_size
        self.dec_units = dec_units
        self.enc_units = dec_units if enc_units is None else enc_units
        self.attention_type = 0 if attention_type is None else attention_type
        self.attention_name = 'luong' if attention_name is None else attention_name
        self.latent_language = False
        self.max_length = max_length

        self.att_step_func = self.luong_att_step
        if self.attention_name.lower() == 'luong':
            self.att_step_func = self.luong_att_step
        elif self.attention_name.lower() == 'bahdanau':
            self.att_step_func = self.bahdanau_att_step

        assert (rnn_type in ('rnn', 'lstm', 'gru'))

        if rnn_type == 'rnn':
            rnn_class = tf.nn.rnn_cell.RNNCell
        elif rnn_type == 'lstm':
            rnn_class = tf.nn.rnn_cell.LSTMCell
        elif rnn_type == 'gru':
            rnn_class = tf.nn.rnn_cell.GRUCell

        rnn_fw_name = 'decode_rnn_fw'
        self.rnn_cell = rnn_class(num_units=self.dec_units, dtype=tf.float32, name=rnn_fw_name)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.enc_W = tf.keras.layers.Dense(self.enc_units)
        self.dec_W = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)

    def decode_bk(self,
               step_id,
               step_output,
               step_attention,
               decoder_outputs,
               decoder_argmaxs,
               decoder_attentions,
               decoder_lengths
               ):

        decoder_dict = {}
        assert (isinstance(decoder_outputs, (list, np.ndarray)))
        assert (isinstance(decoder_argmaxs, (list, np.ndarray)))
        assert (isinstance(decoder_lengths, (list, np.ndarray)))
        assert (isinstance(decoder_attentions, list))
        assert (len(decoder_outputs) == len(decoder_argmaxs))
        assert (len(decoder_attentions) == len(decoder_argmaxs))
        step_argmax = tf.argmax(step_output, 1)
        decoder_outputs.append(step_output)
        decoder_attentions.append(tf.squeeze(step_attention, axis=2))
        decoder_argmaxs.append(step_argmax)

        batchs_eos = tf.equal(step_argmax, self.EOS_ID).numpy()
        update_idx = ((decoder_lengths > step_id) & batchs_eos) != 0
        decoder_lengths[update_idx] = len(decoder_argmaxs)

        decoder_dict[self.DEC_OUTPUTS] = decoder_outputs
        decoder_dict[self.DEC_ARGMAXS] = decoder_argmaxs
        decoder_dict[self.DEC_LENGTHS] = decoder_lengths
        decoder_dict[self.DEC_ATTENTIONS] = decoder_attentions

        return step_argmax, decoder_dict

    def decode(self,
               step_id,
               step_output,
               step_attention,
               decoder_outputs,
               decoder_argmaxs,
               decoder_attentions,
               decoder_lengths
               ):
        print("step_output shape {}".format(step_output.shape))
        print("step_attention shape {}".format(step_attention.shape))
        decoder_dict = {}
        assert (isinstance(decoder_outputs, (list, np.ndarray)))
        assert (isinstance(decoder_argmaxs, (list, np.ndarray)))
        assert (isinstance(decoder_lengths, (list, np.ndarray)))
        assert (isinstance(decoder_attentions, list))
        assert (len(decoder_outputs) == len(decoder_argmaxs))
        assert (len(decoder_attentions) == len(decoder_argmaxs))
        step_argmax = tf.argmax(step_output, 1)
        decoder_outputs.append(step_output)
        decoder_attentions.append(tf.squeeze(step_attention, axis=2))
        decoder_argmaxs.append(step_argmax)

        batchs_eos = tf.equal(step_argmax, self.EOS_ID).numpy()
        update_idx = ((decoder_lengths > step_id) & batchs_eos) != 0
        decoder_lengths[update_idx] = len(decoder_argmaxs)

        decoder_dict[self.DEC_OUTPUTS] = decoder_outputs
        decoder_dict[self.DEC_ARGMAXS] = decoder_argmaxs
        decoder_dict[self.DEC_LENGTHS] = decoder_lengths
        decoder_dict[self.DEC_ATTENTIONS] = decoder_attentions

        return step_argmax, decoder_dict

    def bahdanau_att_step(self,
                          step_input,
                          dec_hidden,
                          enc_output,
                          weight_mask,
                          **kwargs
                          ):
        '''
        input_step:batch
        enc_hidden:batch x hidden
        enc_output:batch x steps x hidden
        seq_len: [len]*batch
        '''

        logger.debug("step_input shape: {}".format(step_input.shape))
        logger.debug("dec_hidden shape: {}".format(dec_hidden.shape))
        logger.debug("enc_output shape: {}".format(enc_output.shape))

        # batch-->batch x 1
        step_input = tf.expand_dims(step_input, axis=1)
        step_embed = self.embedding(step_input)
        logger.debug("step_embed input shape: {}".format(step_input.shape))
        logger.debug("step_embed otput shape: {}".format(step_embed.shape))

        step_rnn_input = step_embed
        if self.latent_language:
            step_rnn_input = tf.concat([step_embed, dec_hidden], axis=1)
        step_rnn_input = tf.squeeze(input=step_rnn_input, axis=1)
        step_rnn_output, dec_hidden = self.rnn_cell(step_rnn_input, dec_hidden)

        logger.debug("decode rnn step output shape: {}".format(step_rnn_output.shape))
        logger.debug("decode rnn step hidden shape: {}".format(dec_hidden.shape))

        step_rnn_output = tf.expand_dims(step_rnn_output, axis=1)
        step_input_feat = self.dec_W(step_rnn_output)
        enc_output_feat = enc_output
        logger.debug("step_input_feat shape: {}".format(step_input_feat.shape))
        logger.debug("enc_output_feat shape: {}".format(enc_output_feat.shape))

        score = None
        if self.attention_type in (0, 1, 2, 3):
            score = tf.reduce_sum(self.V(tf.tanh(step_input_feat + enc_output_feat)), 2)
            score = tf.expand_dims(score, axis=2)
            logger.debug("score shape: {}".format(score.shape))

        # adding weight mask
        if weight_mask is not None:
            score_mask = tf.expand_dims(weight_mask, axis=-1)
            score_weight = tf.fill(score_mask.shape, np.float('-inf'))
            score = tf.where(score_mask, score, score_weight)
            logger.debug("score shape:{}, score_mask shape:{}, score_weight shape:{}".format(
                score.shape, score_mask.shape, score_weight.shape))

        attention_weights = tf.nn.softmax(score, axis=1)
        logger.debug("attention_weights shape:{}".format(attention_weights.shape))
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        output = self.fc(context_vector)

        # output shape == (batch x cocab_size)
        # dec_hidden shape == (batch x hidden_size)
        # attention_weights shape == (batch x steps)
        return output, dec_hidden, tf.squeeze(attention_weights, axis=-1)

    def luong_att_step(self,
                       step_input,
                       dec_hidden,
                       enc_output,
                       weight_mask,
                       **kwargs
                       ):
        '''
        input_step:batch
        enc_hidden:batch x hidden
        enc_output:batch x steps x hidden
        seq_len: [len]*batch
        '''

        logger.debug("step_input shape: {}".format(step_input.shape))
        logger.debug("dec_hidden shape: {}".format(dec_hidden.shape))
        logger.debug("enc_output shape: {}".format(enc_output.shape))

        #batch-->batch x 1
        step_input = tf.expand_dims(step_input, axis=1)
        step_embed = self.embedding(step_input)

        logger.debug("step_embed input shape: {}".format(step_input.shape))
        logger.debug("step_embed otput shape: {}".format(step_embed.shape))

        step_rnn_input = step_embed
        if self.latent_language:
            step_rnn_input = tf.concat([step_embed, dec_hidden], axis=1)
        step_rnn_input = tf.squeeze(input=step_rnn_input, axis=1)
        step_rnn_output, dec_hidden = self.rnn_cell(step_rnn_input, dec_hidden)
        logger.debug("decode rnn step output shape: {}".format(step_rnn_output.shape))
        logger.debug("decode rnn step hidden shape: {}".format(dec_hidden.shape))

        step_rnn_output = tf.expand_dims(step_rnn_output, axis=1)
        step_input_feat = self.dec_W(step_rnn_output)
        enc_output_feat = enc_output
        logger.debug("step_input_feat shape: {}".format(step_input_feat.shape))
        logger.debug("enc_output_feat shape: {}".format(enc_output_feat.shape))

        score = None
        if self.attention_type == 0:
            logger.debug("attention_type: {}".format(self.attention_type))
            score = self.V(step_input_feat * enc_output_feat)
            logger.debug("score shape: {}".format(score.shape))
        elif self.attention_type == 1:
            logger.debug("attention_type: {}".format(self.attention_type))
            score = tf.reduce_sum(step_input_feat * enc_output_feat, axis=2)
            score = tf.expand_dims(score, axis=2)
        elif self.attention_type == 2:
            logger.debug("attention_type: {}".format(self.attention_type))
            score = self.V(tf.tanh(step_input_feat + enc_output_feat))
        elif self.attention_type == 3:
            logger.debug("attention_type: {}".format(self.attention_type))
            step_cnt = enc_output.shape[1]
            step_input_feat = tf.tile(step_input_feat, [1, step_cnt, 1])
            score = self.V(tf.tanh(tf.concat([step_input_feat, enc_output_feat], axis=2)))
        elif self.attention_type == 4:
            pass

        logger.debug("score shape: {}".format(score.shape))
        # adding weight mask
        if weight_mask is not None:
            score_mask = tf.expand_dims(weight_mask, axis=-1)
            score_weight = tf.fill(score_mask.shape, np.float('-inf'))
            score = tf.where(score_mask, score, score_weight)
            logger.debug("score shape:{}, score_mask shape:{}, score_weight shape:{}".format(
                score.shape, score_mask.shape, score_weight.shape))

        attention_weights = tf.nn.softmax(score, axis=1)
        logger.debug("attention_weights shape:{}".format(attention_weights.shape))
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        output = self.fc(context_vector)

        # output shape == (batch x cocab_size)
        # dec_hidden shape == (batch x hidden_size)
        # attention_weights shape == (batch x steps)
        return output, dec_hidden, tf.squeeze(attention_weights, axis=-1)

    def call(self, target_input, enc_status, enc_output, teacher_forcing_ratio=0.5, weight_mask=None, training=True):
        if target_input is not None:
            logger.debug("target input label shape: {}".format(target_input.shape))
        else:
            logger.debug("target input label is none")

        batch_size = int(enc_output.shape[0])
        batch, encoder_steps = enc_output.shape[:2]
        batch, decoder_steps = target_input.shape

        attention_indices = tf.concat([tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=1),
                                       tf.expand_dims([0] * batch_size, axis=1),
                                       tf.expand_dims([0] * batch_size, axis=1)], axis=1)
        decoder_attentions = tf.scatter_nd(attention_indices, [1.0] * batch, [batch, 1, encoder_steps])

        output_indices = tf.concat([tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=1),
                                    tf.expand_dims([0] * batch_size, axis=1),
                                    tf.expand_dims([self.SOS_ID]*batch_size, axis=1)], axis=1)
        decoder_outputs = tf.scatter_nd(output_indices, [1.0] * batch, [batch, 1, self.vocab_size])

        enc_output_feat = self.enc_W(enc_output)
        step_input = tf.convert_to_tensor([self.SOS_ID] * batch_size)
        decoder_hidden = self.rnn_cell.zero_state(batch_size, dtype=tf.float32)

        use_teacher_forcing = False
        if training and random.random() < teacher_forcing_ratio:
            use_teacher_forcing = True

        if use_teacher_forcing:
            for step in range(1, decoder_steps):
                decoder_output, decoder_hidden, attention = self.att_step_func(step_input, decoder_hidden,
                                                                               enc_output_feat, weight_mask)
                decoder_outputs = tf.concat([decoder_outputs, tf.expand_dims(decoder_output, axis=1)], axis=1)
                decoder_attentions = tf.concat([decoder_attentions, tf.expand_dims(attention, axis=1)], axis=1)

                step_input = target_input[:, step]
        else:
            for step in range(1, decoder_steps):
                decoder_output, decoder_hidden, attention = self.att_step_func(step_input, decoder_hidden,
                                                                               enc_output_feat, weight_mask)
                decoder_outputs = tf.concat([decoder_outputs, tf.expand_dims(decoder_output, axis=1)], axis=1)
                decoder_attentions = tf.concat([decoder_attentions, tf.expand_dims(attention, axis=1)], axis=1)
                step_input = tf.argmax(decoder_output, axis=1)


        logger.debug("decoder_outputs shape :{}".format(decoder_outputs.shape))
        logger.debug("decoder_attentions shape :{}".format(decoder_attentions.shape))

        return decoder_outputs, decoder_attentions

    def evaluate(self, enc_status, enc_output, weight_mask=None):
        batch_size = int(enc_output.shape[0])
        batch, encoder_steps = enc_output.shape[:2]
        decoder_steps = self.max_length

        attention_indices = tf.concat([tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=1),
                                       tf.expand_dims([0] * batch_size, axis=1),
                                       tf.expand_dims([0] * batch_size, axis=1)], axis=1)
        decoder_attentions = tf.scatter_nd(attention_indices, [1.0] * batch, [batch, 1, encoder_steps])

        output_indices = tf.concat([tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=1),
                                    tf.expand_dims([0] * batch_size, axis=1),
                                    tf.expand_dims([self.SOS_ID]*batch_size, axis=1)], axis=1)
        decoder_outputs = tf.scatter_nd(output_indices, [1.0] * batch, [batch, 1, self.vocab_size])

        enc_output_feat = self.enc_W(enc_output)
        step_input = tf.convert_to_tensor([self.SOS_ID] * batch_size)
        logger.info("step_input init shape :{}".format(step_input.shape))
        decoder_hidden = self.rnn_cell.zero_state(batch_size, dtype=tf.float32)

        for step in range(1, decoder_steps):
            decoder_output, decoder_hidden, attention = self.att_step_func(step_input,
                                                                           decoder_hidden,
                                                                           enc_output_feat,
                                                                           weight_mask)
            decoder_outputs = tf.concat([decoder_outputs, tf.expand_dims(decoder_output, axis=1)], axis=1)
            decoder_attentions = tf.concat([decoder_attentions, tf.expand_dims(attention, axis=1)], axis=1)
            step_input = tf.argmax(decoder_output, axis=1)

        logger.info("decoder_outputs shape :{}".format(decoder_outputs.shape))
        logger.info("decoder_attentions shape :{}".format(decoder_attentions.shape))

        return decoder_outputs, decoder_attentions


if __name__=='__main__':
    enc_units = 76
    dec_units = 76
    embed_units = 256
    vocab_sizes = 111
    teacher_forcing_ratio = 0.0
    data = np.random.random((6, 32, 128, 3))
    widths = [16, 32, 48, 64, 96, 128]
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    ecnoder = Encoder(enc_units)
    decoder = Decoder(vocab_sizes, embed_units, dec_units=dec_units, enc_units=enc_units)
    features, fw_status, widths = ecnoder(data, widths, True)
    target_input = np.ones((6, 35), dtype=np.int)
    for i in range(6):
        for j in range(35):
            target_input[i][j] = np.random.randint(0, vocab_sizes)
    print("features  shape:", features.shape)
    print("widths    list :", widths)

    weight_mask = None
    target_input = tf.convert_to_tensor(target_input)
    decoder(target_input, fw_status, features, teacher_forcing_ratio, weight_mask, True)
    decoder.evaluate(fw_status, features, weight_mask)




