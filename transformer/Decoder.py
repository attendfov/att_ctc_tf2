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
sys.path.append(os.path.join(abspath, '../utils'))
from Logger import logger
from Encoder import *
from TransferUtils import *


class DecoderLayer(tf.keras.Model):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.BatchNormalization()
        self.layernorm2 = tf.keras.layers.BatchNormalization()
        self.layernorm3 = tf.keras.layers.BatchNormalization()

        #self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        #self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        #self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.Model):
    def __init__(self,
                 num_layers,
                 vocab_size,
                 d_model,
                 num_heads,
                 dff,
                 sos_id=0,
                 eos_id=1,
                 max_length=250,
                 rate=0.1):
        super(Decoder, self).__init__()
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_length, self.d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


if __name__ == '__main__':

    import numpy as np
    batch_size = 6
    image_width = 32
    image_height = 64
    image_channel = 3
    data = np.random.random((batch_size, image_height, image_width, image_channel))
    widths = [2, 4, 6, 8, 10, 12]
    data = tf.convert_to_tensor(data, dtype=tf.float32)

    vocab_size = 8
    num_layers = 2
    sos_id = 0
    eos_id = 1
    d_model = 512
    num_heads = 4
    dff = 1024
    rate = 0.1
    used_rnn = False
    max_width = 1600
    max_length = 256

    ecnoder = Encoder(used_rnn=False)
    features, weight_mask = ecnoder(data, widths, True)
    print("features shape: {}".format(features.shape))
    print("weight   shape: {}".format(weight_mask.shape))

    decoder = Decoder(num_layers,
                      vocab_size,
                      d_model,
                      num_heads,
                      dff,
                      sos_id,
                      eos_id,
                      max_length,
                      rate=0.1)

    padding_len = 10
    target_inputs = []
    target_lengths = []
    for b in range(batch_size):
        target = [sos_id]
        valid_len = random.choice(range(2, padding_len-2))
        target_lengths.append(valid_len+2)
        for i in range(valid_len):
            target.append(random.choice(range(2, vocab_size)))
        for i in range(len(target), padding_len):
            target.append(eos_id)
        target_inputs.append(target)

    target_inputs = tf.convert_to_tensor(target_inputs, dtype=tf.int32)
    target_lengths = tf.convert_to_tensor(target_lengths, dtype=tf.int32)

    print("target_inputs shape: {}".format(target_inputs.shape))
    print("target_lengths shape: {}".format(target_lengths.shape))

    target_mask = tf.sequence_mask(target_lengths, padding_len, dtype=tf.float32)
    print("target_mask shape: {}".format(target_mask.shape))

    look_ahead_mask = tf.sequence_mask(tf.range(1, padding_len + 1), padding_len, dtype=tf.float32)
    print("look_ahead_mask shape {}".format(look_ahead_mask.shape))

    padding_mask = tf.matmul(tf.expand_dims(target_mask, -1),
                             tf.expand_dims(weight_mask, -1), transpose_b=True)
    padding_mask = tf.expand_dims(1.0-padding_mask, axis=1)
    print("padding_mask.shape {}".format(padding_mask.shape))
    look_ahead_mask = 1.0-look_ahead_mask
    x, attentions = decoder(target_inputs, features, True, look_ahead_mask, padding_mask)

    print("x shape: {}".format(x.shape))
    print("attentions keys: {}".format(attentions.keys()))


    '''
    inputs = tf.convert_to_tensor(np.random.random((12, 36, 36, 55)), dtype=tf.float32)
    layer_inst = LayerNormalization(55)
    outputs = layer_inst(inputs)
    print("outputs shape: {}".format(outputs.shape))
    sample_encoder_layer = EncoderLayer(512, 8, 2048)
    sample_encoder_layer_output = sample_encoder_layer(
        tf.random.uniform((64, 43, 512)), False, None)

    print("sample_encoder_layer_output shape {}".format(sample_encoder_layer_output.shape))
    sample_decoder_layer = DecoderLayer(512, 8, 2048)
    sample_decoder_layer_output, _, _ = sample_decoder_layer(
        tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
        False, None, None)
    print("sample_decoder_layer_output shape: {}".format(sample_decoder_layer_output.shape))
    '''





