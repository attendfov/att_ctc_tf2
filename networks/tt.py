import datetime
import os
import numpy as np
import threading
import logging
import tensorflow as tf
from cy_base_layer import ctc_loss_layer
from attention_decoder import Decoder
from attention_decoder import argmax_to_tensor
from attention_decoder import softmax_to_seqprob
from data_utils.tensor_utils import dense_to_sparse


class RESNET_ADA(tf.keras.Model):
    def __init__(self, params):
        super(RESNET_ADA, self).__init__()
        self.layer_params = [
            [64, 3, 'same', 'conv1', False],
            # pool
            [128, 3, 'same', 'conv2', False],
            # pool
            [256, 3, 'same', 'conv3', False],
            [256, 3, 'same', 'conv4', True],
            # hpool
            [512, 3, 'same', 'conv5', False],
            [512, 3, 'same', 'conv6', True],
            # hpool
            [512, 3, 'same', 'conv7', True]
        ]
        self.rnn_size = [256, 256]
        self.dropout_rate = 0.0
        self.num_class_lst = params.num_class_lst
        self.blank_class = params.blank_class
        self.use_beam_search = params.use_beam_search
        self.adapt_learning = True if params.domain_train != 'general' else False
        self.CNN_adapt = [{} for i in range(len(self.layer_params))]
        self.two = tf.constant(2, dtype=tf.int32, name='two')
        # Adaptive learning config
        self.adapter_scope = 'adapter/'
        # adapter index: 0.general  1.disease  2.special TC money.... etc.
        self.domain_name_lst = params.domain_name_lst
        self.backbone_version_dict = params.backbone_version_dict
        # adapter layer & layer_name map.
        self.adapter_1 = {}
        self.adapter_2 = {}
        self.layers_map = {}
        self.backbone_adapter_1 = {}  # {backbone_version}{layer_idx}
        self.backbone_adapter_2 = {}  # {backbone_version}{layer_idx}
        self.attention_decoder_map = {}
        self.attention_sosid_map = {}
        self.attention_eosid_map = {}
        # for squeeze_excitation
        self.squeeze_excitation = params.squeeze_excitation
        self.squeeze_ratio = params.squeeze_ratio
        # basic backbone construction.
        self.ConstructFeatureExtraction()
        # set up modules for each domain.
        self.domain_modules = {}
        self.domain_train = params.domain_train
        self.char_dict_path_dict = params.char_dict_path_dict
        self.domain_modules_dict = params.domain_modules_dict
        self.domain_losstype_dict = params.domain_losstype_dict
        for domain_name in self.domain_losstype_dict:
            loss_type = str(self.domain_losstype_dict[domain_name]).lower()
            assert (loss_type in ('ctc', 'attention'))
            self.domain_losstype_dict[domain_name] = loss_type

        # allow to train domain using data of different domain.
        self.data_transfer = params.data_transfer
        if params.domain_train != '':
            self.train_domain_idx = self.domain_name_lst.index(params.domain_train)
        else:
            self.train_domain_idx = -1

        def checkValidModule(modules_str, domain_name):
            modules = modules_str.split()
            assert len(modules) == 3, 'invalid modules_str, should be "feature_domain context_domain logits_domain"'
            modules_output = []
            for m in modules:
                if m == 'self':
                    m = domain_name
                assert m in self.domain_name_lst, 'invalid modules_str: {}'.format(modules_str)
                modules_output.append(m)
            return modules_output

        for name in self.domain_name_lst:
            self.domain_modules[name] = {}
            modules_str = self.domain_modules_dict[name]
            if 'all_self' in modules_str:
                self.domain_modules[name] = {'feature': name, 'context': name, 'logits': name}
            else:
                modules = checkValidModule(modules_str, name)
                self.domain_modules[name] = {'feature': modules[0], 'context': modules[1], 'logits': modules[2]}

            if 'backbone' in name:
                backbone_version = name.split('backbone')[1]
                self.backbone_adapter_1[backbone_version] = {}
                self.backbone_adapter_2[backbone_version] = {}
            else:
                self.adapter_1[name] = {}
                self.adapter_2[name] = {}

            domain_loss_type = self.domain_losstype_dict[name]
            self.ConstructFeatureAdaptiveLearning(0, name)
            self.ConstructFeatureAdaptiveLearning(1, name)
            self.ConstructFeatureAdaptiveLearning(2, name)
            self.ConstructFeatureAdaptiveLearning(3, name)
            self.ConstructFeatureAdaptiveLearning(4, name)
            self.ConstructFeatureAdaptiveLearning(5, name)
            self.ConstructFeatureAdaptiveLearning(6, name)

            if domain_loss_type == 'ctc':
                self.ConstructContextAdaptiveLearning(name, domain_loss_type=domain_loss_type)
                self.ConstructLogitsAdaptiveLearning(name, domain_loss_type)
            elif domain_loss_type == 'attention':
                domain_char_count = int(self.num_class_lst[name]) - 1
                EOS_ID = domain_char_count
                SOS_ID = domain_char_count + 1
                vocab_size = domain_char_count + 2
                self.attention_sosid_map[name] = SOS_ID
                self.attention_eosid_map[name] = EOS_ID
                embedd_dim = 96
                dec_units = 128
                enc_units = 128
                attention_name = 'luong'
                attention_type = 0
                rnn_type = 'gru'
                max_length = 30
                self.ConstructContextAdaptiveLearning(name,
                                                      SOS_ID=SOS_ID,
                                                      EOS_ID=EOS_ID,
                                                      vocab_size=vocab_size,
                                                      embedd_dim=embedd_dim,
                                                      dec_units=dec_units,
                                                      enc_units=enc_units,
                                                      rnn_type=rnn_type,
                                                      max_length=max_length,
                                                      attention_name=attention_name,
                                                      attention_type=attention_type,
                                                      domain_loss_type=domain_loss_type)

    def widthMask(self, fea_len):
        # calc the receptive field and coordinate.
        # r_l = (r_{l+1}-1) x s_l + k_l
        # x_l = s_l x x_{x+1} + ((k_l - 1)/2 - p_l)
        r_h = 1
        r_w = 1
        y_0 = 0.
        ka = [3, 2, 3, 2, 3, 3, 2, 3, 3, 2, 2]
        sa = [1, 2, 1, 2, 1, 1, (2, 1), 1, 1, (2, 1), 1]
        pa = [1, 0, 1, 0, 1, 1, (0, 0), 1, 1, (0, 0), 0]
        x_mask = []
        rf_w = -1
        rf_h = -1
        for x in range(fea_len):
            for i in range(len(ka) - 1, -1, -1):
                k_h = ka[i][0] if isinstance(ka[i], tuple) else ka[i]
                s_h = sa[i][0] if isinstance(sa[i], tuple) else sa[i]
                p_h = pa[i][0] if isinstance(pa[i], tuple) else pa[i]
                if rf_h == -1:
                    r_h = (r_h - 1) * s_h + k_h
                    # print r_h
                # y_0 = (s_h * y_0) + (float(k_h - 1)/2 - p_h)
                # print y_0
                k_w = ka[i][1] if isinstance(ka[i], tuple) else ka[i]
                s_w = sa[i][1] if isinstance(sa[i], tuple) else sa[i]
                p_w = pa[i][1] if isinstance(pa[i], tuple) else pa[i]
                if rf_w == -1:
                    r_w = (r_w - 1) * s_w + k_w
                    # print r_w
                x = (s_w * x) + (np.ceil(float(k_w - 1) / 2) - p_w)
            x_mask.append(x)
            if rf_w == -1:
                rf_w = r_w
                rf_h = r_h
        return x_mask, (r_h, r_w)

    def ConstructCNNModule(self, idx):
        activation = None
        bias_initializer = tf.constant_initializer(value=0.0)
        kenel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-5)

        cnn = tf.layers.Conv2D(filters=self.layer_params[idx][0],
                               kernel_size=3,
                               padding='same',
                               activation=activation,
                               kernel_initializer='he_normal',
                               bias_initializer=bias_initializer,
                               kernel_regularizer=kenel_regularizer,
                               name='convnet/' + self.layer_params[idx][3],
                               trainable=(not self.adapt_learning))
        var_name = 'self.cnn_' + str(idx)

        self.layers_map[var_name] = cnn
        var_name = 'self.bn_' + str(idx)
        self.layers_map[var_name] = tf.layers.BatchNormalization(trainable=(not self.adapt_learning),
                                                                 name='convnet/' + self.layer_params[idx][3] + '/bn')

        cnn = tf.layers.Conv2D(filters=self.layer_params[idx][0],
                               kernel_size=3,
                               padding='same',
                               activation=activation,
                               kernel_initializer='he_normal',
                               bias_initializer=bias_initializer,
                               kernel_regularizer=kenel_regularizer,
                               name='convnet/' + self.layer_params[idx][3] + '_1',
                               trainable=(not self.adapt_learning))
        var_name = 'self.cnn_' + str(idx) + '_1'
        self.layers_map[var_name] = cnn
        var_name = 'self.bn_' + str(idx) + '_1'
        self.layers_map[var_name] = tf.layers.BatchNormalization(trainable=(not self.adapt_learning),
                                                                 name='convnet/' + self.layer_params[idx][3] + '/bn_1')

        cnn = tf.layers.Conv2D(filters=self.layer_params[idx][0],
                               kernel_size=3,
                               padding='same',
                               activation=activation,
                               kernel_initializer='he_normal',
                               bias_initializer=bias_initializer,
                               kernel_regularizer=kenel_regularizer,
                               name='convnet/' + self.layer_params[idx][3] + '_2',
                               trainable=(not self.adapt_learning))
        var_name = 'self.cnn_' + str(idx) + '_2'
        self.layers_map[var_name] = cnn
        var_name = 'self.bn_' + str(idx) + '_2'
        self.layers_map[var_name] = tf.layers.BatchNormalization(trainable=(not self.adapt_learning),
                                                                 name='convnet/' + self.layer_params[idx][3] + '/bn_2')

        # use squeeze_excitation module in res-type backbone
        if self.squeeze_excitation:
            out_dim = self.layer_params[idx][0]
            var_name = 'self.squeeze_' + str(idx)
            squeeze = tf.layers.Dense(out_dim / self.squeeze_ratio, kernel_initializer='he_normal',
                                      use_bias=False, name='convnet/' + self.layer_params[idx][3] + '/squeeze')
            self.layers_map[var_name] = squeeze

            var_name = 'self.excitation_' + str(idx)
            excitation = tf.layers.Dense(out_dim, kernel_initializer='he_normal', use_bias=False,
                                         name='convnet/' + self.layer_params[idx][3] + '/excitation')
            self.layers_map[var_name] = excitation

    def ConstructFeatureExtraction(self):
        self.ConstructCNNModule(0)
        self.pool1 = tf.layers.MaxPooling2D(pool_size=2, strides=(2, 2), padding='valid')  # 16,?/2

        self.ConstructCNNModule(1)  # 16,?/2
        self.pool2 = tf.layers.MaxPooling2D(pool_size=2, strides=(2, 2), padding='valid')  # 8,?/4

        self.ConstructCNNModule(2)  # 8,?/4
        self.ConstructCNNModule(3)  # 8,?/4
        self.pool3 = tf.layers.MaxPooling2D(pool_size=2, strides=(2, 1), padding='valid')  # 4, ?/4-1

        self.ConstructCNNModule(4)  # 4, ?/4-1
        self.ConstructCNNModule(5)  # 4, ?/4-1
        self.pool4 = tf.layers.MaxPooling2D(pool_size=2, strides=(2, 1), padding='valid')  # 2,?/4-2

        self.ConstructCNNModule(6)  # 2,?/4-2
        #  1,?/4-3
        self.end_cnn = tf.layers.Conv2D(filters=self.layer_params[-1][0],
                                        kernel_size=2,
                                        padding='valid',
                                        activation='relu',
                                        kernel_initializer='he_normal',
                                        bias_initializer='zeros',
                                        name='convnet/end_cnn',
                                        trainable=(not self.adapt_learning))

    def SelectCNNAdapter(self, layer_idx, domain_name, adapter_idx, level='hard'):
        var_name = 'self.cnn_adapter_bn_' + str(layer_idx) + domain_name
        self.layers_map[var_name] = tf.layers.BatchNormalization(
            name=self.adapter_scope + domain_name + str(layer_idx) + '/batch_norm')
        var_name = 'self.cnn_adapter_bn_1_' + str(layer_idx) + domain_name
        self.layers_map[var_name] = tf.layers.BatchNormalization(
            name=self.adapter_scope + domain_name + str(layer_idx) + '/batch_norm1')
        var_name = 'self.cnn_adapter_bn_2_' + str(layer_idx) + domain_name
        self.layers_map[var_name] = tf.layers.BatchNormalization(
            name=self.adapter_scope + domain_name + str(layer_idx) + '/batch_norm2')
        kenel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-5)
        if level == 'easy':
            var_name = 'self.cnn_adapter_1x1_' + str(adapter_idx) + str(layer_idx) + domain_name
            cnn_1x1 = tf.layers.Conv2D(filters=self.layer_params[layer_idx][0],
                                       kernel_size=(1, 1),
                                       padding='same',
                                       activation='relu',
                                       kernel_initializer='he_normal',
                                       bias_initializer='zeros',
                                       kernel_regularizer=kenel_regularizer,
                                       name=self.adapter_scope + domain_name + str(
                                           layer_idx) + '/cnn_adapter_1x1_' + str(adapter_idx))
            self.layers_map[var_name] = cnn_1x1
            return [var_name]
        elif level == 'medium':
            var_name = 'self.cnn_adapter_2x2_' + str(adapter_idx) + str(layer_idx) + domain_name
            cnn_2x2 = tf.layers.Conv2D(filters=self.layer_params[layer_idx][0],
                                       kernel_size=(2, 2),
                                       padding='same',
                                       activation='relu',
                                       kernel_initializer='he_normal',
                                       bias_initializer='zeros',
                                       kernel_regularizer=kenel_regularizer,
                                       name=self.adapter_scope + domain_name + str(
                                           layer_idx) + '/cnn_adapter_2x2_' + str(adapter_idx))
            self.layers_map[var_name] = cnn_2x2
            return [var_name]
        elif level == 'hard':
            var_name_1 = 'self.cnn_adapter_3x1_' + str(adapter_idx) + '_1_' + str(layer_idx) + domain_name
            cnn_3x1 = tf.layers.Conv2D(filters=self.layer_params[layer_idx][0],
                                       kernel_size=(3, 1),
                                       padding='same',
                                       activation='relu',
                                       kernel_initializer='he_normal',
                                       bias_initializer='zeros',
                                       kernel_regularizer=kenel_regularizer,
                                       name=self.adapter_scope + domain_name + str(
                                           layer_idx) + '/cnn_adapter_3x1_' + str(adapter_idx) + '_1')
            self.layers_map[var_name_1] = cnn_3x1

            var_name_2 = 'self.cnn_adapter_1x3_' + str(adapter_idx) + '_2_' + str(layer_idx) + domain_name
            cnn_1x3 = tf.layers.Conv2D(filters=self.layer_params[layer_idx][0],
                                       kernel_size=(1, 3),
                                       padding='same',
                                       activation=None,
                                       kernel_initializer='he_normal',
                                       bias_initializer='zeros',
                                       kernel_regularizer=kenel_regularizer,
                                       name=self.adapter_scope + domain_name + str(
                                           layer_idx) + '/cnn_adapter_1x3_' + str(adapter_idx) + '_2')
            self.layers_map[var_name_2] = cnn_1x3
            return [var_name_1, var_name_2]
        else:
            tf.logging.error('invalid domain level')
            import pdb;
            pdb.set_trace()

    def ConstructFeatureAdaptiveLearning(self, idx, domain_name, level='hard'):
        # select adapter structure according to domain level(difficulty)
        # backbone adapters
        if 'backbone' in domain_name:
            backbone_version = domain_name.split('backbone')[1]
            self.backbone_adapter_1[backbone_version][idx] = self.SelectCNNAdapter(idx, domain_name, 1, level)
            self.backbone_adapter_2[backbone_version][idx] = self.SelectCNNAdapter(idx, domain_name, 2, level)
        elif self.domain_modules[domain_name]['feature'] == domain_name and domain_name != 'general':
            # create domain adapters
            self.adapter_1[domain_name][idx] = self.SelectCNNAdapter(idx, domain_name, 1, level)
            self.adapter_2[domain_name][idx] = self.SelectCNNAdapter(idx, domain_name, 2, level)
            self.CNN_adapt[idx][domain_name] = 1
        else:
            self.CNN_adapt[idx][domain_name] = 0

    def ConstructContextAdaptiveLearning(self, domain_name, **kwargs):
        assert ('domain_loss_type' in kwargs)
        domain_loss_type = kwargs['domain_loss_type']

        if self.domain_modules[domain_name]['context'] == domain_name:
            var_name = 'self.brnn' + domain_name + '1'
            brnn1 = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(self.rnn_size[0],
                                                                           return_sequences=True),
                                                  merge_mode='concat',
                                                  name=self.adapter_scope + domain_name + '/rnn_adapter1')
            self.layers_map[var_name] = brnn1
            var_name = 'self.brnn' + domain_name + '2'
            brnn2 = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(self.rnn_size[1],
                                                                           return_sequences=True),
                                                  merge_mode='concat',
                                                  name=self.adapter_scope + domain_name + '/rnn_adapter2')
            self.layers_map[var_name] = brnn2

        if domain_loss_type == 'attention':
            SOS_ID = kwargs['SOS_ID']
            EOS_ID = kwargs['EOS_ID']
            vocab_size = kwargs['vocab_size']
            embedd_dim = 96
            dec_units = 128
            enc_units = 128
            rnn_type = 'gru'
            max_length = 20
            attention_type = 0
            attention_name = 'luong'

            embedd_dim = kwargs['embedd_dim'] if 'embedd_dim' in kwargs else embedd_dim
            dec_units = kwargs['dec_units'] if 'dec_units' in kwargs else dec_units
            enc_units = kwargs['enc_units'] if 'enc_units' in kwargs else enc_units
            attention_name = kwargs['attention_name'] if 'attention_name' in kwargs else attention_name
            attention_type = kwargs['attention_type'] if 'attention_type' in kwargs else attention_type
            rnn_type = kwargs['rnn_type'] if 'rnn_type' in kwargs else rnn_type
            max_length = kwargs['max_length'] if 'max_length' in kwargs else max_length
            attention_inst = Decoder(vocab_size,
                                     embedd_dim,
                                     SOS_ID, EOS_ID,
                                     dec_units, enc_units,
                                     attention_name,
                                     attention_type,
                                     rnn_type,
                                     max_length,
                                     domain_name)

            name = 'attention_decode_{}'.format(domain_name)
            self.attention_decoder_map[name] = attention_inst

    def ConstructLogitsAdaptiveLearning(self, domain_name, domain_loss_type):
        if domain_loss_type == 'ctc':
            if self.domain_modules[domain_name]['logits'] == domain_name:
                var_name = 'self.linear' + domain_name
                linear = tf.layers.Dense(self.num_class_lst[domain_name],
                                         name=self.adapter_scope + domain_name + '/logits')
                self.layers_map[var_name] = linear
            else:  # use selected domain logits instead.
                # must check the equality of char_dicts
                domain_use = self.domain_modules[domain_name]['logits']
                assert self.char_dict_path_dict[domain_use] == self.char_dict_path_dict[domain_name], \
                    "Logits module reuse: char_dicts are not the same."

    def ForwardCNNModule(self, input, idx, is_training, backbone_version='1.0', domain_name=None):
        # backbone section.
        residual = self.layers_map['self.cnn_' + str(idx)](input)
        # use different BN layers for various domains, even in backbone.
        if domain_name == None:
            if backbone_version == '1.0':
                residual = self.layers_map['self.bn_' + str(idx)](residual, is_training)
            else:
                residual = self.layers_map['self.cnn_adapter_bn_' + str(idx) + 'backbone' + backbone_version](residual,
                                                                                                              is_training)
        else:
            residual = self.layers_map['self.cnn_adapter_bn_' + str(idx) + domain_name](residual, is_training)
        # first backbone residual.
        output = self.layers_map['self.cnn_' + str(idx) + '_1'](residual)
        # first adaptive section.
        if backbone_version != '1.0':
            cnn_ada = residual
            for layer_name in self.backbone_adapter_1[backbone_version][idx]:
                cnn_ada = self.layers_map[layer_name](cnn_ada)
            output += cnn_ada
        if domain_name != None:
            if self.CNN_adapt[idx][domain_name] == 1:
                cnn_ada = residual
                for layer_name in self.adapter_1[domain_name][idx]:
                    cnn_ada = self.layers_map[layer_name](cnn_ada)
                output += cnn_ada
        if domain_name == None:
            if backbone_version == '1.0':
                output = self.layers_map['self.bn_' + str(idx) + '_1'](output, is_training)
            else:
                output = self.layers_map['self.cnn_adapter_bn_1_' + str(idx) + 'backbone' + backbone_version](output,
                                                                                                              is_training)
        else:
            output = self.layers_map['self.cnn_adapter_bn_1_' + str(idx) + domain_name](output, is_training)
        output = tf.nn.relu(output)
        # second backbone residual.
        output_ = self.layers_map['self.cnn_' + str(idx) + '_2'](output)
        # second adaptive section.
        if backbone_version != '1.0':
            cnn_ada = output
            for layer_name in self.backbone_adapter_2[backbone_version][idx]:
                cnn_ada = self.layers_map[layer_name](cnn_ada)
            output_ += cnn_ada
        if domain_name != None:
            if self.CNN_adapt[idx][domain_name] == 1:
                cnn_ada = output
                for layer_name in self.adapter_2[domain_name][idx]:
                    cnn_ada = self.layers_map[layer_name](cnn_ada)
                output_ += cnn_ada
        if domain_name == None:
            if backbone_version == '1.0':
                output = self.layers_map['self.bn_' + str(idx) + '_2'](output_, is_training)
            else:
                output = self.layers_map['self.cnn_adapter_bn_2_' + str(idx) + 'backbone' + backbone_version](output_,
                                                                                                              is_training)
        else:
            output = self.layers_map['self.cnn_adapter_bn_2_' + str(idx) + domain_name](output_, is_training)

        # for squeeze & excitation
        if self.squeeze_excitation:
            out_dim = self.layer_params[idx][0]
            squeeze = tf.keras.layers.GlobalAveragePooling2D()(output)
            excitation = self.layers_map['self.squeeze_' + str(idx)](squeeze)
            excitation = tf.nn.relu(excitation)
            excitation = self.layers_map['self.excitation_' + str(idx)](excitation)
            excitation = tf.math.sigmoid(excitation)
            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            output = output * excitation

        output += residual
        output = tf.nn.relu(output)
        return output

    def ForwardCRNN(self, img, domain_name, is_training):
        cnn_domain = self.domain_modules[domain_name]['feature']
        bb_version = self.backbone_version_dict[domain_name] if 'backbone' not in cnn_domain else \
        cnn_domain.split('backbone')[1]
        cnn_domain = cnn_domain if (cnn_domain != 'general' and 'backbone' not in cnn_domain) else None
        result = self.ForwardCNNModule(img, 0, is_training, bb_version, cnn_domain)
        result = self.pool1(result)
        result = self.ForwardCNNModule(result, 1, is_training, bb_version, cnn_domain)
        result = self.pool2(result)
        result = self.ForwardCNNModule(result, 2, is_training, bb_version, cnn_domain)
        result = self.ForwardCNNModule(result, 3, is_training, bb_version, cnn_domain)
        result = self.pool3(result)
        result = self.ForwardCNNModule(result, 4, is_training, bb_version, cnn_domain)
        result = self.ForwardCNNModule(result, 5, is_training, bb_version, cnn_domain)
        result = self.pool4(result)
        result = self.ForwardCNNModule(result, 6, is_training, bb_version, cnn_domain)
        result = self.end_cnn(result)
        result = tf.squeeze(result, axis=1)
        # Calc the total timestep number.
        ones = tf.ones([img.shape[0]], tf.int32)
        fea_len = tf.multiply(tf.shape(result)[1], ones)

        domain_loss_type = self.domain_losstype_dict[domain_name]
        # context learning modules.
        # use domain_modules['context'] to determine which RNN adapter to use.
        rnn_domain = self.domain_modules[domain_name]['context']
        result = self.layers_map['self.brnn' + rnn_domain + '1'](result)
        result = self.layers_map['self.brnn' + rnn_domain + '2'](result)
        logits_domain = self.domain_modules[domain_name]['context']
        if domain_loss_type == 'ctc':
            logit = self.layers_map['self.linear' + logits_domain](result)
            return logit, fea_len,
        elif domain_loss_type == 'attention':
            return result, fea_len

    def call(self, input, is_training=True, is_dummy=False):
        """Run the model and get the loss.
        Args:
            input: (dict): Input dict from dataset
            is_training: Flag to determine whether is in training phase
            is_dummy: Flag to determine whether is in dummy forwarding(for initialization)
        """
        img = input[1]
        label_dense = input[2]
        domain_idx = input[3]
        resize_widths = input[4]
        batch_size = img.shape[0]
        d_idx = (tf.reduce_sum(domain_idx) / batch_size).numpy()
        if self.train_domain_idx >= 0:
            if d_idx != self.train_domain_idx:
                if self.data_transfer:
                    d_idx = self.train_domain_idx
                else:
                    tf.logging.info('domain error: domain_idx of training data is diff from train_domain')
                    import pdb;
                    pdb.set_trace()
                    # specific domain
        if d_idx < len(self.domain_name_lst):
            if isinstance(domain_idx, list):
                if len(set(domain_idx)) == 1:
                    domain_name = self.domain_name_lst[d_idx]
            else:
                if domain_idx.shape[0] == 1:  # Only one sample
                    domain_name = self.domain_name_lst[d_idx]
                elif len(set(tf.squeeze(domain_idx).numpy().tolist())) == 1:
                    domain_name = self.domain_name_lst[d_idx]
        else:  # undefined domain or not all inputs are from the same domain
            tf.logging.info('domain error: undefined domain or not all inputs are from the same domain')
            import pdb;
            pdb.set_trace()

        if is_dummy:
            is_training = False
            for name in self.domain_name_lst:
                # feature extaction modules.
                domain_loss_type = self.domain_losstype_dict[name]
                logit, fea_len = self.ForwardCRNN(img, name, is_training)
                if domain_loss_type == 'attention':
                    after_pool1 = tf.floordiv(resize_widths, self.two)
                    after_pool2 = tf.floordiv(after_pool1, self.two)
                    after_conv7 = after_pool2 - 3
                    sos_id = self.attention_sosid_map[domain_name]
                    eos_id = self.attention_eosid_map[domain_name]
                    name = 'attention_decode_{}'.format(domain_name)
                    attention_inst = self.attention_decoder_map[name]
                    target_mask = tf.equal(label_dense, -1 * tf.ones_like(label_dense, dtype=label_dense.dtype))
                    target_input = tf.where(target_mask, eos_id * tf.ones_like(label_dense, dtype=label_dense.dtype),
                                            label_dense)
                    fw_status = None
                    features = logit
                    teacher_forcing_ratio = 0.5
                    weight_mask = tf.sequence_mask(lengths=tf.squeeze(after_conv7), maxlen=logit.shape[1])
                    label_sparse = dense_to_sparse(label_dense, eos_id)
                    attention_inst(target_input, fw_status, features,
                                   teacher_forcing_ratio, weight_mask, is_training)
            # is_training = False
            # for name in self.domain_name_lst:
            #    # feature extaction modules.
            #    logit, fea_len = self.ForwardCRNN(img, name, is_training)
            return {}
        else:
            domain_loss_type = self.domain_losstype_dict[domain_name]
            # feature extaction modules.
            logit, fea_len = self.ForwardCRNN(img, domain_name, is_training)

            # Calc the valid width(seq_len).
            after_pool1 = tf.floordiv(resize_widths, self.two)
            after_pool2 = tf.floordiv(after_pool1, self.two)
            after_conv7 = after_pool2 - 3

            if domain_loss_type == 'ctc':
                softmax_out = tf.reshape(logit, [-1, tf.shape(logit)[2]])
                softmax_out = tf.nn.softmax(softmax_out)
                softmax_out = tf.reshape(softmax_out, tf.shape(logit))
                label_sparse = dense_to_sparse(label_dense, self.blank_class)
                ctc_loss, ctc_pred, ctc_prob = ctc_loss_layer(logit, label_sparse, fea_len, time_major=False)
                # Convert sparse pred to dense.
                seq_pred_sparse = tf.cast(ctc_pred[0], tf.int32)
                seq_pred_dense = tf.sparse_tensor_to_dense(seq_pred_sparse, default_value=self.blank_class)

                # output end_points
                end_points = {}
                end_points["img_path"] = input[0]
                end_points["img"] = input[1]
                end_points["softmax"] = softmax_out
                end_points["rnn_logits"] = logit
                end_points["loss"] = ctc_loss
                end_points["sparse_label"] = label_sparse
                end_points["sparse_pred"] = seq_pred_sparse
                end_points["dense_pred"] = seq_pred_dense
                end_points["ctc_prob"] = ctc_prob
                end_points['seq_len'] = after_conv7
                end_points['fea_len'] = fea_len

            elif domain_loss_type == 'attention':
                sos_id = self.attention_sosid_map[domain_name]
                eos_id = self.attention_eosid_map[domain_name]
                name = 'attention_decode_{}'.format(domain_name)
                attention_inst = self.attention_decoder_map[name]
                target_mask = tf.equal(label_dense, -1 * tf.ones_like(label_dense, dtype=label_dense.dtype))
                target_input = tf.where(target_mask, eos_id * tf.ones_like(label_dense, dtype=label_dense.dtype),
                                        label_dense)
                fw_status = None
                features = logit
                teacher_forcing_ratio = 0.5
                weight_mask = tf.sequence_mask(lengths=tf.squeeze(after_conv7), maxlen=logit.shape[1])
                label_sparse = dense_to_sparse(label_dense, eos_id)
                loss_value, decoder_outputs, decoder_hidden, decoder_dict = attention_inst(target_input,
                                                                                           fw_status,
                                                                                           features,
                                                                                           teacher_forcing_ratio,
                                                                                           weight_mask,
                                                                                           is_training)

                decoder_argmaxs = decoder_dict['decoder_argmaxs']
                decoder_outputs = decoder_dict['decoder_outputs']
                attention_prob = softmax_to_seqprob(decoder_outputs, eos_id)
                attention_sparse_pred = argmax_to_tensor(decoder_argmaxs, decoder_outputs, eos_id)

                attention_sparse_pred = tf.cast(attention_sparse_pred, tf.int32)
                attention_dense_pred = tf.sparse_tensor_to_dense(attention_sparse_pred, default_value=eos_id)

                softmax_out = tf.concat([tf.expand_dims(x, axis=1) for x in decoder_dict['decoder_outputs']], axis=1)
                argsmax_out = tf.concat([tf.expand_dims(x, axis=1) for x in decoder_dict['decoder_argmaxs']], axis=1)
                end_points = {}
                end_points["img_path"] = input[0]
                end_points["img"] = input[1]
                end_points["softmax"] = softmax_out
                end_points["rnn_logits"] = logit
                end_points["loss"] = loss_value
                end_points["sparse_label"] = label_sparse
                end_points["sparse_pred"] = attention_sparse_pred
                end_points["dense_pred"] = attention_dense_pred
                end_points["ctc_prob"] = attention_prob
                end_points['seq_len'] = after_conv7
                end_points['fea_len'] = fea_len
                end_points['SOS_ID'] = sos_id
                end_points['EOS_ID'] = eos_id

            return end_points

    """
    Fllowing funcs are for online inference
    """

    def domainInference(self, domain_idx, domain_set,
                        use_beamsearch=False, beam_width=30, top_paths=1,
                        thread_idx=0):
        thread = threading.current_thread()
        start = datetime.datetime.now()
        logging.info('current thread {} start {}'.format(thread.name, start))
        assert domain_idx < len(self.domain_name_lst), "domainInference ERR: domain index out of range."
        img, resize_widths, _ = domain_set
        domain_name = self.domain_name_lst[domain_idx]
        is_training = False
        logit, fea_len = self.ForwardCRNN(img, domain_name, is_training)
        domain_loss_type = self.domain_losstype_dict[domain_name]

        # Calc the valid width(seq_len), duplicate.
        after_pool1 = tf.floordiv(resize_widths, self.two)
        after_pool2 = tf.floordiv(after_pool1, self.two)
        after_conv7 = after_pool2 - 3

        if domain_loss_type == 'ctc':
            softmax_out = tf.reshape(logit, [-1, tf.shape(logit)[2]])
            softmax_out = tf.nn.softmax(softmax_out)
            softmax_out = tf.reshape(softmax_out, tf.shape(logit))
            if use_beamsearch:
                [ctc_pred, ctc_prob] = tf.nn.ctc_beam_search_decoder(tf.transpose(logit, [1, 0, 2]), fea_len,
                                                                     beam_width=beam_width, top_paths=top_paths,
                                                                     merge_repeated=False)
                # ctc_pred, ctc_prob = ctc_beam_search_decoder(tf.transpose(softmax_out, [1, 0, 2]).numpy(), None, beam_width)
                # # merge pred sequence.
                # ctc_pred = ctc_merge(ctc_pred.numpy(), tf.math.reduce_max(ctc_pred).numpy())
            else:
                [ctc_pred, ctc_prob] = tf.nn.ctc_greedy_decoder(tf.transpose(logit, [1, 0, 2]), fea_len)

            # calc the geometric mean prob.
            ctc_prob = np.power(tf.exp(ctc_prob).numpy(), float(1) / logit.shape[1].value)
            # Convert sparse pred to dense.
            seq_preds = []
            for idx, ctc_p in enumerate(ctc_pred):
                seq_pred_sparse = tf.cast(ctc_p, tf.int32)
                seq_pred_dense = tf.sparse_tensor_to_dense(seq_pred_sparse, default_value=self.blank_class)
                seq_preds.append(seq_pred_dense)
            end = datetime.datetime.now()
            logging.info('current thread {} end {} takes {}'.format(thread.name, end, (end - start).total_seconds()))
            return seq_preds, ctc_prob, logit, softmax_out, fea_len, thread_idx
        elif domain_loss_type == 'attention':
            name = 'attention_decode_{}'.format(domain_name)
            attention_inst = self.attention_decoder_map[name]
            eos_id = self.attention_eosid_map[domain_name]
            target_input = None
            fw_status = None
            features = logit
            teacher_forcing_ratio = 0.0
            weight_mask = tf.sequence_mask(lengths=tf.squeeze(after_conv7), maxlen=logit.shape[1])
            loss_value, decoder_outputs, decoder_hidden, decoder_dict = attention_inst(target_input,
                                                                                       fw_status,
                                                                                       features,
                                                                                       teacher_forcing_ratio,
                                                                                       weight_mask,
                                                                                       False)

            decoder_argmaxs = decoder_dict['decoder_argmaxs']
            decoder_outputs = decoder_dict['decoder_outputs']
            # attention_prob = softmax_to_seqprob(decoder_outputs, eos_id)
            attention_sparse_pred = argmax_to_tensor(decoder_argmaxs, decoder_outputs, eos_id)
            attention_sparse_pred = tf.cast(attention_sparse_pred, tf.int32)
            attention_dense_pred = tf.sparse_tensor_to_dense(attention_sparse_pred, default_value=eos_id)
            softmax_out = tf.concat([tf.expand_dims(x, axis=1) for x in decoder_dict['decoder_outputs']], axis=1)

            return attention_dense_pred, logit, softmax_out, fea_len, thread_idx

    @staticmethod
    def GetMinWidth():
        return 16