# _*_ coding:utf-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import cv2
import sys
import time
import shutil
import numpy as np
import tensorflow as tf
import Levenshtein

tf.enable_eager_execution()

if sys.version.startswith('2'):
    reload(sys)
    sys.setdefaultencoding('utf-8')

abspath = os.path.dirname(os.path.realpath(__file__))
abspath = os.path.abspath(abspath)

sys.path.append(abspath)
sys.path.append(os.path.join(abspath, '.'))
sys.path.append(os.path.join(abspath, '../utils'))

from Logger import logger
from Seq2Seq import Seq2Seq
from TensorUtils import *

from Charset import Charset
from Dataset import FileDataset
from Dataset import RecordDataset


class Solover:
    def __init__(self, config):
        self.restore_ckpt = None
        if 'restore_ckpt' in config:
            self.restore_ckpt = config['restore_ckpt']

        self.eval_interval = 1000
        self.save_interval = 10
        self.show_interval = 100
        self.train_epoch = 100
        self.max_iter = 1000000

        #train dataset parse
        self.train_dataset_type = config['train_dataset_type']
        self.train_dataset = None
        if self.train_dataset_type == 'tfrecord':
            self.train_dataset = RecordDataset
        elif self.train_dataset_type == 'filelist':
            self.train_dataset = FileDataset

        train_ds_config = dict()
        train_ds_config['norm_h'] = int(config['norm_h'])
        train_ds_config['expand_rate'] = float(config['expand_rate'])
        train_ds_config['file_list'] = config['train_file_list']
        train_ds_config['num_parallel'] = config['num_parallel']
        train_ds_config['batch_size'] = config['batch_size']
        train_ds_config['model_type'] = config['model_type']
        train_ds_config['char_dict'] = config['char_dict']
        train_ds_config['mode'] = 'train'

        self.train_dataset = self.train_dataset(train_ds_config).data_reader_ctc_attention(self.train_epoch)

        #eval dataset parse
        self.eval_dataset_type = config['eval_dataset_type']
        self.eval_dataset = None
        if self.eval_dataset_type == 'tfrecord':
            self.eval_dataset = RecordDataset
        elif self.eval_dataset_type == 'filelist':
            self.eval_dataset = FileDataset

        eval_ds_config = dict()

        eval_ds_config['norm_h'] = int(config['norm_h'])
        eval_ds_config['expand_rate'] = float(config['expand_rate'])
        eval_ds_config['file_list'] = config['eval_file_list']
        eval_ds_config['num_parallel'] = config['num_parallel']
        eval_ds_config['batch_size'] = config['batch_size']
        eval_ds_config['char_dict'] = config['char_dict']
        eval_ds_config['model_type'] = config['model_type']
        eval_ds_config['mode'] = 'test'
        self.eval_dataset = self.eval_dataset(eval_ds_config).data_reader_ctc_attention()
        
        #charset parse
        self.char_dict = config['char_dict']
        self.model_type = config['model_type']
        self.charset = Charset(self.char_dict, self.model_type)
        self.step_counter = tf.train.get_or_create_global_step()

        self.decoder_dict = {}
        self.loss_value = 0.0

        self.learning_rate = 0.01
        if 'learning_rate' in config:
            self.learning_rate = float(config['learning_rate'])

        self.learning_rate_decay = None
        if 'learning_rate_decay' in config:
            self.learning_rate_decay = config['learning_rate_decay']

        if self.learning_rate_decay == 'piecewise_constant':
            boundaries = [30000, 60000]
            values = [self.learning_rate, self.learning_rate*0.5, self.learning_rate*0.2]
            self.learning_rate = tf.train.piecewise_constant(self.step_counter, boundaries, values)
        elif self.learning_rate_decay == 'exponential_decay':
            decay_rate = 0.8
            decay_steps = 40000
            self.learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                            self.step_counter,
                                                            decay_steps,
                                                            decay_rate)
        elif self.learning_rate_decay == 'linear_cosine_decay':
            decay_steps = 30000
            self.learning_rate = tf.train.linear_cosine_decay(self.learning_rate,
                                                              self.step_counter,
                                                              decay_steps)

        if 'eval_interval' in config:
            self.eval_interval = int(config['eval_interval'])
        if 'save_interval' in config:
            self.save_interval = int(config['save_interval'])
        if 'train_epoch' in config:
            self.train_epoch = int(config['train_epoch'])
        if 'max_iter' in config:
            self.max_iter = int(config['max_iter'])

        self.max_dec_length = 20
        self.eos_id = self.charset.get_eosid()
        self.sos_id = self.charset.get_sosid()
        self.vocab_size = self.charset.get_size()
        self.dec_num_layers = 1
        self.d_model = 512
        self.dec_num_heads = 4
        self.dec_dff = 1024
        self.dec_rate = 0.0

        self.seq2seq = Seq2Seq(dec_num_layers=self.dec_num_layers,
                               d_model=self.d_model,
                               vocab_size=self.vocab_size,
                               dec_num_heads=self.dec_num_heads,
                               dec_dff=self.dec_dff,
                               sos_id=self.sos_id,
                               eos_id=self.eos_id,
                               max_dec_length=self.max_dec_length,
                               dec_rate=self.dec_rate)

        self.optimizer_type = 'adam'
        if 'optimizer_type' in config:
            self.optimizer_type = config['optimizer_type']
        if self.optimizer_type not in ('sgd', 'momentum', 'adam'):
            print("Solover Error: optimizer_type {} not in [sgd, momentum, adam]".format(self.optimizer_type))

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        if self.optimizer_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.optimizer_type == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.95)
        elif self.optimizer_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.checkpoint_dir = 'training_checkpoints'
        if 'checkpoint_dir' in config:
            self.checkpoint_dir = config['checkpoint_dir']
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              seq2seq=self.seq2seq,
                                              global_step=self.step_counter)

        if self.restore_ckpt is not None and os.path.isdir(self.restore_ckpt):
            if tf.train.latest_checkpoint(self.restore_ckpt) is not None:
                self.checkpoint.restore(tf.train.latest_checkpoint(self.restore_ckpt))
        else:
            if tf.train.latest_checkpoint(self.checkpoint_dir) is not None:
                self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def argmax_text(self, argmax_output):
        # decoder_output: tensor --> batch x steps
        assert(isinstance(argmax_output, np.ndarray))
        batch = argmax_output.shape[0]
        argmax_texts = []
        for b_id in range(batch):
            ids = []
            for id in argmax_output[b_id]:
                if int(id) == self.sos_id:
                    continue
                if int(id) == self.eos_id:
                    break
                ids.append(int(id))
            argmax_texts.append(self.charset.get_charstr_by_idxlist(ids))
        return argmax_texts

    def decoder_text(self, decoder_output, is_argmax=False):
        # decoder_output: tensor --> batch x steps if is_argmax is True
        # decoder_output: tensor --> batch x steps x vocb_size if is_argmax is False
        batch = int(decoder_output.shape[0])
        if is_argmax:
            decoder_argmax = decoder_output
        else:
            decoder_argmax = tf.argmax(decoder_output, axis=-1, output_type=tf.int32)

        decoder_argmax = decoder_argmax.numpy()
        infer_texts = self.argmax_text(decoder_argmax)
        return infer_texts

    def att_seqacc(self, label_input, decoder_output, show_flag=False, is_argmax=False):
        #label_input: tensor -->batch x steps
        # decoder_output: tensor --> batch x steps if is_argmax is True
        # decoder_output: tensor --> batch x steps x vocb_size if is_argmax is False
        if is_argmax:
            decoder_argmax = decoder_output
        else:
            decoder_argmax = tf.argmax(decoder_output, axis=-1, output_type=tf.int32)

        decoder_argmax = decoder_argmax.numpy()
        label_input = label_input.numpy()

        label_ids = self.argmax_text(label_input)
        infer_ids = self.argmax_text(decoder_argmax)

        batch_cnt = 0.0
        batch_cor = 0.0
        for label, infer in zip(label_ids, infer_ids):
            batch_cnt = batch_cnt + 1
            if label == infer:
                batch_cor = batch_cor + 1

        if show_flag:
            for label, infer in zip(label_ids, infer_ids):
                logger.info("label: {:^32}, infer: {:^32}".format(label, infer))

            logger.info("corr_rate: {:^32}, corr count: {:^32}, batch count: {:^32}".format(
                batch_cor/batch_cnt, batch_cor, batch_cnt))

        return batch_cnt, batch_cor

    def ctc_seqacc(self, label_input, decoder_output, show_flag=False):
        #label_input: sparse_tensor
        #decoder_output: dense tensor
        batch = tf.shape(decoder_output)[0]
        label_dense = tf.sparse_to_dense(label_input.indices,
                                         [batch, self.vocab_size-1],
                                         label_input.values,
                                         default_value=self.eos_id)

        decoder_argmax = tf.argmax(decoder_output, axis=-1, output_type=tf.int32)
        decoder_softmax = decoder_output

        label_input = label_dense.numpy()
        decoder_argmax = decoder_argmax.numpy()
        decoder_softmax = decoder_softmax.numpy()

        label_ids = []
        for b_id in range(batch):
            ids = []
            for id in label_input[b_id]:
                if int(id) == self.sos_id:
                    continue
                if int(id) == self.eos_id:
                    break
                ids.append(int(id))
            label_ids.append(self.charset.get_charstr_by_idxlist(ids))

        infer_ids = []
        infer_pbs = []
        for b in range(batch):
            b_infer_id = decoder_argmax[b]
            b_infer_pb = decoder_softmax[b]
            pre_id = b_infer_id[0]
            pre_pb = b_infer_pb[0][pre_id]

            ids = []
            pbs = []
            if pre_id != self.eos_id:
                ids.append(pre_id)
                pbs.append(pre_pb)

            for index, cur_id in enumerate(b_infer_id[1:], 1):
                cur_pb = b_infer_pb[index][cur_id]
                if cur_id == self.eos_id:
                    pre_id = cur_id
                    pre_pb = cur_pb
                    continue

                if cur_id == pre_id and cur_pb < pre_pb:
                    pbs[-1] = cur_pb
                    pre_id = cur_id
                    pre_pb = cur_pb
                    continue

                if cur_id != pre_id:
                    ids.append(cur_id)
                    pbs.append(cur_pb)
                    pre_id = cur_id
                    pre_pb = cur_pb
                    continue

            if len(ids) > 0:
                infer_ids.append(self.charset.get_charstr_by_idxlist(ids))
                infer_pbs.append(min(pbs))
            else:
                infer_ids.append('')
                infer_pbs.append(0.0)

        batch_cnt = 0.0
        batch_cor = 0.0
        for label, infer in zip(label_ids, infer_ids):
            batch_cnt = batch_cnt + 1
            if label == infer:
                batch_cor = batch_cor + 1

        if show_flag:
            for label, infer in zip(label_ids, infer_ids):
                logger.info("ctc_seqacc label: {:^32}, infer: {:^32}".format(label, infer))

            logger.info("ctc_seqacc corr_rate: {:^32}, corr count: {:^32}, batch count: {:^32}".format(
                batch_cor/batch_cnt, batch_cor, batch_cnt))

        return batch_cnt, batch_cor

    def train(self, training=True):
        iter_count = 0
        for epoch in range(self.train_epoch):
            logger.info("run epoch {}".format(epoch))
            data_train = None
            for batch, data in enumerate(self.train_dataset):
                if data_train is None:
                    data_train = data
                iter_count = iter_count + 1
                img_path, norm_img, img_text, ctc_idx, ctc_len, att_idx, att_len, coord, norm_w = data_train
                att_dense_label = string2dense(att_idx, [img_path.shape[0], self.max_dec_length], self.eos_id)
                ctc_sparse_label = string2sparse(ctc_idx, [img_path.shape[0], tf.reduce_max(ctc_len)])

                logger.debug("norm_img shape: {}".format(norm_img.shape))
                logger.debug("norm_img texts: {}".format(img_text.numpy()))
                logger.debug("ctc_idx: {}".format(ctc_idx.numpy()))
                logger.debug("ctc_len: {}".format(ctc_len.numpy()))
                logger.debug("att_idx: {}".format(att_idx.numpy()))
                logger.debug("att_len: {}".format(att_len.numpy()))

                logger.debug('att_lbl shape: {}'.format(att_dense_label.shape))
                logger.debug('ctc_lbl shape: {}'.format(ctc_sparse_label.shape))

                with tf.GradientTape() as tape:
                    att_output, ctc_output, attention, loss = self.seq2seq(norm_img,
                                                                           norm_w,
                                                                           att_dense_label,
                                                                           ctc_sparse_label,
                                                                           training)

                    logger.info("loss: {}".format(loss))

                    if iter_count % 100 == 0:
                        self.att_seqacc(att_dense_label, att_output, True)
                        self.ctc_seqacc(ctc_sparse_label, ctc_output, True)

                variables = self.seq2seq.variables
                gradients = tape.gradient(loss, variables)
                self.optimizer.apply_gradients(zip(gradients, variables), global_step=self.step_counter)

                #self.test()

                if iter_count % self.save_interval == 0:
                    self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def test(self):
        att_total_cnt = 0.0000001
        att_total_cor = 0.0000000
        ctc_total_cnt = 0.0000001
        ctc_total_cor = 0.0000000

        iter_count = 0
        for batch, data in enumerate(self.eval_dataset):
            print("run test batch {}".format(batch))
            try:
                iter_count = iter_count + 1
                img_path, norm_img, img_text, ctc_idx, ctc_len, att_idx, att_len, coord, norm_w = data
                att_dense_label = string2dense(att_idx, [img_path.shape[0], self.max_dec_length], self.eos_id)
                ctc_sparse_label = string2sparse(ctc_idx, [img_path.shape[0], tf.reduce_max(ctc_len)])

                ctc_output, ctc_argmax = self.seq2seq.ctc_evaluate(norm_img, input_widths=norm_w)
                att_output, probility, att_weights = self.seq2seq.att_evaluate(norm_img, norm_w, self.max_dec_length)
                show_flag = False
                if iter_count % 4 == 0:
                    show_flag = True
                att_batch_cnt, att_batch_cor = self.att_seqacc(att_dense_label, att_output, show_flag, True)
                att_total_cnt = att_total_cnt + att_batch_cnt
                att_total_cor = att_total_cor + att_batch_cor

                ctc_batch_cnt, ctc_batch_cor = self.ctc_seqacc(ctc_sparse_label, ctc_output, show_flag)
                ctc_total_cnt = ctc_total_cnt + ctc_batch_cnt
                ctc_total_cor = ctc_total_cor + ctc_batch_cor

                if iter_count % 100 == 0:
                    logger.info("att test batch index:{}, corr_rate:{}, total_cor:{}, total_cnt:{}".format(
                        batch, att_total_cor/att_total_cnt, att_total_cor, att_total_cnt))

                    logger.info("ctc test batch index:{}, corr_rate:{}, total_cor:{}, total_cnt:{}".format(
                        batch, ctc_total_cor/ctc_total_cnt, ctc_total_cor, ctc_total_cnt))

            except Exception as e:
                logger.info("test exception:{}".format(e))

        logger.info("att test evalutation: corr_rate:{}, total_cor:{}, total_cnt:{}".format(
            att_total_cor / att_total_cnt, att_total_cor, att_total_cnt))
        logger.info("ctc test evalutation: corr_rate:{}, total_cor:{}, total_cnt:{}".format(
            ctc_total_cor / ctc_total_cnt, ctc_total_cor, ctc_total_cnt))


if __name__ == '__main__':
    configs = {}
    configs['train_dataset_type'] = 'tfrecord'
    configs['eval_dataset_type'] = 'tfrecord'
    configs['model_type'] = 'ctc_attention'
    configs['norm_h'] = 32
    configs['save_interval'] = 100
    configs['learning_rate'] = 0.0001
    configs['expand_rate'] = 1.0
    configs['num_parallel'] = 64
    configs['batch_size'] = 32
    configs['char_dict'] = '../seq2seq/char_dict.lst'

    configs['train_file_list'] = tf.gfile.Glob("/Users/junhuang.hj/Desktop/code_paper/code/crnn_ctc_eager/tfrecord_dir/tfrecord.list.*")
    configs['eval_file_list'] = tf.gfile.Glob("/Users/junhuang.hj/Desktop/code_paper/code/crnn_ctc_eager/tfrecord_dir/tfrecord.list.*")

    print("train_file_list:", configs['train_file_list'])
    print("eval_file_list:", configs['eval_file_list'])
    slv_class = Solover(configs)
    slv_class.train()
    ttf_file = '/Users/junhuang.hj/Desktop/code_paper/code/data_gene/fonts/chinas/simhei.ttf'
    #slv_class.visualize_tfrecord(ttf_file, encoder_feats_height=2)


