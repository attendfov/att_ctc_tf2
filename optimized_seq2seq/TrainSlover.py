# _*_ coding:utf-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import cv2
import sys
import numpy as np
import tensorflow as tf


if sys.version.startswith('2'):
    reload(sys)
    sys.setdefaultencoding('utf-8')
abspath = os.path.dirname(os.path.realpath(__file__))
abspath = os.path.abspath(abspath)

sys.path.append(abspath)
sys.path.append(os.path.join(abspath, '.'))
sys.path.append(os.path.join(abspath, '../utils'))

from Logger import logger
from Logger import time_func
from Seq2Seq import Seq2Seq
from Charset import Charset
from Dataset import FileDataset
from Dataset import RecordDataset
from ShowUtils import show_attention_image


class Solover:
    def __init__(self, config):
        self.restore_ckpt = None
        if 'restore_ckpt' in config:
            self.restore_ckpt = config['restore_ckpt']

        self.eval_interval = 1000
        self.save_interval = 1000
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

        train_ds_config = {}
        train_ds_config['norm_h']         =      int(config['norm_h'])
        train_ds_config['expand_rate']    =      float(config['expand_rate'])
        train_ds_config['file_list']      =      config['train_file_list']
        train_ds_config['num_parallel']   =      config['num_parallel']
        train_ds_config['batch_size']     =      config['batch_size']
        train_ds_config['char_dict']      =      config['char_dict']
        train_ds_config['model_type']     =      config['model_type']
        train_ds_config['mode'] = 'train'

        self.train_dataset = self.train_dataset(train_ds_config).data_reader(self.train_epoch)

        #eval dataset parse
        self.eval_dataset_type = config['eval_dataset_type']
        self.eval_dataset = None
        if self.eval_dataset_type == 'tfrecord':
            self.eval_dataset = RecordDataset
        elif self.eval_dataset_type == 'filelist':
            self.eval_dataset = FileDataset
        eval_ds_config = {}

        eval_ds_config['norm_h']         =      int(config['norm_h'])
        eval_ds_config['expand_rate']    =      float(config['expand_rate'])
        eval_ds_config['file_list']      =      config['eval_file_list']
        eval_ds_config['num_parallel']   =      config['num_parallel']
        eval_ds_config['batch_size']     =      config['batch_size']
        eval_ds_config['char_dict']      =      config['char_dict']
        eval_ds_config['model_type']     =      config['model_type']
        eval_ds_config['mode'] = 'test'
        self.eval_dataset = self.eval_dataset(eval_ds_config).data_reader()
        
        #charset parse
        self.char_dict = config['char_dict']
        self.charset = Charset(self.char_dict)
        self.step_counter = tf.Variable(tf.constant(0), trainable=False, name='step_counter')

        self.decoder_dict = {}
        self.loss_value = 0.0

        self.learning_rate = 0.01
        if 'learning_rate' in config:
            self.learning_rate = float(config['learning_rate'])

        self.learning_rate_decay = None
        if 'learning_rate_decay' in config:
            self.learning_rate_decay = config['learning_rate_decay']

        '''
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
        '''

        if 'eval_interval' in config:
            self.eval_interval = int(config['eval_interval'])
        if 'save_interval' in config:
            self.save_interval = int(config['save_interval'])
        if 'train_epoch' in config:
            self.train_epoch = int(config['train_epoch'])
        if 'max_iter' in config:
            self.max_iter = int(config['max_iter'])

        self.max_padding = 20
        self.eos_id = self.charset.get_eosid()
        self.sos_id = self.charset.get_sosid()
        self.vocab_size = self.charset.get_size()
        self.embedding_dim = 96
        self.enc_units = 128
        self.dec_units = 128
        self.seq2seq = Seq2Seq(
                         vocab_size=self.vocab_size,
                         embedding_dim=self.embedding_dim,
                         SOS_ID=self.sos_id,
                         EOS_ID=self.eos_id,
                         dec_units=self.dec_units,
                         enc_units=self.enc_units,
                         attention_name='luong',
                         attention_type=0,
                         rnn_type='gru',
                         max_length=self.max_padding)

        self.optimizer_type = 'sgd'
        if 'optimizer_type' in config:
            self.optimizer_type = config['optimizer_type']
        if self.optimizer_type not in ('sgd', 'momentum', 'adam'):
            print("Solover Error: optimizer_type {} not in [sgd, momentum, adam]".format(self.optimizer_type))

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        if self.optimizer_type == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(self.learning_rate)
        elif self.optimizer_type == 'momentum':
            self.optimizer = tf.keras.optimizers.SGD(self.learning_rate, 0.95)
        elif self.optimizer_type == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.checkpoint_dir = 'training_checkpoints'
        if 'checkpoint_dir' in config:
            self.checkpoint_dir = config['checkpoint_dir']
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.seq2seq)

        if self.restore_ckpt is not None and os.path.isdir(self.restore_ckpt):
            if tf.train.latest_checkpoint(self.restore_ckpt) is not None:
                self.checkpoint.restore(tf.train.latest_checkpoint(self.restore_ckpt))
        else:
            if tf.train.latest_checkpoint(self.checkpoint_dir) is not None:
                self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def visualize_attention(self,
                            norm_img,
                            norm_w,
                            label_dense,
                            image_path,
                            ttf_file,
                            save_dir='default_show',
                            show_error=False,
                            attention_thr=0.3):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        save_file = os.path.join(save_dir, 'infer.list')
        writer = io.open(save_file, 'w', encoding='utf-8')
        batch = int(label_dense.shape[0])
        decoder_dict, _ = self.seq2seq(norm_img, norm_w, label_dense, False)
        decoder_dict = dict(decoder_dict)
        image_data = norm_img.numpy()
        image_path = image_path.numpy()
        label_input = label_dense.numpy()

        decoder_attentions = decoder_dict['decoder_attentions']
        decoder_output = decoder_dict['decoder_argmaxs']
        decoder_output = [x.numpy().tolist() for x in decoder_output]
        decoder_attentions = [x.numpy().tolist() for x in decoder_attentions]
        decoder_output = np.array(decoder_output).transpose((1, 0))
        decoder_attentions = np.array(decoder_attentions).transpose((1, 0, 2))

        logger.info("attentions: {}, decoder_output: {}".format(decoder_attentions.shape, decoder_output.shape))
        mean = [101, 114, 121]
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

        width_lst = []
        infer_ids = []
        for b_id in range(batch):
            ids = []
            lens = []
            for ix, id in enumerate(decoder_output[b_id]):
                attention = decoder_attentions[b_id][ix]
                if int(id) == self.sos_id:
                    continue
                if int(id) == self.eos_id:
                    break
                ids.append(int(id))
                lens.append([idx for idx, prb in enumerate(attention) if prb>attention_thr])
            width_lst.append(lens)
            infer_ids.append(self.charset.get_charstr_by_idxlist(ids))

            for label, infer, att_lens, img_path, img_data in zip(label_ids, infer_ids, width_lst, image_path, image_data):
                img_path = img_path.decode('utf-8')
                writer.write(img_path + ' ' + infer + ' ' + label + '\n')
                if show_error and label == infer:
                    continue

                att_vers_lens = []
                for att_len in att_lens:
                    att_len = self.seq2seq.encoder.get_reverse_lengths(att_len)
                    att_len_squeeze = []
                    for lens in att_len:
                        att_len_squeeze.extend(lens)
                    att_vers_lens.append(att_len_squeeze)

                img_name = os.path.basename(img_path)
                save_name = os.path.join(save_dir, img_name)
                img_data = img_data + mean

                cv2.imwrite(save_name, img_data, [cv2.IMWRITE_JPEG_QUALITY, 100])
                img_data = show_attention_image(save_name, ttf_file, label, infer, att_vers_lens, imgw_scale=2)
                cv2.imwrite(save_name, img_data, [cv2.IMWRITE_JPEG_QUALITY, 100])
        writer.close()

    def visualize_tfrecord(self,
                           ttf_file,
                           save_dir='default_show',
                           show_error=False,
                           attention_thr=0.3):

        for batch, data in enumerate(self.eval_dataset):
            #try:
                img_path, norm_img, img_text, txt_index, txt_len, coord, norm_w = data
                label_sparse = tf.string_split(txt_index, ',')
                label_indices = label_sparse.indices
                label_values = label_sparse.values
                label_values = tf.string_to_number(label_values, out_type=tf.int32)
                output_shape = [img_path.shape[0], self.max_padding]
                label_dense = tf.sparse_to_dense(label_indices, output_shape, label_values, self.charset.get_eosid())
                self.visualize_attention(norm_img,
                                         norm_w,
                                         label_dense,
                                         img_path,
                                         ttf_file,
                                         save_dir,
                                         show_error,
                                         attention_thr)
            #except Exception as e:
            #    logger.info('Exception {}'.format(str(e)))

    def calc_seqacc(self, label_input, decoder_output, show_flag=False, is_argmax=False):
        #label_input: tensor -->batch x steps
        #is_argmax True  :decoder_output: tensor --> batch x steps
        #is_argmax False :decoder_output: tensor --> batch x steps x vocb_size

        batch = int(label_input.shape[0])
        if is_argmax:
            decoder_argmax = decoder_output
        else:
            decoder_argmax = tf.argmax(decoder_output, axis=-1, output_type=tf.int32)

        label_input = label_input.numpy()
        decoder_argmax = decoder_argmax.numpy()

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
        for b_id in range(batch):
            ids = []
            for id in decoder_argmax[b_id]:
                if int(id) == self.sos_id:
                    continue
                if int(id) == self.eos_id:
                    break
                ids.append(int(id))
            infer_ids.append(self.charset.get_charstr_by_idxlist(ids))

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

    def train(self, training=True):
        iter_count = 0
        for epoch in range(self.train_epoch):
            logger.info("run epoch {}".format(epoch))
            data_train = None
            for batch, data in enumerate(self.train_dataset):
                if data_train is None:
                    data_train = data
                iter_count = iter_count + 1
                img_path, norm_img, img_text, label_dense, txt_len, coord, norm_w = data_train
                logger.debug("norm_img shape: {}".format(norm_img.shape))
                logger.debug('label_dence shape:'.format(label_dense.shape))
                logger.debug("img_text:{} {}".format(type(img_text), img_text))
                logger.debug("label_dense:{}".format(label_dense))

                s2s_time0 = time_func()
                with tf.GradientTape() as tape:
                    outputs, attentions, self.loss_value = self.seq2seq(norm_img, norm_w, label_dense, training)
                    s2s_time1 = time_func()
                    logger.info("seq2seq time consume: {}".format(s2s_time1 - s2s_time0))
                    if iter_count % 100 == 0:
                        self.calc_seqacc(label_dense, outputs, True)

                appgrd_time0 = time_func()
                trainable_variables = self.seq2seq.trainable_variables
                #for var in trainable_variables:
                #    print(var.name)
                gradients = tape.gradient(self.loss_value, trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, trainable_variables))
                appgrd_time1 = time_func()
                logger.info("appgrad time consume: {}".format(appgrd_time1 - appgrd_time0))

                if iter_count % self.eval_interval == 0:
                    self.test()

                if iter_count % self.save_interval == 0:
                    self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def test(self):
        total_cnt = 0.0000001
        total_cor = 0.0000000
        iter_count = 0
        for batch, data in enumerate(self.eval_dataset):
            try:
                iter_count = iter_count + 1
                img_path, norm_img, img_text, label_dense, txt_len, coord, norm_w = data
                outputs, attentions = self.seq2seq.evaluate(norm_img, norm_w)
                show_flag = True if iter_count % 500 == 0 else False
                batch_cnt, batch_cor = self.calc_seqacc(label_dense, outputs, show_flag)
                total_cnt = total_cnt + batch_cnt
                total_cor = total_cor + batch_cor

                if iter_count % 100 == 0:
                    logger.info("test batch index:{}, corr_rate:{}, total_cor:{}, total_cnt:{}".format(
                        batch, total_cor/total_cnt, total_cor, total_cnt))
            except Exception as e:
                logger.info("test exception:{}".format(e))

        logger.info("test evalutation: corr_rate:{}, total_cor:{}, total_cnt:{}".format(
            total_cor / total_cnt, total_cor, total_cnt))


if __name__ == '__main__':

    configs = {}
    configs['train_dataset_type'] = 'tfrecord'
    configs['eval_dataset_type'] = 'tfrecord'
    configs['norm_h'] = 32
    configs['save_interval'] = 10
    configs['learning_rate'] = 0.0002
    configs['expand_rate'] = 1.0
    configs['num_parallel'] = 64
    configs['batch_size'] = 32
    configs['char_dict'] = 'char_dict.lst'
    configs['model_type'] = 'attention'

    configs['train_file_list'] = tf.io.gfile.glob("/Users/junhuang.hj/Desktop/code_paper/code/crnn_ctc_eager/tfrecord_dir/tfrecord.list.*")
    configs['eval_file_list'] = tf.io.gfile.glob("/Users/junhuang.hj/Desktop/code_paper/code/crnn_ctc_eager/tfrecord_dir/tfrecord.list.*")

    print("train_file_list:", configs['train_file_list'])
    print("eval_file_list:", configs['eval_file_list'])
    slv_class = Solover(configs)
    slv_class.train()
    ttf_file = '/Users/junhuang.hj/Desktop/code_paper/code/data_gene/china_font/simhei.ttf'
    slv_class.visualize_tfrecord(ttf_file)


