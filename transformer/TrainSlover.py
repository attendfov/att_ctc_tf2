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
from ShowUtils import show_2dattention_image

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
        self.norm_h = int(config['norm_h'])
        self.expand_rate = float(config['expand_rate'])

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
        self.max_enc_length = 1200
        self.eos_id = self.charset.get_eosid()
        self.sos_id = self.charset.get_sosid()
        self.vocab_size = self.charset.get_size()
        self.enc_used_rnn = False
        self.enc_num_layers = 0
        self.dec_num_layers = 1
        self.d_model = 512
        self.enc_num_heads = 4
        self.dec_num_heads = 4
        self.enc_dff = 1024
        self.dec_dff = 1024
        self.enc_rate = 0.0
        self.dec_rate = 0.0

        self.seq2seq = Seq2Seq(enc_num_layers=self.enc_num_layers,
                               dec_num_layers=self.dec_num_layers,
                               d_model=self.d_model,
                               vocab_size=self.vocab_size,
                               enc_num_heads=self.enc_num_heads,
                               dec_num_heads=self.dec_num_heads,
                               enc_dff=self.enc_dff,
                               dec_dff=self.dec_dff,
                               enc_used_rnn=self.enc_used_rnn,
                               sos_id=self.sos_id,
                               eos_id=self.eos_id,
                               max_enc_length=self.max_enc_length,
                               max_dec_length=self.max_dec_length,
                               enc_rate=self.enc_rate,
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

    def visualize_attention(self,
                            norm_img,
                            norm_w,
                            label_dense,
                            image_path,
                            ttf_file,
                            encoder_feats_height=3,
                            attention_name='decoder_layer1_block2',
                            save_dir='default_show',
                            show_error=False,
                            attention_thr=0.1):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        save_file = os.path.join(save_dir, 'infer.list')
        writer = io.open(save_file, 'w', encoding='utf-8')
        batch = int(label_dense.shape[0])
        output, probility, atten_weight = self.seq2seq.evaluate(norm_img, norm_w, self.max_dec_length)
        image_data = norm_img.numpy()
        image_path = image_path.numpy()
        label_input = label_dense.numpy()

        #batch x vocsize
        decoder_output = output.numpy()
        assert(attention_name in atten_weight)
        #batch x headnums x outputstep x encodestep
        decoder_attentions = atten_weight[attention_name].numpy()
        decode_batch, \
        decode_heads, \
        decode_output_steps, \
        decoder_input_steps = [int(x) for x in decoder_attentions.shape]

        print("decoder_attentions shape:", decoder_attentions.shape)

        assert(batch == decode_batch)
        assert(decoder_input_steps % encoder_feats_height == 0)

        encoder_feats_h = encoder_feats_height
        encoder_feats_w = decoder_input_steps//encoder_feats_h

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

        point_lst = []
        infer_ids = []
        for b_id in range(batch):
            ids = []
            points = []
            for ix, id in enumerate(decoder_output[b_id][:-1]):
                #[heads x encode_steps]
                attention = decoder_attentions[b_id][:, ix, :]
                if int(id) == self.sos_id:
                    continue
                if int(id) == self.eos_id:
                    break
                ids.append(int(id))
                indexs = []
                for head in range(decode_heads):
                    step_attention = attention[head]
                    index_temp = [idx for idx, prb in enumerate(step_attention) if prb>attention_thr]
                    if len(index_temp) == 0:
                        index_temp = [np.argmax(step_attention)]

                    indexs.extend(index_temp)
                indexs = list(set(indexs))
                points.append([[idx % encoder_feats_w, idx // encoder_feats_w] for idx in indexs])

                #logger.info('b_id:{}, step:{}, indexs:{}'.format(b_id, ix, indexs))
                #logger.info('b_id:{}, step:{}, points:{}'.format(b_id, ix, points))

            point_lst.append(points)
            infer_ids.append(self.charset.get_charstr_by_idxlist(ids))

            for label, infer, att_points, img_path, img_data in zip(label_ids, infer_ids, point_lst, image_path, image_data):
                img_path = img_path.decode('utf-8')
                writer.write(img_path + ' ' + infer + ' ' + label + '\n')
                if show_error and label == infer:
                    continue

                att_vers_points = []
                for att_point in att_points:
                    print('att_point', att_point)
                    att_point = self.seq2seq.encoder.get_reverse_points(att_point)
                    att_points_squeeze = []
                    for point in att_point:
                        att_points_squeeze.extend(point)

                    att_vers_points.append(att_points_squeeze)

                img_name = os.path.basename(img_path)
                save_name = os.path.join(save_dir, img_name)
                img_data = img_data + mean

                cv2.imwrite(save_name, img_data, [cv2.IMWRITE_JPEG_QUALITY, 100])
                img_data = show_2dattention_image(save_name, ttf_file, label, infer, att_vers_points, imgw_scale=2)
                cv2.imwrite(save_name, img_data, [cv2.IMWRITE_JPEG_QUALITY, 100])

        writer.close()

    def visualize_tfrecord(self,
                           ttf_file,
                           encoder_feats_height=3,
                           attention_name='decoder_layer1_block2',
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
                output_shape = [img_path.shape[0], self.max_dec_length]
                label_dense = tf.sparse_to_dense(label_indices, output_shape, label_values, self.charset.get_eosid())
                self.visualize_attention(norm_img,
                                         norm_w,
                                         label_dense,
                                         img_path,
                                         ttf_file,
                                         encoder_feats_height,
                                         attention_name,
                                         save_dir,
                                         show_error,
                                         attention_thr)
            #except Exception as e:
            #    logger.info('Exception {}'.format(str(e)))

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

    def calc_seqacc(self, label_input, decoder_output, show_flag=False, is_argmax=False):
        #label_input: tensor -->batch x steps
        # decoder_output: tensor --> batch x steps if is_argmax is True
        # decoder_output: tensor --> batch x steps x vocb_size if is_argmax is False

        batch = int(label_input.shape[0])
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

    def train(self, training=True):
        iter_count = 0
        for epoch in range(self.train_epoch):
            logger.info("run epoch {}".format(epoch))
            data_train = None
            for batch, data in enumerate(self.train_dataset):
                if data_train is None:
                    data_train = data
                iter_count = iter_count + 1
                img_path, norm_img, img_text, txt_index, txt_len, coord, norm_w = data_train
                label_sparse = tf.string_split(txt_index, ',')
                label_indices = label_sparse.indices
                label_values = label_sparse.values
                label_values = tf.string_to_number(label_values, out_type=tf.int32)
                output_shape = [img_path.shape[0], self.max_dec_length]
                label_dense = tf.sparse_to_dense(label_indices, output_shape, label_values, self.charset.get_eosid())
                logger.debug("norm_img shape: {}".format(norm_img.shape))
                logger.debug('label_dence shape:'.format(label_dense.shape))
                logger.debug("img_text:{} {}".format(type(img_text),img_text))
                logger.debug("label_dense:{}".format(label_dense))

                with tf.GradientTape() as tape:
                    decoder_output, attention, loss = self.seq2seq(norm_img, norm_w, label_dense, None, training)
                    logger.info("loss: {}".format(loss))
                    if iter_count % 500 == 0:
                        self.calc_seqacc(label_dense, decoder_output, True)

                variables = self.seq2seq.variables
                gradients = tape.gradient(loss, variables)
                self.optimizer.apply_gradients(zip(gradients, variables), global_step=self.step_counter)

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
                img_path, norm_img, img_text, txt_index, txt_len, coord, norm_w = data
                label_sparse = tf.string_split(txt_index, ',')
                label_indices = label_sparse.indices
                label_values = label_sparse.values
                label_values = tf.string_to_number(label_values, out_type=tf.int32)
                output_shape = [img_path.shape[0], self.max_dec_length]
                label_dense = tf.sparse_to_dense(label_indices, output_shape, label_values, self.charset.get_eosid())

                output, probility, att_weights = self.seq2seq.evaluate(norm_img, norm_w, self.max_dec_length)
                show_flag = False
                if iter_count % 1 == 0:
                    show_flag = True
                batch_cnt, batch_cor = self.calc_seqacc(label_dense, output, show_flag, True)
                total_cnt = total_cnt + batch_cnt
                total_cor = total_cor + batch_cor

                if iter_count % 100 == 0:
                    logger.info("test batch index:{}, corr_rate:{}, total_cor:{}, total_cnt:{}".format(
                        batch, total_cor/total_cnt, total_cor, total_cnt))

            except Exception as e:
                logger.info("test exception:{}".format(e))

        logger.info("test evalutation: corr_rate:{}, total_cor:{}, total_cnt:{}".format(total_cor / total_cnt, total_cor, total_cnt))

    def inference(self, infer_image):
        norm_h = self.norm_h
        expand_rate = self.expand_rate
        imgh, imgw = infer_image.shape[:2]
        ratio = float(norm_h / float(imgh))
        norm_w = int(float(imgw) * expand_rate * ratio)
        norm_img = cv2.resize(infer_image, (norm_w, norm_h))
        norm_img = norm_img.astype(np.float32)

        mean = [127.5, 127.5, 127.5]
        norm_img = norm_img[:, :, ::-1]
        norm_img = norm_img - mean

        norm_w = tf.convert_to_tensor([norm_w], dtype=tf.int32)
        norm_img = tf.convert_to_tensor(np.expand_dims(norm_img, axis=0), dtype=tf.float32)

        output, probility, att_weights = self.seq2seq.evaluate(norm_img, [norm_w], self.max_dec_length)
        output_texts = self.decoder_text(output, is_argmax=True)
        return output_texts


if __name__ == '__main__':
    from HtmlShow import *

    configs = {}
    configs['train_dataset_type'] = 'tfrecord'
    configs['eval_dataset_type'] = 'tfrecord'
    configs['model_type'] = 'attention'
    configs['norm_h'] = 48
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
    #slv_class.test()
    #slv_class.train()
    #ttf_file = '/Users/junhuang.hj/Desktop/code_paper/code/data_gene/fonts/chinas//simhei.ttf'
    #slv_class.visualize_tfrecord(ttf_file, encoder_feats_height=2)

    img_list = []
    lbl_list = []
    inf_list = []
    imgs_dir = "/Users/junhuang.hj/Desktop/CTW/bbox_imgs"
    src_imgs = [os.path.join(imgs_dir, img_name) for img_name in os.listdir(imgs_dir) if img_name.endswith('.jpg')]
    for index, img_file in enumerate(src_imgs):
        if index>100:
            break
        try:
          image = cv2.imread(img_file)
          imgh, imgw = image.shape[:2]
          if imgh > imgw:
              continue
        except Exception as e:
            continue

        output_texts = slv_class.inference(image)

        img_list.append(img_file)
        lbl_list.append('###')
        inf_list.append(output_texts[0])

    html_file = 'html_file.html'
    write_html(html_file, img_list, lbl_list, inf_list)









