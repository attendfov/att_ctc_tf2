# _*_ coding:utf-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf


if sys.version.startswith('2'):
    reload(sys)
    sys.setdefaultencoding('utf-8')

abspath = os.path.dirname(os.path.realpath(__file__))
abspath = os.path.abspath(abspath)

sys.path.append(abspath)
sys.path.append(os.path.join(abspath, '.'))
sys.path.append(os.path.join(abspath, '../utils'))

from ctc_utils import *
from Logger import logger
from Seq2Seq import Seq2Seq
from Charset import Charset
from Dataset import FileDataset
from Dataset import RecordDataset


class Solover:
    def __init__(self, config):
        self.restore_ckpt = None
        if 'restore_ckpt' in config:
            self.restore_ckpt = config['restore_ckpt']

        self.clip_max_gradient = 3.0
        if 'clip_max_gradient' in config:
            self.clip_max_gradient = float(config['clip_max_gradient'])

        self.clip_min_gradient = -3.0
        if 'clip_min_gradient' in config:
            self.clip_min_gradient = float(config['clip_min_gradient'])

        self.eval_interval = 2000
        self.save_interval = 1000
        self.show_interval = 10
        self.train_epoch = 100
        self.max_iter = 1000000

        self.mirrored_strategy = None
        self.num_replicas_in_sync = 1
        self.batch_per_replica = int(config['batch_size'])
        self.devices = configs['devices'].split(',')
        assert(len(self.devices) > 1)
        self.mirrored_strategy = tf.distribute.MirroredStrategy(devices=self.devices)
        self.num_replicas_in_sync = self.mirrored_strategy.num_replicas_in_sync
        self.global_batch_size = self.batch_per_replica * self.num_replicas_in_sync

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
        train_ds_config['batch_size'] = self.global_batch_size
        train_ds_config['model_type'] = config['model_type']
        train_ds_config['char_dict'] = config['char_dict']
        train_ds_config['mode'] = 'train'

        self.train_dataset = self.train_dataset(train_ds_config).data_reader(self.train_epoch)

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
        eval_ds_config['batch_size'] = self.global_batch_size
        eval_ds_config['char_dict'] = config['char_dict']
        eval_ds_config['model_type'] = config['model_type']
        eval_ds_config['mode'] = 'test'
        self.eval_dataset = self.eval_dataset(eval_ds_config).data_reader()

        self.train_dataset = self.mirrored_strategy.experimental_distribute_dataset(self.train_dataset)
        self.eval_dataset = self.mirrored_strategy.experimental_distribute_dataset(self.eval_dataset)

        #charset parse
        self.char_dict = config['char_dict']
        self.model_type = config['model_type']
        self.charset = Charset(self.char_dict, self.model_type)
        self.step_counter = 0
        #self.step_counter = tf.Variable(tf.constant(0), trainable=False, name='step_counter')

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

        self.optimizer_type = 'adam'
        if 'optimizer_type' in config:
            self.optimizer_type = config['optimizer_type']

        if self.optimizer_type not in ('sgd', 'momentum', 'adam', 'rmsprop', 'adadelte'):
            print("Solover Error: optimizer_type {} not in [sgd, momentum, adam, rmsprop, adadelte]".format(self.optimizer_type))

        self.eos_id = self.charset.get_eosid()
        self.vocab_size = self.charset.get_size()
        with self.mirrored_strategy.scope():
            self.seq2seq = Seq2Seq(vocab_size=self.vocab_size, eos_id=self.eos_id)
            self.train_loss = tf.keras.metrics.Mean('train_loss')
            self.seq_err_cnt = tf.keras.metrics.Sum('seqerr_count')
            self.seq_all_cnt = tf.keras.metrics.Sum('seqerr_count')
            self.char_err_cnt = tf.keras.metrics.Sum('charerr_count')
            self.char_all_cnt = tf.keras.metrics.Sum('charall_count')

            self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
            if self.optimizer_type == 'sgd':
                self.optimizer = tf.keras.optimizers.SGD(self.learning_rate)
            elif self.optimizer_type == 'momentum':
                self.optimizer = tf.keras.optimizers.SGD(self.learning_rate, 0.95)
            elif self.optimizer_type == 'adam':
                self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
            elif self.optimizer_type == 'rmsprop':
                self.optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
            elif self.optimizer_type == 'adadelte':
                self.optimizer = tf.keras.optimizers.Adadelta(self.learning_rate)

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

    def train_step(self, inputs):
        with self.mirrored_strategy.scope():
            img_path, norm_img, img_text, dense_label, label_len, coord, norm_w = inputs
            with tf.GradientTape() as tape:
                ctc_features, ctc_output, step_widths = self.seq2seq(norm_img, norm_w, True)
                ctc_loss = tf.nn.ctc_loss(labels=dense_label, logits=ctc_output,
                                          label_length=label_len, logit_length=step_widths,
                                          logits_time_major=False, blank_index=self.eos_id)
                ctc_loss = tf.reduce_sum(ctc_loss) / self.global_batch_size
                self.train_loss.update_state(ctc_loss)
            gradients = tape.gradient(ctc_loss, self.seq2seq.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.seq2seq.trainable_variables))
            return ctc_loss

    def test_step(self, inputs):
        with self.mirrored_strategy.scope():
            img_path, norm_img, img_text, dense_label, label_len, coord, norm_w = inputs
            ctc_features, ctc_logits, step_widths = self.seq2seq(norm_img, norm_w, True)
            inf_tensor, inf_probs = tf.nn.ctc_greedy_decoder(tf.transpose(ctc_logits, [1, 0, 2]), step_widths)
            seq_error, seq_count, char_error, char_count = ctc_metrics(inf_tensor[0],
                                                                       dense_label,
                                                                       label_len,
                                                                       sparse_val=self.eos_id,
                                                                       inf_sparse=True,
                                                                       grt_sparse=False)

            self.seq_err_cnt.update_state(seq_error)
            self.seq_all_cnt.update_state(seq_count)
            self.char_err_cnt.update_state(char_error)
            self.char_all_cnt.update_state(char_count)

    @tf.function
    def distributed_train_step(self, dataset_inputs):
        per_replica_losses = self.mirrored_strategy.experimental_run_v2(self.train_step, args=(dataset_inputs,))
        return self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tf.function
    def distributed_test_step(self, dataset_inputs):
        return self.mirrored_strategy.experimental_run_v2(self.test_step, args=(dataset_inputs,))

    def train(self):
        with self.mirrored_strategy.scope():
            iter_count = 0
            for epoch in range(self.train_epoch):
                logger.info("run epoch {}".format(epoch))
                for batch, data in enumerate(self.train_dataset):
                    try:
                        iter_count = iter_count + 1
                        self.distributed_train_step(data)
                        if iter_count % self.show_interval == 0:
                            print("train_loss:", self.train_loss.result().numpy())
                            self.train_loss.reset_states()
                        if iter_count % self.save_interval == 0:
                            for eval_data in self.eval_dataset:
                                self.distributed_test_step(eval_data)

                            seq_err_cnt = self.seq_err_cnt.result().numpy()
                            seq_all_cnt = self.seq_all_cnt.result().numpy()
                            char_err_cnt = self.char_err_cnt.result().numpy()
                            char_all_cnt = self.char_all_cnt.result().numpy()

                            seq_acc = str(round(1.0 - seq_err_cnt / seq_all_cnt, 4))
                            char_acc = str(round(1.0 - char_err_cnt / char_all_cnt, 4))
                            model_acc_str = '-'.join(['seqacc', seq_acc, 'characc', char_acc])
                            checkpoint_prefix = '-'.join([self.checkpoint_prefix, str(iter_count), model_acc_str])
                            self.checkpoint.save(file_prefix=checkpoint_prefix)

                            self.seq_err_cnt.reset_states()
                            self.seq_all_cnt.reset_states()
                            self.char_err_cnt.reset_states()
                            self.char_all_cnt.reset_states()
                    except Exception as e:
                        print("Exception:", str(e))


if __name__ == '__main__':
    configs = {}
    configs['train_dataset_type'] = 'tfrecord'
    configs['eval_dataset_type'] = 'tfrecord'
    configs['model_type'] = 'ctc'
    configs['norm_h'] = 32
    configs['save_interval'] = 100
    configs['optimizer_type'] = 'adadelte'
    configs['learning_rate'] = 0.01
    configs['expand_rate'] = 1.0
    configs['num_parallel'] = 64
    configs['batch_size'] = 32
    configs['devices'] = '/gpu:0,/gpu:1'
    configs['clip_max_gradient'] = 1.0
    configs['clip_min_gradient'] = -1.0

    configs['char_dict'] = '../seq2seq/char_dict.lst'

    configs['train_file_list'] = tf.io.gfile.glob("/Users/junhuang.hj/Desktop/code_paper/code/crnn_ctc_eager/tfrecord_dir/tfrecord.list.*")
    configs['eval_file_list'] = tf.io.gfile.glob("/Users/junhuang.hj/Desktop/code_paper/code/crnn_ctc_eager/tfrecord_dir/tfrecord.list.*")

    print("train_file_list:", configs['train_file_list'])
    print("eval_file_list:", configs['eval_file_list'])
    slv_class = Solover(configs)
    slv_class.train()



