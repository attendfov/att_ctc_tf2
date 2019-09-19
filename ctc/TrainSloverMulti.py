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

from Logger import logger
from Seq2Seq import Seq2Seq
from ctc_utils import *
from tensor_utils import *
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

        self.mirrored_strategy = None
        self.num_replicas_in_sync = 1
        self.batch_per_replica = int(config['batch_size'])
        self.devices = configs['devices'].split(',')
        if len(self.devices) > 1:
            self.mirrored_strategy = tf.distribute.MirroredStrategy(devices=self.devices)
            self.num_replicas_in_sync = mirrored_strategy.num_replicas_in_sync
            assert (len(self.devices) == self.num_replicas_in_sync)

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
        train_ds_config['batch_size'] = self.batch_per_replica*self.num_replicas_in_sync
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
        eval_ds_config['batch_size'] = self.batch_per_replica*self.num_replicas_in_sync
        eval_ds_config['char_dict'] = config['char_dict']
        eval_ds_config['model_type'] = config['model_type']
        eval_ds_config['mode'] = 'test'
        self.eval_dataset = self.eval_dataset(eval_ds_config).data_reader()
        
        #charset parse
        self.char_dict = config['char_dict']
        self.model_type = config['model_type']
        self.charset = Charset(self.char_dict, self.model_type)
        self.step_counter = tf.Variable(tf.constant(0), trainable=False, name='step_counter')

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

        #seq2seq params
        self.eos_id = self.charset.get_eosid()
        self.vocab_size = self.charset.get_size()

        #optimizer params
        self.optimizer_type = 'adam'
        if 'optimizer_type' in configs:
            self.optimizer_type = configs['optimizer_type']
        self.optimizer_cls = tf.keras.optimizers.Adam
        if self.optimizer_type == 'sgd':
            self.optimizer_cls = tf.keras.optimizers.SGD
        elif self.optimizer_type == 'adam':
            self.optimizer_cls = tf.keras.optimizers.Adam

        #checkpoint params
        self.checkpoint_dir = 'training_checkpoints'
        if 'checkpoint_dir' in config:
            self.checkpoint_dir = config['checkpoint_dir']
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

        self.global_batch_size = self.batch_per_replica * self.num_replicas_in_sync
        if self.mirrored_strategy is None:
            self.create_solo_env()
        else:
            self.train_dataset = mirrored_strategy.experimental_distribute_dataset(self.train_dataset)
            self.eval_dataset = mirrored_strategy.experimental_distribute_dataset(self.eval_dataset)
            self.create_multi_env()

    def create_solo_env(self):
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_seq_error = tf.keras.metrics.Sum(name='test_seq_error')
        self.test_seq_count = tf.keras.metrics.Sum(name='test_seq_count')
        self.test_char_error = tf.keras.metrics.Accuracy(name='test_char_error')
        self.test_char_count = tf.keras.metrics.Accuracy(name='test_char_count')

        self.seq2seq = Seq2Seq(vocab_size=self.vocab_size, eos_id=self.eos_id)
        self.optimizer = self.optimizer_cls(self.learning_rate)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, seq2seq=self.seq2seq)

        if self.restore_ckpt is not None and os.path.isdir(self.restore_ckpt):
            if tf.train.latest_checkpoint(self.restore_ckpt) is not None:
                self.checkpoint.restore(tf.train.latest_checkpoint(self.restore_ckpt))
        else:
            if tf.train.latest_checkpoint(self.checkpoint_dir) is not None:
                self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def create_multi_env(self):
        with self.mirrored_strategy.scope():
            self.train_loss = tf.keras.metrics.Mean(name='train_loss')
            self.test_loss = tf.keras.metrics.Mean(name='test_loss')
            self.test_seq_error = tf.keras.metrics.Sum(name='test_seq_error')
            self.test_seq_count = tf.keras.metrics.Sum(name='test_seq_count')
            self.test_char_error = tf.keras.metrics.Accuracy(name='test_char_error')
            self.test_char_count = tf.keras.metrics.Accuracy(name='test_char_count')

            self.seq2seq = Seq2Seq(vocab_size=self.vocab_size, eos_id=self.eos_id)
            self.optimizer = self.optimizer_cls(self.learning_rate)
            self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, seq2seq=self.seq2seq)

            if self.restore_ckpt is not None and os.path.isdir(self.restore_ckpt):
                if tf.train.latest_checkpoint(self.restore_ckpt) is not None:
                    self.checkpoint.restore(tf.train.latest_checkpoint(self.restore_ckpt))
            else:
                if tf.train.latest_checkpoint(self.checkpoint_dir) is not None:
                    self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def train_step_multi(self, inputs):
        with self.mirrored_strategy.scope():
            img_path, norm_img, img_text, dense_label, label_len, coord, norm_w = inputs
            with tf.GradientTape() as tape:
                sparse_label = dense_to_sparse(dense_label, self.eos_id)
                ctc_features, ctc_logits, seq_lens = self.seq2seq(norm_img, norm_w, sparse_label, label_len, True)
                loss = sparse_ctc_loss(ctc_logits, sparse_label, seq_lens, self.charset.get_eosid())
                loss = tf.reduce_sum(loss) * (1.0 / self.global_batch_size)
                self.train_loss.update_state(loss)
            gradients = tape.gradient(loss, self.seq2seq.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.seq2seq.trainable_variables))

    def test_step_multi(self, inputs):
        with self.mirrored_strategy.scope():
            img_path, norm_img, img_text, dense_label, label_len, coord, norm_w = inputs
            sparse_label = dense_to_sparse(dense_label, self.eos_id)
            ctc_features, ctc_logits, seq_lens = self.seq2seq(norm_img, norm_w, sparse_label, label_len, False)
            loss = sparse_ctc_loss(ctc_logits, sparse_label, seq_lens, self.charset.get_eosid())
            loss = tf.reduce_sum(loss) * (1.0 / self.global_batch_size)
            self.test_loss.update_state(loss)
            inf_tensor, inf_probs = tf.nn.ctc_greedy_decoder(tf.transpose(ctc_logits, [1, 0, 2]), seq_lens)
            seq_error, seq_count, char_error, char_count = ctc_metrics(inf_tensor, sparse_label, label_len)
            self.test_seq_error.update_state(seq_error)
            self.test_seq_count.update_state(seq_count)
            self.test_char_error.update_state(char_error)
            self.test_char_count.update_state(char_count)

    def train_step_solo(self, inputs):
        img_path, norm_img, img_text, dense_label, label_len, coord, norm_w = inputs
        with tf.GradientTape() as tape:
            sparse_label = dense_to_sparse(dense_label, self.eos_id)
            ctc_features, ctc_logits, seq_lens = self.seq2seq(norm_img, norm_w, True)
            loss = sparse_ctc_loss(ctc_logits, sparse_label, seq_lens, self.charset.get_eosid())
            loss = tf.reduce_sum(loss) * (1.0 / self.global_batch_size)
            self.train_loss.update_state(loss)
        gradients = tape.gradient(loss, self.seq2seq.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.seq2seq.trainable_variables))

    def test_step_solo(self, inputs):
        img_path, norm_img, img_text, dense_label, label_len, coord, norm_w = inputs
        sparse_label = dense_to_sparse(dense_label, self.eos_id)
        ctc_features, ctc_logits, seq_lens = self.seq2seq(norm_img, norm_w, False)
        loss = sparse_ctc_loss(ctc_logits, sparse_label, seq_lens, self.charset.get_eosid())
        loss = tf.reduce_sum(loss) * (1.0 / self.global_batch_size)
        self.test_loss.update_state(loss)
        print("seq_lens", len(seq_lens), "ctc_logits:", ctc_logits.shape)
        inf_tensor, inf_probs = tf.nn.ctc_greedy_decoder(tf.transpose(ctc_logits, [1, 0, 2]), seq_lens)
        seq_error, seq_count, char_error, char_count = ctc_metrics(inf_tensor, sparse_label, label_len)
        self.test_seq_error.update_state(seq_error)
        self.test_seq_count.update_state(seq_count)
        self.test_char_error.update_state(char_error)
        self.test_char_count.update_state(char_count)

    def train_solo_gpu(self):
        iter_count = 0
        for epoch in range(self.train_epoch):
            logger.info("run epoch {}".format(epoch))
            for batch, data in enumerate(self.train_dataset):
                try:
                    self.train_step_solo(data)
                    if iter_count % self.show_interval == 0:
                        print("train_loss:", self.train_loss.result().numpy())
                        self.train_loss.reset_states()
                    if iter_count % self.save_interval == 0:
                        for data in self.eval_dataset:
                            self.test_step_solo(data)

                            test_loss = float(self.test_loss.result().numpy())
                            test_seq_error = float(self.test_seq_error.result.numpy())
                            test_seq_count = float(self.test_seq_count.result.numpy())
                            test_char_error = float(self.test_char_error.result.numpy())
                            test_char_count = float(self.test_char_count.result.numpy())

                            test_seq_acc = (1.0-test_seq_error/test_seq_count)*100
                            test_char_acc = (1.0-test_char_error/test_char_count)*100
                            print("test_loss:{:^10.5f}".format(test_loss))
                            print("test_seq_acc:{:^10.5f}, test_char_acc:{:^10.5f}".format(test_seq_acc, test_char_acc))

                            self.test_loss.reset_states()
                            self.test_seq_error.reset_states()
                            self.test_seq_count.reset_states()
                            self.test_char_error.reset_states()
                            self.test_char_count.reset_states()
                except Exception as e:
                    print("Exception:", str(e))

    @tf.function
    def distributed_train_step(self, dataset_inputs):
        per_replica_losses = self.mirrored_strategy.experimental_run_v2(self.train_step_multi, args=(dataset_inputs,))
        return self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tf.function
    def distributed_test_step(self, dataset_inputs):
        return self.mirrored_strategy.experimental_run_v2(self.test_step_multi, args=(dataset_inputs,))

    def train_multi_gpu(self):
        with self.mirrored_strategy.scope():
            iter_count = 0
            for epoch in range(self.train_epoch):
                logger.info("run epoch {}".format(epoch))
                for batch, data in enumerate(self.train_dataset):
                    try:
                        self.distributed_train_step(data)
                        if iter_count % self.show_interval == 0:
                            print("train_loss:", self.train_loss.result().numpy())
                            self.train_loss.reset_states()
                        if iter_count % self.save_interval == 0:
                            for data in self.eval_dataset:
                                self.distributed_test_step(data)
                                test_loss = float(self.test_loss.result().numpy())
                                test_seq_error = float(self.test_seq_error.result.numpy())
                                test_seq_count = float(self.test_seq_count.result.numpy())
                                test_char_error = float(self.test_char_error.result.numpy())
                                test_char_count = float(self.test_char_count.result.numpy())

                                test_seq_acc = (1.0 - test_seq_error / test_seq_count) * 100
                                test_char_acc = (1.0 - test_char_error / test_char_count) * 100
                                print("test_loss:{:^10.5f}".format(test_loss))
                                print("test_seq_acc:{:^10.5f}, test_char_acc:{:^10.5f}".format(test_seq_acc, test_char_acc))

                                self.test_loss.reset_states()
                                self.test_seq_error.reset_states()
                                self.test_seq_count.reset_states()
                                self.test_char_error.reset_states()
                                self.test_char_count.reset_states()
                    except Exception as e:
                        print("Exception:", str(e))

    def train(self):
        if self.mirrored_strategy is None:
            self.train_solo_gpu()
        else:
            self.train_multi_gpu()


if __name__ == '__main__':
    configs = {}
    configs['train_dataset_type'] = 'tfrecord'
    configs['eval_dataset_type'] = 'tfrecord'
    configs['devices'] = ''
    #configs['devices'] = '/gpu:0,/gpu:1'
    configs['model_type'] = 'ctc'
    configs['norm_h'] = 32
    configs['save_interval'] = 100
    configs['learning_rate'] = 0.0001
    configs['expand_rate'] = 1.0
    configs['num_parallel'] = 64
    configs['batch_size'] = 32
    configs['char_dict'] = '../seq2seq/char_dict.lst'
    configs['optimizer_type'] = 'sgd'

    configs['train_file_list'] = tf.io.gfile.glob("/Users/junhuang.hj/Desktop/code_paper/code/crnn_ctc_eager/tfrecord_dir/tfrecord.list.*")
    configs['eval_file_list'] = tf.io.gfile.glob("/Users/junhuang.hj/Desktop/code_paper/code/crnn_ctc_eager/tfrecord_dir/tfrecord.list.*")

    print("train_file_list:", configs['train_file_list'])
    print("eval_file_list:", configs['eval_file_list'])
    slv_class = Solover(configs)
    slv_class.train()

    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
    batch_per_replica = 32
    global_batch_size = (batch_per_replica * mirrored_strategy.num_replicas_in_sync)
    print("mirrored_strategy.num_replicas_in_sync:", mirrored_strategy.num_replicas_in_sync)
    learning_rate = 0.001
    dir_with_mnist_data_files = '/disk6/junhuang.hj/code/dir_with_mnist_data_files'

    dataset_train = tf.data.Dataset.from_generator(mnist_gene_train, (tf.float32, tf.int32),
                                                   (tf.TensorShape([28, 28, 1]), tf.TensorShape([])))
    dataset_train = dataset_train.batch(global_batch_size)
    # dataset_train = dataset_train.repeat(1000).batch(global_batch_size)
    dataset_test = tf.data.Dataset.from_generator(mnist_gene_test, (tf.float32, tf.int32),
                                                  (tf.TensorShape([28, 28, 1]), tf.TensorShape([])))
    dataset_test = dataset_test.batch(global_batch_size)
    # for index, data in enumerate(dataset_train):
    #    if index==100:
    #        break
    #    print("train:", index, type(data), len(data), data[0].numpy().shape, data[1].numpy().shape)
    #
    # for index, data in enumerate(dataset_test):
    #    if index==50:
    #        break
    #    print("test:", index, type(data), len(data), data[0].numpy().shape, data[1].numpy().shape)
    # global_batch_size = 32
    # model = Encoder(128, 10)
    # optimizer = tf.keras.optimizers.SGD()
    ##optimizer = tf.keras.optimizers.Adadelta()
    # for index, inputs in enumerate(dataset):
    #    images, labels = inputs
    #    with tf.GradientTape() as tape:
    #        features, logits = model(images, True)
    #        #print("logits:", logits.shape, "labels:", labels.shape, tf.reduce_max(images), tf.reduce_min(images))
    #        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    #        loss = tf.reduce_sum(cross_entropy)/global_batch_size
    #        print(loss)
    #    grads = tape.gradient(loss, model.trainable_variables)
    #    optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
    #    #print(loss)


