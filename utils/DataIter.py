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
tf.enable_eager_execution()


def parse_tfrecord(serial_example):
    feat_dict = tf.parse_single_example(serial_example,
                                        features={
                                            'img_raw' : tf.FixedLenFeature([], tf.string),\
                                            'height'  : tf.FixedLenFeature([], tf.int64), \
                                            'width'   : tf.FixedLenFeature([], tf.int64), \
                                            'channel' : tf.FixedLenFeature([], tf.int64), \
                                            'domain_idx': tf.FixedLenFeature([], tf.int64), \
                                            'img_path': tf.FixedLenFeature([], tf.string),\
                                            'coord'   : tf.FixedLenFeature([], tf.string), \
                                            'label'   : tf.FixedLenFeature([], tf.string)})

    img_raw  =  feat_dict['img_raw']
    height   =  feat_dict['height']
    width    =  feat_dict['width']
    channel  =  feat_dict['channel']
    img_path =  feat_dict['img_path']
    coord    =  feat_dict['coord']
    label    =  feat_dict['label']
    domain_idx = feat_dict['domain_idx']

    img_raw = tf.decode_raw(img_raw, tf.uint8)
    img_raw = tf.reshape(img_raw, (height, width, channel))
    return img_raw, label, height, width, channel, img_path, coord, domain_idx


class DataIter(object):
    def __init__(self, file_names, file_types, save_dir='default'):
        assert (isinstance(file_names, (list, str)))
        assert (isinstance(file_types, (list, str)))

        self.index = 0
        self.save_dir = save_dir
        self.image_files = []
        if isinstance(file_names, str):
            self.file_names = [file_name for file_name in file_names.split(',')]
        else:
            self.file_names = file_names

        if isinstance(file_types, str):
            self.file_types = [file_type for file_type in file_types.split(',')]
        else:
            self.file_types = file_types

        #file type check
        for file_type in self.file_types:
            assert(file_type in ('tfrecord', 'list'))
        for file_name in self.file_names:
            assert(os.path.isfile(file_name))

        #make sure file type same
        assert(len(set(self.file_types))==1)
        self.file_type = self.file_types[0]

        if self.file_type == 'tfrecord':
            if save_dir is not None:
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
            self.save_dir = os.path.abspath(self.save_dir)
            self.dataset = tf.data.TFRecordDataset(self.file_names)
            self.dataset = self.dataset.map(parse_tfrecord)
            self.dataset = iter(self.dataset)
        elif self.file_type == 'list':
            self.file_names.sort()
            self.dataset = []
            for file_name in self.file_names:
                reader = io.open(file_name, 'r', encoding='utf-8')
                for line in reader:
                    if len(line.strip()) == 0:
                        continue
                    line_sp = line.strip().split(' ')
                    if len(line_sp) < 2:
                        continue

                    img_coord = ''
                    domain_idx = '0'
                    img_file = line_sp[0]
                    img_text = line_sp[1]
                    if len(line_sp) == 3:
                        img_coord = line_sp[2]
                    if len(line_sp) == 4:
                        img_coord = line_sp[2]
                        domain_idx = line_sp[3]

                    if not os.path.isfile(img_file):
                        continue
                    if len(img_text) == 0:
                        continue
                    if len(img_coord) == 0:
                        try:
                            image = cv2.imread(img_file)
                            img_w, img_h = image.shape[:2]
                            img_coord = ','.join([str(x) for x in [0, 0, img_w, img_h]])
                        except:
                            continue

                    line = ' '.join([img_file, img_text, img_coord, domain_idx])
                    self.dataset.append(line.strip())
                reader.close()

    def __iter__(self):
        return self

    '''python2 used'''
    def next(self):
        try:
            if self.file_type == 'tfrecord':
                line = self.dataset.next()
                img_raw, label, height, width, channel, img_path, coord, domain_idx = line
                domain_idx = int(domain_idx)
                label = label.numpy().decode('utf-8')
                coord = coord.numpy().decode('utf-8')
                img_path = img_path.numpy().decode('utf-8')
                image = img_raw.numpy()
                image_name = os.path.basename(img_path)
                name, posfix = os.path.splitext(image_name)
                image_name = name + '_' + str(posfix) + posfix
                image_file = os.path.join(self.save_dir, image_name)
                self.image_files.append(image_file)
                cv2.imwrite(image_file, image)
                return ' '.join([image_file, label, coord, str(domain_idx)])
            elif self.file_type == 'list':
                line = self.dataset[self.index]
                self.index = self.index + 1
                if len(line) > 0:
                    return line
        except:
            raise StopIteration()
        raise StopIteration()

    '''python3 used'''
    def __next__(self):
        try:
            if self.file_type == 'tfrecord':
                line = self.dataset.next()
                img_raw, label, height, width, channel, img_path, coord, domain_idx = line
                domain_idx = int(domain_idx)
                label = label.numpy().decode('utf-8')
                coord = coord.numpy().decode('utf-8')
                img_path = img_path.numpy().decode('utf-8')
                image = img_raw.numpy()
                image_name = os.path.basename(img_path)
                name, posfix = os.path.splitext(image_name)
                image_name = name + '_' + str(posfix) + posfix
                image_file = os.path.join(self.save_dir, image_name)
                self.image_files.append(image_file)
                cv2.imwrite(image_file, image)
                return ' '.join([image_file, label, coord, str(domain_idx)])
            elif self.file_type == 'list':
                line = self.dataset[self.index]
                self.index = self.index + 1
                if len(line) > 0:
                    return line
        except:
            raise StopIteration()
        raise StopIteration()

    def __del__(self):
        for img_file in self.image_files:
            if os.path.isfile(img_file):
                os.remove(img_file)


if __name__ == '__main__':

    file_names = '/Users/junhuang.hj/Desktop/code_paper/code/crnn_ctc_eager/tfrecord_dir/tfrecord.list.0'
    file_types = 'tfrecord'
    dataiter = DataIter(file_names, file_types)

    for line in dataiter:
        image_file, image_text, image_coord, domain_index = line.strip().split(' ')
        xmin, ymin, xmax, ymax = [int(x) for x in image_coord.split(',')]
        image = cv2.imread(image_file)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
        cv2.imshow('image:', image)
        cv2.waitKey(0)
        print(line)