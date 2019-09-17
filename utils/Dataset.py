# _*_ coding:utf-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import sys
import numpy as np
import tensorflow as tf

sys.path.append('.')
from Charset import Charset


def coord_augmentation(tf_coord, width, height, lefoff=15, rigoff=15, upoff=5, downoff=5):
    xmin = tf_coord[0]
    ymin = tf_coord[1]
    xmax = tf_coord[2]
    ymax = tf_coord[3]

    width = tf.cast(width, dtype=xmin.dtype)
    height = tf.cast(height, dtype=ymin.dtype)

    xmin = xmin - tf.random_shuffle(tf.range(lefoff))[0]
    ymin = ymin - tf.random_shuffle(tf.range(upoff))[0]
    xmax = xmax + tf.random_shuffle(tf.range(rigoff))[0]
    ymax = ymax + tf.random_shuffle(tf.range(downoff))[0]

    xmin = tf.minimum(tf.maximum(0, xmin), width)
    ymin = tf.minimum(tf.maximum(0, ymin), height)
    xmax = tf.minimum(tf.maximum(0, xmax), width)
    ymax = tf.minimum(tf.maximum(0, ymax), height)

    return (xmin, ymin, xmax, ymax)


def distort_color(image, color_ordering=0):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255.)
    elif color_ordering == 2:
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 3:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 4:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif color_ordering == 5:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255.)

    image = tf.clip_by_value(image, 0.0, 1.0)
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    return image


def encode_decode_static(image, quality):
    def encode_decode_default(image):
        return image

    def encode_decode_95(image):
        image = tf.image.encode_jpeg(image, 'rgb', 95)
        image = tf.image.decode_jpeg(image)
        return image

    image = tf.cond(tf.equal(quality, 0),
                    true_fn=lambda: encode_decode_95(image),
                    false_fn=lambda: encode_decode_default(image))

    def encode_decode_90(image):
        image = tf.image.encode_jpeg(image, 'rgb', 90)
        image = tf.image.decode_jpeg(image)
        return image

    image = tf.cond(tf.equal(quality, 1),
                    true_fn=lambda: encode_decode_90(image),
                    false_fn=lambda: encode_decode_default(image))

    def encode_decode_85(image):
        image = tf.image.encode_jpeg(image, 'rgb', 85)
        image = tf.image.decode_jpeg(image)
        return image

    image = tf.cond(tf.equal(quality, 2),
                    true_fn=lambda: encode_decode_85(image),
                    false_fn=lambda: encode_decode_default(image))

    def encode_decode_80(image):
        image = tf.image.encode_jpeg(image, 'rgb', 80)
        image = tf.image.decode_jpeg(image)
        return image

    image = tf.cond(tf.equal(quality, 3),
                    true_fn=lambda: encode_decode_80(image),
                    false_fn=lambda: encode_decode_default(image))

    return image


def random_noise_static(image, noise_type):

    def random_default_noise(image):
        return image

    def random_normal_noise(image):
        noise = tf.random_normal(tf.shape(image), mean=0.0, stddev=6.0)
        image = tf.cast(image, dtype=tf.float32) + noise
        image = tf.clip_by_value(image, 0.0, 255.0)
        return tf.cast(image, dtype=tf.uint8)

    def random_uniform_noise(image):
        noise = tf.random_uniform(tf.shape(image), minval=-6, maxval=6)
        image = tf.cast(image, dtype=tf.float32) + noise
        image = tf.clip_by_value(image, 0.0, 255.0)
        return tf.cast(image, dtype=tf.uint8)

    image = tf.cond(tf.equal(noise_type, 0),
                    true_fn=lambda: random_normal_noise(image),
                    false_fn=lambda: random_default_noise(image))

    image = tf.cond(tf.equal(noise_type, 1),
                    true_fn=lambda: random_uniform_noise(image),
                    false_fn=lambda: random_default_noise(image))
    return image


def distort_color_static(image, color_ordering):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    def distort_default(image):
        return image

    def distort_color0(image):
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        return image

    def distort_color1(image):
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255.)
        return image

    def distort_color2(image):
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        return image

    def distort_color3(image):
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        return image

    def distort_color4(image):
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        return image

    def distort_color5(image):
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255.)
        return image

    image = tf.cond(tf.equal(color_ordering, 0),
                    true_fn=lambda: distort_color0(image),
                    false_fn=lambda: distort_default(image))
    image = tf.cond(tf.equal(color_ordering, 1),
                    true_fn=lambda: distort_color1(image),
                    false_fn=lambda: distort_default(image))
    image = tf.cond(tf.equal(color_ordering, 2),
                    true_fn=lambda: distort_color2(image),
                    false_fn=lambda: distort_default(image))
    image = tf.cond(tf.equal(color_ordering, 3),
                    true_fn=lambda: distort_color3(image),
                    false_fn=lambda: distort_default(image))
    image = tf.cond(tf.equal(color_ordering, 4),
                    true_fn=lambda: distort_color4(image),
                    false_fn=lambda: distort_default(image))
    image = tf.cond(tf.equal(color_ordering, 5),
                    true_fn=lambda:distort_color5(image),
                    false_fn=lambda: distort_default(image))
    image = tf.clip_by_value(image, 0.0, 1.0)
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    return image


def augmentation_test(img_path):
    decode_image0 = tf.image.decode_image(tf.read_file(img_path))
    for quality in range(100, 50, -5):
        encode_image = tf.image.encode_jpeg(decode_image0, 'rgb', quality)
        decode_image = tf.image.decode_jpeg(encode_image)
        print(type(decode_image), decode_image.shape, decode_image.dtype)
        decode_image_np = decode_image.numpy()
        decode_image_np = cv2.cvtColor(decode_image_np, cv2.COLOR_RGB2BGR)
        if quality==100:
            cv2.imshow('decode_image0', decode_image_np)
        else:
            cv2.imshow('decode_image1', decode_image_np)

        color_ordering = tf.random_shuffle(tf.range(6))
        print(color_ordering)
        color_ordering = int(color_ordering[0])

        decode_image = distort_color(decode_image, color_ordering)
        decode_image_np = decode_image.numpy()
        decode_image_np = cv2.cvtColor(decode_image_np, cv2.COLOR_RGB2BGR)

        cv2.imshow('decode_color', decode_image_np)
        cv2.waitKey(0)


class FileDataset:
    def __init__(self, configs):
        assert(isinstance(configs, dict))
        self.norm_h = 32
        if 'norm_h' in configs:
            self.norm_h = int(configs['norm_h'])
        self.expand_rate = 1.0
        if 'expand_rate' in configs:
            self.expand_rate = float(configs['expand_rate'])
        self.file_list = []
        if 'file_list' in configs:
            self.set_files(configs['file_list'])
        self.num_parallel=4
        if 'num_parallel' in configs:
            self.num_parallel = int(configs['num_parallel'])
        self.batch_size = 32
        if 'batch_size' in configs:
            self.batch_size = int(configs['batch_size'])

        self.max_txtlen = 32
        if 'max_txtlen' in configs:
            self.max_txtlen = int(configs['max_txtlen'])
        self.max_imglen = 1024
        if 'max_imglen' in configs:
            self.max_imglen = int(configs['max_imglen'])
        self.min_imglen = 16
        if 'min_imglen' in configs:
            self.min_imglen = int(configs['min_imglen'])
        self.BUFFER_SIZE = 4096
        if 'BUFFER_SIZE' in configs:
            self.BUFFER_SIZE = int(configs['BUFFER_SIZE'])

        self.mode = configs['mode'].lower()
        self.char_dict = configs['char_dict']
        self.model_type = configs['model_type']
        self.charset = Charset(self.char_dict, self.model_type)

    def set_files(self, file_list):
        assert(isinstance(file_list, (list, tuple)))
        self.file_list = [file for file in file_list if os.path.isfile(file) and os.path.getsize(file)>0]

    def get_idstr_by_charstr(self, charstr):
        if isinstance(charstr, bytes):
            charstr = charstr.decode('utf-8')

        if self.model_type in ('ctc', 'attention'):
            idxstr = self.charset.get_idxstr_by_charstr(charstr)
            idxlen = len(idxstr.split(','))
            return idxstr, idxlen
        elif self.model_type in ('ctc_attention', 'attention_ctc'):
            ctcidx, attidx = self.charset.get_idxstr_by_charstr(charstr)
            ctclen = len(ctcidx.split(','))
            attlen = len(attidx.split(','))
            return ctcidx, ctclen, attidx, attlen

    def parse_example(self, line):
        norm_h = self.norm_h
        expand_rate = self.expand_rate
        debug = False
        field_delim = ' '
        use_quote_delim = False
        record_defaults = ['', '', '']
        img_path, img_text, coord = tf.decode_csv(line, record_defaults, field_delim, use_quote_delim)
        txt_index, txt_len = tf.py_func(self.get_idstr_by_charstr, [img_text], [tf.string, tf.int64])
        txt_len = tf.to_int32(txt_len)
        coord_val = tf.string_split([coord], ',').values
        coord_val = tf.string_to_number(coord_val, out_type=tf.int32)
        orig_img = tf.image.decode_image(tf.read_file(img_path))
        img_shape = tf.shape(orig_img)
        width = img_shape[1]
        height = img_shape[0]

        prob = tf.random_uniform([])
        invert_flg = tf.logical_and(tf.greater(prob, 0.0), tf.equal(self.mode, 'train'))
        orig_img = tf.cond(invert_flg,
                           true_fn=lambda: tf.cast(255 - orig_img, dtype=tf.uint8),
                           false_fn=lambda: orig_img)

        prob = tf.random_uniform([])
        noise_flg = tf.logical_and(tf.greater(prob, 0.0), tf.equal(self.mode, 'train'))
        noise_idx = tf.random_shuffle(tf.range(2))[0]
        orig_img = tf.cond(noise_flg,
                           true_fn=lambda: random_noise_static(orig_img, noise_idx),
                           false_fn=lambda: random_noise_static(orig_img, -1))

        prob = tf.random_uniform([])
        encode_flg = tf.logical_and(tf.greater(0.3, prob), tf.equal(self.mode, 'train'))
        encode_idx = tf.random_shuffle(tf.range(4))[0]

        orig_img = tf.cond(encode_flg,
                           true_fn=lambda: encode_decode_static(orig_img, encode_idx),
                           false_fn=lambda: encode_decode_static(orig_img, -1))

        prob = tf.random_uniform([])
        color_flg = tf.logical_and(tf.greater(prob, 0), tf.equal(self.mode, 'train'))
        color_idx = tf.random_shuffle(tf.range(6))[0]
        orig_img = tf.cond(color_flg,
                           true_fn=lambda: distort_color_static(orig_img, color_idx),
                           false_fn=lambda: distort_color_static(orig_img, -1))

        prob = tf.random_uniform([])
        coord_flg = tf.logical_and(tf.greater(prob, 0), tf.equal(self.mode, 'train'))
        coord_val1 = tf.cond(coord_flg,
                             true_fn=lambda: coord_augmentation(coord_val, width, height),
                             false_fn=lambda: (coord_val[0], coord_val[1], coord_val[2], coord_val[3]))

        offset_w = coord_val1[0]
        offset_h = coord_val1[1]
        target_w = coord_val1[2] - coord_val1[0]
        target_h = coord_val1[3] - coord_val1[1]
        crop_img = tf.image.crop_to_bounding_box(orig_img, offset_h, offset_w, target_h, target_w)

        ratio = tf.to_float(norm_h / tf.to_float(target_h))
        norm_w = tf.to_int32(tf.to_float(target_w) * expand_rate * ratio)
        norm_img = tf.image.resize_images(crop_img, (norm_h, norm_w))
        if debug:
            norm_img = tf.cast(norm_img, tf.uint8)
        else:
            # convert RGB-->BGR
            mean = [127.5, 127.5, 127.5]
            norm_img = norm_img[:, :, ::-1]
            norm_img = norm_img - mean
        return img_path, norm_img, img_text, txt_index, txt_len, coord, norm_w

    def filter(self, img_path, norm_img, img_text, txt_index, txt_len, coord, norm_w):
        img_len = tf.cast(norm_w, dtype=tf.int32)
        txt_len = tf.cast(txt_len, dtype=tf.int32)
        txt_len_logical = tf.logical_and(txt_len <= self.max_txtlen, txt_len >= 0)
        img_len_logical = tf.logical_and(img_len <= self.max_imglen,img_len >= self.min_imglen)
        return tf.logical_and(txt_len_logical, img_len_logical)

    def data_reader(self, repeat=0):
        padded_shapes = ([], [self.norm_h, None, 3], [], [], [], [], [])
        padding_values = ('', 0.0, '', '', 0, '', 0)

        dataset = tf.data.TextLineDataset(self.file_list)
        if repeat != 0:
            dataset = dataset.repeat(repeat)
        else:
            dataset = dataset.repeat()

        dataset = dataset.map(map_func=self.parse_example, num_parallel_calls=self.num_parallel)
        dataset = dataset.filter(self.filter)
        dataset = dataset.shuffle(self.BUFFER_SIZE).padded_batch(self.batch_size, padded_shapes, padding_values)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def parse_example_ctc_attention(self, line):
        norm_h = self.norm_h
        expand_rate = self.expand_rate
        debug = False
        field_delim = ' '
        use_quote_delim = False
        record_defaults = ['', '', '']
        img_path, img_text, coord = tf.decode_csv(line, record_defaults, field_delim, use_quote_delim)
        ctc_idx, ctc_len,  att_idx, att_len = tf.py_func(self.get_idstr_by_charstr,
                                                         [img_text],
                                                         [tf.string, tf.int64, tf.string, tf.int64])
        ctc_len = tf.to_int32(ctc_len)
        att_len = tf.to_int32(att_len)
        coord_val = tf.string_split([coord], ',').values
        coord_val = tf.string_to_number(coord_val, out_type=tf.int32)
        orig_img = tf.image.decode_image(tf.read_file(img_path))
        img_shape = tf.shape(orig_img)
        width = img_shape[1]
        height = img_shape[0]

        prob = tf.random_uniform([])
        invert_flg = tf.logical_and(tf.greater(prob, 0.0), tf.equal(self.mode, 'train'))
        orig_img = tf.cond(invert_flg,
                           true_fn=lambda: tf.cast(255 - orig_img, dtype=tf.uint8),
                           false_fn=lambda: orig_img)

        prob = tf.random_uniform([])
        noise_flg = tf.logical_and(tf.greater(prob, 0.0), tf.equal(self.mode, 'train'))
        noise_idx = tf.random_shuffle(tf.range(2))[0]
        orig_img = tf.cond(noise_flg,
                           true_fn=lambda: random_noise_static(orig_img, noise_idx),
                           false_fn=lambda: random_noise_static(orig_img, -1))

        prob = tf.random_uniform([])
        encode_flg = tf.logical_and(tf.greater(0.3, prob), tf.equal(self.mode, 'train'))
        encode_idx = tf.random_shuffle(tf.range(4))[0]

        orig_img = tf.cond(encode_flg,
                           true_fn=lambda: encode_decode_static(orig_img, encode_idx),
                           false_fn=lambda: encode_decode_static(orig_img, -1))

        prob = tf.random_uniform([])
        color_flg = tf.logical_and(tf.greater(prob, 0), tf.equal(self.mode, 'train'))
        color_idx = tf.random_shuffle(tf.range(6))[0]
        orig_img = tf.cond(color_flg,
                           true_fn=lambda: distort_color_static(orig_img, color_idx),
                           false_fn=lambda: distort_color_static(orig_img, -1))

        prob = tf.random_uniform([])
        coord_flg = tf.logical_and(tf.greater(prob, 0), tf.equal(self.mode, 'train'))
        coord_val1 = tf.cond(coord_flg,
                             true_fn=lambda: coord_augmentation(coord_val, width, height),
                             false_fn=lambda: (coord_val[0], coord_val[1], coord_val[2], coord_val[3]))

        offset_w = coord_val1[0]
        offset_h = coord_val1[1]
        target_w = coord_val1[2] - coord_val1[0]
        target_h = coord_val1[3] - coord_val1[1]
        crop_img = tf.image.crop_to_bounding_box(orig_img, offset_h, offset_w, target_h, target_w)

        ratio = tf.to_float(norm_h / tf.to_float(target_h))
        norm_w = tf.to_int32(tf.to_float(target_w) * expand_rate * ratio)
        norm_img = tf.image.resize_images(crop_img, (norm_h, norm_w))
        if debug:
            norm_img = tf.cast(norm_img, tf.uint8)
        else:
            # convert RGB-->BGR
            mean = [127.5, 127.5, 127.5]
            norm_img = norm_img[:, :, ::-1]
            norm_img = norm_img - mean

        return img_path, norm_img, img_text, ctc_idx, ctc_len, att_idx, att_len, coord, norm_w

    def filter_ctc_attention(self, img_path, norm_img, img_text, ctc_idx, ctc_len, att_idx, att_len, coord, norm_w):
        img_len = tf.cast(norm_w, dtype=tf.int32)
        ctc_len = tf.cast(ctc_len, dtype=tf.int32)
        att_len = tf.cast(att_len, dtype=tf.int32)
        ctc_len_logical = tf.logical_and(ctc_len <= self.max_txtlen, ctc_len >= 0)
        att_len_logical = tf.logical_and(att_len <= self.max_txtlen, att_len >= 0)
        txt_len_logical = tf.logical_and(ctc_len_logical, att_len_logical)
        img_len_logical = tf.logical_and(img_len <= self.max_imglen,img_len >= self.min_imglen)
        return tf.logical_and(txt_len_logical, img_len_logical)

    def data_reader_ctc_attention(self, repeat=0):
        padded_shapes = ([], [self.norm_h, None, 3], [], [], [], [], [], [], [])
        padding_values = ('', 0.0, '', '', 0, '', 0, '', 0)

        dataset = tf.data.TextLineDataset(self.file_list)
        if repeat != 0:
            dataset = dataset.repeat(repeat)
        else:
            dataset = dataset.repeat()

        dataset = dataset.map(map_func=self.parse_example_ctc_attention, num_parallel_calls=self.num_parallel)
        dataset = dataset.filter(self.filter_ctc_attention)
        dataset = dataset.shuffle(self.BUFFER_SIZE).padded_batch(self.batch_size, padded_shapes, padding_values)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


class RecordDataset:
    def __init__(self, configs):
        assert(isinstance(configs, dict))
        self.norm_h = 32
        if 'norm_h' in configs:
            self.norm_h = int(configs['norm_h'])
        self.expand_rate = 1.0
        if 'expand_rate' in configs:
            self.expand_rate = float(configs['expand_rate'])
        self.file_list = []
        if 'file_list' in configs:
            self.set_files(configs['file_list'])
        self.num_parallel=4
        if 'num_parallel' in configs:
            self.num_parallel = int(configs['num_parallel'])
        self.batch_size = 32
        if 'batch_size' in configs:
            self.batch_size = int(configs['batch_size'])

        self.max_txtlen = 32
        if 'max_txtlen' in configs:
            self.max_txtlen = int(configs['max_txtlen'])
        self.max_imglen = 1024
        if 'max_imglen' in configs:
            self.max_imglen = int(configs['max_imglen'])
        self.min_imglen = 16
        if 'min_imglen' in configs:
            self.min_imglen = int(configs['min_imglen'])
        self.BUFFER_SIZE = 4096
        if 'BUFFER_SIZE' in configs:
            self.BUFFER_SIZE = int(configs['BUFFER_SIZE'])
        self.mode = 'train'
        if 'mode' in configs:
            self.mode = configs['mode'].lower()
            assert(self.mode in ('train', 'test', 'eval'))

        self.char_dict = configs['char_dict']
        self.model_type = configs['model_type']
        self.charset = Charset(self.char_dict, self.model_type)

    def set_files(self, file_list):
        assert(isinstance(file_list, (list, tuple)))
        self.file_list = [file for file in file_list if os.path.isfile(file) and os.path.getsize(file)>0]

    def get_idstr_by_charstr(self, charstr):
        if isinstance(charstr, bytes):
            charstr = charstr.decode('utf-8')

        if self.model_type in ('ctc', 'attention'):
            idxstr = self.charset.get_idxstr_by_charstr(charstr)
            idxlen = len(idxstr.split(','))
            return idxstr, idxlen
        elif self.model_type in ('ctc_attention', 'attention_ctc'):
            ctcidx, attidx = self.charset.get_idxstr_by_charstr(charstr)
            ctclen = len(ctcidx.split(','))
            attlen = len(attidx.split(','))
            return ctcidx, ctclen, attidx, attlen

    def parse_example(self, serial_example):
        norm_h = self.norm_h
        expand_rate = self.expand_rate
        debug = False

        feat_dict = tf.parse_single_example(serial_example,features={
                                            'img_raw' : tf.FixedLenFeature([], tf.string), \
                                            'height'  : tf.FixedLenFeature([], tf.int64),  \
                                            'width'   : tf.FixedLenFeature([], tf.int64),  \
                                            'channel' : tf.FixedLenFeature([], tf.int64),  \
                                            'img_path': tf.FixedLenFeature([], tf.string), \
                                            'coord'   : tf.FixedLenFeature([], tf.string), \
                                            'label'   : tf.FixedLenFeature([], tf.string)})

        img_raw = feat_dict['img_raw']
        height = feat_dict['height']
        width = feat_dict['width']
        channel = feat_dict['channel']
        img_path = feat_dict['img_path']
        coord = feat_dict['coord']
        img_text = feat_dict['label']

        txt_index, txt_len = tf.py_func(self.get_idstr_by_charstr, [img_text], [tf.string, tf.int64])
        txt_len = tf.to_int32(txt_len)

        coord_val = tf.string_split([coord], ',').values
        coord_val = tf.string_to_number(coord_val, out_type=tf.int32)

        img_raw = tf.decode_raw(img_raw, tf.uint8)
        orig_img = tf.reshape(img_raw, (height, width, channel))

        prob = tf.random_uniform([])
        invert_flg = tf.logical_and(tf.greater(prob, 0.75), tf.equal(self.mode, 'train'))
        orig_img = tf.cond(invert_flg,
                           true_fn=lambda: tf.cast(255-orig_img, dtype=tf.uint8),
                           false_fn=lambda: orig_img)

        prob = tf.random_uniform([])
        noise_flg = tf.logical_and(tf.greater(prob, 0.75), tf.equal(self.mode, 'train'))
        noise_idx = tf.random_shuffle(tf.range(2))[0]
        orig_img = tf.cond(noise_flg,
                           true_fn=lambda: random_noise_static(orig_img, noise_idx),
                           false_fn=lambda: random_noise_static(orig_img, -1))

        prob = tf.random_uniform([])
        encode_flg = tf.logical_and(tf.greater(prob, 0.75), tf.equal(self.mode, 'train'))
        encode_idx = tf.random_shuffle(tf.range(4))[0]

        orig_img = tf.cond(encode_flg,
                           true_fn=lambda: encode_decode_static(orig_img, encode_idx),
                           false_fn=lambda: encode_decode_static(orig_img, -1))

        prob = tf.random_uniform([])
        color_flg = tf.logical_and(tf.greater(prob, 0.75), tf.equal(self.mode, 'train'))
        color_idx = tf.random_shuffle(tf.range(6))[0]
        orig_img = tf.cond(color_flg,
                           true_fn=lambda: distort_color_static(orig_img, color_idx),
                           false_fn=lambda: distort_color_static(orig_img, -1))

        prob = tf.random_uniform([])
        coord_flg = tf.logical_and(tf.greater(prob, 0.4), tf.equal(self.mode, 'train'))
        coord_val1 = tf.cond(coord_flg,
                             true_fn=lambda: coord_augmentation(coord_val, width, height),
                             false_fn=lambda: (coord_val[0], coord_val[1], coord_val[2], coord_val[3]))

        offset_w = coord_val1[0]
        offset_h = coord_val1[1]
        target_w = coord_val1[2] - coord_val1[0]
        target_h = coord_val1[3] - coord_val1[1]
        crop_img = tf.image.crop_to_bounding_box(orig_img, offset_h, offset_w, target_h, target_w)

        ratio = tf.to_float(norm_h / tf.to_float(target_h))
        norm_w = tf.to_int32(tf.to_float(target_w) * expand_rate * ratio)
        norm_img = tf.image.resize_images(crop_img, (norm_h, norm_w))

        if debug:
            norm_img = tf.cast(norm_img, tf.uint8)
        else:
            # convert RGB-->BGR
            mean = [102.9801, 115.9465, 122.7717]
            norm_img = norm_img[:, :, ::-1]
            norm_img = norm_img - mean
        return img_path, norm_img, img_text, txt_index, txt_len, coord, norm_w

    def filter(self, img_path, norm_img, img_text, txt_index, txt_len, coord, norm_w):
        img_len = tf.cast(norm_w, dtype=tf.int32)
        txt_len = tf.cast(txt_len, dtype=tf.int32)
        txt_len_logical = tf.logical_and(txt_len <= self.max_txtlen, txt_len >= 0)
        img_len_logical = tf.logical_and(img_len <= self.max_imglen,img_len >= self.min_imglen)
        return tf.logical_and(txt_len_logical, img_len_logical)

    def data_reader_v0(self, repeat=0):
        padded_shapes = ([], [self.norm_h, None, 3], [], [], [], [], [])
        padding_values = ('', 0.0, '', '', 0, '', 0)
        dataset = tf.data.TFRecordDataset(self.file_list)
        if repeat != 0:
            dataset = dataset.repeat(repeat)
        else:
            dataset = dataset.repeat()
        dataset = dataset.map(map_func=self.parse_example, num_parallel_calls=self.num_parallel)
        dataset = dataset.padded_batch(self.batch_size, padded_shapes, padding_values)
        return dataset

    def data_reader(self, repeat=0):
        padded_shapes = ([], [self.norm_h, None, 3], [], [], [], [], [])
        padding_values = ('', 0.0, '', '', 0, '', 0)
        fileset = tf.data.Dataset.list_files(self.file_list)
        dataset = fileset.apply(
                    tf.data.experimental.parallel_interleave(
                        lambda filename: tf.data.TFRecordDataset(
                            filename, num_parallel_reads=self.num_parallel),
                        cycle_length=32))

        if repeat != 0:
            dataset = dataset.repeat(repeat)
        dataset = dataset.map(map_func=self.parse_example, num_parallel_calls=self.num_parallel)
        dataset = dataset.filter(self.filter)
        dataset = dataset.shuffle(self.BUFFER_SIZE).padded_batch(self.batch_size, padded_shapes, padding_values)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        #dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def parse_example_ctc_attention(self, serial_example):
        norm_h = self.norm_h
        expand_rate = self.expand_rate
        debug = False

        feat_dict = tf.parse_single_example(serial_example, features={
            'img_raw': tf.FixedLenFeature([], tf.string),\
            'height': tf.FixedLenFeature([], tf.int64),\
            'width': tf.FixedLenFeature([], tf.int64),\
            'channel': tf.FixedLenFeature([], tf.int64),\
            'img_path': tf.FixedLenFeature([], tf.string),\
            'coord': tf.FixedLenFeature([], tf.string),\
            'label': tf.FixedLenFeature([], tf.string)})

        img_raw = feat_dict['img_raw']
        height = feat_dict['height']
        width = feat_dict['width']
        channel = feat_dict['channel']
        img_path = feat_dict['img_path']
        coord = feat_dict['coord']
        img_text = feat_dict['label']
        ctc_idx, ctc_len, att_idx, att_len = tf.py_func(self.get_idstr_by_charstr,
                                                        [img_text],
                                                        [tf.string, tf.int64, tf.string, tf.int64])
        ctc_len = tf.to_int32(ctc_len)
        att_len = tf.to_int32(att_len)

        coord_val = tf.string_split([coord], ',').values
        coord_val = tf.string_to_number(coord_val, out_type=tf.int32)

        img_raw = tf.decode_raw(img_raw, tf.uint8)
        orig_img = tf.reshape(img_raw, (height, width, channel))

        prob = tf.random_uniform([])
        invert_flg = tf.logical_and(tf.greater(prob, 0.75), tf.equal(self.mode, 'train'))
        orig_img = tf.cond(invert_flg,
                           true_fn=lambda: tf.cast(255 - orig_img, dtype=tf.uint8),
                           false_fn=lambda: orig_img)

        prob = tf.random_uniform([])
        noise_flg = tf.logical_and(tf.greater(prob, 0.75), tf.equal(self.mode, 'train'))
        noise_idx = tf.random_shuffle(tf.range(2))[0]
        orig_img = tf.cond(noise_flg,
                           true_fn=lambda: random_noise_static(orig_img, noise_idx),
                           false_fn=lambda: random_noise_static(orig_img, -1))

        prob = tf.random_uniform([])
        encode_flg = tf.logical_and(tf.greater(prob, 0.75), tf.equal(self.mode, 'train'))
        encode_idx = tf.random_shuffle(tf.range(4))[0]

        orig_img = tf.cond(encode_flg,
                           true_fn=lambda: encode_decode_static(orig_img, encode_idx),
                           false_fn=lambda: encode_decode_static(orig_img, -1))

        prob = tf.random_uniform([])
        color_flg = tf.logical_and(tf.greater(prob, 0.75), tf.equal(self.mode, 'train'))
        color_idx = tf.random_shuffle(tf.range(6))[0]
        orig_img = tf.cond(color_flg,
                           true_fn=lambda: distort_color_static(orig_img, color_idx),
                           false_fn=lambda: distort_color_static(orig_img, -1))

        prob = tf.random_uniform([])
        coord_flg = tf.logical_and(tf.greater(prob, 0.4), tf.equal(self.mode, 'train'))
        coord_val1 = tf.cond(coord_flg,
                             true_fn=lambda: coord_augmentation(coord_val, width, height),
                             false_fn=lambda: (coord_val[0], coord_val[1], coord_val[2], coord_val[3]))

        offset_w = coord_val1[0]
        offset_h = coord_val1[1]
        target_w = coord_val1[2] - coord_val1[0]
        target_h = coord_val1[3] - coord_val1[1]
        crop_img = tf.image.crop_to_bounding_box(orig_img, offset_h, offset_w, target_h, target_w)

        ratio = tf.to_float(norm_h / tf.to_float(target_h))
        norm_w = tf.to_int32(tf.to_float(target_w) * expand_rate * ratio)
        norm_img = tf.image.resize_images(crop_img, (norm_h, norm_w))

        if debug:
            norm_img = tf.cast(norm_img, tf.uint8)
        else:
            # convert RGB-->BGR
            mean = [127.5, 127.5, 127.5]
            norm_img = norm_img[:, :, ::-1]
            norm_img = norm_img - mean
        return img_path, norm_img, img_text, ctc_idx, ctc_len, att_idx, att_len, coord, norm_w

    def filter_ctc_attention(self, img_path, norm_img, img_text, ctc_idx, ctc_len, att_idx, att_len, coord, norm_w):
        img_len = tf.cast(norm_w, dtype=tf.int32)
        ctc_len = tf.cast(ctc_len, dtype=tf.int32)
        att_len = tf.cast(att_len, dtype=tf.int32)
        ctc_len_logical = tf.logical_and(ctc_len <= self.max_txtlen, ctc_len >= 0)
        att_len_logical = tf.logical_and(att_len <= self.max_txtlen, att_len >= 0)
        txt_len_logical = tf.logical_and(ctc_len_logical, att_len_logical)
        img_len_logical = tf.logical_and(img_len <= self.max_imglen,img_len >= self.min_imglen)
        return tf.logical_and(txt_len_logical, img_len_logical)

    def data_reader_ctc_attention(self, repeat=0):
        padded_shapes = ([], [self.norm_h, None, 3], [], [], [], [], [], [], [])
        padding_values = ('', 0.0, '', '', 0, '', 0, '', 0)

        fileset = tf.data.Dataset.list_files(self.file_list)
        dataset = fileset.apply(
            tf.data.experimental.parallel_interleave(
                lambda filename: tf.data.TFRecordDataset(
                    filename, num_parallel_reads=self.num_parallel),
                cycle_length=32))

        if repeat != 0:
            dataset = dataset.repeat(repeat)
        dataset = dataset.map(map_func=self.parse_example_ctc_attention, num_parallel_calls=self.num_parallel)
        dataset = dataset.filter(self.filter_ctc_attention)
        dataset = dataset.shuffle(self.BUFFER_SIZE).padded_batch(self.batch_size, padded_shapes, padding_values)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset



def FileDatasetTestSoloModel():
    tf.enable_eager_execution()
    configs = {}
    configs['norm_h'] = 32
    configs['expand_rate'] = 1.0
    configs['file_list'] = ['/Users/junhuang.hj/Desktop/code_paper/data/sample.legal']
    configs['num_parallel'] = 4
    configs['batch_size'] = 1
    configs['char_dict'] = 'char_dict.lst'
    configs['model_type'] = 'attention'
    configs['mode'] = 'train'

    dataset = FileDataset(configs)
    dataset = dataset.data_reader()

    for line in dataset:
        img_path, norm_img, img_text, txt_index, txt_len, coord, norm_w = line
        norm_img = norm_img.numpy()
        print("norm_img0:", np.max(norm_img), np.min(norm_img))
        norm_img = norm_img + [127.5, 127.5, 127.5]
        print("norm_img1:", np.max(norm_img), np.min(norm_img))
        image = np.array(norm_img, np.uint8)
        img_path = img_path.numpy()
        img_text = img_text.numpy()
        txt_len  = txt_len.numpy()
        txt_index = txt_index.numpy()
        coord = coord.numpy()
        norm_w = norm_w.numpy()
        xmin, ymin, xmax, ymax = [int(x) for x in coord[0].decode('utf-8').split(',')]
        print(img_path[0])
        print(coord[0])
        print(txt_index[0])
        print(img_text[0])
        print(txt_len[0])
        print(norm_w[0])
        cv2.rectangle(image[0], (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
        cv2.imshow("image", image[0])
        cv2.waitKey(0)


def FileDatasetTestJoinModel():
    tf.enable_eager_execution()
    configs = {}
    configs['norm_h'] = 32
    configs['expand_rate'] = 1.0
    configs['file_list'] = ['/Users/junhuang.hj/Desktop/code_paper/data/sample.legal']
    configs['num_parallel'] = 4
    configs['batch_size'] = 1
    configs['char_dict'] = 'char_dict.lst'
    configs['model_type'] = 'ctc_attention'
    configs['mode'] = 'train'

    dataset = FileDataset(configs)
    dataset = dataset.data_reader_ctc_attention()

    for line in dataset:
        img_path, norm_img, img_text, ctc_idx, ctc_len, att_idx, att_len, coord, norm_w = line
        norm_img = norm_img.numpy()
        print("norm_img0:", np.max(norm_img), np.min(norm_img))
        norm_img = norm_img + [127.5, 127.5, 127.5]
        print("norm_img1:", np.max(norm_img), np.min(norm_img))
        image = np.array(norm_img, np.uint8)
        img_path = img_path.numpy()
        img_text = img_text.numpy()
        ctc_len = ctc_len.numpy()
        ctc_idx = ctc_idx.numpy()
        att_len = att_len.numpy()
        att_idx = att_idx.numpy()
        coord = coord.numpy()
        norm_w = norm_w.numpy()
        xmin, ymin, xmax, ymax = [int(x) for x in coord[0].decode('utf-8').split(',')]
        print('img_path:', img_path[0])
        print('img_text:', img_text[0])
        print('ctc_idx:', ctc_idx)
        print('ctc_len:', ctc_len)
        print('att_idx:', att_idx)
        print('att_idx:', att_len)
        print('img_coord:', coord[0])
        print('img_normw:', norm_w[0])
        cv2.rectangle(image[0], (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
        cv2.imshow("image", image[0])
        cv2.waitKey(0)



def RecordDatasetTestSoloModel():
    configs = {}
    configs['norm_h'] = 32
    configs['expand_rate'] = 1.0
    configs['file_list'] = ['../seq2seq/tfrecord_dir/tfrecord.list.0']
    configs['num_parallel'] = 4
    configs['batch_size'] = 1
    configs['char_dict'] = 'char_dict.lst'
    configs['model_type'] = 'attention'
    configs['mode'] = 'train'

    dataset = RecordDataset(configs)
    dataset = dataset.data_reader()

    for line in dataset:
        img_path, norm_img, img_text, txt_index, txt_len, coord, norm_w = line
        norm_img = norm_img + [102.9801, 115.9465, 122.7717]
        image = np.array(norm_img.numpy(), np.uint8)
        img_path = img_path.numpy()
        img_text = img_text.numpy()
        txt_index = txt_index.numpy()
        txt_len = txt_len.numpy()
        coord = coord.numpy()
        norm_w = norm_w.numpy()
        xmin, ymin, xmax, ymax = [int(x) for x in coord[0].decode('utf-8').split(',')]
        print(img_path[0])
        print(coord[0])
        print(txt_index[0])
        print(img_text[0])
        print(txt_len[0])
        print(norm_w[0])
        cv2.rectangle(image[0], (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
        cv2.imshow("image", image[0])
        cv2.waitKey(0)


def RecordDatasetTestJoinModel():
    tf.enable_eager_execution()
    configs = {}
    configs['norm_h'] = 32
    configs['expand_rate'] = 1.0
    configs['file_list'] = ['../seq2seq/tfrecord_dir/tfrecord.list.0']
    configs['num_parallel'] = 4
    configs['batch_size'] = 1
    configs['char_dict'] = 'char_dict.lst'
    configs['model_type'] = 'ctc_attention'
    configs['mode'] = 'train'


    dataset = RecordDataset(configs)

    tt = dataset.get_idstr_by_charstr('123467HIJG')

    print(tt)

    dataset = dataset.data_reader_ctc_attention()

    for line in dataset:
        img_path, norm_img, img_text, ctc_idx, ctc_len, att_idx, att_len, coord, norm_w = line
        norm_img = norm_img + [127.5, 127.5, 127.5]
        image = np.array(norm_img.numpy(), np.uint8)
        img_path = img_path.numpy()
        img_text = img_text.numpy()
        ctc_idx = ctc_idx.numpy()
        ctc_len = ctc_len.numpy()
        att_idx = att_idx.numpy()
        att_len = att_len.numpy()
        coord = coord.numpy()
        norm_w = norm_w.numpy()
        xmin, ymin, xmax, ymax = [int(x) for x in coord[0].decode('utf-8').split(',')]
        print('img_path:', img_path[0])
        print("img_text:", img_text[0])
        print('ctc_index:', ctc_idx[0])
        print('ctc_len:', ctc_len[0])
        print('att_index:', att_idx[0])
        print('att_len:', att_len[0])
        print('norm_width:', norm_w[0])
        print('img_coord:', coord[0])
        cv2.rectangle(image[0], (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
        cv2.imshow("image", image[0])
        cv2.waitKey(0)


if __name__ == '__main__':
    #augmentation_test(img_path)
    #FileDatasetTestSoloModel()
    #FileDatasetTestJoinModel()
    RecordDatasetTestSoloModel()
    #RecordDatasetTestJoinModel()




