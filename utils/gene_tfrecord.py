# _*_ coding:utf-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import cv2
import multiprocessing
import tensorflow as tf


def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def datalist_to_tfrecord(lines, record_type, tfrecord_name, posfix):
    assert(record_type in ('aug', 'std'))
    writer = tf.python_io.TFRecordWriter(tfrecord_name + str(posfix))

    for line in lines:
        try:
            line_sp = line.strip().split(' ')
            if len(line_sp) < 3:
                continue
            img_path, = line_sp[0]
            label = ' '.join(line_sp[1:-1])
            coord = line_sp[-1]
            if not os.path.isfile(img_path) or os.path.getsize(img_path) == 0:
                continue

            xmin, ymin, xmax, ymax = [int(x) for x in coord.split(',')]
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            imgh, imgw, imgc = [0, 0, 0]
            shape = image.shape
            imgh, imgw = shape[:2]
            if len(shape) >= 3:
                imgh,imgw,imgc = shape[:3]

            if xmax > imgw or ymax > imgh:
                continue

            patch_w_off = xmax-xmin
            patch_h_off = ymax-ymin

            patch_xmin = xmin
            patch_ymin = ymin
            patch_xmax = xmax
            patch_ymax = ymax
            if record_type == 'aug':
                patch_size = min(patch_xmax-patch_xmin, patch_ymax-patch_xmin)
                patch_xmin = max(0, int(xmin-max(0.2*patch_w_off, patch_size)))
                patch_ymin = max(0, int(ymin-max(0.2*patch_h_off, patch_size)))
                patch_xmax = min(imgw, int(xmax+max(0.2*patch_w_off, patch_size)))
                patch_ymax = min(imgh, int(ymax+max(0.2*patch_h_off, patch_size)))

            image = image[patch_ymin:patch_ymax, patch_xmin:patch_xmax]
            imgh, imgw = image.shape[:2]
            image = image.tostring()
            xmin = xmin - patch_xmin
            ymin = ymin - patch_ymin
            xmax = xmin + patch_w_off
            ymax = ymin + patch_h_off
            coord = ','.join([str(xmin), str(ymin), str(xmax), str(ymax)])

            example = tf.train.Example(features=tf.train.Features(feature={
                'width': _int64_features(imgw),
                'height': _int64_features(imgh),
                'channel': _int64_features(imgc),
                'image_raw': _bytes_features(image),
                'img_path': _bytes_features(img_path.encode('utf8')),
                'label': _bytes_features(label.encode('utf8')),
                'coord': _bytes_features(coord.encode('utf8'))
            }))

            writer.write(example.SerializeToString())
        except Exception as e:
            print("datalist_to_tfrecord Exception:{}".format(str(e)))
    writer.close()


def gene_tfrecord(list_file, save_dir, tfrecord_name, record_type='aug', processes=32):
    assert(os.path.isfile(list_file))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    tfrecord_name = os.path.join(save_dir, tfrecord_name)

    reader = io.open(list_file, 'r', encoding='utf-8')
    lines = reader.readlines()
    print('lines:', len(lines))
    reader.close()

    proc_size = int(len(lines)/processes+1)
    proc_pool = multiprocessing.Pool(processes=processes)
    for proc_id in range(processes):
        posfix = '.' + str(proc_id)
        block_list = lines[proc_id*proc_size:(proc_id+1)*proc_size]
        block_param = [block_list, record_type, tfrecord_name, posfix]
        proc_pool.apply_async(datalist_to_tfrecord, block_param)

    proc_pool.close()
    proc_pool.join()


def parse_tfrecord(serial_example):
    feat_dict = tf.parse_single_example(serial_example,
                                        features={
                                            'width': tf.FixedLenFeature([], tf.int64),
                                            'height': tf.FixedLenFeature([], tf.int64),
                                            'channel': tf.FixedLenFeature([], tf.int64),
                                            'coord': tf.FixedLenFeature([], tf.string),
                                            'label': tf.FixedLenFeature([], tf.string),
                                            'img_raw': tf.FixedLenFeature([], tf.string),
                                            'img_path': tf.FixedLenFeature([], tf.string)
                                        })

    width = feat_dict['width']
    height = feat_dict['height']
    channel = feat_dict['channel']
    img_raw = feat_dict['img_raw']
    img_path = feat_dict['img_path']
    coord = feat_dict['coord']
    label = feat_dict['label']

    img_raw = tf.decode_raw(img_raw, tf.uint8)
    img_raw = tf.reshape(img_raw, (height, width, channel))
    return img_raw, label, height, width, channel, img_path, coord


def show_tfrecord(tfrecord_pattern_file):
    file_names = tf.gfile.Glob(tfrecord_pattern_file)
    dataset = tf.data.TFRecordDataset(file_names)
    dataset = dataset.map(parse_tfrecord)
    for line in dataset:
        img_raw, label, height, width, channel, img_path, coord = line
        label = label.numpy().decode('utf-8')
        coord = coord.numpy().decode('utf-8')
        img_path = img_path.numpy().decode('utf-8')
        image = img_raw.numpy()
        xmin,ymin,xmax,ymax = [int(x) for x in coord.split(',')]
        print('label:', label)
        print('coord:', coord)
        print('imgpath:', img_path)
        print('shape:', image.shape)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
        cv2.imshow('image:', image)
        cv2.waitKey(0)


if __name__=='__main__':
    tf.enable_eager_execution()
    list_file = 'tt.list'
    save_dir = 'tfrecord_dir'
    tfrecord_name = 'tfrecord.list'
    gene_tfrecord(list_file, save_dir, tfrecord_name, record_type='aug', processes=32)

    tfrecord = 'tfrecord_dir/tfrecord.list.*'
    show_tfrecord(tfrecord)











