# _*_ coding:utf-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import sys
import cv2
import time
import shutil
import numpy as np
import tensorflow as tf

sys.path.append('.')
from Logger import logger

global_chr_fun = unichr if sys.version.startswith('2') else chr


def q2b_function(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif inside_code>=65281 and inside_code<=65374:
            inside_code -= 65248

        rstring += global_chr_fun(inside_code)
    return rstring


class Charset:
    def __init__(self, charset_file, model_type='attention', ignore_unk=False):
        assert(model_type.lower() in ('ctc', 'attention', 'ctc_attention', 'attention_ctc'))
        self.unk_char = '☠'
        self.model_type = model_type.lower()
        if not os.path.isfile(charset_file):
            logger.info('Charset Error: charset_file {} is not exists!!'.format(charset_file))
        if os.path.getsize(charset_file) == 0:
            logger.info('Charset Error: charset_file {} is empty!!'.format(charset_file))

        reader = io.open(charset_file, 'r', encoding='utf-8')
        lines = reader.readlines()
        reader.close()

        self.char2idx = {}
        self.idx2char = {}
        self.ignore_unk = ignore_unk
        self.start_chr = '<sos>'
        self.start_idx = -1
        self.end_chr = '<eos>'
        self.end_idx = -1

        max_idx = 0
        idx_off = len(self.char2idx)
        for index, line in enumerate(lines, idx_off):
            line = line.strip()
            if len(line) == 0:
                continue
            char = line.strip().split(' ')[0]
            if char in self.char2idx:
                continue
            self.char2idx[char] = max_idx
            self.idx2char[max_idx] = char
            max_idx = max_idx + 1

        if ignore_unk is False:
            self.char2idx[self.unk_char] = max_idx
            self.idx2char[max_idx] = self.unk_char
            max_idx = max_idx + 1

        if model_type.lower() in ('attention', 'ctc_attention', 'attention_ctc'):
            self.end_idx = max_idx
            self.start_idx = max_idx + 1
            self.char2idx[self.start_chr] = self.start_idx
            self.idx2char[self.start_idx] = self.start_chr
            self.char2idx[self.end_chr] = self.end_idx
            self.idx2char[self.end_idx] = self.end_chr

        if model_type.lower() == 'ctc':
            self.end_idx = max_idx
            self.end_chr = ' '
            self.char2idx[self.end_chr] = self.end_idx
            self.idx2char[self.end_idx] = self.end_chr

        self.char_count = len(self.idx2char)

        if len(self.char2idx) != len(self.idx2char):
            logger.info("Charset Error: char2idx size {} not equal idx2char size {}".
                        format(len(self.char2idx), len(self.idx2char)))

    def get_idxstr_by_charstr(self, charstr):
        if self.model_type in ('attention', ):
            return self.get_idxstr_by_charstr_in_att(charstr)
        elif self.model_type in ('ctc', ):
            return self.get_idxstr_by_charstr_in_ctc(charstr)
        elif self.model_type in ('attention_ctc', 'ctc_attention'):
            return self.get_idxstr_by_charstr_in_ctc(charstr), self.get_idxstr_by_charstr_in_att(charstr)

    def get_idxstr_by_charstr_in_att(self, charstr):
        id_list = [self.start_idx]
        charstr = q2b_function(charstr)
        for char in charstr:
            if char not in self.char2idx:
                if self.ignore_unk is False:
                    char = self.unk_char
                else:
                    continue
            id_list.append(self.char2idx[char])

        id_list.append(self.end_idx)
        idstr = ','.join([str(id) for id in id_list])
        return idstr

    def get_idxstr_by_charstr_in_ctc(self, charstr):
        id_list = []
        charstr = q2b_function(charstr)
        for char in charstr:
            if char not in self.char2idx:
                if self.ignore_unk is False:
                    char = self.unk_char
                else:
                    continue
            id_list.append(self.char2idx[char])

        idstr = ','.join([str(id) for id in id_list])
        return idstr

    def get_idxlist_by_charstr(self, charstr):
        if self.model_type in ('attention', ):
            return self.get_idxlist_by_charstr_in_att(charstr)
        elif self.model_type in ('ctc', ):
            return self.get_idxlist_by_charstr_in_ctc(charstr)
        elif self.model_type in ('attention_ctc', 'ctc_attention'):
            return self.get_idxlist_by_charstr_in_ctc(charstr), self.get_idxlist_by_charstr_in_att(charstr)

    def get_idxlist_by_charstr_in_att(self, charstr):
        id_list = [self.start_idx]
        charstr = q2b_function(charstr)
        for char in charstr:
            if char not in self.char2idx:
                if self.ignore_unk is False:
                    char = self.unk_char
                else:
                    continue
            id_list.append(self.char2idx[char])
        id_list.append(self.end_idx)
        return id_list

    def get_idxlist_by_charstr_in_ctc(self, charstr):
        id_list = []
        charstr = q2b_function(charstr)
        for char in charstr:
            if char not in self.char2idx:
                if self.ignore_unk is False:
                    char = self.unk_char
                else:
                    continue
            id_list.append(self.char2idx[char])

        return id_list

    def get_charstr_by_idxlist(self, id_list):
        char_list = []
        for idx in id_list:
            char = self.idx2char[idx]
            char_list.append(char)

        return ''.join(char_list)

    def get_size(self):
        return self.char_count

    def get_eosid(self):
        return self.end_idx

    def get_sosid(self):
        return self.start_idx


if __name__ == '__main__':

    char_file = 'char_dict.lst'
    model_type = 'ctc'
    charset = Charset(char_file, model_type, False)

    char_count = charset.get_size()
    index_str = charset.get_idxstr_by_charstr("kindest")
    index_lst = charset.get_idxlist_by_charstr("kindest")
    index_list = [int(x) for x in index_str.split(',')]
    char_str = charset.get_charstr_by_idxlist(index_list)
    print(index_lst)
    print(char_count)
    print(index_list)
    print(char_str)

    string = 'ｍｎ123abc中华人民共和国'
    string = q2b_function(string)
    print("string:{}".format(string))
    index_str = charset.get_idxstr_by_charstr(string)
    index_lst = charset.get_idxlist_by_charstr(string)
    index_list = [int(x) for x in index_str.split(',')]
    char_str = charset.get_charstr_by_idxlist(index_list)
    print(index_lst)
    print(index_list)
    print(char_str)
