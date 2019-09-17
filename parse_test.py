import os
import numpy as np

import argparse  # 导入模块，这个没什么说的


def parse_test():
    parser = argparse.ArgumentParser(description='datacube parser')
    parser.add_argument()
    parser.parse_args()

import argparse
parser = argparse.ArgumentParser(description='解析命令行参数')
parser.add_argument('--defatut', help='default help', default='default111')
parser.add_argument('--require', help='require help', required=True)
args = parser.parse_args()
print(args.defatut, type(args.defatut))
print(args.require, type(args.require))
