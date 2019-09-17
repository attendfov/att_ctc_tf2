# _*_ coding:utf-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(lineno)s - %(funcName)s - %(message)s')
logger = logging.getLogger()
#time_func = lambda: time.time()*1000
time_func = lambda: time.clock()*1000