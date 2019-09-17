# _*_ coding:utf-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import sys
import time
import numpy as np
import tensorflow as tf
import threading
import multiprocessing
try:
    import queue
except ImportError:
    import Queue as queue

'''
this file is modified from keras implemention of data process multi-threading,
see https://github.com/fchollet/keras/blob/master/keras/utils/data_utils.py
'''


class GeneratorEnqueuer():
    """Builds a queue out of a data generator.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        generator: a generator function which endlessly yields data
        use_multiprocessing: use multiprocessing if True, otherwise threading
        wait_time: time to sleep in-between calls to `put()`
        random_seed: Initial seed for workers,
            will be incremented by one for each workers.
    """

    def __init__(self, generator,
                 use_multiprocessing=False,
                 wait_time=0.05,
                 random_seed=None):
        self.wait_time = wait_time
        self._generator = generator
        self._use_multiprocessing = use_multiprocessing
        self._threads = []
        self._stop_event = None
        self.queue = None
        self.random_seed = random_seed

    def start(self, workers=1, max_queue_size=10):
        """Kicks off threads which add data from the generator into the queue.
        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, threads could block on `put()`)
        """

        def data_generator_task():
            while not self._stop_event.is_set():
                try:
                    if self._use_multiprocessing or self.queue.qsize() < max_queue_size:
                        generator_output = next(self._generator)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(self.wait_time)
                except Exception:
                    self._stop_event.set()
                    raise

        try:
            if self._use_multiprocessing:
                self.queue = multiprocessing.Queue(maxsize=max_queue_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = queue.Queue()
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._use_multiprocessing:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed(self.random_seed)
                    thread = multiprocessing.Process(target=data_generator_task)
                    thread.daemon = True
                    if self.random_seed is not None:
                        self.random_seed += 1
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.
        Should be called by the same thread which called `start()`.
        # Arguments
            timeout: maximum time to wait on `thread.join()`.
        """
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                if self._use_multiprocessing:
                    thread.terminate()
                else:
                    thread.join(timeout)

        if self._use_multiprocessing:
            if self.queue is not None:
                self.queue.close()

        self._threads = []
        self._stop_event = None
        self.queue = None

    def get(self):
        """Creates a generator to extract data from the queue.
        Skip the data if it is `None`.
        # Returns
            A generator
        """
        while self.is_running():
            if not self.queue.empty():
                inputs = self.queue.get()
                if inputs is not None:
                    yield inputs
            else:
                time.sleep(self.wait_time)


def gene_digital(count=100):
    while True:
        for i in range(count):
            yield i


def get_batch(generator, num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        print('Generator use 128 batches for buffering, this may take a while, you can tune this yourself.')
        enqueuer.start(max_queue_size=128, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


class generator:
    def __init__(self, gene_fun, work_nums, **kwargs):
        self.data_generator = get_batch(gene_fun, work_nums, **kwargs)

    def __call__(self):
        return self.data_generator


def RecordGenerator():
    tf.enable_eager_execution()
    fake_generator = generator(gene_digital, 10, count=100)
    ds = tf.data.Dataset.from_generator(fake_generator, tf.int64, tf.TensorShape([]))

    for data in ds:
        print(data)


if __name__ == '__main__':
    RecordGenerator()


