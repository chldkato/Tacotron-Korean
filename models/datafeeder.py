import numpy as np
import tensorflow as tf
import os, random, threading, traceback, glob
from tensorflow.keras.preprocessing.sequence import pad_sequences
from util.hparams import *


data_dir = './data'
text_list = glob.glob(os.path.join(data_dir + '/text', '*.npy'))
mel_list = glob.glob(os.path.join(data_dir + '/mel', '*.npy'))
dec_list = glob.glob(os.path.join(data_dir + '/dec', '*.npy'))
spec_list = glob.glob(os.path.join(data_dir + '/spec', '*.npy'))
text_len = np.load(os.path.join(data_dir + '/text_len.npy'))
mel_len = np.load(os.path.join(data_dir + '/mel_len.npy'))


class DataFeeder(threading.Thread):
    def __init__(self, coordinator, mode):
        super(DataFeeder, self).__init__()
        self.coord = coordinator
        self.mode = mode

        if self.mode == 1:
            self.placeholder = [tf.placeholder(tf.int32, [batch_size, None]),
                                tf.placeholder(tf.int32, [batch_size]),
                                tf.placeholder(tf.float32, [batch_size, None, mel_dim]),
                                tf.placeholder(tf.float32, [batch_size, None, mel_dim])]

            queue = tf.FIFOQueue(8, [tf.int32, tf.int32, tf.float32, tf.float32])
            self.enqueue = queue.enqueue(self.placeholder)
            self.enc_input, self.sequence_length, self.dec_input, self.mel_target = queue.dequeue()
            self.enc_input.set_shape(self.placeholder[0].shape)
            self.sequence_length.set_shape(self.placeholder[1].shape)
            self.dec_input.set_shape(self.placeholder[2].shape)
            self.mel_target.set_shape(self.placeholder[3].shape)

        else:
            self.placeholder = [tf.placeholder(tf.float32, [batch_size, None, mel_dim]),
                                tf.placeholder(tf.float32, [batch_size, None, n_fft // 2 + 1])]

            queue = tf.FIFOQueue(8, [tf.float32, tf.float32])
            self.enqueue = queue.enqueue(self.placeholder)
            self.mel_input, self.spec_target = queue.dequeue()
            self.mel_input.set_shape(self.placeholder[0].shape)
            self.spec_target.set_shape(self.placeholder[1].shape)

    def start_in_session(self, session):
        self.session = session
        self.start()

    def run(self):
        try:
            while not self.coord.should_stop():
                if self.mode == 1:
                    idx_list = np.random.choice(len(mel_list), batch_size ** 2, replace=False)
                    idx_list = sorted(idx_list)
                    idx_list = [idx_list[i: i + batch_size] for i in range(0, len(idx_list), batch_size)]
                    random.shuffle(idx_list)

                    batches = []
                    for idx in idx_list:
                        random.shuffle(idx)

                        text = [np.load(text_list[mel_len[i][1]]) for i in idx]
                        dec = [np.load(dec_list[mel_len[i][1]]) for i in idx]
                        mel = [np.load(mel_list[mel_len[i][1]]) for i in idx]
                        text_length = [text_len[mel_len[i][1]] for i in idx]

                        text = pad_sequences(text, padding='post')
                        dec = pad_sequences(dec, padding='post', dtype='float32')
                        mel = pad_sequences(mel, padding='post', dtype='float32')

                        batches.append([text, text_length, dec, mel])

                    for batch in batches:
                        feed_dict = dict(zip(self.placeholder, batch))
                        self.session.run(self.enqueue, feed_dict=feed_dict)

                if self.mode == 2:
                    idx_list = np.random.choice(len(mel_list), batch_size ** 2, replace=False)
                    idx_list = sorted(idx_list)
                    idx_list = [idx_list[i: i + batch_size] for i in range(0, len(idx_list), batch_size)]
                    random.shuffle(idx_list)

                    batches = []
                    for idx in idx_list:
                        random.shuffle(idx)

                        mel = [np.load(mel_list[mel_len[i][1]]) for i in idx]
                        spec = [np.load(spec_list[mel_len[i][1]]) for i in idx]

                        mel = pad_sequences(mel, padding='post', dtype='float32')
                        spec = pad_sequences(spec, padding='post', dtype='float32')

                        batches.append([mel, spec])

                    for batch in batches:
                        feed_dict = dict(zip(self.placeholder, batch))
                        self.session.run(self.enqueue, feed_dict=feed_dict)

        except Exception as e:
            traceback.print_exc()
            self.coord.request_stop(e)
