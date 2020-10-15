import librosa
import tensorflow as tf
import numpy as np
from copy import deepcopy
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.contrib.rnn import GRUCell, RNNCell
from util.hparams import *


def pre_net(input_data, training):
    x = Dense(256)(input_data)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x, training=training)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x, training=training)
    return x


def CBHG(input_data, sequence_length, K, conv_dim):
    x = tf.concat([
        Activation('relu')(BatchNormalization()(
            Conv1D(128, kernel_size=k, strides=1, padding='same')(input_data))) for k in range(1, K+1)], axis=-1)

    x = MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
    x = Conv1D(conv_dim[0], kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(conv_dim[1], kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    highway_input = input_data + x

    if K == 8:
        highway_input = Dense(128)(highway_input)

    for _ in range(4):
        H = Dense(128)(highway_input)
        H = Activation('relu')(H)
        T = Dense(128, bias_initializer=tf.constant_initializer(-1.0))(highway_input)
        T = Activation('sigmoid')(T)
        highway_input = H * T + highway_input * (1.0 - T)

    x, _ = tf.nn.bidirectional_dynamic_rnn(
        GRUCell(128),
        GRUCell(128),
        highway_input,
        sequence_length=sequence_length,
        dtype=tf.float32)
    x = tf.concat(x, axis=2)

    return x


class ConcatWrapper(RNNCell):
    def __init__(self, cell):
        super(ConcatWrapper, self).__init__()
        self.cell = cell

    @property
    def state_size(self):
        return self.cell.state_size

    @property
    def output_size(self):
        return self.cell.output_size + self.cell.state_size.attention

    def zero_state(self, batch_size, dtype):
        return self.cell.zero_state(batch_size, dtype)

    def call(self, inputs, state):
        output, res_state = self.cell(inputs, state)
        return tf.concat([output, res_state.attention], axis=-1), res_state


def griffin_lim(spectrogram):
    spec = deepcopy(spectrogram)
    for i in range(50):
        est_wav = librosa.istft(spec, hop_length=hop_length, win_length=win_length)
        est_stft = librosa.stft(est_wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        phase = est_stft / np.maximum(1e-8, np.abs(est_stft))
        spec = spectrogram * phase
    wav = librosa.istft(spec, hop_length=hop_length, win_length=win_length)
    return np.real(wav)
