from tensorflow.contrib.seq2seq import AttentionWrapper, BahdanauAttention
from tensorflow.keras.layers import Embedding, GRU
from tensorflow.keras.losses import MAE
from .modules import *
from util.hparams import *


class Tacotron():

    def initialize(self, enc_input, sequence_length, dec_input, mel_target=None):
        is_training = 1 if mel_target is not None else 0
        batch = enc_input.shape[0]

        embedding = Embedding(symbol_length, embedding_dim)(enc_input)
        enc_pre = pre_net(embedding, is_training)
        enc_out = CBHG(enc_pre, sequence_length, K=16, conv_dim=[128, 128])

        dec_pre = pre_net(dec_input, is_training)

        attention_cell = AttentionWrapper(GRUCell(decoder_dim),
                                          BahdanauAttention(decoder_dim, enc_out),
                                          alignment_history=True,
                                          output_attention=False)

        concat_cell = ConcatWrapper(attention_cell)

        attention_out, state = tf.nn.dynamic_rnn(concat_cell, dec_pre, dtype=tf.float32)
        alignment = tf.transpose(state.alignment_history.stack(), [1, 2, 0])

        residual_gru_input = Dense(decoder_dim)(attention_out)

        for _ in range(2):
            residual_gru_input += GRU(decoder_dim, return_sequences=True)(residual_gru_input)

        dec_out = Dense(mel_dim * reduction)(residual_gru_input)

        mel_output = tf.reshape(dec_out, [batch, -1, mel_dim])

        self.enc_input = enc_input
        self.sequence_length = sequence_length
        self.dec_input = dec_input
        self.mel_output = mel_output
        self.alignment = alignment
        self.mel_target = mel_target

        if is_training:
            self.loss = tf.reduce_mean(MAE(self.mel_target, self.mel_output))
            self.global_step = tf.Variable(0)
            optimizer = tf.train.AdamOptimizer()
            gv = optimizer.compute_gradients(self.loss)
            self.optimize = optimizer.apply_gradients(gv, global_step=self.global_step)


class post_CBHG():

    def initialize(self, mel_input, spec_target=None):
        is_training = 1 if spec_target is not None else 0

        spec_output = CBHG(mel_input, None, K=8, conv_dim=[256, mel_dim])
        spec_output = Dense(n_fft // 2 + 1)(spec_output)

        self.mel_input = mel_input
        self.spec_target = spec_target
        self.spec_output = spec_output

        if is_training:
            self.loss = tf.reduce_mean(MAE(self.spec_target, self.spec_output))
            self.global_step = tf.Variable(0)
            optimizer = tf.train.AdamOptimizer()
            gv = optimizer.compute_gradients(self.loss)
            self.optimize = optimizer.apply_gradients(gv, global_step=self.global_step)
