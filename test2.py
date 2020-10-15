import numpy as np
import tensorflow as tf
import os, argparse, glob, scipy, librosa
from scipy import signal
from scipy.io.wavfile import write
from models.tacotron import post_CBHG
from models.modules import griffin_lim
from util.hparams import *

mel_list = glob.glob(os.path.join('./output', '*.npy'))

checkpoint_dir = './checkpoint'


class Synthesizer:
    def load(self, step):
        mel_input = tf.placeholder(tf.float32, [1, None, mel_dim])

        self.model = post_CBHG()
        self.model.initialize(mel_input)

        self.mel_input = self.model.mel_input[0]
        self.spec_output = self.model.spec_output[0]

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.session, os.path.join(checkpoint_dir, '2/model.ckpt-{}'.format(step)))


    def synthesize(self, args, mel, idx):
        mel = np.expand_dims(mel, axis=0)
        pred = self.session.run(self.spec_output, feed_dict={self.model.mel_input: mel})

        pred = np.transpose(pred)

        pred = (np.clip(pred, 0, 1) * max_db) - max_db + ref_db
        pred = np.power(10.0, pred * 0.05)
        wav = griffin_lim(pred ** 1.5)
        wav = scipy.signal.lfilter([1], [1, -preemphasis], wav)
        endpoint = librosa.effects.split(wav, frame_length=win_length, hop_length=hop_length)[0, 1]
        wav = wav[:endpoint]
        wav = wav.astype(np.float32)
        scipy.io.wavfile.write(os.path.join(args.save_dir, '{}.wav'.format(idx)), sample_rate, wav)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', required=True)
    parser.add_argument('--save_dir', default='./output')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    synth = Synthesizer()
    synth.load(args.step)
    for i, fn in enumerate(mel_list):
        mel = np.load(fn)
        synth.synthesize(args, mel, i)
