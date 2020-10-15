import tensorflow as tf
import argparse, os, traceback
from models.datafeeder import DataFeeder
from models.tacotron import Tacotron
from util.plot_alignment import plot_alignment
from util.text import sequence_to_text
from util.hparams import *


def train(args):
    save_dir = './checkpoint/1'
    checkpoint_path = os.path.join(save_dir, 'model.ckpt')

    coord = tf.train.Coordinator()
    feeder = DataFeeder(coord, mode=1)

    model = Tacotron()
    model.initialize(feeder.enc_input, feeder.sequence_length, feeder.dec_input, feeder.mel_target)
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

    with tf.Session() as sess:
        try:
            sess.run(tf.global_variables_initializer())

            if args.step:
                load_dir = '{}-{}'.format(checkpoint_path, args.step)
                saver.restore(sess, load_dir)

            feeder.start_in_session(sess)

            while not coord.should_stop():
                step, loss, optimize = sess.run([model.global_step, model.loss, model.optimize])
                print('Step: {}, Loss: {:.5f}'.format(step, loss))

                if step % checkpoint_step == 0:
                    saver.save(sess, checkpoint_path, global_step=step)
                    input_seq, alignment, pred, target = \
                        sess.run([model.enc_input[0], model.alignment[0], model.mel_output[0], model.mel_target[0]])

                    input_seq = sequence_to_text(input_seq)
                    alignment_dir = os.path.join(save_dir, 'step-{}-align.png'.format(step))
                    plot_alignment(alignment, alignment_dir, input_seq)

        except Exception as e:
            traceback.print_exc()
            coord.request_stop(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int)
    args = parser.parse_args()
    os.makedirs('./checkpoint/1', exist_ok=True)
    train(args)
