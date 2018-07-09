#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os

import tensorflow as tf

import utils
from model import Model

from flags import parse_args
FLAGS, unparsed = parse_args()


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


vocabulary = utils.read_data(FLAGS.text)
print('Data size', len(vocabulary))


with open(FLAGS.dictionary, encoding='utf-8') as inf:
    dictionary = json.load(inf, encoding='utf-8')

with open(FLAGS.reverse_dictionary, encoding='utf-8') as inf:
    reverse_dictionary = json.load(inf, encoding='utf-8')


model = Model(learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps)
model.build(FLAGS.embedding_npy)


with tf.Session() as sess:
    summary_string_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph)

    saver = tf.train.Saver(max_to_keep=5)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    logging.debug('Initialized')

    try:
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.output_dir)
        saver.restore(sess, checkpoint_path)
        logging.debug('restore from [{0}]'.format(checkpoint_path))

    except Exception:
        logging.debug('no check point found....')

    for x in range(1):
        logging.debug('epoch [{0}]....'.format(x))
        state = sess.run(model.state_tensor)
        for dl in utils.get_train_data(vocabulary, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps):

            X_batch = [utils.index_data(ch, dictionary) for ch in dl[0]]
            Y_batch = [utils.index_data(ch, dictionary) for ch in dl[1]]
            feed_dict = {model.X:X_batch, model.Y:Y_batch, model.state_tensor:state, model.keep_prob:FLAGS.keep_prob}
            
            gs, _, state, l, summary_string = sess.run(
                [model.global_step, model.optimizer, model.outputs_state_tensor, model.loss, model.merged_summary_op], feed_dict=feed_dict)
            summary_string_writer.add_summary(summary_string, gs)

            if gs % 50 == 0:
                logging.debug('step [{0}] loss [{1}]'.format(gs, l))
            if gs % 500 == 0:
                logging.debug('save model.ckpt @{0} '.format(gs))
                save_path = saver.save(sess, os.path.join(FLAGS.output_dir, "model.ckpt"), global_step=gs)

    save_path = saver.save(sess, os.path.join(FLAGS.output_dir, "model.ckpt"), global_step=gs)
    summary_string_writer.close()
