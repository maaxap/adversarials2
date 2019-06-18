import os
from os.path import join
import math
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf

from model import VGG19
from util import init_logger
from data import make_dataset, resize, read_image, decode_png


logger = init_logger("predictions")


def fetch_data(sess, model, dataset, attack_dir, iter_dir):
  assert dataset.batch_size
  assert dataset.is_shuffled == False

  batch_size = dataset.batch_size
  filenames = dataset.filenames
  labels = dataset.labels

  data = []

  iterator = tf.data.Iterator.from_structure(
    dataset.dataset.output_types, dataset.dataset.output_shapes)
  dataset_init_op = iterator.make_initializer(dataset.dataset)

  x, y = iterator.get_next()
  probs = model.get_probs(x)

  num_batches = int(math.ceil(float(dataset.size) / batch_size))

  with sess.as_default():
    sess.run(dataset_init_op)
    for i in range(num_batches):
      batch_slice = slice(i * batch_size, (i + 1) * batch_size)
      filenames_batch, labels_batch = filenames[batch_slice], labels[batch_slice]
      x_batch_np, y_batch_np, probs_batch_np = sess.run([x, y, probs])

      gen = zip(x_batch_np, y_batch_np, probs_batch_np, filenames_batch, labels_batch)
      for x_np, y_np, probs_np, filename, label in gen:
        assert np.argmax(y_np) == label
        data_chunk = (filename, attack_dir, int(iter_dir), label,
                      probs_np[0], probs_np[1], np.linalg.norm(x_np))
        data.append(data_chunk)

  return data


def main(args):
  os.environ['CUDA_VISIBLE_DEVICES'] = args.device

  sess = tf.Session()
  model = VGG19(nb_classes=2, is_training=False)
  img_height, img_width, _ = model.SHAPE

  saver = tf.train.Saver()
  saver.restore(sess, args.checkpoint)

  columns = ['filename', 'attack', 'iteration', 'label',
             'class_0_pred', 'class_1_pred', 'l2_norm']

  data = []

  def preprocess_func(x, y):
    x = read_image(x)
    x = decode_png(x)
    x = resize(x, img_height, img_width)
    return x, y

  allowed_attacks = set([a.strip() for a in args.attacks.split(',')
                         if len(a.strip()) > 0])


  for attack_dir in os.listdir(args.data):

    if not attack_dir in allowed_attacks:
      continue

    logger.debug("Start processing attack {}".format(attack_dir))

    for iter_dir in os.listdir(join(args.data, attack_dir)):
      logger.debug("Start processing iteration #{}".format(int(iter_dir)))

      source_dir = join(args.data, attack_dir, iter_dir)
      dataset = make_dataset(source_dir, 128, preprocess_func, shuffle=False, repeat=False)
      iter_data = fetch_data(sess, model, dataset, attack_dir, iter_dir)
      data.extend(iter_data)

  dataframe = pd.DataFrame(data=data, columns=columns)
  dataframe.to_csv(args.csv, index=False)


if __name__ == '__main__':
  tf.set_random_seed(2019)

  parser = argparse.ArgumentParser()
  parser.add_argument('--data', required=True)
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--csv', required=True)
  parser.add_argument('--attacks', required=True)
  parser.add_argument('--device', required=True)
  args = parser.parse_args()

  main(args)

