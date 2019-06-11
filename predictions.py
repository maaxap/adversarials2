import os
from os.path import join
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf

from model import VGG19
from util import init_logger


logger = init_logger("adversarials")


def fetch_data(sess, x, probs, img_shape, adv_dir, attack_dir, iter_dir):
  img_height, img_width, _ = img_shape

  data = []

  with sess.as_default():
    iter_dir_path = join(adv_dir, attack_dir, iter_dir)
    for class_dir in sorted(os.listdir(iter_dir_path)):
      for filename in os.listdir(join(iter_dir_path, class_dir)):
        label = int(filename.split('_')[3][1:-4])
        image = Image.open(join(iter_dir_path, class_dir, filename))
        image_resized = image.resize((img_width, img_height), Image.NEAREST)
        image_np = np.array(image_resized)[:, :, :3]
        probs_np, = sess.run(probs, feed_dict={x : np.expand_dims(image_np, axis=0)})
        data_chunk = (filename, attack_dir, int(iter_dir),
                      label, probs_np[0], probs_np[1])
        data.append(data_chunk)

  return data


def main(args):
  sess = tf.Session()
  model = VGG19(is_training=False)
  img_shape = model.SHAPE

  saver = tf.train.Saver()
  saver.restore(sess, args.checkpoint)

  x, y = model.make_input_placeholder(), model.make_label_placeholder()
  probs = model.get_probs(x)

  columns = ['filename', 'attack', 'iteration', 'label',
             'class_0_pred', 'class_1_pred']

  data = []
  for attack_dir in os.listdir(args.data):
    logger.debug("Start processing attack {}".format(attack_dir))

    for iter_dir in os.listdir(join(args.data, attack_dir)):
      logger.debug("Start processing iteration #{}".format(int(iter_dir)))

      iter_data = fetch_data(sess, x, probs, img_shape,
                             args.data, attack_dir, iter_dir)
      data.extend(iter_data)

  dataframe = pd.DataFrame(data=data, columns=columns)
  dataframe.to_csv(args.csv, index=False)


if __name__ == '__main__':
  tf.set_random_seed(2019)

  parser = argparse.ArgumentParser()
  parser.add_argument('--data', required=True)
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--csv', required=True)
  args = parser.parse_args()

  main(args)
