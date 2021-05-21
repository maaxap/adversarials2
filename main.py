import math
import os
from os.path import join
import argparse
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf

from data import make_dataset, read_image, decode_png
from data import random_crop, random_flip, resize, save_image_np
from util import init_logger
from model import VGG19, InceptionV3
from attack import MomentumIterativeMethod, NesterovIterativeMethod
from attack import AdagradIterativeMethod, AdadeltaIterativeMethod
from attack import RMSPropIterativeMethod, AdamIterativeMethod
from cleverhans.attacks.fast_gradient_method import FastGradientMethod


# Common parameters
DEFAULT_TRAIN_DIR = '/path/to/train'
DEFAULT_VAL_DIR = '/path/to/val'
DEFAULT_TEST_DIR = '/path/to/test'
DEFAULT_CHECKPOINT_PATH = 'models/vgg/vgg19_model_v2.ckpt'

# Train parameters
DEFAULT_NUM_EPOCHS = 100
DEFAULT_TRAIN_BATCH_SIZE = 128
DEFAULT_INITIAL_LEARNING_RATE = 1e-4
DEFAULT_LEARNING_RATE_DECAY = 1e-1
DEFAULT_ACCURACY_DELTA = 1e-5

# Attack parameters
DEFAULT_ADV_DIR = '/path/to/data/hist/adv'
DEFAULT_ATTACK_BATCH_SIZE = 128
DEAFULT_EPS = 60. / 255
DEFAULT_NUM_ITER = 20
# DEFAULT_EPS_ITER = 1. / 255
# DEFAULT_DECAY_FACTOR = 0.9


def main(args, logger):
  # Load model
  sess = tf.Session()
  model = VGG19(nb_classes=2, is_training=False)

  chkp_path = args.checkpoint_path

  logger.debug("Loading model from: {}".format(chkp_path))

  saver = tf.train.Saver()
  saver.restore(sess, chkp_path)

  img_shape = model.SHAPE
  leg_dir = args.test_dir
  batch_size = args.attack_batch_size

  img_height, img_width, num_channels = img_shape

  def preprocess_func(x, y):
    x = read_image(x)
    x = decode_png(x)
    x = resize(x, img_height, img_width)
    return x, y

  logger.debug("Initializing legitimate dataset from: {}".format(leg_dir))

  dataset = make_dataset(leg_dir, batch_size, preprocess_func,
                         shuffle=False, repeat=False)

  iterator = tf.data.Iterator.from_structure(
    dataset.dataset.output_types, dataset.dataset.output_shapes)
  dataset_init_op = iterator.make_initializer(dataset.dataset)

  x, y = iterator.get_next()

  num_batches = int(math.ceil(float(dataset.size) / batch_size))

  adv_dir = args.adv_dir
  eps = args.eps
  nb_iter = args.num_iter
  eps_iter = eps/nb_iter
  ord = np.inf

  attack_param_grid = {
    AdadeltaIterativeMethod: {
      'adv_dir': join(adv_dir, 'adadelta'),
      'init_op': dataset_init_op,
      'nb_batches': num_batches,
      'logger': logger,
      'clip_min': 0.0,
      'clip_max': 1.0,
      'nb_iter': nb_iter,
      'eps_iter': eps_iter,
      'ord': ord,
      'y': y,

      'eps': eps,
      'gamma': 0.9
    },
  }

  for attack_class, attack_params in attack_param_grid.items():
    logger.debug("Starting attack images with {}".format(attack_class.__name__))

    attack = attack_class(model, sess)

    if not os.path.exists(attack_params['adv_dir']):
      os.mkdir(attack_params['adv_dir'])

    attack.generate(x, **attack_params)


if __name__ == '__main__':
  tf.set_random_seed(2019)

  logger = init_logger("adversarials-2dadelta")

  parser = argparse.ArgumentParser()

  # Common parameters
  parser.add_argument('--train-dir', default=DEFAULT_TRAIN_DIR)
  parser.add_argument('--val-dir', default=DEFAULT_VAL_DIR)
  parser.add_argument('--test-dir', default=DEFAULT_TEST_DIR)
  parser.add_argument('--checkpoint-path', default=DEFAULT_CHECKPOINT_PATH)

  # Train parameters
  parser.add_argument('--num-epochs', default=DEFAULT_NUM_EPOCHS)
  parser.add_argument('--train-batch-size', default=DEFAULT_TRAIN_BATCH_SIZE)
  parser.add_argument('--initial-learning-rate',
                      default=DEFAULT_INITIAL_LEARNING_RATE)
  parser.add_argument('--learning-rage-decay',
                      default=DEFAULT_LEARNING_RATE_DECAY)
  parser.add_argument('--accuracy-delta', default=DEFAULT_ACCURACY_DELTA)

  # Attack parameters
  parser.add_argument('--eps', default=DEAFULT_EPS)
  parser.add_argument('--attack-batch-size', default=DEFAULT_ATTACK_BATCH_SIZE)
  parser.add_argument('--adv-dir', default=DEFAULT_ADV_DIR)
  parser.add_argument('--num-iter', default=DEFAULT_NUM_ITER)
  # parser.add_argument('--eps-iter', default=DEFAULT_EPS_ITER)
  # parser.add_argument('--decay_factor', default=DEFAULT_DECAY_FACTOR)

  # Parse parameters
  args = parser.parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = '2'

  main(args, logger)
