import math
import os
import argparse
import numpy as np
import tensorflow as tf
from cleverhans.loss import CrossEntropy
from cleverhans.attacks import FastGradientMethod

from data import make_dataset, read_image
from data import decode_png, resize
from util import init_logger
from model import InceptionV3


def eval_acc(sess, logits, labels, num_batches, is_training, dataset_init_op):
  sess.run(dataset_init_op)

  pred = tf.equal(tf.to_int32(tf.argmax(logits, 1)),
                  tf.to_int32(tf.argmax(labels, 1)))

  num_correct, num_samples = 0, 0
  for batch in range(num_batches):
    pred_val = sess.run(pred, {is_training: False})
    num_correct += pred_val.sum()
    num_samples += pred_val.shape[0]

  acc = num_correct / num_samples
  return acc


def main(args):
  logger = init_logger(args.run_name)

  # Datasets
  img_height, img_width, _ = InceptionV3.SHAPE

  def prep_func(f, x, y):
    x = read_image(x)
    x = decode_png(x)
    x = resize(x, img_height, img_width)
    return f, x, y

  trn_ds = make_dataset(args.train_dir, args.batch_size, prep_func,
                        shuffle=True, repeat=True, add_filenames=True)
  val_ds = make_dataset(args.train_dir, args.batch_size, prep_func,
                        shuffle=False, repeat=False, add_filenames=True)
  tst_ds = make_dataset(args.train_dir, args.batch_size, prep_func,
                        shuffle=False, repeat=False, add_filenames=True)

  num_classes = len(trn_ds.labels_map)

  it = tf.data.Iterator.from_structure(
    trn_ds.dataset.output_types, trn_ds.dataset.output_shapes)

  num_trn_batches = int(math.ceil(float(trn_ds.size) / args.batch_size))
  num_val_batches = int(math.ceil(float(val_ds.size) / args.batch_size))
  num_tst_batches = int(math.ceil(float(tst_ds.size) / args.batch_size))

  trn_init_op = it.make_initializer(trn_ds.dataset)
  val_init_op = it.make_initializer(val_ds.dataset)
  tst_init_op = it.make_initializer(tst_ds.dataset)

  # Filename, input image and corrsponding one hot encoded label
  f, x, y = it.get_next()

  sess = tf.Session()

  # Model and logits
  is_training = tf.placeholder(dtype=tf.bool)
  model = InceptionV3(nb_classes=num_classes, is_training=is_training)
  logits = model.get_logits(x)

  attacks_ord = {
    'inf': np.inf,
    '1': 1,
    '2': 2
  }

  # FGM attack
  attack_params = {
    'eps': args.eps,
    'clip_min': 0.0,
    'clip_max': 1.0,
    'ord': attacks_ord[args.ord],
  }
  attack = FastGradientMethod(model, sess)

  # Learning rate with exponential decay
  global_step = tf.Variable(0, trainable=False)
  global_step_update_op = tf.assign(global_step, tf.add(global_step, 1))
  lr = tf.train.exponential_decay(
    args.initial_lr, global_step, args.lr_decay_steps,
    args.lr_decay_factor, staircase=True)

  cross_entropy = CrossEntropy(model, attack=attack,
                               smoothing=args.label_smth,
                               attack_params=attack_params)
  loss = cross_entropy.fprop(x, y)

  # Gradients clipping
  opt = tf.train.RMSPropOptimizer(learning_rate=lr, decay=args.opt_decay,
                                  epsilon=1.0)
  gvs = opt.compute_gradients(loss)
  clip_min, clip_max = -args.grad_clip, args.grad_clip

  capped_gvs = []
  for g, v in gvs:
    capped_g = tf.clip_by_value(g, clip_min, clip_max) \
      if g is not None else tf.zeros_like(v)
    capped_gvs.append((capped_g, v))

  train_op = opt.apply_gradients(capped_gvs)

  saver = tf.train.Saver()
  global_init_op = tf.global_variables_initializer()


  with sess.as_default():
    sess.run(global_init_op)

    best_val_acc = -1
    for epoch in range(args.num_epochs):
      logger.info("Epoch: {:04d}/{:04d}".format(epoch + 1, args.num_epochs))
      sess.run(trn_init_op)

      for batch in range(num_trn_batches):
        loss_np, lr_np, _ = sess.run([loss, lr, train_op],
                                     feed_dict={is_training: True})
        logger.info("Batch: {:04d}/{:04d}, loss: {:.05f}, lr: {:.05f}"
          .format(batch + 1, num_trn_batches, loss_np, lr_np))

      logger.info("Epoch completed...")

      sess.run(global_step_update_op)
      val_acc = eval_acc(sess, logits, y, num_val_batches,
                         is_training, val_init_op)
      logger.info("Validation set accuracy: {:.05f}".format(val_acc))

      if best_val_acc < val_acc:
        output_path = saver.save(sess, args.model_path)
        logger.info("Model was successfully saved: {}".format(output_path))
        best_val_acc = val_acc
        pass

    tst_acc = eval_acc(sess, logits, y, num_tst_batches,
                       is_training, tst_init_op)
    logger.info("Test set accuracy: {:.05f}".format(tst_acc))


if __name__ == '__main__':
  tf.set_random_seed(2019)

  parser = argparse.ArgumentParser()

  parser.add_argument('--run-name', required=True, type=str)
  parser.add_argument('--train-dir', required=True, type=str)
  parser.add_argument('--val-dir', required=True, type=str)
  parser.add_argument('--test-dir', required=True, type=str)
  parser.add_argument('--model-path', required=True, type=str)
  parser.add_argument('--batch-size', default=64, type=int)
  parser.add_argument('--num-epochs', default=100, type=int)
  parser.add_argument('--initial-lr', default=0.045, type=float)
  parser.add_argument('--lr-decay-factor', default=0.94, type=float)
  parser.add_argument('--lr-decay-steps', default=2, type=int)
  parser.add_argument('--opt-decay', default=0.9, type=float)
  parser.add_argument('--grad-clip', default=2.0, type=float)
  parser.add_argument('--label-smth', default=1.0, type=float)
  parser.add_argument('--adv-coeff', default=0.5, type=float)
  parser.add_argument('--device', default='2', choices=['0', '1', '2'], type=str)
  parser.add_argument('--eps', default=0.0784, type=float)
  parser.add_argument('--ord', default='inf', choices=['inf', '1', '2'], type=str)

  args = parser.parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.device

  main(args)
