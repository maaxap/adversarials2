import warnings

from os.path import join
import numpy as np
import tensorflow as tf

from cleverhans.attacks.attack import Attack
from cleverhans.attacks.fast_gradient_method import optimize_linear
from cleverhans.compat import reduce_sum, reduce_mean
from cleverhans.compat import softmax_cross_entropy_with_logits
from cleverhans import utils_tf
from data import save_image_np


class MomentumIterativeMethod(Attack):

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    super(MomentumIterativeMethod, self).__init__(
      model, sess, dtypestr, **kwargs)
    self.feedable_kwargs = ('eps', 'eps_iter', 'y', 'y_target',
                            'clip_min', 'clip_max')
    self.structural_kwargs = ['ord', 'nb_iter', 'gamma',
                              'sanity_checks', 'clip_grad']

  def generate(self, x, **kwargs):
    assert self.parse_params(**kwargs)

    asserts = []

    if self.clip_min is not None:
      asserts.append(utils_tf.assert_greater_equal(
        x, tf.cast(self.clip_min, x.dtype)))
    if self.clip_max is not None:
      asserts.append(utils_tf.assert_less_equal(
        x, tf.cast(self.clip_max, x.dtype)))

    momentum = tf.zeros_like(x)
    adv_x = x

    y, _nb_classes = self.get_or_guess_labels(x, kwargs)
    y = y / reduce_sum(y, 1, keepdims=True)
    targeted = (self.y_target is not None)

    def save_batch(directory, images, labels, iteration, batch_idx):
      for idx, (image, label) in enumerate(zip(images, labels)):
        filename = "id{}_b{}_it{}_l{}.png".format(
          idx, batch_idx, iteration, np.argmax(label))
        save_image_np(join(directory, filename), image)

    for i in range(self.nb_iter):

      self.logger.debug("Starting #{} iteration".format(i + 1))

      logits = self.model.get_logits(adv_x)
      loss = softmax_cross_entropy_with_logits(labels=y, logits=logits)
      if targeted:
        loss = -loss

      grad, = tf.gradients(loss, adv_x)

      red_ind = list(range(1, len(grad.get_shape())))
      avoid_zero_div = tf.cast(1e-8, grad.dtype)
      grad = grad / tf.maximum(
        avoid_zero_div,
        reduce_mean(tf.abs(grad), red_ind, keepdims=True))

      momentum = self.gamma * momentum + grad

      optimal_perturbation = optimize_linear(
        momentum, self.eps_iter, self.ord)
      if self.ord == 1:
        raise NotImplementedError("This attack hasn't been tested for ord=1."
                                  "It's not clear that FGM makes a good inner "
                                  "loop step for iterative optimization since "
                                  "it updates just one coordinate at a time.")

      adv_x = adv_x + optimal_perturbation
      adv_x = x + utils_tf.clip_eta(adv_x - x, self.ord, self.eps)

      if self.clip_min is not None and self.clip_max is not None:
        adv_x = utils_tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

      adv_x = tf.stop_gradient(adv_x)

      if self.sanity_checks:
        with tf.control_dependencies(asserts):
          adv_x = tf.identity(adv_x)

      with self.sess.as_default():
        self.sess.run(self.init_op)
        for batch in range(self.nb_batches):
          adv_x_np, y_np = self.sess.run([adv_x, y])
          self.logger.debug("Saving attacked batch #{}".format(batch + 1))
          save_batch(self.adv_dir, adv_x_np, y_np, i, batch)

  def parse_params(self,
                   init_op=None,
                   adv_dir=None,
                   nb_batches=None,
                   logger=None,
                   eps=0.3,
                   eps_iter=0.06,
                   nb_iter=10,
                   y=None,
                   ord=np.inf,
                   gamma=0.8,
                   clip_min=None,
                   clip_max=None,
                   y_target=None,
                   sanity_checks=True,
                   **kwargs):

    # Save attack-specific parameters
    self.init_op = init_op
    self.adv_dir = adv_dir
    self.nb_batches = nb_batches
    self.logger = logger
    self.eps = eps
    self.eps_iter = eps_iter
    self.nb_iter = nb_iter
    self.y = y
    self.y_target = y_target
    self.ord = ord
    self.gamma = gamma
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.sanity_checks = sanity_checks

    if self.init_op is None or self.y is None or self.adv_dir is None \
            or self.nb_batches is None or self.logger is None:
      raise ValueError("Not all required parameters specified")

    if self.y is not None and self.y_target is not None:
      raise ValueError("Must not set both y and y_target")

    if self.ord not in [np.inf, 1, 2]:
      raise ValueError("Norm order must be either np.inf, 1, or 2.")

    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True


class NesterovIterativeMethod(Attack):

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    super(NesterovIterativeMethod, self).__init__(
      model, sess, dtypestr, **kwargs)
    self.feedable_kwargs = ('eps', 'eps_iter', 'y', 'y_target',
                            'clip_min', 'clip_max')
    self.structural_kwargs = ['ord', 'nb_iter', 'gamma',
                              'sanity_checks', 'clip_grad']

  def generate(self, x, **kwargs):
    assert self.parse_params(**kwargs)

    asserts = []

    if self.clip_min is not None:
      asserts.append(utils_tf.assert_greater_equal(
        x, tf.cast(self.clip_min, x.dtype)))

    if self.clip_max is not None:
      asserts.append(utils_tf.assert_less_equal(
        x, tf.cast(self.clip_max, x.dtype)))

    momentum = tf.zeros_like(x)
    adv_x = x

    y, _nb_classes = self.get_or_guess_labels(x, kwargs)
    y = y / reduce_sum(y, 1, keepdims=True)
    targeted = (self.y_target is not None)

    def update_and_clip(ax, perturbation):
      ax = ax + perturbation
      ax = x + utils_tf.clip_eta(ax - x, self.ord, self.eps)

      if self.clip_min is not None and self.clip_max is not None:
        ax = utils_tf.clip_by_value(ax, self.clip_min, self.clip_max)

      ax = tf.stop_gradient(ax)

      return ax

    def save_batch(directory, images, labels, iteration, batch_idx):
      for idx, (image, label) in enumerate(zip(images, labels)):
        filename = "id{}_b{}_it{}_l{}.png".format(idx, batch_idx,
                                                  iteration, np.argmax(label))
        save_image_np(join(directory, filename), image)

    for i in range(self.nb_iter):

      self.logger.debug("Starting #{} iteration".format(i + 1))

      lookup_perturbation = optimize_linear(
        self.gamma * momentum, self.eps_iter, self.ord)
      lookup_x = update_and_clip(adv_x, lookup_perturbation)
      logits = self.model.get_logits(lookup_x)
      loss = softmax_cross_entropy_with_logits(labels=y, logits=logits)
      if targeted:
        loss = -loss

      grad, = tf.gradients(loss, lookup_x)

      red_ind = list(range(1, len(grad.get_shape())))
      avoid_zero_div = tf.cast(1e-8, grad.dtype)
      grad = grad / tf.maximum(
        avoid_zero_div,
        reduce_mean(tf.abs(grad), red_ind, keepdims=True))

      momentum = self.gamma * momentum + grad

      optimal_perturbation = optimize_linear(momentum, self.eps_iter, self.ord)
      if self.ord == 1:
        raise NotImplementedError("This attack hasn't been tested for ord=1."
                                  "It's not clear that FGM makes a good inner "
                                  "loop step for iterative optimization since "
                                  "it updates just one coordinate at a time.")

      adv_x = update_and_clip(adv_x, optimal_perturbation)

      if self.sanity_checks:
        with tf.control_dependencies(asserts):
          adv_x = tf.identity(adv_x)

      with self.sess.as_default():
        self.sess.run(self.init_op)
        for batch in range(self.nb_batches):
          adv_x_np, y_np = self.sess.run([adv_x, y])
          self.logger.debug("Saving attacked batch #{}".format(batch + 1))
          save_batch(self.adv_dir, adv_x_np, y_np, i, batch)

  def parse_params(self,
                   init_op=None,
                   adv_dir=None,
                   nb_batches=None,
                   logger=None,
                   eps=0.3,
                   eps_iter=0.06,
                   nb_iter=10,
                   y=None,
                   ord=np.inf,
                   gamma=0.8,
                   clip_min=None,
                   clip_max=None,
                   y_target=None,
                   sanity_checks=True,
                   **kwargs):

    # Save attack-specific parameters
    self.init_op = init_op
    self.adv_dir = adv_dir
    self.nb_batches = nb_batches
    self.logger = logger
    self.eps = eps
    self.eps_iter = eps_iter
    self.nb_iter = nb_iter
    self.y = y
    self.y_target = y_target
    self.ord = ord
    self.gamma = gamma
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.sanity_checks = sanity_checks

    if self.init_op is None or self.y is None or self.adv_dir is None \
            or self.nb_batches is None or self.logger is None:
      raise ValueError("Not all required parameters specified")

    if self.y is not None and self.y_target is not None:
      raise ValueError("Must not set both y and y_target")

    if self.ord not in [np.inf, 1, 2]:
      raise ValueError("Norm order must be either np.inf, 1, or 2.")

    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True


class AdagradIterativeMethod(Attack):

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    super(AdagradIterativeMethod, self).__init__(
      model, sess, dtypestr, **kwargs)
    self.feedable_kwargs = ('eps', 'eps_iter', 'y', 'y_target',
                            'clip_min', 'clip_max')
    self.structural_kwargs = ['ord', 'nb_iter', 'gamma',
                              'sanity_checks', 'clip_grad']

  def generate(self, x, **kwargs):
    assert self.parse_params(**kwargs)

    asserts = []

    if self.clip_min is not None:
      asserts.append(utils_tf.assert_greater_equal(
        x, tf.cast(self.clip_min, x.dtype)))

    if self.clip_max is not None:
      asserts.append(utils_tf.assert_less_equal(
        x, tf.cast(self.clip_max, x.dtype)))

    g_cached = tf.zeros_like(x)
    adv_x = x

    y, _nb_classes = self.get_or_guess_labels(x, kwargs)
    y = y / reduce_sum(y, 1, keepdims=True)
    targeted = (self.y_target is not None)

    def save_batch(directory, images, labels, iteration, batch_idx):
      for idx, (image, label) in enumerate(zip(images, labels)):
        filename = "id{}_b{}_it{}_l{}.png".format(idx, batch_idx,
                                                  iteration, np.argmax(label))
        save_image_np(join(directory, filename), image)

    for i in range(self.nb_iter):
      self.logger.debug("Starting #{} iteration".format(i + 1))

      logits = self.model.get_logits(adv_x)
      loss = softmax_cross_entropy_with_logits(labels=y, logits=logits)
      if targeted:
        loss = -loss

      grad, = tf.gradients(loss, adv_x)

      red_ind = list(range(1, len(grad.get_shape())))
      avoid_zero_div = tf.cast(1e-8, grad.dtype)
      grad = grad / tf.maximum(
        avoid_zero_div,
        reduce_mean(tf.abs(grad), red_ind, keepdims=True))

      g_cached = g_cached + tf.square(grad)
      update = tf.divide(grad, tf.sqrt(g_cached + avoid_zero_div))

      optimal_perturbation = optimize_linear(update, self.eps_iter, self.ord)
      if self.ord == 1:
        raise NotImplementedError("This attack hasn't been tested for ord=1."
                                  "It's not clear that FGM makes a good inner "
                                  "loop step for iterative optimization since "
                                  "it updates just one coordinate at a time.")

      # Update and clip adversarial example in current iteration
      adv_x = adv_x + optimal_perturbation
      adv_x = x + utils_tf.clip_eta(adv_x - x, self.ord, self.eps)

      if self.clip_min is not None and self.clip_max is not None:
        adv_x = utils_tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

      adv_x = tf.stop_gradient(adv_x)

      if self.sanity_checks:
        with tf.control_dependencies(asserts):
          adv_x = tf.identity(adv_x)

      with self.sess.as_default():
        self.sess.run(self.init_op)
        for batch in range(self.nb_batches):
          adv_x_np, y_np = self.sess.run([adv_x, y])
          self.logger.debug("Saving attacked batch #{}".format(batch + 1))
          save_batch(self.adv_dir, adv_x_np, y_np, i, batch)


  def parse_params(self,
                   init_op=None,
                   adv_dir=None,
                   nb_batches=None,
                   logger=None,
                   eps=0.3,
                   eps_iter=0.06,
                   nb_iter=10,
                   y=None,
                   ord=np.inf,
                   gamma=0.8,
                   clip_min=None,
                   clip_max=None,
                   y_target=None,
                   sanity_checks=True,
                   **kwargs):

    # Save attack-specific parameters
    self.init_op = init_op
    self.adv_dir = adv_dir
    self.nb_batches = nb_batches
    self.logger = logger
    self.eps = eps
    self.eps_iter = eps_iter
    self.nb_iter = nb_iter
    self.y = y
    self.y_target = y_target
    self.ord = ord
    self.gamma = gamma
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.sanity_checks = sanity_checks

    if self.init_op is None or self.y is None or self.adv_dir is None \
            or self.nb_batches is None or self.logger is None:
      raise ValueError("Not all required parameters specified")

    if self.y is not None and self.y_target is not None:
      raise ValueError("Must not set both y and y_target")
    # Check if order of the norm is acceptable given current implementation
    if self.ord not in [np.inf, 1, 2]:
      raise ValueError("Norm order must be either np.inf, 1, or 2.")

    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True


class AdadeltaIterativeMethod(Attack):

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    super(AdadeltaIterativeMethod, self).__init__(
      model, sess, dtypestr, **kwargs)
    self.feedable_kwargs = ('eps', 'eps_iter', 'y', 'y_target',
                            'clip_min', 'clip_max')
    self.structural_kwargs = ['ord', 'nb_iter', 'gamma',
                              'sanity_checks', 'clip_grad']

  def generate(self, x, **kwargs):
    assert self.parse_params(**kwargs)

    asserts = []

    if self.clip_min is not None:
      asserts.append(utils_tf.assert_greater_equal(
        x, tf.cast(self.clip_min, x.dtype)))

    if self.clip_max is not None:
      asserts.append(utils_tf.assert_less_equal(
        x, tf.cast(self.clip_max, x.dtype)))

    g_cached = tf.zeros_like(x)
    u_cached = tf.zeros_like(x)
    u_iter = tf.zeros_like(x)
    adv_x = x

    y, _nb_classes = self.get_or_guess_labels(x, kwargs)
    y = y / reduce_sum(y, 1, keepdims=True)
    targeted = (self.y_target is not None)

    def save_batch(directory, images, labels, iteration, batch_idx):
      for idx, (image, label) in enumerate(zip(images, labels)):
        filename = "id{}_b{}_it{}_l{}.png".format(idx, batch_idx,
                                                  iteration, np.argmax(label))
        save_image_np(join(directory, filename), image)

    for i in range(self.nb_iter):
      self.logger.debug("Starting #{} iteration".format(i + 1))

      logits = self.model.get_logits(adv_x)
      loss = softmax_cross_entropy_with_logits(labels=y, logits=logits)
      if targeted:
        loss = -loss

      grad, = tf.gradients(loss, adv_x)

      red_ind = list(range(1, len(grad.get_shape())))
      avoid_zero_div = tf.cast(1e-8, grad.dtype)
      grad = grad / tf.maximum(
        avoid_zero_div,
        reduce_mean(tf.abs(grad), red_ind, keepdims=True))

      g_cached = self.gamma * g_cached + (1 - self.gamma) * tf.square(grad)
      u_cached = self.gamma * u_cached + (1 - self.gamma) * tf.square(u_iter)
      u_iter = grad * tf.divide(tf.sqrt(u_cached + avoid_zero_div), tf.sqrt(tf.sqrt(g_cached) + avoid_zero_div))

      optimal_perturbation = optimize_linear(u_iter, self.eps_iter, self.ord)
      if self.ord == 1:
        raise NotImplementedError("This attack hasn't been tested for ord=1."
                                  "It's not clear that FGM makes a good inner "
                                  "loop step for iterative optimization since "
                                  "it updates just one coordinate at a time.")

      adv_x = adv_x + optimal_perturbation
      adv_x = x + utils_tf.clip_eta(adv_x - x, self.ord, self.eps)

      if self.clip_min is not None and self.clip_max is not None:
        adv_x = utils_tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

      adv_x = tf.stop_gradient(adv_x)

      if self.sanity_checks:
        with tf.control_dependencies(asserts):
          adv_x = tf.identity(adv_x)

      with self.sess.as_default():
        self.sess.run(self.init_op)
        for batch in range(self.nb_batches):
          adv_x_np, y_np = self.sess.run([adv_x, y])
          self.logger.debug("Saving attacked batch #{}".format(batch + 1))
          save_batch(self.adv_dir, adv_x_np, y_np, i, batch)

  def parse_params(self,
                   init_op=None,
                   adv_dir=None,
                   nb_batches=None,
                   logger=None,
                   eps=0.3,
                   eps_iter=0.06,
                   nb_iter=10,
                   y=None,
                   ord=np.inf,
                   gamma=0.8,
                   clip_min=None,
                   clip_max=None,
                   y_target=None,
                   sanity_checks=True,
                   **kwargs):

    self.init_op = init_op
    self.adv_dir = adv_dir
    self.nb_batches = nb_batches
    self.logger = logger
    self.eps = eps
    self.eps_iter = eps_iter
    self.nb_iter = nb_iter
    self.y = y
    self.y_target = y_target
    self.ord = ord
    self.gamma = gamma
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.sanity_checks = sanity_checks

    if self.init_op is None or self.y is None or self.adv_dir is None \
            or self.nb_batches is None or self.logger is None:
      raise ValueError("Not all required parameters specified")

    if self.y is not None and self.y_target is not None:
      raise ValueError("Must not set both y and y_target")

    if self.ord not in [np.inf, 1, 2]:
      raise ValueError("Norm order must be either np.inf, 1, or 2.")

    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True


class RMSPropIterativeMethod(Attack):

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    super(RMSPropIterativeMethod, self).__init__(
      model, sess, dtypestr, **kwargs)
    self.feedable_kwargs = ('eps', 'eps_iter', 'y', 'y_target',
                            'clip_min', 'clip_max')
    self.structural_kwargs = ['ord', 'nb_iter', 'gamma',
                              'sanity_checks', 'clip_grad']

  def generate(self, x, **kwargs):
    assert self.parse_params(**kwargs)

    asserts = []

    if self.clip_min is not None:
      asserts.append(utils_tf.assert_greater_equal(
        x, tf.cast(self.clip_min, x.dtype)))

    if self.clip_max is not None:
      asserts.append(utils_tf.assert_less_equal(
        x, tf.cast(self.clip_max, x.dtype)))

    g_cached = tf.zeros_like(x)
    adv_x = x

    y, _nb_classes = self.get_or_guess_labels(x, kwargs)
    y = y / reduce_sum(y, 1, keepdims=True)
    targeted = (self.y_target is not None)

    def save_batch(directory, images, labels, iteration, batch_idx):
      for idx, (image, label) in enumerate(zip(images, labels)):
        filename = "id{}_b{}_it{}_l{}.png".format(idx, batch_idx,
                                                  iteration, np.argmax(label))
        save_image_np(join(directory, filename), image)

    for i in range(self.nb_iter):
      self.logger.debug("Starting #{} iteration".format(i + 1))

      logits = self.model.get_logits(adv_x)
      loss = softmax_cross_entropy_with_logits(labels=y, logits=logits)
      if targeted:
        loss = -loss

      grad, = tf.gradients(loss, adv_x)

      red_ind = list(range(1, len(grad.get_shape())))
      avoid_zero_div = tf.cast(1e-8, grad.dtype)
      grad = grad / tf.maximum(
        avoid_zero_div,
        reduce_mean(tf.abs(grad), red_ind, keepdims=True))

      g_cached = self.gamma * g_cached + (1 - self.gamma) * tf.square(grad)
      update = tf.divide(grad, tf.sqrt(g_cached + avoid_zero_div))

      optimal_perturbation = optimize_linear(update, self.eps_iter, self.ord)
      if self.ord == 1:
        raise NotImplementedError("This attack hasn't been tested for ord=1."
                                  "It's not clear that FGM makes a good inner "
                                  "loop step for iterative optimization since "
                                  "it updates just one coordinate at a time.")

      adv_x = adv_x + optimal_perturbation
      adv_x = x + utils_tf.clip_eta(adv_x - x, self.ord, self.eps)

      if self.clip_min is not None and self.clip_max is not None:
        adv_x = utils_tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

      adv_x = tf.stop_gradient(adv_x)

      if self.sanity_checks:
        with tf.control_dependencies(asserts):
          adv_x = tf.identity(adv_x)

      with self.sess.as_default():
        self.sess.run(self.init_op)
        for batch in range(self.nb_batches):
          adv_x_np, y_np = self.sess.run([adv_x, y])
          self.logger.debug("Saving attacked batch #{}".format(batch + 1))
          save_batch(self.adv_dir, adv_x_np, y_np, i, batch)


  def parse_params(self,
                   init_op=None,
                   adv_dir=None,
                   nb_batches=None,
                   logger=None,
                   eps=0.3,
                   eps_iter=0.06,
                   nb_iter=10,
                   y=None,
                   ord=np.inf,
                   gamma=0.8,
                   clip_min=None,
                   clip_max=None,
                   y_target=None,
                   sanity_checks=True,
                   **kwargs):

    # Save attack-specific parameters
    self.init_op = init_op
    self.adv_dir = adv_dir
    self.nb_batches = nb_batches
    self.logger = logger
    self.eps = eps
    self.eps_iter = eps_iter
    self.nb_iter = nb_iter
    self.y = y
    self.y_target = y_target
    self.ord = ord
    self.gamma = gamma
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.sanity_checks = sanity_checks

    if self.init_op is None or self.y is None or self.adv_dir is None \
            or self.nb_batches is None or self.logger is None:
      raise ValueError("Not all required parameters specified")

    if self.y is not None and self.y_target is not None:
      raise ValueError("Must not set both y and y_target")

    if self.ord not in [np.inf, 1, 2]:
      raise ValueError("Norm order must be either np.inf, 1, or 2.")

    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True


class AdamIterativeMethod(Attack):

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    super(AdamIterativeMethod, self).__init__(model, sess, dtypestr, **kwargs)
    self.feedable_kwargs = ('eps', 'eps_iter', 'y', 'y_target',
                            'clip_min', 'clip_max')
    self.structural_kwargs = ['ord', 'nb_iter', 'betha1', 'betha2',
                              'sanity_checks', 'clip_grad']

  def generate(self, x, **kwargs):
    assert self.parse_params(**kwargs)

    asserts = []

    if self.clip_min is not None:
      asserts.append(utils_tf.assert_greater_equal(
        x, tf.cast(self.clip_min,x.dtype)))

    if self.clip_max is not None:
      asserts.append(utils_tf.assert_less_equal(
        x, tf.cast(self.clip_max, x.dtype)))

    m_cache = tf.zeros_like(x)
    v_cache = tf.zeros_like(x)
    adv_x = x

    y, _nb_classes = self.get_or_guess_labels(x, kwargs)
    y = y / reduce_sum(y, 1, keepdims=True)
    targeted = (self.y_target is not None)

    def save_batch(directory, images, labels, iteration, batch_idx):
      for idx, (image, label) in enumerate(zip(images, labels)):
        filename = "id{}_b{}_it{}_l{}.png".format(idx, batch_idx,
                                                  iteration, np.argmax(label))
        save_image_np(join(directory, filename), image)

    for i in range(self.nb_iter):
      self.logger.debug("Starting #{} iteration".format(i + 1))

      logits = self.model.get_logits(adv_x)
      loss = softmax_cross_entropy_with_logits(labels=y, logits=logits)
      if targeted:
        loss = -loss

      grad, = tf.gradients(loss, adv_x)

      red_ind = list(range(1, len(grad.get_shape())))
      avoid_zero_div = tf.cast(1e-8, grad.dtype)
      grad = grad / tf.maximum(
        avoid_zero_div,
        reduce_mean(tf.abs(grad), red_ind, keepdims=True))

      m_cache = self.betha1 * m_cache + (1 - self.betha1) * grad
      v_cache = self.betha2 * v_cache + (1 - self.betha2) * tf.square(grad)
      update = tf.divide(m_cache, tf.sqrt(v_cache + avoid_zero_div))

      optimal_perturbation = optimize_linear(update, self.eps_iter, self.ord)
      if self.ord == 1:
        raise NotImplementedError("This attack hasn't been tested for ord=1."
                                  "It's not clear that FGM makes a good inner "
                                  "loop step for iterative optimization since "
                                  "it updates just one coordinate at a time.")

      adv_x = adv_x + optimal_perturbation
      adv_x = x + utils_tf.clip_eta(adv_x - x, self.ord, self.eps)

      if self.clip_min is not None and self.clip_max is not None:
        adv_x = utils_tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

      adv_x = tf.stop_gradient(adv_x)

      if self.sanity_checks:
        with tf.control_dependencies(asserts):
          adv_x = tf.identity(adv_x)

      with self.sess.as_default():
        self.sess.run(self.init_op)
        for batch in range(self.nb_batches):
          adv_x_np, y_np = self.sess.run([adv_x, y])
          self.logger.debug("Saving attacked batch #{}".format(batch + 1))
          save_batch(self.adv_dir, adv_x_np, y_np, i, batch)


  def parse_params(self,
                   init_op=None,
                   adv_dir=None,
                   nb_batches=None,
                   logger=None,
                   betha1=0.9,
                   betha2=0.999,
                   eps=0.3,
                   eps_iter=0.06,
                   nb_iter=10,
                   y=None,
                   ord=np.inf,
                   gamma=1.0,
                   clip_min=None,
                   clip_max=None,
                   y_target=None,
                   sanity_checks=True,
                   **kwargs):

    # Save attack-specific parameters
    self.init_op = init_op
    self.adv_dir = adv_dir
    self.nb_batches = nb_batches
    self.logger = logger
    self.betha1 = betha1
    self.betha2 = betha2
    self.eps = eps
    self.eps_iter = eps_iter
    self.nb_iter = nb_iter
    self.y = y
    self.y_target = y_target
    self.ord = ord
    self.gamma = gamma
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.sanity_checks = sanity_checks

    if self.init_op is None or self.y is None or self.adv_dir is None \
            or self.nb_batches is None or self.logger is None:
      raise ValueError("Not all required parameters specified")

    if self.y is not None and self.y_target is not None:
      raise ValueError("Must not set both y and y_target")

    if self.ord not in [np.inf, 1, 2]:
      raise ValueError("Norm order must be either np.inf, 1, or 2.")

    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True
