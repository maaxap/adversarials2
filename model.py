import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import vgg, inception

from cleverhans.model import Model


class VGG19(Model):
    SHAPE = (224, 224, 3)

    def __init__(self, scope='vgg_19', nb_classes=2,
                 dropout_keep_prob=0.5, is_training=True):
        super(VGG19, self).__init__(
            scope=scope, nb_classes=nb_classes, hparams=locals())
        self.is_training = is_training
        self.dropout_keep_prob = dropout_keep_prob
        self.scope = scope
        self.built = False

        self.fprop(self.make_input_placeholder())
        self.params = self.get_params()

    def fprop(self, x, **kwargs):
        del kwargs

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            with slim.arg_scope(vgg.vgg_arg_scope()):
                logits, _ = vgg.vgg_19(
                    x, num_classes=self.nb_classes,
                    dropout_keep_prob=self.dropout_keep_prob,
                    is_training=self.is_training, scope=self.scope)

        probs = tf.nn.softmax(logits)

        return {self.O_LOGITS: logits, self.O_PROBS: probs}

    def make_input_placeholder(self):
        shape = (None,) + VGG19.SHAPE
        input_placeholder = tf.placeholder(
            tf.float32, shape=shape, name='input_placeholder')
        return input_placeholder

    def make_label_placeholder(self):
        label_placeholder = tf.placeholder(
            tf.int32, shape=(None,), name='label_placeholder')

        return label_placeholder


class InceptionV3(Model):
  SHAPE = (299, 299, 3)

  def __init__(self, scope='InceptionV3', nb_classes=2,
               dropout_keep_prob=0.5, is_training=True):
    super(InceptionV3, self).__init__(
      scope=scope, nb_classes=nb_classes, hparams=locals())
    self.is_training = is_training
    self.dropout_keep_prob = dropout_keep_prob
    self.scope = scope
    self.built = False

    self.fprop(self.make_input_placeholder())
    self.params = self.get_params()

  def fprop(self, x, **kwargs):
    del kwargs
    reuse = True if self.built else None

    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(
        x, num_classes=self.nb_classes,
        dropout_keep_prob=self.dropout_keep_prob,
        is_training=self.is_training, reuse=reuse, scope=self.scope)

    self.built = True
    logits, probs = (end_points['Logits'],
                     end_points['Predictions'].op.inputs[0])

    return {self.O_LOGITS: logits, self.O_PROBS: probs}

  def make_input_placeholder(self):
    shape = (None,) + InceptionV3.SHAPE
    input_placeholder = tf.placeholder(
      tf.float32, shape=shape, name='input_placeholder')
    return input_placeholder

  def make_label_placeholder(self):
    label_placeholder = tf.placeholder(
      tf.int32, shape=(None,), name='label_placeholder')
    return label_placeholder