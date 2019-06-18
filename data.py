import os
from os.path import join
import numpy as np
import tensorflow as tf
from PIL import Image


class Dataset(object):

  def __init__(self, data_directory, add_filenames=False):
    assert os.path.isdir(data_directory), ("`data_directory` expected "
                                           "to be a directory")

    class_directories = sorted(os.listdir(data_directory))
    num_classes = len(class_directories)

    assert num_classes > 1, "Expected more than one class"

    image_lists = {class_dir: os.listdir(join(data_directory, class_dir))
                   for class_dir in class_directories}
    image_lists = {y: sorted(x) for y, x in image_lists.items()}
    labels_map = dict(list(zip(class_directories, range(num_classes))))

    flat_images_list, flat_labels_list = [], []
    for class_dir, filenames in image_lists.items():
      label = labels_map[class_dir]
      for filename in filenames:
        filepath = join(data_directory, class_dir, filename)
        flat_images_list.append(filepath)
        flat_labels_list.append(label)

    self.filenames = flat_images_list
    self.labels = flat_labels_list
    self.size = len(flat_images_list)
    self.labels_map = labels_map
    self.is_shuffled = False
    self.batch_size = None

    if add_filenames:
      self.dataset = tf.data.Dataset.from_tensor_slices(
        (flat_images_list, flat_images_list, flat_labels_list))
    else:
      self.dataset = tf.data.Dataset.from_tensor_slices(
        (flat_images_list, flat_labels_list))


def make_dataset(datadir, batch_size, preprocess_func=None,
                 shuffle=False, repeat=False, add_filenames=False):
  dataset = Dataset(datadir, add_filenames=add_filenames)

  # Apply preprocess function to image paths
  if preprocess_func:
    dataset.dataset = dataset.dataset.map(preprocess_func)

  # Convert labels to OHE
  num_classes = len(dataset.labels_map)
  def labels_to_one_hot(x, y):
    return x, tf.one_hot(y, num_classes)
  dataset.dataset = dataset.dataset.map(labels_to_one_hot)

  if shuffle is True:
    dataset.is_shuffled = True
    dataset.dataset = dataset.dataset.shuffle(buffer_size=10000)

  dataset.batch_size = batch_size
  dataset.dataset = dataset.dataset.batch(batch_size)

  if repeat is True:
    dataset.dataset = dataset.dataset.repeat()

  return dataset


def read_image(image_path):
  image_buffer = tf.read_file(image_path)
  return image_buffer


def decode_png(image_buffer):
  image = tf.image.decode_png(image_buffer, channels=3)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  return image


def random_crop(image, height, width, nb_channels, seed=None):
  image = tf.random_crop(image, (height, width, nb_channels), seed=seed)
  return image


def random_flip(image, seed=None):
  image = tf.image.random_flip_left_right(image, seed=seed)
  return image


def resize(image, height, width):
  image = tf.image.resize_images(
    image, (height, width),
    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return image


def save_image_np(filepath, image):
  image = (image * 255).astype(np.uint8)
  pil_image = Image.fromarray(image)
  pil_image.save(filepath)

  assert os.path.exists(filepath), ("Error while saving image: {} does not "
                                    "exist".format(filepath))
