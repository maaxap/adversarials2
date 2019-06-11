import logging


def init_logger(name):
  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)

  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

  stream_handler = logging.StreamHandler()
  stream_handler.setLevel(logging.DEBUG)
  stream_handler.setFormatter(formatter)

  logger.addHandler(stream_handler)

  file_handler = logging.FileHandler('{}.log'.format(name))
  file_handler.setLevel(logging.DEBUG)
  file_handler.setFormatter(formatter)

  logger.addHandler(file_handler)

  return logger