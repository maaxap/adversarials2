import os
from os.path import join
import shutil
import argparse


if __name__ == '__main__':
  # Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir', required=True)
  args = parser.parse_args()

  # Combine files accoding to the number of iterations
  for filename in os.listdir(args.dir):
    it = filename.split('_')[2]
    it_dir = join(args.dir, it)

    if not os.path.exists(it_dir):
      os.mkdir(it_dir)

    shutil.move(join(args.dir, filename), join(it_dir, filename))

