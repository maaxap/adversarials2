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
    if os.path.isdir(join(args.dir, filename)):
      continue

    tokens = filename.split('_')
    
    if len(tokens) != 4:
      continue

    if len(tokens) != 4:
      continue

    it = "{:02d}".format(int(tokens[2][2:]))
    it_dir = join(args.dir, it)

    if not os.path.exists(it_dir):
      os.mkdir(it_dir)

    lb = tokens[3][1:-4]
    class_dir = '00-normal' if lb == '0' else '01-tumor'

    if not os.path.exists(join(it_dir, class_dir)):
      os.mkdir(join(it_dir, class_dir))

    shutil.move(join(args.dir, filename), join(it_dir, class_dir, filename))
