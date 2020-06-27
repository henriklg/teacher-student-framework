"""
Split dataset into train/test/val and resize images to correct size.

The dataset comes in the following format:
    data_dir/
        class1
        ... images
        class2/
        ... images
        
Example: py build_dataset.py --data_dir labeled --output_dir labeled_ttv --split 0.7 0.15 --size 128
"""

import argparse
import random
import os
import cv2
import numpy as np

from math import ceil, floor
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory with the dataset")
parser.add_argument('--output_dir', default='data', help="Where to write the new data")
parser.add_argument('--split', nargs=2, help="Train, test, val split given as '0.7 0.15'")
parser.add_argument('--size', default=0, help="Resize to this resolution. Default is to use orginal resolution.")
parser.add_argument('--seed', default=2511, help="What seed to use when shuffling the data")

def resize_and_save(filename, output_dir, size):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    if size > 1:
      image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    path = os.path.join(output_dir, filename.split('/')[-1])
    cv2.imwrite(path, image)
    
    
if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    data_dir = args.data_dir
    output_dir = args.output_dir
    size = int(args.size)
    seed = int(args.seed)
    splits = args.split
    train_split = float(splits[0])
    test_split = float(splits[0])+float(splits[1])
    random.seed(seed)
    
    class_names = os.listdir(data_dir)
    num_classes = len(class_names)
    num_samples = len(list(Path(data_dir).glob('*/*')))

    image_folders = [os.path.join(data_dir, class_name) for class_name in class_names]

    # Create folders for train, test and val with subfolders
    train_folders = [os.path.join(output_dir, 'train', class_name) for class_name in class_names]
    test_folders = [os.path.join(output_dir, 'test', class_name) for class_name in class_names]
    val_folders = [os.path.join(output_dir, 'val', class_name) for class_name in class_names]

    # Create all folders
    [Path(train_folder).mkdir(parents=True, exist_ok=True) for train_folder in train_folders]
    [Path(test_folder).mkdir(parents=True, exist_ok=True) for test_folder in test_folders]
    [Path(val_folder).mkdir(parents=True, exist_ok=True) for val_folder in val_folders];
    
    tqdm_img = tqdm(total=num_samples, desc='Images', position=0)

    # Iterate over categories
    for idx, directory in enumerate(image_folders):

        random.seed(seed)
        # Get filenames in category, sort and shuffle (for reproducible split)
        filenames = os.listdir(directory)
        filenames.sort()
        random.shuffle(filenames)
        num_samples = len(filenames)

        # Calculate number of samples for each dataset
        # NB: minimum 4 samples to get one in each split
        filenames = np.array(filenames)
        filenames_split = np.split(filenames, [floor(num_samples*train_split), floor(num_samples*test_split)])

        # Split dataset into train test val
        ds = {'train': filenames_split[0],
              'test': filenames_split[1],
              'val': filenames_split[2]}

        # Copy files to correct folder/split
        for split in ds:
            output = os.path.join(output_dir, split, directory.split("/")[-1])
            for filename in ds[split]:
                filename = os.path.join(directory, filename)
                resize_and_save(filename, output, size=size)
                tqdm_img.update(1)
