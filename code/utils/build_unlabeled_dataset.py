"""
Split dataset into train/test/val and resize images to correct size.

The dataset comes in the following format:
    data_dir/
        image1
        image2
        image3
        ...

Example: py build_unlabeled_dataset.py --data_dir unlabeled --output_dir unlabeled_ttv --size 128
"""

import argparse
import os
import cv2

from tqdm import tqdm
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory with the dataset")
parser.add_argument('--output_dir', default='data', help="Where to write the new data")
parser.add_argument('--size', default=256, help="Image size")

def resize_and_save(filename, output_dir, size):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    path = os.path.join(output_dir, filename.split('/')[-1])
    cv2.imwrite(path, image)
    
    
if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    data_dir = args.data_dir
    output_dir = args.output_dir
    size = int(args.size)
    
    # Create output dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    images = os.listdir(data_dir)
    num_images = len(images)
    images = [os.path.join(data_dir, image) for image in images]
    
    tqdm_img = tqdm(total=num_images, desc="Resized images", position=0)
    for image in images:
        resize_and_save(image, output_dir, size)
        tqdm_img.update(1)
