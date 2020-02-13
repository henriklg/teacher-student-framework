from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt

def create_dataset():
    """
    Create a tf.data dataset
    
    Args:
    data_dir: path to directory containing folders with class names
    outcast: list of classes to exclude
    binary: binary dataset or not
    verbose: print info or not
    
    Return:
    tf.data.Dataset   
    """
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    MODEL = 'cnn' 
    DS_INFO = 'complete'
    NUM_EPOCHS = 50
    BATCH_SIZE = 64

    IMG_HEIGHT = 32
    IMG_WIDTH = 32
    NUM_CHANNELS = 3
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)

    # epoch*batch_size*img_size
    model_name = '{}x{}x{}_{}_{}'.format(NUM_EPOCHS, BATCH_SIZE, IMG_WIDTH, DS_INFO, MODEL)

    data_dir = pathlib.Path('/mnt/sdb/cifar10/')
    outcast = 'None'

    DATASET_SIZE = len(list(data_dir.glob('*/*/*.*g')))
    STEPS_PER_EPOCH = np.ceil(DATASET_SIZE/BATCH_SIZE)

    directories = np.array([item.name for item in data_dir.glob('train/*') if item.name != 'metadata.json'])

    class_names = directories
    NUM_CLASSES = len(directories)
    print ("Class names: ", class_names)

    
    # Create a dataset of the file paths
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
    
    samples_per_class = []
    for class_name in class_names:
        class_samples = len(list(data_dir.glob('*/'+class_name+'/*.*g')))
        samples_per_class.append(class_samples)
        print('{0:18}: {1:3d}'.format(class_name, class_samples))

    print ('\nTotal number of images: {}, in {} classes'.format(DATASET_SIZE, NUM_CLASSES))

    # If one class contains more than half of the entire sample size
    if np.max(samples_per_class) > DATASET_SIZE//2:
        print ("But the dataset is mainly shit")

    # Create a dataset of the file paths
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*/*.png'))
        
    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # get class integer from class-list
        label_int = tf.reduce_min(tf.where(tf.equal(parts[-2], class_names)))
        # cast to tensor array with dtype=uint8
        return tf.dtypes.cast(label_int, tf.int32)

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    # Set 'num_parallel_calls' so multiple images are loaded and processed in parallel
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    
    
    train_size = int(0.7 * DATASET_SIZE)
    val_size = int(0.15 * DATASET_SIZE)
    test_size = int(0.15 * DATASET_SIZE)

    train_ds = labeled_ds.take(train_size)
    test_ds = labeled_ds.skip(train_size)
    val_ds = test_ds.skip(val_size)
    test_ds = test_ds.take(test_size)
    
    
    

    # Create training, test and validation dataset
    train_ds = prepare_for_training(train_ds, BATCH_SIZE, cache="./cache/{}_train.tfcache".format(IMG_WIDTH))
    test_ds = prepare_for_training(test_ds, BATCH_SIZE,cache="./cache/{}_test.tfcache".format(IMG_WIDTH))
    val_ds = prepare_for_training(val_ds, BATCH_SIZE,cache="./cache/{}_val.tfcache".format(IMG_WIDTH))
    
    return train_ds, test_ds, val_ds
    
    
    

def prepare_for_training(ds, bs, cache=True, shuffle_buffer_size=3000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(bs)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds
    
    
    
def print_class_info(directories, data_dir, pos_class, neg_class):
    # Extract and print info about the class split 
    
    idx = 0
    for class_ in [pos_class, neg_class]:
        print ("{} class names:".format(['Positive', 'Negative'][idx]))
        for cl in class_:
            print ("{}- {}".format(" "*8, cl))
        idx += 1
    
    neg_count = 0
    pos_count = 0
    for class_name in directories:
        # Number of samples in 'class_name' folder
        class_samples = len(list(data_dir.glob(class_name+'/*.jpg')))
        
        if (class_name == neg_class[0]):
            neg_count += class_samples
        else:
            pos_count += class_samples

    DS_SIZE = neg_count+pos_count
    print ('\nNegative samples: {0:5} | {1:5.2f}%'.format(neg_count, neg_count/DS_SIZE*100))
    print ('Positive samples: {0:5} | {1:5.2f}%'.format(pos_count, pos_count/DS_SIZE*100))
    # Print number of images in dataset (excluded samples in outcast)
    print ('\nTotal number of images:', DS_SIZE)
    
    
    
def show_image(img, class_names):
    for image, label in img:
        class_index = int(label.numpy()[0])
        plt.figure()
        plt.figure(frameon=False, facecolor='white')
        plt.title(class_names[class_index], fontdict={'color':'white','size':20})
        plt.imshow(image.numpy())
        plt.axis('off')