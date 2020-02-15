from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt

def create_dataset(config):
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
    data_dir = config["data_dir"]
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    DS_SIZE = len(list(data_dir.glob('*/*.*g')))
    directories = np.array([item.name for item in data_dir.glob('*') if item.name != 'metadata.json'])
    
    # Remove the outcast folder
    outcast = config["outcast"]
    if outcast != None:
        directories = np.delete(directories, np.where(outcast == directories))
        if config["verbosity"] > 0: print ("Removed outcast:", outcast)

    if config["DS_INFO"] == 'binary':
        class_names = np.array(['Negative','Positive'])
        neg_class_name = ['ship'] # 'normal'-class
        pos_class_names = np.delete(directories, np.where(neg_class_name == directories))
        # Print info about neg/pos split
        if config["verbosity"] > 0: print_class_info(directories, data_dir, DS_SIZE, class_names, neg_class_name, pos_class_names)
    else:     
        class_names = directories
    
    NUM_CLASSES = len(directories)
    
    # Print info about classes
    if config["verbosity"] > 0:
        print ("Directories: ", directories, end='\n\n')
        
        samples_per_class = []
        for class_name in class_names:
            class_samples = len(list(data_dir.glob(class_name+'/*.*g')))
            samples_per_class.append(class_samples)
            print('{0:18}: {1:3d}'.format(class_name, class_samples))

        print ('\nTotal number of images: {}, in {} classes'.format(DS_SIZE, NUM_CLASSES))

        # If one class contains more than half of the entire sample size
        if np.max(samples_per_class) > DS_SIZE//2:
            print ("But the dataset is mainly shit")

    # Create a dataset of the file paths
    list_ds = tf.data.Dataset.list_files(str(data_dir/'[!')+str(outcast+']*/*'))
#     list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.*g'))
        
    def get_label(file_path):
        # if binary ds
        if config["DS_INFO"] == 'binary':
            parts = tf.strings.split(file_path, os.path.sep)
            bc = parts[-2] == pos_class_names
            nz_cnt = tf.math.count_nonzero(bc)
            if (nz_cnt > 0):
                return tf.constant(1, tf.int32)
            return tf.constant(0, tf.int32)
        # if complete ds
        else:
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
        return tf.image.resize(img, [config["IMG_SIZE"][0], config["IMG_SIZE"][1]])

    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    # Set 'num_parallel_calls' so multiple images are loaded and processed in parallel
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    
    
    train_size = int(0.7 * DS_SIZE)
    val_size = int(0.15 * DS_SIZE)
    test_size = int(0.15 * DS_SIZE)

    train_ds = labeled_ds.take(train_size)
    test_ds = labeled_ds.skip(train_size)
    val_ds = test_ds.skip(val_size)
    test_ds = test_ds.take(test_size)
    
    
    # Create training, test and validation dataset
    cache_dir = config["cache_dir"]
    img_width = config["IMG_SIZE"][0]
    ds_info = config["DS_INFO"]
    train_ds = prepare_for_training(
        train_ds, config["BATCH_SIZE"], cache="{}/{}_train.tfcache".format(cache_dir, img_width, ds_info))
    test_ds = prepare_for_training(
        test_ds, config["BATCH_SIZE"],cache="{}/{}_test.tfcache".format(cache_dir, img_width, ds_info))
    val_ds = prepare_for_training(
        val_ds, config["BATCH_SIZE"],cache="{}/{}_val.tfcache".format(cache_dir, img_width, ds_info))
    
    return_config = {
        "NUM_CLASSES": NUM_CLASSES,
        "DS_SIZE": DS_SIZE,
        "train_size": train_size,
        "test_size": test_size,
        "val_size": val_size,
        "class_names": class_names
    }
    
    return train_ds, test_ds, val_ds, return_config
    
    
    

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
    
    
    
def print_class_info(directories, data_dir, DS_SIZE, class_names, pos_class, neg_class):
    # Extract and print info about the class split 
    
    for i, class_ in enumerate([neg_class, pos_class]):
        print ("{} class names:".format(class_names[i]))
        for cl in class_:
            print ("{}- {}".format(" "*8, cl))
    
    neg_count = pos_count = 0
    for dir_name in directories:
        # Number of samples in 'class_name' folder
        class_samples = len(list(data_dir.glob(dir_name+'/*.*g')))

        if (dir_name == neg_class[0]):
            neg_count += class_samples
        else:
            pos_count += class_samples

    print ('\nNegative samples: {0:5} | {1:5.2f}%'.format(neg_count, neg_count/DS_SIZE*100))
    print ('Positive samples: {0:5} | {1:5.2f}%'.format(pos_count, pos_count/DS_SIZE*100))
    # Print number of images in dataset (excluded samples in outcast)
    print ('\nTotal number of images:', DS_SIZE)
    
    
    
def show_image(img, class_names):
    if (isinstance(img, tf.data.Dataset)):
        for image, label in img:
            plt.figure(frameon=False, facecolor='white')
            title = class_names[label.numpy()]+" ["+str(label.numpy())+"]"
            plt.title(title, fontdict={'color':'white','size':20})
            plt.imshow(image.numpy())
            plt.axis('off')
    else:
        plt.figure(frameon=False, facecolor='white')
        plt.title("None", fontdict={'color':'white','size':20})
        plt.imshow(img.numpy())
        plt.axis('off')