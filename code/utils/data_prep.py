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
    outcast = config["outcast"]
    verbosity = config["verbosity"]
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    DS_SIZE = len(list(data_dir.glob('*/*.*g')))
    directories = np.array([item.name for item in data_dir.glob('*') if item.name != 'metadata.json'])
    
    # Remove the outcast folder
    if outcast != None:
        directories = np.delete(directories, np.where(outcast == directories))
        if verbosity > 0: print ("Removed outcast:", outcast, end="\n\n")

    # If the dataset is to be split in two classes
    if config["DS_INFO"] == 'binary':
        class_names = np.array(['Negative','Positive'])
        NUM_CLASSES = len(class_names)
        neg_class_name = ['ship'] # 'normal'-class
        pos_class_names = np.delete(directories, np.where(neg_class_name == directories))
        # Print info about neg/pos split
        if verbosity > 0: 
            DS_SIZE = print_bin_class_info(directories, data_dir, DS_SIZE, outcast, class_names, neg_class_name, pos_class_names)
    else:     
        class_names = directories
        NUM_CLASSES = len(class_names)
        # Print info about classes
        if verbosity > 0: 
            DS_SIZE = print_class_info(directories, data_dir, DS_SIZE, outcast, NUM_CLASSES)
    
    # Create a dataset of the file paths
    if outcast == None:
        files_string = str(data_dir/'*/*.*g')
    else:
        files_string = str(data_dir/'[!{}]*/*'.format(outcast))
    if verbosity > 0: print ("Dataset.list_files: ",files_string, "\n")
    list_ds = tf.data.Dataset.list_files(files_string)
    
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
    
    if verbosity > 0:
        for images, labels in labeled_ds.batch(10).take(10):
            print(labels.numpy())
    
    if config["resample"] and config["DS_INFO"] == 'binary':
        print ("\nResamplng the dataset..")
        labeled_ds = labeled_ds.batch(1024)
        
        counts = labeled_ds.take(10).reduce(
        initial_state={'class_0': 0, 'class_1': 0},
        reduce_func = count)

        counts = np.array([counts['class_0'].numpy(),
                   counts['class_1'].numpy()]).astype(np.float32)

        counts_sum = counts.sum()
        assert counts_sum != 0, "Can't divide by zero"

        fractions = counts/counts_sum
        if verbosity > 0:
            print("\nFractions: ", fractions)
            print("Counts: ", counts, end="\n\n")
        
        negative_ds = labeled_ds.unbatch().filter(lambda image, label: label==0).repeat()
        positive_ds = labeled_ds.unbatch().filter(lambda image, label: label==1).repeat()
    
        balanced_ds = tf.data.experimental.sample_from_datasets(
            [negative_ds, positive_ds], [0.5, 0.5])
        
        if verbosity > 0:
            for images, labels in balanced_ds.batch(10).take(10):
                print(labels.numpy())
        
        # Overwrite the old dataset with the new and resampled one
        labeled_ds = balanced_ds
    
    
    train_size = int(0.7 * DS_SIZE)
    val_size = int(0.15 * DS_SIZE)
    test_size = int(0.15 * DS_SIZE)

    train_ds = labeled_ds.take(train_size)
    test_ds = labeled_ds.skip(train_size)
    val_ds = test_ds.skip(val_size)
    test_ds = test_ds.take(test_size)

    # Print info about the dataset split
    if verbosity > 0:
        def get_size(ds):
            return tf.data.experimental.cardinality(ds).numpy()

        print ("\n{:32} {:>5}".format("Full dataset sample size:", get_size(labeled_ds)))
        print ("{:32} {:>5}".format("Train dataset sample size:", get_size(train_ds)))
        print ("{:32} {:>5}".format("Test dataset sample size:", get_size(test_ds)))
        print ("{:32} {:>5}".format("Validation dataset sample size:", get_size(val_ds)))
    
    # Create training, test and validation dataset
    cache_dir = config["cache_dir"]
    img_width = config["IMG_SIZE"][0]
    ds_info = config["DS_INFO"]
    train_ds = prepare_for_training(
        train_ds, config["BATCH_SIZE"], cache="{}/{}_{}_train.tfcache".format(cache_dir, img_width, ds_info))
    test_ds = prepare_for_training(
        test_ds, config["BATCH_SIZE"],cache="{}/{}_{}_test.tfcache".format(cache_dir, img_width, ds_info))
    val_ds = prepare_for_training(
        val_ds, config["BATCH_SIZE"],cache="{}/{}_{}_val.tfcache".format(cache_dir, img_width, ds_info))
    
   
    
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
    
    
    
def print_bin_class_info(directories, data_dir, DS_SIZE, outcast, class_names, neg, pos):
    # Extract and print info about the class split 
    
    for i, class_ in enumerate([neg, pos]):
        print ("{} class names:".format(class_names[i]))
        for cl in class_:
            print ("{}- {}".format(" "*8, cl))
    
    neg_count = pos_count = 0
    for dir_name in directories:
        # Number of samples in 'class_name' folder
        class_samples = len(list(data_dir.glob(dir_name+'/*.*g')))

        if (dir_name == neg[0]):
            neg_count += class_samples
        else:
            pos_count += class_samples

    DS_SIZE = neg_count+pos_count
    print ('\nNegative samples: {0:5} | {1:5.2f}%'.format(neg_count, neg_count/DS_SIZE*100))
    print ('Positive samples: {0:5} | {1:5.2f}%'.format(pos_count, pos_count/DS_SIZE*100))
    # Print number of images in dataset (excluded samples in outcast)
    print ('\nTotal number of images:', DS_SIZE)
    return DS_SIZE

    
def print_class_info(directories, data_dir, DS_SIZE, outcast, NUM_CLASSES):
    print ("Directories: ", directories, end='\n\n')

    samples_per_class = []
    for dir_name in directories:
        class_samples = len(list(data_dir.glob(dir_name+'/*.*g')))
        samples_per_class.append(class_samples)
        print('{0:18}: {1:3d}'.format(dir_name, class_samples))

    DS_SIZE = sum(samples_per_class)
    print ('\nTotal number of images: {}, in {} classes'.format(DS_SIZE, NUM_CLASSES))

    # If one class contains more than half of the entire sample size
    if np.max(samples_per_class) > DS_SIZE//2:
        print ("But the dataset is mainly shit")
    return DS_SIZE

    
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
        
def count(counts, batch):
    images, labels = batch
    
    class_1 = labels == 1
    class_1 = tf.cast(class_1, tf.int32)
    
    class_0 = labels == 0
    class_0 = tf.cast(class_0, tf.int32)

    counts['class_0'] += tf.reduce_sum(class_0)
    counts['class_1'] += tf.reduce_sum(class_1)

    return counts