from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os
import glob
import pathlib
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

from utils import class_distribution, print_class_info_ttv, print_class_info, print_split_info

def create_dataset(conf):
    """
    Create a tf.data Dataset based on a config file.
    Expects dataset to be stored in data_dir/split/categories. 
    Don't currently accept binary ds.
    
    Pipeline: list_files -> get_label -> read_file -> decode_image -> resample
                -> prepare_for_training
    
    Args:
    conf - a dictionary with configuration settings
    
    Return:
    3x tf.data.Dataset for each train, test and val
    Dictionary with some settings like ds_size, class-names etc.
    """
    # Some parameters
    data_dir = conf["data_dir"]
    outcast = conf["outcast"]
    verbosity = conf["verbosity"]
    shuffle_buffer_size = conf["shuffle_buffer_size"]
    seed = conf["seed"]
    
    # Create cache-dir if not already exists
    pathlib.Path(conf["cache_dir"]).mkdir(parents=True, exist_ok=True)
    np.random.seed(seed=seed)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds_size = len(list(data_dir.glob('*/*/*.*g')))
    class_names = np.array(
        [item.name for item in data_dir.glob('train/*') if item.name != 'metadata.json']
    )
    
    # Remove the outcast folder
    if outcast != None:
        class_names = np.delete(class_names, np.where(outcast == class_names))
        if verbosity > 0: print ("Removed outcast:", outcast, end="\n\n")
    num_classes = len(class_names)
    
    # Print info about classes
    if verbosity > 0: 
        ds_size = print_class_info_ttv(class_names, data_dir, ds_size, outcast)
    
    # Create a tf.dataset of the file paths
    ds = {split: str(data_dir/split/'*/*.*g') for split in ["train","test","val"]}

    split_size = [len(list(glob.glob(ds[split]))) for split in ds] # train/test/val
    if verbosity > 0: print ("Dataset split:", split_size)
        
    if verbosity > 0:
        [print(key) for key in ds.values()]
        print ()
    
    for split in ds:
        ds[split] = tf.data.Dataset.list_files(
                                    ds[split],
                                    shuffle=shuffle_buffer_size>1,
                                    seed=tf.constant(seed, tf.int64) if seed else None
                                    )
    
    # Functions for the data pipeline
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
        return tf.image.resize(img, [conf["img_shape"][0], conf["img_shape"][1]])

    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label
    
    def process_path_bin(file_path):
        label = get_label_bin(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label
    
    # Create dataset of label, image pairs from the tf.dataset of image paths
    for split in ds:
        ds[split] = ds[split].map(process_path, num_parallel_calls=AUTOTUNE)
    
    # Save a clean copy of training data
    clean_train = ds["train"]
    
    # print info about the dataset split
    if verbosity:
        print_split_info(ds, num_classes, class_names)
    
    # Cache, shuffle, repeat, batch, prefetch pipeline
    for split in ds:
        ds[split] = prepare_for_training(ds[split], split, num_classes, conf, cache=True)
    
    # Return some parameters
    return_params = {
        "num_classes": num_classes,
        "sizes": {"total":ds_size, 
                  "train":split_size[0], 
                  "test":split_size[1], 
                  "val":split_size[2]},
        "steps": {"train":split_size[0]//conf["batch_size"], 
                  "test":split_size[1]//conf["batch_size"],
                  "val":split_size[2]//conf["batch_size"]}
        "class_names": class_names,
    }
    return clean_train, ds["train"], ds["test"], ds["val"], return_params



def prepare_for_training(ds, ds_name, num_classes, conf, cache):
    """
    Cache -> shuffle -> repeat -> augment -> batch -> prefetch
    """
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    # Resample dataset. NB: dataset is cached in resamler
    if conf["resample"] and 'train' in ds_name:
        ds = oversample(ds, ds_name, num_classes, conf)
    
    # Cache to SSD
    elif cache:
        cache_string = "{}/{}_{}_{}".format(
            conf["cache_dir"], conf["img_shape"][0], conf["ds_info"], ds_name
        )
        ds = ds.cache(cache_string)
    
    # Shuffle
    if conf["shuffle_buffer_size"]>1:
        ds = ds.shuffle(
            buffer_size=conf["shuffle_buffer_size"], 
            seed=tf.constant(conf["seed"], tf.int64) if conf["seed"] else None
        )
    # Repeat forever
    ds = ds.repeat()
    
    #Augment
    if conf["augment"] and ds_name=='train':
        ds = augment_ds(ds, conf, AUTOTUNE)
    
    # Batch
    ds = ds.batch(conf["batch_size"], drop_remainder=False)

    # Prefetch - lets the dataset fetch batches in the background while the model is training. 
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds



def oversample(ds, cache_name, num_classes, conf):
    """
    Resample the dataset. Accepts both binary and multiclass datasets.
    
    Args:
    - ds: dataset to balance/resample. Should not be repeated or batched
    - number of classes
    - verbosity
    
    Returns:
    - Resampled, repeated, and unbatched dataset
    """
    ## Check the original sample distribution
    if conf["verbosity"] > 0:
        print ("\n---- Ratios before resampling ---- ")
        initial_dist, count = class_distribution(ds, num_classes)
        print (initial_dist)

    ####################################
    ## Resample
    cache_dir = './cache/{}_{}_{}_resampled/'.format(
        conf["img_shape"][0], 
        conf["ds_info"],
        cache_name
    )
    # create directory if not already exist
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Beginning resampling
    datasets = []
    for i in range(num_classes):
        # Get all samples from class i [0 -> num_classes], repeat the dataset
        # indefinitely and store in datasets list
        data = ds.filter(lambda img, lab: lab==i)
        data = data.cache(cache_dir+'{}_ds'.format(i))
        data = data.repeat()
        datasets.append(data)
    
    target_dist = [ 1.0/num_classes ] * num_classes
    
    balanced_ds = tf.data.experimental.sample_from_datasets(datasets, target_dist)
    
    ####################################
    ## Check the sample distribution after oversampling the dataset
    if conf["verbosity"] > 0:
        print ("\n---- Ratios after resampling ----")
        final_distribution, _ = class_distribution(balanced_ds, num_classes)
        print (final_distribution)
    
    return balanced_ds



def augment_ds(ds, conf, AUTOTUNE):
    def random_rotate_image(img):
        img = ndimage.rotate(img, np.random.uniform(-30, 30), reshape=False)
        return img
    def augment(img, label):
        # Augment the image using tf.image
        if "rotate" in conf["augment"]:
            im_shape = img.shape
            [img,] = tf.py_function(random_rotate_image, [img], [tf.float32])
            img.set_shape(im_shape)
        if "crop" in conf["augment"]:
            # Pad image with 10 percent og image size, and randomly crop back to size
            pad = int(conf["img_shape"][0]*0.15)
            img = tf.image.resize_with_crop_or_pad(
                    img, conf["img_shape"][0] + pad, conf["img_shape"][1] + pad)
            img = tf.image.random_crop(img, conf["img_shape"], seed=conf["seed"])
        if "flip" in conf["augment"]:
              # Randomly flip image
            img = tf.image.random_flip_left_right(img, seed=conf["seed"])
            img = tf.image.random_flip_up_down(img, seed=conf["seed"])
        if "brightness" in conf["augment"]:
            # Change brightness and saturation
            img = tf.image.random_brightness(img, max_delta=0.15, seed=conf["seed"])
        if "saturation" in conf["augment"]:
            img = tf.image.random_saturation(img, lower = 0.5, upper =1.5, seed=conf["seed"])
        if "contrast" in conf["augment"]:
              img = tf.image.random_contrast(img, lower=0.6, upper=1.6, seed=conf["seed"])
        # Make sure imgae is still in [0, 1]
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label
    
    return ds.map(augment, num_parallel_calls=AUTOTUNE)




def split_and_create_dataset(conf):
    """
    Create a tf.data Dataset based on a config file.
    Expects dataset to be stored in data_dir/categories. This also accepts binary datasets.
    pipeline: list_files -> get_label -> read_file -> decode_image -> split -> resample
                -> prepare_for_training
    
    Args:
    conf - a dictionary with configuration settings
    
    Return:
    3x tf.data.Dataset for each train, test and val
    Dictionary with some settings like ds_size, class-names etc.
    """
    # Some parameters
    data_dir = conf["data_dir"]
    outcast = conf["outcast"]
    verbosity = conf["verbosity"]
    shuffle_buffer_size = conf["shuffle_buffer_size"]
    seed = conf["seed"]

    np.random.seed(seed=seed)
    train_cache = 'train'
    neg_count = pos_count = 0
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds_size = len(list(data_dir.glob('*/*.*g')))
    directories = np.array([item.name for item in data_dir.glob('*') if item.name != 'metadata.json'])
    
    # Remove the outcast folder
    if outcast != None:
        directories = np.delete(directories, np.where(outcast == directories))
        if verbosity > 0: print ("Removed outcast:", outcast, end="\n\n")

    ## Print info about the dataset
    # Binary dataset
    if conf["ds_info"] == 'binary':
        class_names = np.array(['Negative','Positive'])
        num_classes = len(class_names)
        neg_class_name = conf['neg_class'] # 'normal'-class
        pos_class_names = np.delete(directories, np.where(neg_class_name == directories))
        # Print info about neg/pos split
        if verbosity > 0: 
            ds_size, neg_count, pos_count = print_bin_class_info(
                                                directories, data_dir, 
                                                ds_size, outcast, class_names, 
                                                neg_class_name, pos_class_names
            )
    # Full dataset
    else:     
        class_names = directories
        num_classes = len(class_names)
        # Print info about classes
        if verbosity > 0: 
            ds_size = print_class_info(directories, data_dir, ds_size, outcast)
    
    # Create a tf.dataset of the file paths
    if outcast == None:
        files_string = str(data_dir/'*/*.*g')
    else:
        files_string = str(data_dir/'[!{}]*/*'.format(outcast))
    
    if verbosity > 0: print ("Dataset.list_files: ",files_string, "\n")
    
    
    list_ds = tf.data.Dataset.list_files(
            files_string, 
            shuffle=shuffle_buffer_size>1, 
            seed=tf.constant(seed, tf.int64) if seed else None
    )
    
    # Functions for the data pipeline
    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # get class integer from class-list
        label_int = tf.reduce_min(tf.where(tf.equal(parts[-2], class_names)))
        # cast to tensor array with dtype=uint8
        return tf.dtypes.cast(label_int, tf.int32)

    def get_label_bin(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        bc = parts[-2] == pos_class_names
        nz_cnt = tf.math.count_nonzero(bc)
        if (nz_cnt > 0):
            return tf.constant(1, tf.int32)
        return tf.constant(0, tf.int32)
        
    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [conf["img_shape"][0], conf["img_shape"][1]])

    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label
    
    def process_path_bin(file_path):
        label = get_label_bin(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    # Create dataset of label, image pairs from the tf.dataset of image paths
    if conf["ds_info"] == 'binary':
        labeled_ds = list_ds.map(process_path_bin, num_parallel_calls=AUTOTUNE)
    else:
        labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
        
    # Split into train, test and validation data
    train_size = int(0.7 * ds_size)
    test_size = int(0.15 * ds_size)
    val_size = int(0.15 * ds_size)
    
    train_ds = labeled_ds.take(train_size)
    test_ds = labeled_ds.skip(train_size)
    val_ds = test_ds.skip(val_size)
    test_ds = test_ds.take(test_size)

    # Print info about the dataset split
    if verbosity > 0:
        print ("\n{:32} {:>5}".format("Full dataset sample size:", ds_size))
        print ("{:32} {:>5}".format("Train dataset sample size:", train_size))
        print ("{:32} {:>5}".format("Test dataset sample size:", test_size))
        print ("{:32} {:>5}".format("Validation dataset sample size:", val_size))
    
    # Resample the dataset. NB: dataset is cached in resamler
    if conf["resample"]:
        train_ds = oversample(train_ds, num_classes, conf)
        train_cache = None
    
    # Create cache-dir if not already exists
    pathlib.Path(conf["cache_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Cache, shuffle, repeat, batch, prefetch pipeline
    train_ds = prepare_for_training(train_ds, 'train', conf, cache=train_cache)
    test_ds = prepare_for_training(test_ds, 'test', conf, cache=True)
    val_ds = prepare_for_training(val_ds, 'val', conf, cache=True)
            
    # Return some parameters
    return_params = {
        "num_classes": num_classes,
        "ds_size": ds_size,
        "train_size": train_size,
        "test_size": test_size,
        "val_size": val_size,
        "class_names": class_names,
        "neg_count": neg_count,
        "pos_count": pos_count
    }
    return train_ds, test_ds, val_ds, return_params
