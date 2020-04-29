from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

def create_dataset(conf):
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
            ds_size, neg_count, pos_count = print_bin_class_info(directories, data_dir, 
                                            ds_size, outcast, class_names, neg_class_name, pos_class_names)
    # Full dataset
    else:     
        class_names = directories
        num_classes = len(class_names)
        # Print info about classes
        if verbosity > 0: 
            ds_size = print_class_info(directories, data_dir, ds_size, outcast, num_classes)
    
    # Create a tf.dataset of the file paths
    if outcast == None:
        files_string = str(data_dir/'*/*.*g')
    else:
        files_string = str(data_dir/'[!{}]*/*'.format(outcast))
    
    if verbosity > 0: print ("Dataset.list_files: ",files_string, "\n")
    list_ds = tf.data.Dataset.list_files(
            files_string, 
            shuffle=shuffle_buffer_size>1, 
            seed=tf.constant(seed, tf.int64)
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
    
    # Print some labels (to check the class-distribution)
#     if verbosity > 0:
#         for images, labels in labeled_ds.batch(10).take(10):
#             print(labels.numpy())
    
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
#         train_ds = resample(train_ds, num_classes, conf)
        train_ds = reject_resample(train_ds, num_classes, conf)
        train_cache = None

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
            img = tf.image.random_crop(img, conf["img_shape"], seed=seed)
        if "flip" in conf["augment"]:
            # Randomly flip image
            img = tf.image.random_flip_left_right(img, seed=seed)
            img = tf.image.random_flip_up_down(img, seed=seed)
        if "brightness" in conf["augment"]:
            # Change brightness and saturation
            img = tf.image.random_brightness(img, max_delta=0.15, seed=seed)
        if "saturation" in conf["augment"]:
            img = tf.image.random_saturation(img, lower = 0.5, upper =1.5, seed=seed)
        
        # Make sure imgae is still in [0, 1]
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

     # Augment the training data
    if conf["augment"]:
        train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
        
    # Create cache-dir if not already exists
    pathlib.Path(conf["cache_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Cache, shuffle, repeat, batch, prefetch pipeline
    train_ds = prepare_for_training(train_ds, train_cache, conf)
    test_ds = prepare_for_training(test_ds, 'test', conf)
    val_ds = prepare_for_training(val_ds, 'val', conf)

    # Print some labels (to check the class-distribution)
#     print ()
#     if conf["verbosity"] > 0:
#         for images, labels in train_ds.unbatch().batch(10).take(10):
#             print(labels.numpy())
            
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
    
    
    

def prepare_for_training(ds, cache, conf):
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            cache_string = "{}/{}_{}_{}.tfcache".format(
                conf["cache_dir"], conf["img_shape"][0], conf["ds_info"], cache
            )
            ds = ds.cache(cache_string)
        else:
            ds = ds.cache()
    
    if conf["shuffle_buffer_size"]>1:
        ds = ds.shuffle(buffer_size=conf["shuffle_buffer_size"], 
                        seed=tf.constant(conf["seed"], tf.int64))

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(conf["batch_size"], drop_remainder=False)

    # `prefetch` lets the dataset fetch batches in the background while the model is training. 
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
    
    
    
def print_bin_class_info(directories, data_dir, ds_size, outcast, class_names, neg, pos):
    """
    Extract and print info about the class split of binary dataset
    """
    # Count the samples in each folder
    count_dir = {}
    negpos = [0, 0]
    for dir_name in directories:
        # Number of samples in 'class_name' folder
        count = len(list(data_dir.glob(dir_name+'/*.*g')))
        count_dir[dir_name] = count
        
        if (dir_name == neg[0]):
            negpos[0] += count
        else:
            negpos[1] += count
    
    tot = np.sum(negpos)
    assert tot != 0, "Can't divide by zero."
    
    # Print folder name and amount of samples
    for i, class_ in enumerate([neg, pos]):
        print ("\n{:27} : {:5} | {:2.2f}%".format(class_names[i], negpos[i], negpos[i]/tot*100))
        print ("-"*45)
        for cl in class_:
            print ("{:5}- {:20} : {:5} | {:>2.2f}%".format(" "*5, cl, count_dir[cl], count_dir[cl]/tot*100))
    print ('\nTotal number of image {} : {}\n'.format(" "*5, tot))
    
    return tot, negpos[0], negpos[1]

    
    
def print_class_info(directories, data_dir, ds_size, outcast, num_classes):
    # print ("Directories: ", directories, end='\n\n')
    
    # Count number of samples for each folder
    count_dir = {}
    for dir_name in directories:
        count_dir[dir_name] = len(list(data_dir.glob(dir_name+'/*.*g')))

    tot = sum(count_dir.values())
    assert tot != 0, "Can't divide by zero."
    
    for folder, count in count_dir.items():
        print ("{:28}: {:4d} | {:2.2f}%".format(folder, count, count/tot*100))
        
    print ('\nTotal number of images: {}, in {} classes.\n'.format(ds_size, num_classes))

    return tot

    
def show_image(img, class_names=None, title=None):
    if (isinstance(img, tf.data.Dataset)):
        for image, label in img:
            plt.figure(frameon=False, facecolor='white')
            class_name = class_names[label.numpy()]+" ["+str(label.numpy())+"]"
            plt.title(class_name, fontdict={'color':'white','size':20})
            plt.imshow(image.numpy())
            plt.axis('off')
    else:
        plt.figure(frameon=False, facecolor='white')
        if type(title) == str:
            plt.title(title, fontdict={'color':'white','size':20})
        plt.imshow(img)
        plt.axis('off')
    


def resample(ds, num_classes, conf):
    """
    Resample the dataset. Accepts both binary and multiclass datasets.
    
    Args:
    - ds: dataset to balance/resample. Should not be repeated or batched
    - number of classes
    - verbosity
    
    Returns:
    - Resampled, repeated, and unbatched dataset
    """
    # How many batches to use when counting the dataset
    certainty_bs = 10
    
    ### Counting functions
    def count(counts, batch):
        images, labels = batch

        for i in range(num_classes):
            counts['class_{}'.format(i)] += tf.reduce_sum(tf.cast(labels == i, tf.int32))

        return counts
    
    def count_samples(count_ds):
        # Set the initial states to zero
        initial_state = {}
        for i in range(num_classes):
            initial_state['class_{}'.format(i)] = 0
        
        counts = count_ds.take(certainty_bs).reduce(
                    initial_state = initial_state,
                    reduce_func = count)

        final_counts = []
        for class_, value in counts.items():
                    final_counts.append(value.numpy().astype(np.float32))

        final_counts = np.asarray(final_counts)
        fractions = final_counts/final_counts.sum()
        print (fractions)
    
    ## Count
    if conf["verbosity"] > 0:
        print ("\n---- Ratios before resampling ---- ")
        count_samples(ds.batch(1024))

    ####################################
    ## Resample
    
    cache_dir = './cache/{}_train_cache/'.format(conf["img_shape"][0])
    # create directory if not already exist
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Beginning resampling
    datasets = []
    for i in range(num_classes):
        # Get all samples from class i [0 -> num_classes], repeat the dataset
        # indefinitely and store in datasets list
        data = ds.filter(lambda image, label: label==i)
        data = data.cache(cache_dir+'{}_ds'.format(i))
        data = data.repeat() # temp removed unbatch and added cache
        datasets.append(data)
    
    target_dist = [ 1.0/num_classes ] * num_classes
    
    balanced_ds = tf.data.experimental.sample_from_datasets(datasets, target_dist)
    
    ####################################
    
    ## Count
    if conf["verbosity"] > 0:
        print ("\n---- Ratios after resampling ----")
        count_samples(balanced_ds.batch(1024))
    
    return balanced_ds



def reject_resample(ds, num_classes, conf):
    # How many batches to use when counting the dataset
    certainty_bs = 10
    
    if conf["verbosity"]: 
        print ('\nResample the dataset with rejection_resample')
    ####################################
    ### Counting functions
    def count(counts, batch):
        images, labels = batch

        for i in range(num_classes):
            counts['class_{}'.format(i)] += tf.reduce_sum(tf.cast(labels == i, tf.int32))

        return counts
    
    def count_samples(count_ds):
        # Set the initial states to zero
        initial_state = {}
        for i in range(num_classes):
            initial_state['class_{}'.format(i)] = 0
        
        counts = count_ds.take(certainty_bs).reduce(
                    initial_state = initial_state,
                    reduce_func = count)

        final_counts = []
        for class_, value in counts.items():
                    final_counts.append(value.numpy().astype(np.float32))

        final_counts = np.asarray(final_counts)
        fractions = final_counts/final_counts.sum()
        return fractions
    ####################################
    ## Count before resample
    print ("\n---- Ratios before resampling ---- ")
    initial_dist = count_samples(ds.batch(1024))
    print (initial_dist)
    ####################################
    ## Resample
    
    def class_func(img, lab):
        return lab
    
    target_dist = [ 1.0/num_classes ] * num_classes
    
    resampler = tf.data.experimental.rejection_resample(
        class_func, target_dist=target_dist, initial_dist=initial_dist
    )
    
    resample_ds = ds.apply(resampler)
    
    balanced_ds = resample_ds.map(lambda extra_label, img_and_label: img_and_label)
    
    ####################################
    ## Count after resample
    if conf["verbosity"] > 0:
        print ("\n---- Ratios after resampling ----")
        print(count_samples(balanced_ds.batch(1024)))
    ####################################
    return balanced_ds