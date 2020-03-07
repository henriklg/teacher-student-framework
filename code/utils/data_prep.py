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
    # Some parameters
    data_dir = config["data_dir"]
    outcast = config["outcast"]
    verbosity = config["verbosity"]
    neg_count = pos_count = 0
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    DS_SIZE = len(list(data_dir.glob('*/*.*g')))
    directories = np.array([item.name for item in data_dir.glob('*') if item.name != 'metadata.json'])
    
    # Remove the outcast folder
    if outcast != None:
        directories = np.delete(directories, np.where(outcast == directories))
        if verbosity > 0: print ("Removed outcast:", outcast, end="\n\n")

    ## Print info about the dataset
    # Binary dataset
    if config["ds_info"] == 'binary':
        class_names = np.array(['Negative','Positive'])
        NUM_CLASSES = len(class_names)
        neg_class_name = config['neg_class'] # 'normal'-class
        pos_class_names = np.delete(directories, np.where(neg_class_name == directories))
        # Print info about neg/pos split
        if verbosity > 0: 
            DS_SIZE, neg_count, pos_count = print_bin_class_info(directories, data_dir, 
                                            DS_SIZE, outcast, class_names, neg_class_name, pos_class_names)
    # Full dataset
    else:     
        class_names = directories
        NUM_CLASSES = len(class_names)
        # Print info about classes
        if verbosity > 0: 
            DS_SIZE = print_class_info(directories, data_dir, DS_SIZE, outcast, NUM_CLASSES)
    
    # Create a tf.dataset of the file paths
    if outcast == None:
        files_string = str(data_dir/'*/*.*g')
    else:
        files_string = str(data_dir/'[!{}]*/*'.format(outcast))
    
    if verbosity > 0: print ("Dataset.list_files: ",files_string, "\n")
    list_ds = tf.data.Dataset.list_files(files_string)
    
    # Functions for the data pipeline
    def get_label(file_path):
        # if binary ds
        if config["ds_info"] == 'binary':
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
        return tf.image.resize(img, [config["img_shape"][0], config["img_shape"][1]])

    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    # Create dataset of label, image pairs from the tf.dataset of image paths
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    
    # Print some labels (to check the class-distribution)
    if verbosity > 0:
        for images, labels in labeled_ds.batch(10).take(10):
            print(labels.numpy())
    
    # Resample the binary dataset
    if config["resample"]:
        labeled_ds = resample(labeled_ds.batch(1024), NUM_CLASSES, verbosity=1)
    
    # Split into train, test and validation data
    train_size = int(0.7 * DS_SIZE)
    test_size = int(0.15 * DS_SIZE)
    val_size = int(0.15 * DS_SIZE)
    
    train_ds = labeled_ds.take(train_size)
    test_ds = labeled_ds.skip(train_size)
    val_ds = test_ds.skip(val_size)
    test_ds = test_ds.take(test_size)

    # Print info about the dataset split
    if verbosity > 0 and not config["resample"]:
        def get_size(ds):
            return tf.data.experimental.cardinality(ds).numpy()

        print ("\n{:32} {:>5}".format("Full dataset sample size:", get_size(labeled_ds)))
        print ("{:32} {:>5}".format("Train dataset sample size:", get_size(train_ds)))
        print ("{:32} {:>5}".format("Test dataset sample size:", get_size(test_ds)))
        print ("{:32} {:>5}".format("Validation dataset sample size:", get_size(val_ds)))
    elif verbosity > 0 and config["resample"]:
        print ("\n{:32} {:>5}".format("Full dataset sample size:", DS_SIZE))
        print ("{:32} {:>5}".format("Train dataset sample size:", train_size))
        print ("{:32} {:>5}".format("Test dataset sample size:", test_size))
        print ("{:32} {:>5}".format("Validation dataset sample size:", val_size))
    
    
#     def augment(img, label):
#         # Augment the image using tf.image
#         # Standardize
#         img = tf.image.per_image_standardization(img)
#         # Pad with 8 pixels
#         img = tf.image.resize_with_crop_or_pad(img, IMG_HEIGHT + 8, IMG_WIDTH + 8)
#         # Randomly crop the image back to original size
#         img = tf.image.random_crop(img, [IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS])
#         # Randomly flip image
#         img = tf.image.random_flip_left_right(img)
#         return img, label

#     # Augment the training data
#     train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
    
    
    # Create training, test and validation dataset
    cache_dir = config["cache_dir"]
    img_width = config["img_shape"][0]
    ds_info = config["ds_info"]
    # Create cache-dir if not already exists
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    train_ds = prepare_for_training(
        train_ds, config["batch_size"], 
        cache="{}/{}_{}_train.tfcache".format(cache_dir, img_width, ds_info),
        shuffle_buffer_size=config["shuffle_buffer_size"]
    )
    test_ds = prepare_for_training(
        test_ds, config["batch_size"], 
        cache="{}/{}_{}_test.tfcache".format(cache_dir, img_width, ds_info),
        shuffle_buffer_size=config["shuffle_buffer_size"]
    )
    val_ds = prepare_for_training(
        val_ds, config["batch_size"], 
        cache="{}/{}_{}_val.tfcache".format(cache_dir, img_width, ds_info),
        shuffle_buffer_size=config["shuffle_buffer_size"]
    )
    
   
    # Return some of the data
    return_params = {
        "num_classes": NUM_CLASSES,
        "ds_size": DS_SIZE,
        "train_size": train_size,
        "test_size": test_size,
        "val_size": val_size,
        "class_names": class_names,
        "neg_count": neg_count,
        "pos_count": pos_count
    }
    
    return train_ds, test_ds, val_ds, return_params
    
    
    

def prepare_for_training(ds, bs, cache=True, shuffle_buffer_size=4000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    
    if shuffle_buffer_size > 0:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(bs, drop_remainder=False)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds
    
    
    
def print_bin_class_info(directories, data_dir, DS_SIZE, outcast, class_names, neg, pos):
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

    
    
def print_class_info(directories, data_dir, DS_SIZE, outcast, NUM_CLASSES):
    # print ("Directories: ", directories, end='\n\n')
    
    # Count number of samples for each folder
    count_dir = {}
    for dir_name in directories:
        count_dir[dir_name] = len(list(data_dir.glob(dir_name+'/*.*g')))

    tot = sum(count_dir.values())
    assert tot != 0, "Can't divide by zero."
    
    for folder, count in count_dir.items():
        print ("{:28}: {:4d} | {:2.2f}%".format(folder, count, count/tot*100))
        
    print ('\nTotal number of images: {}, in {} classes.\n'.format(DS_SIZE, NUM_CLASSES))

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
    


def resample(ds, num_classes, verbosity=0):
    """
    Resample the dataset
    
    Args:
    - ds: dataset to balance/resample. Should be batched to ~1024 samples per batch
    - number of classes
    - verbosity
    
    Returns:
    - Resampled, repeated, and unbatched dataset
    """
    certainty_bs = 5
    
    if verbosity > 0: print ("\nBeginning resampling..")
    def count(counts, batch):
        images, labels = batch

        for i in range(num_classes):
            counts['class_{}'.format(i)] += tf.reduce_sum(tf.cast(labels == i, tf.int32))

        return counts
    
    # Set the initial states
    initial_state = {}
    for i in range(num_classes):
        initial_state['class_{}'.format(i)] = 0

    # Count samples
    counts = ds.take(certainty_bs).reduce(
        initial_state = initial_state,
        reduce_func = count)

    final_counts = []
    for class_, value in counts.items():
        final_counts.append(value.numpy().astype(np.float32))

    final_counts = np.asarray(final_counts)

    fractions = final_counts/final_counts.sum()
    if verbosity > 0:
        print ("---- Fractions ---- ")
        print (fractions)
        
    
    # Beginning resampling
    datasets = []
    for i in range(num_classes):
        data = ds.unbatch().filter(lambda image, label: label==i).repeat()
        datasets.append(data)
    
    target_dist = [1.0/num_classes] * num_classes
    
    balanced_ds = tf.data.experimental.sample_from_datasets(datasets, target_dist)
    
    if verbosity > 0:
        print ("\n---- Fractions after resampling ---.")
        counts = balanced_ds.batch(10).take(certainty_bs).reduce(
            initial_state = initial_state,
            reduce_func = count)

        final_counts = []
        for class_, value in counts.items():
            final_counts.append(value.numpy().astype(np.float32))

        final_counts = np.asarray(final_counts)

        fractions = final_counts/final_counts.sum()
        print (fractions)
    
    return balanced_ds