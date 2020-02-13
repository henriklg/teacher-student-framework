import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from os.path import join

def create_dataset(data_dir, outcasts=None, binary=False, neg_class=None, verbose=False):
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
    files = []
    directories = list(os.walk(data_dir))[0][1]
    
    # Remove all outcasts
    if outcasts != None:
        for outcast in outcasts:
            if verbose: print ("Removed outcasts: ", outcasts)
            directories.remove(outcast)
    
    # For each folder add files
    for folder in directories:
        files.extend(glob(join(data_dir,folder,'*.jpg')))
        
    
    DS_SIZE = len(files)
    
    if binary:
        assert neg_class!=None, "Must give a negative class"
        
        pos_class = directories.copy()
        pos_class.remove(neg_class[0])
        if verbose:
            print ("Positive class names: ", pos_class)
            print ("Negative class names: ", neg_class)
            print ()
    
    ds = tf.data.Dataset.list_files(files)
    
    return [ds, DS_SIZE, directories, pos_class]


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