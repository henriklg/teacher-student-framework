import os
import numpy as np
import tensorflow as tf
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
            directories.remove(outcast)
    
    # For each folder add files
    for folder in directories:
        files.extend(glob(join(data_dir,folder,'*.jpg')))
        
    
    DS_SIZE = len(files)
    
    if binary:
        assert neg_class!=None, "Must give a negative class"
        
        pos_class = directories
        pos_class.remove(neg_class)
        if verbose:
            print ("Positive class names: ", pos_class)
            print ("Negative class names: ", neg_class)
            print ()
    
    else: 
        continue()
    
    return_dict = {'ds': tf.data.Dataset.list_files(files),
                   'ds_size': DS_SIZE}