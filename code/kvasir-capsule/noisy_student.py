from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import datetime
import pickle
import time
import os
import pathlib
import matplotlib.pyplot as plt
import sys
import shutil

# Some stuff to make utils-function work
sys.path.append('../utils')
from pipeline import *
from create_model import *
from utils import *
from unlabeled_utils import *
from evaluate_model import *

begin_time = time.time()

project_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

##### CONFIG #####
data_dir = pathlib.Path('/home/henriklg/master-thesis/data/kvasir-capsule/labeled_new_splits/')
unlab_dir = pathlib.Path('/home/henriklg/master-thesis/data/kvasir-capsule/unlabeled_ttv/')

conf = {
    # Dataset
    "data_dir": data_dir,
    "unlab_dir": unlab_dir,
    "ds_info": 'hypcap',
    "neg_class": None,                 # select neg class for binary ds (normal class)
    "augment": ["crop","flip","brightness","saturation","contrast","rotate"],
#     "aug_mult": 0.8,
    "resample": True,
    "class_weight": False,
    "shuffle_buffer_size": 2000,       # 0=no shuffling
    "seed": 2511,
    "outcast": None,                   # list of folders to drop - currently only works for 1 item
    # Model
#     "model": 'EfficientNetB4',         # EfficientNet or ResNet50
    "weights": "imagenet",             # which weights to initialize the model with
#     "dropout": 0.2,
    "num_epochs": 2,
    "batch_size": 8,
    "img_shape": (128, 128, 3),
    "learning_rate": 0.001,
    "optimizer": 'Adam',
    "final_activation": 'softmax',     # sigmoid for binary ds
    # Callbacks
    "tensorboard": False,
    "decay_rate": 0,                   # 128:0.25   64:1.0   32:4.0   16:16   8:64
    "checkpoint": False,
    "early_stopp_patience": 0,         # disable: 0
    # Misc
    "verbosity": 1,
    "keep_thresh": 0.95,                # probability threshold for inferring pseudo labels
    "pseudo_thresh": 2000,
    "class_limit": 2000,
    "cache_dir": "./cache",
    }


##### RUN ITERATION #####
def run_iteration(conf, ds, datasets_bin, sanity):
    """
    """
    model = create_model(conf)
    callbacks = create_callbacks(conf)
    class_weights = get_class_weights(ds["train"], conf)

    start_time = time.time()
    history = model.fit(
            ds["train"],
            steps_per_epoch = conf["steps"]["train"],
            epochs = conf["num_epochs"],
            validation_data = ds["test"],
            validation_steps = conf["steps"]["test"],
            validation_freq = 1,
            class_weight = class_weights,
            callbacks = callbacks,
            verbose = 1
    )
    if conf["verbosity"]:
        print ("Time spent on training: {:.2f} minutes.".format(np.round(time.time() - start_time)/60))

    evaluate_model(model, history, ds, conf)

    count = {"findings": 0, "total": 0}
    pseudo = {"pred_list": [], "lab_list": [], "name_list": []}

    pseudo, count = generate_labels(pseudo, count, ds["unlab"], model, conf)

    # Sort in order of highest confidence to lowest
    pseudo_sorted = custom_sort(pseudo)

    checkout_findings(pseudo_sorted, conf, show=False)

    datasets_bin, added_samples = resample_and_combine(ds, conf, pseudo, pseudo_sorted, datasets_bin, limit=conf["class_limit"])

    # Update unlab_ds
    ds["unlab"] = reduce_dataset(ds["unlab"], remove=added_samples)

    sanity, conf = update_sanity(sanity, len(added_samples), datasets_bin, conf)


##### EXPERIMENT VALUES #####
teacher = {
    "name": "teacher",
    "model": "EfficientNetB0",
    "aug_mult": 0.2,
    "dropout": 0.1
}
student = {
    "name": "student",
    "model": "EfficientNetB0",
    "aug_mult": 0.6,
    "dropout": 0.2
}
models_list = [teacher, student]
sanity = []


##### RUN EXPERIMENT #####
for idx, curr_model in enumerate(models_list):
    iteration = int((np.floor(idx/2.0)))   # 0,0,1,1 etc
    dir_name = str(iteration)+'_'+curr_model["name"]
    print ("\n#### {} ####\n".format(dir_name))
    conf["log_dir"] = "./logs/{}/{}".format(project_time, dir_name)

    # Update model hyper-parameters
    for (key, value) in curr_model.items():
        conf[key] = value

    # Prepare the dataset
    if idx is 0:
        # First iteration only - create dataset
        ds = create_dataset(conf)
        ds["unlab"] = create_unlab_ds(conf)
        datasets_bin = [tf_bincount(ds["clean_train"], conf["num_classes"])]
        ds["combined_train"] = ds["clean_train"]
    else:
        # refresh training data
        ds["train"] = prepare_for_training(
            ds=ds["combined_train"],
            ds_name='train_'+dir_name,
            conf=conf,
            cache=True
        )

    run_iteration(conf, ds, datasets_bin, sanity)

print ("Time spent: {:.2f} minutes.".format(np.round(time.time() - begin_time)/60))
