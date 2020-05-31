import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from tqdm.notebook import tqdm
from IPython.display import clear_output

from utils import print_bar_chart, write_to_file, fn2img



def generate_labels(count, unlab, unlab_ds, unlab_size, model, conf):
    """
    Generate new psuedo labels from unlabeled dataset
    """
    def plot_and_save(lab_list):
        lab_array = np.asarray(unlab["lab_list"], dtype=np.uint8)
        findings = np.bincount(lab_array, minlength=int(conf["num_classes"]))
        print_bar_chart(
            data=[findings],
            conf=conf,
            title=None,
            fname="bar_chart-findings",
            figsize=(16,7)
        )
        lab_array=None
    
    
    total_time = time.time()

    tqdm_predicting, tqdm_findings = get_tqdm(unlab_size, count["findings"], count["total"])

    print ("Press 'Interrupt Kernel' to save and exit.")
    try:
        for tot_cnt, (image,path) in enumerate(unlab_ds, start=count["total"]):
#             if tot_cnt > 10000:
#                 break
                    
            img = np.expand_dims(image, 0)
            pred = model.predict(img)
            highest_pred = np.max(pred)
            if highest_pred > conf["keep_threshold"]:
                pred_idx = np.argmax(pred).astype(np.uint8)

                unlab["lab_list"].append(pred_idx)
                unlab["pred_list"].append(highest_pred)
                unlab["name_list"].append(path)

                # Clear old bar chart, generate new one and refresh the tqdm progress bars
                # NB, tqdm run-timer is also reset, unfortunately
                if not count["findings"]%500 and count["findings"]>100:
                    clear_output(wait=True)
                    tqdm_predicting, tqdm_findings = get_tqdm(unlab_size, count["findings"], tot_cnt)
                    plot_and_save(unlab["lab_list"])
                    
                count["findings"] += 1   # previously findings_cnt
                tqdm_findings.update(1)
            tqdm_predicting.update(1)
    except KeyboardInterrupt:
        clear_output(wait=True)
        count["total"] = tot_cnt
        print ("Exiting")
        tqdm_predicting, tqdm_findings = get_tqdm(unlab_size, count["findings"], count["total"])
        plot_and_save(unlab["lab_list"])

    finally:
        print ("\nTotal run time: {:.1f} min.".format( (time.time() - total_time)/60 ))
        print ("Found {} new samples in unlabeled_ds after looking at {} images.".format(count["findings"], tot_cnt))
        
    return unlab, count



def get_tqdm(unlab_size, findings_cnt, tot_cnt):
    """
    Return a tqdm hbox
    """
    tqdm_predicting = tqdm(total=unlab_size, desc='Predicting', position=0, initial=tot_cnt)
    tqdm_findings = tqdm(total=unlab_size, desc='Findings', 
                     position=1, bar_format='{desc}:{bar}{n_fmt}', initial=findings_cnt)
    
    return tqdm_predicting, tqdm_findings



def custom_sort(unlab_findings):
    """
    Takes three lists and return three sorted list based on prediction confidence
    """
    # Unpack
    pred = unlab_findings["pred_list"]
    lab = unlab_findings["lab_list"]
    name = unlab_findings["name_list"]
    
    # Sort
    sorted_list = list(zip(pred, lab, name))
    sorted_list.sort(key=lambda x: x[0], reverse=True)
    
    pred_sorted = [row[0] for row in sorted_list]
    lab_sorted = [row[1] for row in sorted_list]
    name_sorted = [row[2] for row in sorted_list]
    
    # Re-pack
    unlab_findings = {
        "pred_list": pred_sorted,
        "lab_list": lab_sorted,
        "name_list": name_sorted
    }
    
    return unlab_findings



def resample_unlab(unlab, orig_dist,  conf):
    """
    Resample unlabeled dataset based upon a given distribution.
    """
    added = {}
    
    num_to_match = np.max(orig_dist)
    idx_to_match = np.argmax(orig_dist)
    print ('Limit set by {} with {} samples'.format(conf["class_names"][idx_to_match], int(num_to_match)))
    print ("-"*40)

    new_findings = ([], [])
    new_findings_filepaths = []
    lab_arr = np.asarray(unlab["lab_list"], dtype=np.uint8)

    for class_idx in range(conf["num_classes"]):
        # how many samples already in this class
        in_count = orig_dist[class_idx]

        indexes = np.where(lab_arr==class_idx)[0]
        num_new_findings = len(indexes)

        count = 0
        for count, idx in enumerate(indexes, start=1):
            if in_count >= num_to_match:
                count -= 1 # reduce by one cuz of enumerate updates index early
                break
            fn = unlab["name_list"][idx]
            img = fn2img(fn, conf["unlab_dir"], conf["img_shape"][0])
            
            new_findings[0].append(img)         # image
            new_findings[1].append(unlab["lab_list"][idx])         # label
            new_findings_filepaths.append(unlab["name_list"][idx]) # filepath
            in_count += 1
        
        if conf["verbosity"]:
            print ("{:27}: added {}/{} samples.".format(conf["class_names"][class_idx], count, num_new_findings))
            added[conf["class_names"][class_idx]] = [count, num_new_findings]
            
    write_to_file(added, conf, 'samples_added_to_train')
        
    return new_findings, new_findings_filepaths



def resample_unlab_orig(unlab, orig_dist,  conf):
    """
    Resample unlabeled dataset based upon a given distribution.
    """
    num_to_match = np.max(orig_dist)
    idx_to_match = np.argmax(orig_dist)
    print ('Limit set by {} with {} samples'.format(conf["class_names"][idx_to_match], int(num_to_match)))
    print ("-"*40)

    new_findings = ([], [])
    new_findings_filepaths = []
    lab_arr = np.asarray(unlab["lab_list"], dtype=np.uint8)

    for class_idx in range(conf["num_classes"]):
        # how many samples already in this class
        in_count = orig_dist[class_idx]

        indexes = np.where(lab_arr==class_idx)[0]
        num_new_findings = len(indexes)

        count = 0
        for count, idx in enumerate(indexes, start=1):
            if in_count >= num_to_match:
                count -= 1 # reduce by one cuz of enumerate updates index early
                break
            fn = unlab["name_list"][idx]
            img = fn2img(fn, conf["unlab_dir"], conf["img_shape"][0])
            
            new_findings[0].append(img)         # image
            new_findings[1].append(unlab["lab_list"][idx])         # label
            new_findings_filepaths.append(unlab["name_list"][idx]) # filepath
            in_count += 1
        
        if conf["verbosity"]:
            print ("{:27}: added {}/{} samples.".format(conf["class_names"][class_idx], count, num_new_findings))
    
    return new_findings, new_findings_filepaths



def reduce_dataset(ds, remove=None):
    """
    Remove samples which are combined into the training data from the unlabeled data
    """
    def remove_samples(img, path):
        """
        Filter out images which filename exists in new_findings_filepaths.
        Return boolean.
        """
        bool_list = tf.equal(path, remove)
        in_list = tf.math.count_nonzero(bool_list) > 0
        return not in_list
    
    return ds.filter(remove_samples)