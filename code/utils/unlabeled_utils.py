import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from tqdm.notebook import tqdm
from IPython.display import clear_output

from utils import print_bar_chart
from utils import fn2img



def generate_labels(findings_cnt, tot_cnt, lab_list, pred_list, name_list, unlab_ds, unlab_size, model, conf):
    """
    Generate new psuedo labels from unlabeled dataset
    """
    def plot_and_save(lab_list):
        lab_array = np.asarray(lab_list, dtype=np.uint8)
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

    tqdm_predicting, tqdm_findings = get_tqdm(unlab_size, tot_cnt, findings_cnt)

    print ("Press 'Interrupt Kernel' to save and exit.")
    try:
        for tot_cnt, (image,path) in enumerate(unlab_ds, start=tot_cnt):
            img = np.expand_dims(image, 0)
            pred = model.predict(img)
            highest_pred = np.max(pred)
            if highest_pred > conf["keep_threshold"]:
                pred_idx = np.argmax(pred).astype(np.uint8)

                lab_list.append(pred_idx)
                pred_list.append(highest_pred)
                name_list.append(path)

                # Clear old bar chart, generate new one and refresh the tqdm progress bars
                # NB, tqdm run-timer is also reset, unfortunately
                if not findings_cnt%500 and findings_cnt>100:
                    clear_output(wait=True)
                    tqdm_predicting, tqdm_findings = get_tqdm(unlab_size, tot_cnt, findings_cnt)
                    plot_and_save(lab_list)

                findings_cnt += 1
                tqdm_findings.update(1)
            tqdm_predicting.update(1)
    except KeyboardInterrupt:
        clear_output(wait=True)
        print ("Exiting")
        tqdm_predicting, tqdm_findings = get_tqdm(unlab_size, tot_cnt, findings_cnt)
        plot_and_save(lab_list)

    finally:
        print ("\nTotal run time: {:.3f} s".format( time.time() - total_time ))
        print ("Found {} new samples in unlabeled_ds after looking at {} images.".format(findings_cnt, tot_cnt))
        
    return lab_list, pred_list, name_list



def get_tqdm(unlab_size, count, new_findings):
    """
    Return a tqdm hbox
    """
    tqdm_predicting = tqdm(total=unlab_size, desc='Predicting', position=0, initial=count)
    tqdm_findings = tqdm(total=unlab_size, desc='Findings', 
                     position=1, bar_format='{desc}:{bar}{n_fmt}', initial=new_findings)
    
    return tqdm_predicting, tqdm_findings



def resample_unlab(unlab, orig_dist,  conf):
    """
    Resample unlabeled dataset based upon a given distribution.
    unlab: pred, lab, name
    """
    lab_list = unlab[1]
    name_list = unlab[2]
    
    num_to_match = np.max(orig_dist)
    idx_to_match = np.argmax(orig_dist)
    print ('Limit set by {} with {} samples'.format(conf["class_names"][idx_to_match], int(num_to_match)))
    print ("-"*40)

    new_findings = ([], [])
    new_findings_filepaths = []
    lab_arr = np.asarray(lab_list, dtype=np.uint8)

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
            fn = name_list[idx]
            img = fn2img(fn, conf["unlab_dir"], conf["img_shape"][0])
            
            new_findings[0].append(img)         # image
            new_findings[1].append(lab_list[idx])         # label
            new_findings_filepaths.append(name_list[idx]) # filepath
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