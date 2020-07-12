import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import textwrap

from tqdm.notebook import tqdm
from IPython.display import clear_output
from PIL import Image, ImageDraw, ImageFont
from utils import print_bar_chart, write_to_file, fn2img, tf_bincount



def generate_labels(pseudo, count, unlab_ds, model, conf):
    """
    Generate new psuedo labels from unlabeled dataset
    """
    def plot_and_save(lab_list, show=False):
        lab_array = np.asarray(lab_list, dtype=np.uint8)
        findings = np.bincount(lab_array, minlength=int(conf["num_classes"]))
        print_bar_chart(
            data=[findings],
            conf=conf,
            title=None,
            fname="pseudo_labels_distribution",
            figsize=(16,7),
            show=show
        )
        lab_array=None
    
    total_time = time.time()
    tqdm_predicting, tqdm_findings = get_tqdm(conf["ds_sizes"]["unlab"], count["findings"], count["total"])
    
    if conf["verbosity"]:
        print ("Press 'Interrupt Kernel' to save and exit.")
    try:
        for tot_cnt, (image,path) in enumerate(unlab_ds, start=count["total"]):
            if tot_cnt > conf["pseudo_thresh"] and conf["pseudo_thresh"] is not 0:
                break
                    
            img = np.expand_dims(image, 0)
            pred = model.predict(img)
            highest_pred = np.max(pred)
            if highest_pred > conf["keep_thresh"]:
                pred_idx = np.argmax(pred).astype(np.uint8)

                pseudo["lab_list"].append(pred_idx)
                pseudo["pred_list"].append(highest_pred)
                pseudo["name_list"].append(path)

                # For every 1000 pseudo label found, save results in case of crash
                if not count["findings"]%2000 and count["findings"]>100:
                    plot_and_save(pseudo["lab_list"])
                    with open(conf["log_dir"]+"/pseudo_labels.pkl", 'wb') as f:
                        pickle.dump(pseudo, f)
                    
                count["findings"] += 1
                tqdm_findings.update(1)
            tqdm_predicting.update(1)
    except KeyboardInterrupt:
        count["total"] = tot_cnt
        print ("Exiting")

    finally:
        # Plot and save
        plot_and_save(pseudo["lab_list"], show=True)
        with open(conf["log_dir"]+"/pseudo_labels.pkl", 'wb') as f:
            pickle.dump(pseudo, f)
            
        print ("\nTotal run time: {:.1f} min.".format( (time.time() - total_time)/60 ))
        print ("Found {} new samples in unlabeled_ds after looking at {} images.".format(count["findings"], tot_cnt))
        
    return pseudo, count



def get_tqdm(unlab_size, findings_cnt, tot_cnt):
    """
    Return a tqdm hbox
    """
    tqdm_predicting = tqdm(total=unlab_size, desc='Predicting', position=0, initial=tot_cnt)
    tqdm_findings = tqdm(total=unlab_size, desc='Findings', 
                     position=1, bar_format='{desc}:{bar}{n_fmt}', initial=findings_cnt)
    
    return tqdm_predicting, tqdm_findings



def custom_sort(pseudo):
    """
    Takes three lists and return three sorted list based on prediction confidence
    """
    # Unpack
    pred = pseudo["pred_list"]
    lab = pseudo["lab_list"]
    name = pseudo["name_list"]
    
    # Sort
    sorted_list = list(zip(pred, lab, name))
    sorted_list.sort(key=lambda x: x[0], reverse=True)
    
    pred_sorted = [row[0] for row in sorted_list]
    lab_sorted = [row[1] for row in sorted_list]
    name_sorted = [row[2] for row in sorted_list]
    
    # Re-pack
    pseudo = {
        "pred_list": pred_sorted,
        "lab_list": lab_sorted,
        "name_list": name_sorted
    }
    
    return pseudo



def resample_unlab(pseudo, orig_dist, conf, limit=0):
    """
    Resample unlabeled dataset based upon a given distribution.
    """
    added = {}
    total_added = 0
    
    if limit is 0:
        limit = int(np.max(orig_dist))
        limit_set_by = conf["class_names"][np.argmax(orig_dist)]
    else:
        limit_set_by = 'user'
        
    if conf["verbosity"]:
        print ('Limit set by {} with {} samples'.format(limit_set_by, limit))
        print ("-"*50)

    new_findings = ([], [])
    new_findings_filepaths = []
    lab_arr = np.asarray(pseudo["lab_list"], dtype=np.uint8)

    for class_idx in range(conf["num_classes"]):
        # how many samples already in this class
        in_count = orig_dist[class_idx]

        indexes = np.where(lab_arr==class_idx)[0]
        num_new_findings = len(indexes)

        count = 0
        for count, idx in enumerate(indexes, start=1):
            if in_count >= limit:
                count -= 1 # reduce by one cuz of enumerate updates index early
                break
            fn = pseudo["name_list"][idx]
            img = fn2img(fn, conf["unlab_dir"], conf["img_shape"][0])
            
            new_findings[0].append(img)                             # image
            new_findings[1].append(pseudo["lab_list"][idx])         # label
            new_findings_filepaths.append(pseudo["name_list"][idx]) # filepath
            in_count += 1
            
        total_added += count
        if conf["verbosity"]:
            print ("{:27}: added {}/{} samples".format(conf["class_names"][class_idx], count, num_new_findings))
            
        added[conf["class_names"][class_idx]] = [count, num_new_findings]
    write_to_file(added, conf, 'samples_added_to_train')
    
    if conf["verbosity"]:
        print ("-"*50)
        text = "Added a total of {} samples to the training dataset. New dataset size is {}."
        print (text.format(total_added, conf["ds_sizes"]["train"] + total_added))
        
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
    
    new_unlab_ds = ds.filter(remove_samples)
    return new_unlab_ds




def resample_and_combine(ds, conf, pseudo, pseudo_sorted, datasets_bin, limit=0):
    """
    """
    if conf["resample"]:
        new_findings, added_samples = resample_unlab(pseudo_sorted, datasets_bin[-1], conf, limit=limit)
        # create tf.tensor of the new findings
        findings_tensor = tf.data.Dataset.from_tensor_slices(new_findings)
    else:
        # NB: this uses the un-sorted list of labels
        added_samples = pseudo["name_list"]
        img_list = [fn2img(name, conf["unlab_dir"], conf["img_shape"][0]) for name in pseudo["name_list"]]
        findings_tensor = tf.data.Dataset.from_tensor_slices((img_list, pseudo["lab_list"]))

    # combine with original training_ds (using clean_ds which is not augmented/repeated etc)
    if len(added_samples) != 0: # if no samples are added just re-use previous combined_train
        ds["combined_train"] = ds["combined_train"].concatenate(findings_tensor)

    # count samples in the new/combined dataset
    datasets_bin.append(tf_bincount(ds["combined_train"], conf["num_classes"]))
    with open(conf["log_dir"]+"/datasets_bin.pkl", 'wb') as f:
        pickle.dump(datasets_bin, f)

    # History of class distribution
    print_bar_chart(
        data=datasets_bin,
        conf=conf,
        title=None,
        fname="bar_chart-distribution"
    )
    return datasets_bin, added_samples



def update_sanity(sanity, added_count, datasets_bin, conf):
    """
    """
    sanity.append({"added_samples": added_count,
               "last_unlab_size": conf["ds_sizes"]["unlab"],
               "curr_unlab_size": conf["ds_sizes"]["unlab"] - added_count,
               "last_train_size": int(np.sum(datasets_bin[-2])),
               "curr_train_size": int(np.sum(datasets_bin[-1]))
              })
    write_to_file(sanity, conf, "sanity")
    
    # Update dataset sizes
    conf["ds_sizes"]["unlab"] -= added_count
    conf["ds_sizes"]["train"] = sanity[-1]["curr_train_size"]
    conf["steps"]["train"] = sanity[-1]["curr_train_size"]//conf["batch_size"] 
    return sanity, conf




def checkout_findings(pseudo, conf, show=True):
    """
    Create a large plot of 6 samples from every class found in the unlabeled dataset
    """
    ### Create images with label names
    class_label_img = []
#     font_path = '/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf'
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
    img_width = 512
    font_size = int(img_width*0.14) #hypkva 15
    letters_per_line = 12           #hypkva 13

    for i in range(conf["num_classes"]):
        img = Image.new('RGB', (img_width, img_width), color = (0, 0, 0))
        fnt = ImageFont.truetype(font_path, font_size)
        d = ImageDraw.Draw(img)
        if (len(conf["class_names"][i])>letters_per_line):
            text = textwrap.fill(conf["class_names"][i], width=letters_per_line)
        else:
            text = conf["class_names"][i]
        linebreaks = text.count('\n')
        d.text((1,(img_width//2.2)-linebreaks*img_width*0.1), text, font=fnt, fill=(255, 255, 255))

        class_label_img.append(img)
        
    ### Create a list with 6 samples per class
    # black image
    img_black = Image.new('RGB', (conf["img_shape"][0], conf["img_shape"][1]), color = (0, 0, 0))
    
    lab_arr = np.asarray(pseudo["lab_list"], dtype=np.uint8)
    class_examples = []
    class_preds = []
    for class_idx in range(conf["num_classes"]):
        curr_class_examples = []
        curr_class_preds = []

        indekser = np.where(lab_arr==class_idx)[0]
        for i in range(6):
            # get 6 finding images from class_idx-class
            # no image - index out of bounds
            if not indekser.size > i:
                curr_class_examples.append(img_black)
                curr_class_preds.append(0)
            # found image
            else:
                fn = pseudo["name_list"][indekser[i]]
                img = fn2img(fn, conf["unlab_dir"], img_width)
                
                curr_class_examples.append(img)
                curr_class_preds.append(pseudo["pred_list"][indekser[i]])

        class_examples.append(curr_class_examples)
        class_preds.append(curr_class_preds)

    assert (len(conf["class_names"])==len(class_examples)), 'must be same length'
    
    ### Display the predicted images in each class
    # settings
    nrows, ncols = conf["num_classes"], 7  # array of sub-plots
    figsize = [ncols*3, conf["num_classes"]*3]     # figure size, inches

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, 
                           figsize=figsize, frameon=False, facecolor='white')

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        rowid = i // ncols
        colid = i % ncols

        if colid == 0:
            img = class_label_img[rowid]
        else:
            pred = class_preds[rowid][colid-1]
            title = "conf: "+str(round(pred, 3))
            if pred: axi.set_title(title)
            img = class_examples[rowid][colid-1]
            
        axi.imshow(img)
        axi.set_axis_off()

    plt.axis('off')
    plt.tight_layout(True)
    plt.savefig("{}/checkout-all.pdf".format(conf["log_dir"]), format='pdf')
    if show:
        plt.show()
    else:
        plt.close()



def checkout_class(checkout, pseudo, conf, show=True):
    """
    Display a grid with sample images from one specified class
    """    
    # Which class number correspond to that class name
    class_idx = np.where(conf["class_names"] == checkout)[0]
    if len(class_idx) == 0:
        raise NameError('Error: class-name not found. Check spelling.')
        
    # List of img_list-indexes with images corresponding to that class number
    idx_list = np.where(pseudo["lab_list"] == class_idx[0])[0]
    num_images = len(idx_list)
    if (num_images == 0):
        raise IndexError("No findings in this class.")
    
    # settings
    figint = np.sqrt(num_images)
    nrows = int(np.floor(figint)) if figint <= 5 else 5
    ncols = int(np.ceil(figint)) if figint <= 5 else 5
    figsize = [ncols*3, nrows*3]     # figure size, inches

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, 
                           figsize=figsize, frameon=False, facecolor='white')
    
    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        idx = idx_list[i]
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        try:
            fn = pseudo["name_list"][idx]
            img = fn2img(fn, conf["unlab_dir"], 256)
            
            pred = pseudo["pred_list"][idx_list[i]]
            title = "conf: "+str(round(pred, 5))
            axi.set_title(title)
            axi.imshow(img)
        except IndexError:
            # No more images - skip last suplots
            pass
        finally:
            axi.set_axis_off()
    
    plt.axis('off')
    plt.tight_layout(True)
    plt.savefig("{}/checkout-{}.pdf".format(conf["log_dir"], checkout), format='pdf')
    if show:
        plt.show()
    else:
        plt.close()