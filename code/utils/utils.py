import tensorflow as tf
import numpy as np
import os
import textwrap
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from PIL import Image, ImageDraw, ImageFont



def print_split_info(ds, num_classes, class_names):
    """
    """
    # Count samples in each dataset by calling class_distribution
    line = "{:28}: ".format('Category')
    cnt_list = []
    for split in ds:
        _, cnt = class_distribution(ds[split], num_classes)
        cnt_list.append(cnt)
        line += "{:5} | ".format(split)
    print (line, '\n-------------')
    
    for i in range(num_classes):
        line = "{:28}: ".format(class_names[i])
        for j in range(len(ds)):
#             path = str(conf["data_dir"])+'/'+split+'/'+conf["class_names"][i]
#             num_files = len([name for name in os.listdir(path)])
            line += "{:5d} | ".format(int(cnt_list[j][i]))
        print (line)



def print_class_info(directories, data_dir, ds_size, outcast):
    """
    Extract and print info about the class split of multiclass dataset
    
    return:
    total numbeer of samples
    """
    num_classes = len(directories)
    # Count number of samples for each folder
    count_dir = {}
    for dir_name in directories:
        count_dir[dir_name] = len(list(data_dir.glob(dir_name+'/*.*g')))

    total = sum(count_dir.values())
    assert total != 0, "Can't divide by zero."
    
    for folder, count in count_dir.items():
        print ("{:28}: {:4d} | {:2.2f}%".format(folder, count, count/total*100))
        
    print ('\nTotal number of images: {}, in {} classes.\n'.format(ds_size, num_classes))
    return total



def print_class_info_ttv(directories, data_dir, ds_size, outcast):
    """
    Extract and print info about the class split of multiclass dataset
    
    return:
    total numbeer of samples
    """
    # Count number of samples for each folder
    num_classes = len(directories)
    count_dir = {}
    for dir_name in directories:
        count_dir[dir_name] = len(list(data_dir.glob('*/'+dir_name+'/*.*g')))
    
    total = sum(count_dir.values())
    assert total != 0, "Dataset is empty."
    
    for folder, count in count_dir.items():
        print ("{:28}: {:4d} | {:2.2f}%".format(folder, count, count/total*100))
        
    print ('\nTotal number of images: {}, in {} classes.\n'.format(ds_size, num_classes))
    return total



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



def class_distribution(count_ds, num_classes, count_batches=10, bs=1024):
    """
    Find distribution of dataset by counting a subset.
    
    Args: count_ds - dataset to be counted. Not batched or repeated.
    Return: a list of class distributions
    """
    def count(counts, batch):
        _, labels = batch
        for i in range(num_classes):
            counts['class_{}'.format(i)] += tf.reduce_sum(tf.cast(labels == i, tf.int32))
        return counts
    
    count_ds = count_ds.batch(bs)
    # Set the initial states to zero
    initial_state = {}
    for i in range(num_classes):
        initial_state['class_{}'.format(i)] = 0
        
    counts = count_ds.take(count_batches).reduce(
                initial_state = initial_state,
                reduce_func = count)

    final_counts = []
    for class_, value in counts.items():
                final_counts.append(value.numpy().astype(np.float32))

    final_counts = np.asarray(final_counts)
    distribution = final_counts/final_counts.sum()
    return distribution, final_counts



def get_class_weights(ds, conf):
    """
    """
    if not conf["class_weight"]:
        return None
    
    assert not conf["resample"], "Should only use resample or class_weight. Not both." 
    
    ds = unpipe(ds, conf["ds_sizes"]["train"])
    
    _, cnt = class_distribution(ds, conf["num_classes"])
    total = cnt.sum()
    score = total / (cnt*conf["num_classes"])
    # Set scores lower than 1.0 to 1
    score[score<1.0] = 1.0

    class_weights = dict(enumerate(score))
    
    if conf["verbosity"]:
        print ("---- Class weights ----")
        print (class_weights)
    
    return class_weights



def show_image(img, class_names=None, title=None):
    """
    Display a image given as either a tensor, list or ndarray.
    """
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


def print_bar_chart(data, conf, title=None, fname=None, figsize=(15,6)):
    """
    Takes in list of data and makes a bar chart of it.
    Dynamically allocates placement for bars.
    """
    x = np.arange(conf["num_classes"])
    width = 0.7      # 1.0 = bars side by side
    width = width/len(data)

    num_bars = len(data)
    if num_bars == 1:
        bar_placement = [0]
    # even number of bars
    elif (num_bars % 2) == 0:
        bar_placement = np.arange(-num_bars/2, num_bars/2+1)    #[-2, -1, 0, 1, 2]
        bar_placement = np.delete(bar_placement, num_bars//2)   #delete 0
        bar_placement = [bar+0.5 if bar<0 else bar-0.5 for bar in bar_placement]
    # odd number of bars
    else:
        bar_placement = np.arange(-np.floor(num_bars/2), np.floor(num_bars/2)+1)


    fig, ax = plt.subplots(figsize=figsize)

    rects = []
    for cnt, (dat, placement) in enumerate(zip(data, bar_placement)):
        rects.append(ax.bar(x+placement*width, dat, width, label='Iter {}'.format(cnt)))

    ax.set_ylabel('Number of samples')
    if title:
        title_string = title
        ax.set_title(title_string)
    ax.set_xticks(x)
    ax.set_xticklabels(conf["class_names"])
    ax.set_axisbelow(True)
    ax.legend(loc='upper left');

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right",
                 rotation_mode="anchor")
    plt.grid(axis='y')

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = int(rect.get_height())
            ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

    # autolabel(rects1)
    autolabel(rects[-1])

    fig.tight_layout()
    if fname:
        plt.savefig('{}/{}.pdf'.format(conf["log_dir"], fname), format='pdf')
    plt.show()



def unpipe(ds, size):
    return ds.unbatch().take(size)



def custom_sort(pred, lab, path):
    """
    Takes three lists and return three sorted list based on prediction confidence
    """
    sorted_list = list(zip(pred, lab, path))
    sorted_list.sort(key=lambda x: x[0], reverse=True)
    
    pred_sorted = [row[0] for row in sorted_list]
    lab_sorted = [row[1] for row in sorted_list]
    path_sorted = [row[2] for row in sorted_list]
    
    return pred_sorted, lab_sorted, path_sorted





def checkout_findings(unlab, conf):
    """
    unlab: pred, lab, path
    
    Todo; remove empty rows
    """
    pred_list = unlab[0]
    lab_list = unlab[1]
    name_list = unlab[2]
    ### Create images with label names
    class_label_img = []
    font_path = '/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf'
    img_width = 512
    font_size = int(img_width*0.15)
    letters_per_line = 13

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
    
    lab_arr = np.asarray(lab_list, dtype=np.uint8)
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
                fn = name_list[indekser[i]]
                img = fn2img(fn, conf["unlab_dir"], img_width)
                
                curr_class_examples.append(img)
                curr_class_preds.append(pred_list[indekser[i]])

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
    plt.show()



def checkout_class(checkout, unlab, conf):
    """
    unlab: pred, lab, name
    """
    pred_list = unlab[0]
    lab_list = unlab[1]
    name_list = unlab[2]
    
    # Which class number correspond to that class name
    class_idx = np.where(conf["class_names"] == checkout)[0]
    if len(class_idx) == 0:
        raise NameError('Error: class-name not found. Check spelling.')
        
    # List of img_list-indexes with images corresponding to that class number
    idx_list = np.where(lab_list == class_idx[0])[0]
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
            fn = name_list[idx]
            img = fn2img(fn, conf["unlab_dir"], 256)
            
            pred = pred_list[idx_list[i]]
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
    plt.show()



def checkout_dataset(ds, conf=None):
    """
    ds is assumed to be from prepare_for_training - so batched, repeated etc
    """
    # Show some images from training dataset (mainly to check augmentation)
    batch = next(iter(ds))
    images, labels = batch
    images = images.numpy()
    labels = labels.numpy()

    nrows, ncols = 3, 4  # array of sub-plots
    figsize = [ncols*4, nrows*4]     # figure size, inches

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, 
                           figsize=figsize, frameon=False, facecolor='white')

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        img = images[i]
        axi.imshow(img)
        if conf:
            lab = labels[i]
            title = conf["class_names"][lab]
            axi.set_title(title)
        axi.set_axis_off()

    plt.axis('off')
    plt.tight_layout(True)
    if conf:
        plt.savefig("{}/checkout-train_ds.pdf".format(conf["log_dir"]), format='pdf')
    plt.show()



def write_to_file(var, conf, fname):
    """
    Write 'var' to a .txt file inside log_dir. 
    """
    if type(var) == dict:
        # Write conf and params dictionary to text file
        list_of_strings = [ '{:20} : {}'.format(key, var[key]) for key in var ]
        with open("{}/{}.txt".format(conf["log_dir"], fname),"w") as f:
            [ f.write(f'{st}\n') for st in list_of_strings ]
        f.close()
    else:
        f = open("{}/{}.txt".format(conf["log_dir"], fname),"w")
        f.write( str(var) )
        f.close()



def resample_unlab(unlab, orig_dist,  conf):
    """
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



def get_tqdm(unlab_size, count, new_findings):
    tqdm_predicting = tqdm(total=unlab_size, desc='Predicting', position=0, initial=count)
    tqdm_findings = tqdm(total=unlab_size, desc='Findings', 
                     position=1, bar_format='{desc}:{bar}{n_fmt}', initial=new_findings)
    
    return tqdm_predicting, tqdm_findings



def fn2img(fn, folder, size):
    """
    Used for reading unlab_ds images
    """    
    fn = fn.numpy().decode("utf-8")
    img = tf.io.read_file("{}/{}".format(folder, fn))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [size, size])
    return img



def tf_bincount(ds, num_classes):
    """
    Counts samples for each class in dataset (NB: NOT REPEATED OR BATCHED)
    """
    count = np.zeros(num_classes)
    for img, lab in ds:
        count[lab] += 1
    return count