import tensorflow as tf
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm




def print_split_info(ds, class_names, cnt_per_class, ds_sizes):
    """
    Print info about the dataset
    """
    line = "{:28}: ".format('Category')
    cnt_list = []
    for split in ds: # train/test/val
        cnt = tf_bincount(ds[split], len(class_names))
        cnt_list.append(cnt)
        line += "{:5} | ".format(split)
    
    line += "{:5} | ".format("total")
    line += "{:5}".format("% of total")
    print (line, '\n'+'-'*72)
    
    for i in range(len(class_names)):
        line = "{:28}: ".format(class_names[i])
        for j in range(len(ds)):
            line += "{:5d} | ".format(int(cnt_list[j][i]))
        
        line+="{:5d} | ".format(cnt_per_class[class_names[i]][0])
        line+="{:5.2f}%".format(cnt_per_class[class_names[i]][1])
        print (line)
    
    # print totals
    print ('-'*72)
    line = "{:28}: ".format("Total")
    for ds_name in ds_sizes:
        line+= "{:5} | ".format(ds_sizes[ds_name])
    print (line)



def get_dataset_info(directories, data_dir, ds_size, ttv=True):
    """
    Get number of samples per class in dataset.
    
    return:
    a dictionary with number of samples per class and percentage of total
    """
    # Count number of samples for each folder
    count_dir = {}
    for dir_name in directories:
        if ttv:
            class_cnt = len(list(data_dir.glob('*/'+dir_name+'/*.*g')))
            count_dir[dir_name] = [class_cnt, class_cnt/ds_size*100]
        else:
            class_cnt = len(list(data_dir.glob(dir_name+'/*.*g')))
            count_dir[dir_name] = [class_cnt, class_cnt/ds_size*100]
    return count_dir



def print_bin_class_info(conf, directories, neg, pos):
    """
    Extract and print info about the class split of binary dataset
    """
    neg = [neg]
    # Count the samples in each folder
    count_dir = {}
    negpos = [0, 0]
    for dir_name in directories:
        # Number of samples in 'class_name' folder
        count = len(list(conf["data_dir"].glob('*/'+dir_name+'/*.*g')))
        count_dir[dir_name] = count
        
        if (dir_name == neg[0]):
            negpos[0] += count
        else:
            negpos[1] += count
    
    tot = np.sum(negpos)
    assert tot != 0, "Can't divide by zero."
    conf["neg_count"] = negpos[0]
    conf["pos_count"] = negpos[1]
    
    # Print folder name and amount of samples
    if conf["verbosity"]:
        for i, class_ in enumerate([neg, pos]):
            print ("\n{:33} : {:5} | {:2.2f}%".format(conf["class_names"][i], negpos[i], negpos[i]/tot*100))
            print ("-"*45)
            for cl in class_:
                print ("{:5}- {:26} : {:5} | {:>2.2f}%".format(" "*5, cl, count_dir[cl], count_dir[cl]/tot*100))
        print ('\nTotal number of image {} : {}\n'.format(" "*5, tot))



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




def better_class_dist(count_ds, num_classes):
    counts = tf_bincount(count_ds, num_classes)
    return counts/counts.sum()




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




def print_bar_chart(data, conf, title=None, fname=None, figsize=(15,6), show=True):
    """
    Takes in list of data and makes a bar chart of it.
    Dynamically allocates placement for bars.
    """
    import seaborn as sns
    sns.set()

    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
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
    plt.grid(axis='x')

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
    if show:
        plt.show()
    else:
        plt.close()



def unpipe(ds, size):
    return ds.unbatch().take(size)





def checkout_dataset(ds, conf=None):
    """
    Show some images from training dataset (mainly to check augmentation)
    ds is assumed to be from prepare_for_training - so batched, repeated etc
    """
    pathlib.Path(conf["log_dir"]).mkdir(parents=True, exist_ok=True)
    
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
    # make sure folder exists:
    pathlib.Path(conf["log_dir"]).mkdir(parents=True, exist_ok=True)
    
    if type(var) == dict:
        # Write conf and params dictionary to text file
        list_of_strings = [ '{:25} : {}'.format(key, var[key]) for key in var ]
        with open("{}/{}.txt".format(conf["log_dir"], fname),"w") as f:
            [ f.write(f'{st}\n') for st in list_of_strings ]
        f.close()
    else:
        f = open("{}/{}.txt".format(conf["log_dir"], fname),"w")
        f.write( str(var) )
        f.close()



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
