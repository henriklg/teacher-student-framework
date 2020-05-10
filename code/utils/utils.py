import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# for checkout_unlab
from PIL import Image, ImageDraw, ImageFont
import textwrap



def print_split_info(ds, conf, params):
    """
    """
    # Count samples in each dataset by calling class_distribution
    line = "{:28}: ".format('Category')
    for split in ds:
        _, ds[split] = class_distribution(ds[split], params["num_classes"])
        line += "{:5} | ".format(split)
    print (line, '\n-------------')
    
    for i in range(params["num_classes"]):
        line = "{:28}: ".format(params["class_names"][i])
        for split in ds:
#             path = str(conf["data_dir"])+'/'+split+'/'+params["class_names"][i]
#             num_files = len([name for name in os.listdir(path)])
            line += "{:5d} | ".format(int(ds[split][i]))
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




def calculate_weights(count_ds, num_classes):
    """
    Find distribution of dataset by counting .
    
    Args: count_ds - dataset to be counted.
    Return: a list of class distributions
    """
    _, final_counts = class_distribution(count_ds, num_classes)
    
    total = final_counts.sum()
    
    score = total / (final_counts*num_classes)
    # Set scores lower than 1.0 to 1
    score[score<1.0] = 1.0
    return score



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



def print_bar_chart(lab_list, new_findings, count, params, log_dir=None, figsize=(15,6)):
    lab_array = np.asarray(lab_list, dtype=np.int64)
    findings = np.bincount(lab_array, minlength=int(params["num_classes"]))
    assert len(params["class_names"]) == len(findings), "Must be same length."

    # x = findings[:,0]
    x = np.arange(params["num_classes"])
    width = 0.5

    fig, ax = plt.subplots(figsize=figsize)
    # rects1 = ax.bar(x, findings[:,1], width, label='Findings')
    rects1 = ax.bar(x, findings, width, label='Findings')
    #rects2 = ax.bar(x + width/2, women_means, width, label='Women')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of samples')
    title_string = "Found {} new samples in unlabeled_ds after looking at {} images."
    ax.set_title(title_string.format(new_findings, count))
    ax.set_xticks(x)
    ax.set_xticklabels(params["class_names"])
    ax.set_axisbelow(True)
    ax.legend()

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right",
             rotation_mode="anchor")
    plt.grid(axis='y')

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)

    fig.tight_layout()
    if log_dir:
        plt.savefig(log_dir+'/unlab_data_prediction.pdf', format='pdf')
    plt.show()



def unpipe(ds, size):
    return ds.unbatch().take(size)



def custom_sort(pred, lab, img):
    """
    Takes three lists and return three sorted list based on prediction confidence
    """
    sorted_list = list(zip(pred, lab, img))
    sorted_list.sort(key=lambda x: x[0], reverse=True)
    
    pred_sorted = [row[0] for row in sorted_list]
    lab_sorted = [row[1] for row in sorted_list]
    img_sorted = [row[2] for row in sorted_list]
    
    return pred_sorted, lab_sorted, img_sorted





def checkout_unlab(unlab, conf, params, log_dir):
    """
    unlab: pred, lab, img
    
    Todo; remove empty rows
    """
    ### Create images with label names
    class_label_img = []
    font_path = '/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf'
    img_width = 512
    font_size = int(img_width*0.15)
    letters_per_line = 13

    for i in range(params["num_classes"]):
        img = Image.new('RGB', (img_width, img_width), color = (0, 0, 0))
        fnt = ImageFont.truetype(font_path, font_size)
        d = ImageDraw.Draw(img)
        if (len(params["class_names"][i])>letters_per_line):
            text = textwrap.fill(params["class_names"][i], width=letters_per_line)
        else:
            text = params["class_names"][i]
        linebreaks = text.count('\n')
        d.text((1,(img_width//2.2)-linebreaks*img_width*0.1), text, font=fnt, fill=(255, 255, 255))

        class_label_img.append(img)
        
    ### Create a list with 6 samples per class
    # black image
    img_black = Image.new('RGB', (conf["img_shape"][0], conf["img_shape"][1]), color = (0, 0, 0))

    class_examples = []
    class_preds = []
    for class_idx in range(params["num_classes"]):
        curr_class_examples = []
        curr_class_preds = []

        indekser = np.where(np.asarray(unlab[1], dtype=np.int64)==class_idx)[0]
        for i in range(6):
            # get 6 finding images from class_idx-class
            # no image - index out of bounds
            if not indekser.size > i:
                curr_class_examples.append(img_black)
                curr_class_preds.append(0)
            # found image
            else:
                curr_class_examples.append(unlab[2][indekser[i]])
                curr_class_preds.append(unlab[1][indekser[i]])

        class_examples.append(curr_class_examples)
        class_preds.append(curr_class_preds)

    assert (len(params["class_names"])==len(class_examples)), 'must be same length'
    
    ### Display the predicted images in each class
    # settings
    nrows, ncols = params["num_classes"], 7  # array of sub-plots
    figsize = [ncols*3, params["num_classes"]*3]     # figure size, inches

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, 
                           figsize=figsize, frameon=False, facecolor='white')

    # plot simple raster image on each sub-plot
    try:
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
    except IndexError:
        pass

    plt.axis('off')
    plt.tight_layout(True)
    plt.savefig("{}/checkout-{}.pdf".format(log_dir, 'all'), format='pdf')
    plt.show()




def checkout_class(checkout, unlab, conf, params, log_dir):
    """
    unlab: pred, lab, img
    """
    # Which class number correspond to that class name
#     try:
    class_idx = np.where(params["class_names"] == checkout)[0]
    if len(class_idx) == 0:
        raise NameError('Error: class-name not found. Check spelling.')
        
    # List of img_list-indexes with images corresponding to that class number
    idx_list = np.where(unlab[1] == class_idx[0])[0]
    
    # settings
    nrows, ncols = 4, 6    # array of sub-plots
    figsize = [15, 10]     # figure size, inches

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, 
                           figsize=figsize, frameon=False, facecolor='white')
    
    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        img = unlab[2][idx_list[i]]
        pred = unlab[0][idx_list[i]]
        title = "conf: "+str(round(pred, 5))
        axi.set_title(title)
        axi.imshow(img)
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        #axi.set_title("Row:"+str(rowid)+", Col:"+str(colid))
        axi.set_axis_off()
    
    plt.axis('off')
    plt.tight_layout(True)
    plt.savefig("{}/checkout-{}.pdf".format(log_dir, checkout), format='pdf')
    plt.show()