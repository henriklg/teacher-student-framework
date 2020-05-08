import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt



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