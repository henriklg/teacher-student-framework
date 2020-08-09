import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

from sklearn import metrics as m
from utils import write_to_file, unpipe


def get_metrics(true_labels, predicted_labels):
    acc = np.round(m.accuracy_score(true_labels, predicted_labels),4)
    prec = np.round(m.precision_score(true_labels,predicted_labels,average='weighted'),4)
    rec = np.round(m.recall_score(true_labels, predicted_labels, average='weighted'), 4)
    fsc = np.round(m.f1_score(true_labels, predicted_labels, average='weighted', zero_division=0), 4)
    print('Accuracy:', acc)
    print('Precision:', prec)
    print('Recall:', rec)
    print('F1 Score:', fsc)
    
    
    
def get_classification_report(true_labels, predicted_labels, 
                                  classes, target_names=None, digits=3):
    """
    Returns a classification report with recall, precission and f1 score.
    """
    report = m.classification_report(
                y_true=true_labels,
                y_pred=predicted_labels,
                labels=classes,
                target_names=target_names,
                digits=digits,
                zero_division=0
    )
    return report
    
    
    
def display_model_performance_metrics(true_labels, predicted_labels, classes):
    print('Model Performance metrics:')
    print('-'*30)
    get_metrics(
        true_labels=true_labels, 
        predicted_labels=predicted_labels)
    print('\nModel Classification report:')
    print('-'*30)
    display_classification_report(
        true_labels=true_labels, 
        predicted_labels=predicted_labels, 
        classes=classes)
    print('\nPrediction Confusion Matrix:')
    print('-'*30)
    display_confusion_matrix(
        true_labels=true_labels, 
        predicted_labels=predicted_labels, 
        classes=classes)



def get_confusion_matrix(true_labels, predicted_labels):
    return m.confusion_matrix(true_labels, predicted_labels)



def plot_confusion_matrix(cm, log_dir, names=None, cmap='Blues', figsize=(15,13), show=True):
    """
    """
    # Font sizes
    axis_font = 10 # font size of x,y labels
    cell_font = 7 # font size of sns heatmap
    plt.rc('xtick', labelsize=axis_font)
    plt.rc('ytick', labelsize=axis_font)
    plt.rc('axes', titlesize=16) # font size of title
    plt.rc('axes', labelsize=12) # size of 'predicted','true label'
    
    plt.figure(figsize=figsize)
    ax = plt.subplot()

    # Show percentages inside cells and hide empty cells
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)

    #sns.set(font_scale=0.7) # for label size
    hm = sns.heatmap(cm_perc, annot=annot, fmt='', cmap=cmap, 
                linewidths=0.4, linecolor="white", annot_kws={"size": cell_font});
    cbar = hm.collections[0].colorbar
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.set_ticklabels(['0%', '20%', '50&', '75%', '100%'])

    #labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels'); 
    ax.set_title("Confusion matrix");
    if names is None:
        ax.xaxis.set_ticklabels(range(nrows), rotation=40, ha="right");
        ax.yaxis.set_ticklabels(range(nrows), rotation=40, ha="right");
    else:
        ax.xaxis.set_ticklabels(names, rotation=40, ha="right");
        ax.yaxis.set_ticklabels(names, rotation=40, ha="right");
    
    plt.tight_layout()
    plt.savefig(log_dir+"/confusion_matrix.pdf", format="pdf")
    if show:
        plt.show()
    else:
        plt.close()



def display_confusion_matrix(true_labels, predicted_labels, classes):
    total_classes = len(classes)
    level_labels = [total_classes*[0], list(range(total_classes))]

    cm = m.confusion_matrix(y_true=true_labels, y_pred=predicted_labels, labels=classes)
    cm_frame = pd.DataFrame(
        data=cm, 
        index=pd.MultiIndex(levels=[['Actual:'], classes], codes=level_labels),
        columns=pd.MultiIndex(levels=[['Predicted:'], classes], codes=level_labels), 
        )
    print(cm_frame)
    


def plot_lr_and_accuracy(history, conf):
    """
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

    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(history.epoch[-1]+1)
    
    if conf["decay_rate"] > 0:
        lr = history.history['lr']
        # Plot the learning rate
        plt.figure(figsize=(8, 6))
        plt.plot(epochs_range, lr, label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learnign rate')
        plt.title('Learning Rate development during training');
        plt.tight_layout()
        plt.savefig(conf["log_dir"]+'/learning_rate.pdf', format='pdf')
    
    # Plot train-val accuracy and loss
    plt.figure(figsize=(14, 6))

    # Subplot 1
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    # Subplot 2
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylim([0.0, 3])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.tight_layout()
    plt.savefig(conf["log_dir"]+'/accuracy_and_loss.pdf', format='pdf')
    plt.show()
    


def show_dataset_predictions(true_labels, pred_labels, pred_confidence, images, conf):
    grid_width = 5
    grid_height = 5
    f, ax = plt.subplots(grid_width, grid_height)
    f.set_size_inches(22, 22)

    img_idx = 0
    for i in range(0, grid_width):
        for j in range(0, grid_height):
            actual = conf["class_names"][true_labels[img_idx]]
            lab = conf["class_names"][pred_labels[img_idx]]
            pred = np.round(pred_confidence[img_idx], 2)

            title = 'Actual: '+actual+'\nPred: '+lab+ '\nConf: '+str(pred)
            ax[i][j].axis('off')
            ax[i][j].set_title(title)
            ax[i][j].imshow(images[img_idx])
            img_idx += 1

    plt.subplots_adjust(left=0, bottom=0, right=2, top=2, wspace=0.5, hspace=0.5)
    plt.tight_layout()
    plt.savefig("{}/checkout-eval_ds-pred.pdf".format(conf["log_dir"]), format="pdf")




def evaluate_model(model, history, ds, conf):
    # Save the metrics and model from training
    write_to_file(history.history, conf, "history")
    write_to_file(conf, conf, "conf")
    with open(conf["log_dir"]+"/history.pkl", 'wb') as f:
        pickle.dump(history.history, f)
    if conf["num_epochs"] > 9:
        model.save(conf["log_dir"]+'/model')
    
    # Plot learning rate and loss
    plot_lr_and_accuracy(history, conf)
    
    # Create true_labels and pred_labels for later evaluations
    eval_ds = unpipe(ds["test"], conf["ds_sizes"]["test"]).as_numpy_iterator()
    eval_ds = np.array(list(eval_ds))
    true_labels = list(eval_ds[:,1])
    eval_images = np.stack(eval_ds[:,0], axis=0)
    
    # Evaluate model on test dataset
    model_evaluation = model.evaluate(ds["test"], verbose=0, steps=conf["steps"]["test"])
    write_to_file(model_evaluation, conf, "evaluate_val")
    
    # Create predictions and pred_labels
    predictions = model.predict(eval_images, verbose=1)
    pred_confidence = [np.max(pred) for pred in predictions]
    pred_labels = [np.argmax(pred) for pred in predictions]
    
    # Classification report
    report = get_classification_report(
            true_labels, 
            pred_labels, 
            range(conf["num_classes"]), 
            target_names=conf["class_names"]
    )
    print (report)
    write_to_file(report, conf, "classification_report")

    # Confusion matrix
    cm = get_confusion_matrix(true_labels, pred_labels)
    plot_confusion_matrix(cm, conf["log_dir"], conf["class_names"], figsize=(12,10), show=False)