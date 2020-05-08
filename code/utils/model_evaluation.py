import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics as m


def get_metrics(true_labels, predicted_labels):
    acc = np.round(m.accuracy_score(true_labels, predicted_labels),4)
    prec = np.round(m.precision_score(true_labels,predicted_labels,average='weighted'),4)
    rec = np.round(m.recall_score(true_labels, predicted_labels, average='weighted'), 4)
    fsc = np.round(m.f1_score(true_labels, predicted_labels, average='weighted', zero_division=0), 4)
    print('Accuracy:', acc)
    print('Precision:', prec)
    print('Recall:', rec)
    print('F1 Score:', fsc)
    
    
    
def display_classification_report(true_labels, predicted_labels, 
                                  classes, target_names=None, digits=3):
    """
    Does not work for some reason..
    """
    report = m.classification_report(
                y_true=true_labels,
                y_pred=predicted_labels,
                labels=classes,
                target_names=target_names,
                digits=digits,
                zero_division=0
    )
    print(report)
    
    
    
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



def plot_confusion_matrix(cm, log_dir, names=None, cmap='Blues', figsize=(15,13)):
    """
    TODO: adde toggle for save_fig
    """
    # Font sizes
    axis_font = 8 # font size of x,y labels
    cell_font = 7 # font size of sns heatmap
    plt.rc('xtick', labelsize=axis_font)
    plt.rc('ytick', labelsize=axis_font)
    plt.rc('axes', titlesize=16) # font size of title
    plt.rc('axes', labelsize=10) # size of 'predicted','true label'
    
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
    sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, annot_kws={"size": cell_font});

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
    plt.show()



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