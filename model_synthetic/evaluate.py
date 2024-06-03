import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns

def plot_history(history):

    fig, (ax1, ax2) = plt.subplots(2)

    fig.set_size_inches(18.5, 10.5)

    # Plot loss
    ax1.set_title('Loss variation through epochs',fontsize=16)
    ax1.plot(history.history['loss'], linewidth=1.5)
    ax1.set_ylabel('Loss',fontsize=14)
    ax1.set_xlabel('Epoch',fontsize=14)
    ax1.grid(True, which='both')
      
    max_loss = max(history.history['loss'])

    ax1.set_ylim([0, np.ceil(max_loss)])

    # Plot accuracy
    ax2.set_title('Accuracy variation through epochs',fontsize=16)
    ax2.plot(history.history['accuracy'], linewidth=1.5)
    ax2.set_ylabel('Accuracy',fontsize=14)
    ax2.set_xlabel('Epoch',fontsize=14)
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, which='both')

    plt.tight_layout(pad=1, h_pad=1, w_pad=1)

    return fig

def plot_confusion_matrix(actual, predicted, labels, ds_type):
      fig, ax = plt.subplots()
      cm = tf.math.confusion_matrix(actual, predicted)
      ax = sns.heatmap(cm, annot=True, fmt='g')
      ax.set_title('Confusion matrix for ' + ds_type)
      ax.set_xlabel('Prediction')
      ax.set_ylabel('Actual')
      plt.xticks(rotation=90)
      plt.yticks(rotation=0)
      ax.xaxis.set_ticklabels(labels)
      ax.yaxis.set_ticklabels(labels)

      plt.subplots_adjust(bottom=0.3,left=0.25)
      
      return fig

def calculate_classification_metrics(y_actual, y_pred, labels):
    
      #Calculate the precision and recall of a classification model using the ground truth and
      #predicted values. 

      #Args:
        #y_actual: Ground truth labels.
        #y_pred: Predicted labels.
        #labels: List of classification labels.

      #Return:
        #Precision and recall measures.
    
    cm = tf.math.confusion_matrix(y_actual, y_pred)
    tp = np.diag(cm) # Diagonal represents true positives
    precision = dict()
    recall = dict()
    for i in range(len(labels)):
      col = cm[:, i]
      fp = np.sum(col) - tp[i] # Sum of column minus true positive is false negative

      row = cm[i, :]
      fn = np.sum(row) - tp[i] # Sum of row minus true positive, is false negative

      precision[list(labels)[i]] = tp[i] / (tp[i] + fp) # Precision 

      recall[list(labels)[i]] = tp[i] / (tp[i] + fn) # Recall

    return precision, recall