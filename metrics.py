from sklearn.metrics import confusion_matrix

import itertools
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class ImageMetrics:
    def __init__(self, image_dims):
        self.dim_W = image_dims(0)
        self.dim_H = image_dims(1)

    @staticmethod
    def semantic_seg_accuracy(actual, predicted, level='batch'):
        """
        :param actual: 4D matrix of dimension dim_B x dim_H x dim_W x n_classes
        :param predicted: 4D matrix of dimension dim_B x dim_H x dim_W x n_classes. Values contain pixel level class
        probabilities.
        :param level:batch/image
        :return: accuracy score float if level='batch' | (dim_B,) if level='image'
        """
        y_true = np.argmax(actual, axis=-1)  # B x H x W
        y_pred = np.argmax(predicted, axis=-1)  # B x H x W
        correct = y_true == y_pred  # B x H x W
        if level == 'batch':
            return np.mean(correct)
        elif level == 'image':
            return np.mean(correct, axis=(1, 2))

    @staticmethod
    def semantic_seg_metric(actual, predicted, metric='precision', level='batch', classes='all'):
        """
        :param actual: 4D array of dimension dim_B x dim_H x dim_W x n_classes (float32)
        :param predicted: 4D array of dimension dim_B x dim_H x dim_W x n_classes. Values contain pixel level class
                          probabilities.
        :param metric: precision/recall/iou
        :param level: batch/image
        :param classes: all/each
        :return: precision/recall/iou batch,all->scalar
                                      batch,each->(num_classes,)
                                      image,all->(dim_B,)
                                      image,each->(dim_B,num_classes)
        """
        num_classes = actual.shape[-1]
        y_true = actual
        y_pred = tf.one_hot(np.argmax(predicted, axis=-1), depth=num_classes)
        true_positives = np.logical_and(y_true, y_pred)

        if level == 'batch' and classes == 'all':
            axis_to_reduce = (0, 1, 2, 3)
        elif level == 'batch' and classes == 'each':
            axis_to_reduce = (0, 1, 2)
        elif level == 'image' and classes == 'all':
            axis_to_reduce = (1, 2, 3)
        elif level == 'image' and classes == 'each':
            axis_to_reduce = (1, 2)

        if metric == 'precision':
            denom = y_pred
        elif metric == 'recall':
            denom = y_true
        elif metric == 'iou':
            denom = np.logical_or(y_true, y_pred)  # union

        score = np.sum(true_positives, axis=axis_to_reduce) / np.sum(denom, axis=axis_to_reduce)
        return score

    @staticmethod
    def semantic_seg_confusion_matrix(actual, predicted):
        """
        :param actual: 4D array of dimension dim_B x dim_H x dim_W x n_classes with class labels (0,1,2,3...) as values
        :param predicted: 4D array of dimension dim_B x dim_H x dim_W x n_classes. Values contain pixel level class
                          probabilities.
        :return: n_classes x n_classes array for the batch
        """
        batch_size, height, width, num_classes = actual.shape
        y_true = np.argmax(actual, axis=-1).reshape(batch_size, height * width)  # b,h*w
        y_pred = np.argmax(predicted, axis=-1).reshape(batch_size, height * width)  # b,h*w
        labels = [i for i in range(num_classes)]
        cm = np.zeros((num_classes, num_classes))
        for i in range(batch_size):
            cm = cm + confusion_matrix(y_true[i, :], y_pred[i, :], labels=labels)
        return cm

    @staticmethod
    def plot_confusion_matrix(cm, target_names, title='Confusion Matrix', cmap=None, normalize=True):
        """
        given a sklearn confusion matrix (cm), make a nice plot

        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix

        target_names: given classification classes such as [0, 1, 2]
                      the class names, for example: ['high', 'medium', 'low']

        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                      see http://matplotlib.org/examples/color/colormaps_reference.html
                      plt.get_cmap('jet') or plt.cm.Blues

        normalize:    If False, plot the raw numbers
                      If True, plot the proportions

        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                  # sklearn.metrics.confusion_matrix
                              normalize    = True,                # show proportions
                              target_names = y_labels_vals,       # list of names of the classes
                              title        = best_estimator_name) # title of graph

        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        """
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=90)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.show()
        #plt.savefig('foo.png')



