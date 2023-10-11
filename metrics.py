"""
metrics.py
Computes scores:
* Overall accuracy
* Mean accuracy
* Mean IoU
* Frequency weighted accuracy
* Class wise IoU

Running score adapted from meeetps and wkentaro
https://github.com/meetps/pytorch-semseg
https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
"""
import numpy as np


class RunningScore():
    """Computes scores"""
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label, pred):
        # Extract the valid pixel locations from label
        mask = (label >= 0) & (label < self.n_classes)
        # bincount(n * L + P).reshape(square)
        hist = np.bincount(
            self.n_classes * label[mask].astype(int) + pred[mask],
            minlength=self.n_classes**2
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def update(self, labels, preds):
        "Computes confusion matrix over a batch"
        for lbl, pred in zip(labels, preds):
            self.confusion_matrix += self._fast_hist(lbl.flatten(),
                                                     pred.flatten())

    def get_scores(self):
        """Returns scores:
            - Overall accuracy (float)
            - Mean accuracy (list, one for each class)
            - Mean IoU (float)
            - Frequency weighted accuracy (float)
        """
        conf_mx = self.confusion_matrix
        accuracy = conf_mx.diagonal().sum() / conf_mx.sum()
        acc_class = conf_mx.diagonal() / conf_mx.sum(axis=1)
        acc_class_mean = np.nanmean(acc_class)
        intersection = conf_mx.diagonal().sum()
        union = conf_mx.sum(axis=0) + conf_mx.sum(axis=1) - intersection
        iu = intersection / union
        mean_iu = np.nanmean(iu)
        freq = conf_mx.sum(axis=1) / conf_mx.sum()
        fwav = (freq[freq > 0] * iu[freq > 0]).sum()
        class_iu = dict(zip(range(self.n_classes), iu))
        score_dict = {
            "Overall Accuracy: \t": accuracy,
            "Mean Accuracy: \t": acc_class_mean,
            "FreqW Acc \t": fwav,
            "Mean IoU: \t": mean_iu
        }
        return score_dict, class_iu

    def reset(self):
        "Resets confusion matrix to zeroes"
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageMeter():
    """Keeps track of running mean"""
    def __init__(self):
        self.reset()

    def reset(self):
        "resets variables to 0"
        self.value = 0
        self.mean = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        "updates mean, count with value"
        self.value = value
        self.sum += value * n
        self.count += n
        self.mean = self.sum / self.count
