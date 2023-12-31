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
import torch
import numpy as np
from data.cityscapes_labels import trainId2name


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
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()

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
        # diag and sum(ax=1) can sometimes have 0s, ignore 0.0/0.0 error
        with np.errstate(divide="ignore", invalid="ignore"):
            conf_mx = self.confusion_matrix
            accuracy = conf_mx.diagonal().sum() / conf_mx.sum()
            acc_class = conf_mx.diagonal() / conf_mx.sum(axis=1)
            acc_class_mean = np.nanmean(acc_class)
            intersection = conf_mx.diagonal()
            union = conf_mx.sum(axis=0) + conf_mx.sum(axis=1) - intersection
            iu = intersection / union
            mean_iu = np.nanmean(iu)
            freq = conf_mx.sum(axis=1) / conf_mx.sum()
            fwav = (freq[freq > 0] * iu[freq > 0]).sum()
            class_iu = {trainId2name.get(i, "ignore"): class_iu_
                        for i, class_iu_ in enumerate(iu)}
            # class_iu = dict(zip(range(self.n_classes), iu))
        score_dict = {
            "Accuracy": accuracy,
            "Mean Accuracy": acc_class_mean,
            "Global IoU": fwav,
            "Mean IoU": mean_iu
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
