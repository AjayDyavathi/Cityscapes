"""
schedulers.py
"""

from torch.optim.lr_scheduler import _LRScheduler


class ConstantLR(_LRScheduler):
    "Constant lr scheduler"
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self):
        return list(self.base_lrs)
