"""
loss.py
Computes batch wise loss between predictions and targets

TODO: Add more loss functions
"""
import torch.nn.functional as F


# Credits to https://github.com/meetps/pytorch-semseg
# Using NEAREST interpolation instead of BILINEAR.
def cross_entropy_2d(input_, target, weight=None, reduction="mean"):
    "Compute CE Loss"
    n, c, h, w = input_.size()
    nt, ht, wt = target.size()

    # handle inconsistent size between target and input
    # If height or width doesn't match, interpolate pixels to match target.
    if h != ht or w != wt:
        input_ = F.interpolate(input_, size=(ht, wt),
                               mode="nearest", align_corners=True)

    # Reshaping to match axes [NCHW -> NHWC -> XC] (X = N*H*W).
    input_ = input_.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    # Reshaping to Y (Y = NT*HT*WT)
    target = target.view(-1)
    loss = F.cross_entropy(input=input_, target=target,
                           weight=weight, reduction=reduction,
                           ignore_index=255)
    return loss
