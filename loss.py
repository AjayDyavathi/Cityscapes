"""
loss.py
Computes batch wise loss between predictions and targets

TODO: Add more loss functions
"""
import torch
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


def focal_loss_2d(input_, target, gamma=0.5, reduction="mean"):
    """ Compute Focal loss
        Introduced in Focal Loss for Dense Object Detection
        Paper: https://arxiv.org/abs/1708.02002
    """
    ignore_index = 255
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
    # Compute log(softmax(x))
    log_softmax_op = F.log_softmax(input_, dim=1)
    # Make a copy of target to replace ignore_index with 0
    target_safe = target.detach().clone()
    target_safe[target_safe == ignore_index] = 0
    # Gather log softmax values at target indices
    target_log_probs = log_softmax_op.gather(1,
                                             target_safe.view(-1, 1)).view(-1)
    # Compute focus loss -1 * (1 - Pt)**gamma * target_log_softmax values
    focus_losses = -1.0 * ((1-target_log_probs)**gamma) * target_log_probs
    # Substitute 0s at ignore index locations
    focus_losses_adjusted = torch.where(target != ignore_index,
                                        focus_losses,
                                        0)
    # perform mean only at non-ignoring target locations (acc. to PyTorch doc)
    if reduction == "mean":
        n_ignore_indices = (target == ignore_index).sum()
        return torch.sum(focus_losses_adjusted) / (len(focus_losses_adjusted) -
                                                   n_ignore_indices)
    if reduction == "sum":
        return torch.sum(focus_losses_adjusted)
    if reduction == "none":
        return focus_losses_adjusted
    raise ValueError(f"{reduction} is not a valid value for reduction")
