import numpy as np
import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weighted_loss
from .smooth_l1_loss import smooth_l1_loss


@weighted_loss
def balanced_l1_loss(pred,
                     target,
                     beta=1.0,
                     alpha=0.5,
                     gamma=1.5,
                     reduction='mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0

    diff = torch.abs(pred - target)
    b = np.e**(gamma / alpha) - 1
    loss = torch.where(
        diff < beta, alpha / b *
        (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta)

    return loss


def reweight(pred, target, alpha, gamma, beta):
    diff = torch.abs(pred - target)
    ones = torch.ones_like(diff)
    b = np.e**(gamma / alpha) - 1
    base = torch.where(diff < beta, diff / beta, ones)
    enhanced = torch.where(diff < beta, alpha * torch.log(b * diff / beta + 1),
                           gamma * ones)
    rho = enhanced / base
    return rho


@LOSSES.register_module
class BalancedL1Loss(nn.Module):
    """Balanced L1 Loss

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """

    def __init__(self,
                 alpha=0.5,
                 gamma=1.5,
                 beta=1.0,
                 reduction='mean',
                 use_reweight=False,
                 loss_weight=1.0):
        super(BalancedL1Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.use_reweight = use_reweight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.use_reweight:
            rho = reweight(pred, target, self.alpha, self.gamma,
                           self.beta).detach()
            loss_bbox = self.loss_weight * rho * smooth_l1_loss(
                pred,
                target,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs)
        else:
            loss_bbox = self.loss_weight * balanced_l1_loss(
                pred,
                target,
                weight,
                beta=self.beta,
                alpha=self.alpha,
                gamma=self.gamma,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs)
        return loss_bbox
