from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VIDLoss(nn.Module):
    """Variational Information Distillation for Knowledge Transfer (CVPR 2019),
    code from author: https://github.com/ssahn0215/variational-information-distillation"""
    def __init__(self,
                 num_input_channels,
                 num_mid_channel,
                 num_target_channels,
                 init_pred_var=5.0,
                 eps=1e-5):
        super(VIDLoss, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, padding=0,
                bias=False, stride=stride)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var-eps)-1.0) * torch.ones(num_target_channels)
            )
        self.eps = eps

    def forward(self, input, target):
        # pool for dimension match
        if input.shape[2:] != target.shape[2:]:
            target_size = (min(input.shape[2], target.shape[2]), min(input.shape[3], target.shape[3]))
            if input.shape[2:] > target.shape[2:]:
                input = F.adaptive_avg_pool2d(input, target_size)
            else:
                target = F.adaptive_avg_pool2d(target, target_size)
        pred_mean = self.regressor(input)
        # Clamp pred_var to a minimum of 0.1: prevents collapse toward eps where any
        # non-trivial residual creates (residual^2 / tiny_var) -> inf gradient.
        pred_var = torch.clamp(F.softplus(self.log_scale) + self.eps, min=0.1)
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5 * ((pred_mean - target)**2 / pred_var + torch.log(pred_var))
        # Clamp per-element loss to prevent outlier spatial locations from producing
        # inf/nan in backward. At optimum, neg_log_prob is in [-5, 5]; 50.0 only
        # activates for pathological residual/variance ratios.
        loss = torch.mean(neg_log_prob.clamp(max=50.0))
        return loss
