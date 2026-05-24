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
        # Clamp pred_var to a minimum of 1.0.
        # Critical: log(pred_var) >= log(1.0) = 0 ensures neg_log_prob >= 0 always.
        # With min=0.1, log(0.1)=-2.3 dominates once residuals are small, making the
        # VID loss negative. A negative loss breaks DynamicLossWeighter (Kendall et al.)
        # because its gradient w.r.t. log_vars is always positive when L_i<0, driving
        # exp(-log_vars[i]) -> +inf and total loss -> -inf. min=1.0 closes this path.
        pred_var = torch.clamp(F.softplus(self.log_scale) + self.eps, min=1.0)
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5 * ((pred_mean - target)**2 / pred_var + torch.log(pred_var))
        # Clamp per-element max to prevent catastrophic spikes from large residuals
        # early in training from corrupting weights through inf/nan gradients.
        loss = torch.mean(neg_log_prob.clamp(max=50.0))
        return loss
