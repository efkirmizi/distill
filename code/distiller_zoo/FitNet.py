from __future__ import print_function

import torch.nn as nn


class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, g_s, g_t):
        loss = [self.crit(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]


        return sum(loss)
