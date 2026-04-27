### Modified:

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

'''
Author: Jiajie Chen, Helong Zhou.

Implemented the following paper:
Helong Zhou, Liangchen Song, Jiajie Chen, Ye Zhou, Guoli Wang, Junsong Yuan, Qian Zhang. "Rethinking Soft Labels for Knowledge Distillation: A Bias-Variance Tradeoff Perspective" (ICLR2021)
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


class WSLLoss(nn.Module):
    def __init__(self, T):
        super(WSLLoss, self).__init__()


        self.T = T
        self.softmax = nn.Softmax(dim=1)#.cuda()
        self.logsoftmax = nn.LogSoftmax(dim=1)#.cuda()



    def forward(self, g_s, g_t, target):


        s_input_for_softmax = g_s / self.T
        t_input_for_softmax = g_t / self.T

        t_soft_label = self.softmax(t_input_for_softmax)

        softmax_loss = - torch.sum(t_soft_label * self.logsoftmax(s_input_for_softmax), 1, keepdim=True)

        fc_s_auto = g_s.detach()
        fc_t_auto = g_t.detach()
        log_softmax_s = self.logsoftmax(fc_s_auto)
        log_softmax_t = self.logsoftmax(fc_t_auto)
        one_hot_label = F.one_hot(target, num_classes=g_s.size(1)).float() #for cifar100
        softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
        softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)

        focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
        ratio_lower = torch.zeros(1).cuda()
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(- focal_weight)
        softmax_loss = focal_weight * softmax_loss

        soft_loss = (self.T ** 2) * torch.mean(softmax_loss)

        loss = soft_loss

        return loss
