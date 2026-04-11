from __future__ import print_function, division

import sys
import time
import torch
import numpy as np
import math
from .util import AverageMeter, accuracy



def adjust_learning_rate(optimizer, epoch, opt, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = opt.learning_rate*(0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt, scaler=None):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            output = model(input)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt, scaler=None, teacher_dict=None):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()


    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]
    if opt.distill == 'ATT_crd':
        criterion_kd1 = criterion_list[3] #crd loss

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_cls = AverageMeter()
    losses_div = AverageMeter()
    losses_kd = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd'] or opt.distill == 'WSL_crd'  or opt.distill == 'ATT_crd':
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd'] or opt.distill == 'WSL_crd'  or opt.distill == 'ATT_crd':
                contrast_idx = contrast_idx.cuda()



        # ===================forward=====================
        preact = False
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            feat_s, logit_s = model_s(input, is_feat=True, preact=preact)

            #s_points are the last layers of blocks, unless it is specified:
            #NOTE: s_points should be specified by args for ATT+CRD experiments.
            if opt.s_points:
                s_points = opt.s_points
            else:

                if opt.model_s == 'resnet8' or opt.model_s == 'resnet8x4':
                    s_points = '1,2,3'
                elif opt.model_s == 'resnet20':
                    s_points = '3,6,9'
                elif opt.model_s == 'resnet32':
                    s_points = '5,10,15'
                elif opt.model_s == 'resnet110':
                    s_points = '18,36,54'
                elif opt.model_s == 'ShuffleV1':
                    s_points = '4,12,16'
                elif opt.model_s == 'wrn_16_2':
                    s_points = '2,4,6'
                elif opt.model_s in ['vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19']:
                    s_points = opt.hint_points
                else:
                    raise NotImplementedError

            feat_s = [feat_s[int(i)] for i in s_points.split(',')]

            if teacher_dict is not None:
                feat_t = []
                # Fetch precomputed teacher outputs for this batch
                batch_feat_t = []
                batch_logit_t = []
                for idx_val in index:
                    f_t, l_t = teacher_dict[idx_val.item()]
                    batch_feat_t.append(f_t)
                    batch_logit_t.append(l_t)
                
                logit_t = torch.stack(batch_logit_t).cuda()
                # Unpack list of features
                num_hint_points = len(batch_feat_t[0])
                for pt in range(num_hint_points):
                    feat_t_pt = torch.stack([batch_feat_t[b][pt] for b in range(len(batch_feat_t))]).cuda()
                    feat_t.append(feat_t_pt)
            else:
                with torch.no_grad():
                    feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
                    feat_t = [feat_t[int(i)] for i in opt.hint_points.split(',')]
            
                    feat_t = [f.detach() for f in feat_t]

            # cls + kl div
            loss_cls = criterion_cls(logit_s, target)

            if opt.distill == 'WSL_att' or opt.distill == 'WSL_crd':
                loss_div = criterion_div(logit_s, logit_t, target)
            else:
                loss_div = criterion_div(logit_s, logit_t)

            # other kd beyond KL divergence
            if opt.distill == 'kd':
                loss_kd = torch.tensor(0.0).cuda()
            elif opt.distill == 'hint':
                f_s = [module_list[1+i](feat_s[i]) for i in range(len(feat_s))]
                f_t = feat_t
                loss_kd = criterion_kd(f_s, f_t)
            elif opt.distill == 'crd':
                f_s = feat_s[-1]
                f_t = feat_t[-1]
                loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
            elif opt.distill == 'attention':
                g_s = feat_s
                g_t = feat_t
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)
            elif opt.distill == 'ATT_crd':
                #first loss for ATT (criterion_kd), second for crd (criterion_kd1)
                # 4+1(CRD) points should be chosen for s_points for ATT+CRD experiments:

                f_s = feat_s[-1]
                f_t = feat_t[-1]
                loss_kd_crd = criterion_kd1(f_s, f_t, index, contrast_idx) # for CRD

                g_s = feat_s[:-1] #last HP for CRD
                g_t = feat_t[:-1]
                loss_group = criterion_kd(g_s, g_t) # for ATT
                loss_kd_att = sum(loss_group)


                loss_kd = loss_kd_crd + loss_kd_att * opt.beta2

            elif opt.distill == 'WSL_att':
                g_s = feat_s
                g_t = feat_t
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)       
            elif opt.distill == 'WSL_crd':
                f_s = feat_s[-1]
                f_t = feat_t[-1]
                loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)

                    
            elif opt.distill == 'vid':
                g_s = feat_s
                g_t = feat_t
                loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
                loss_kd = sum(loss_group)
            elif opt.distill == 'pursuhint_cmtf':
                f_s = feat_s
                f_t = feat_t
                loss_kd = criterion_kd(f_s, f_t)
            else:
                raise NotImplementedError(opt.distill)

            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd


        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        losses_cls.update(loss_cls.item(), input.size(0))
        losses_div.update(loss_div.item(), input.size(0))
        losses_kd.update(loss_kd.item() if hasattr(loss_kd, 'item') else loss_kd, input.size(0))
        
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg, losses_cls.avg, losses_div.avg, losses_kd.avg



def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

