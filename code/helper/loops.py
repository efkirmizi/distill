from __future__ import print_function, division

import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import math
from .util import AverageMeter, accuracy



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

    return top1.avg, top5.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt,
                  scaler=None, teacher_cache=None,
                  module_list_2=None, criterion_list_2=None, optimizer_2=None, scaler_2=None,
                  loss_weighter=None, loss_weighter_2=None):
    """One epoch distillation. Pass module_list_2 and friends for per-batch dual-student training."""
    dual = module_list_2 is not None

    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if dual:
        for module in module_list_2:
            module.train()
        module_list_2[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]
    if opt.distill == 'ATT_crd':
        criterion_kd1 = criterion_list[3]

    if dual:
        criterion_cls_2 = criterion_list_2[0]
        criterion_div_2 = criterion_list_2[1]
        criterion_kd_2 = criterion_list_2[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    if dual:
        model_s2 = module_list_2[0]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_cls = AverageMeter()
    losses_div = AverageMeter()
    losses_kd = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if dual:
        losses_2 = AverageMeter()
        losses_cls_2 = AverageMeter()
        losses_div_2 = AverageMeter()
        losses_kd_2 = AverageMeter()
        top1_2 = AverageMeter()
        top5_2 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd'] or opt.distill == 'WSL_crd'  or opt.distill == 'ATT_crd':
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        index_cpu = index  # keep CPU reference before CUDA move
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd'] or opt.distill == 'WSL_crd'  or opt.distill == 'ATT_crd':
                contrast_idx = contrast_idx.cuda()

        proj_cp = None  # CP student's batch projectors, passed to Tucker for coupling

        # ===================forward CP student=====================
        preact = getattr(opt, 'preact', False)
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            feat_s, logit_s = model_s(input, is_feat=True, preact=preact)

            # s_points is always pre-populated by train_student.py before the loop starts
            feat_s = [feat_s[int(i)] for i in opt.s_points.split(',')]

            if teacher_cache is not None:
                # Fast vectorized lookup from pre-stacked tensors
                teacher_logits, teacher_feats = teacher_cache
                logit_t = teacher_logits[index_cpu].cuda(non_blocking=True)
                feat_t = [f[index_cpu].cuda(non_blocking=True) for f in teacher_feats]
            else:
                with torch.no_grad():
                    feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
                    feat_t = [feat_t[int(i)] for i in opt.hint_points.split(',')]
                    feat_t = [f.detach().float() for f in feat_t]
                    logit_t = logit_t.detach().float()

            # cls + kl div
            loss_cls = criterion_cls(logit_s, target)

            if opt.distill == 'WSL_att' or opt.distill == 'WSL_crd':
                loss_div = criterion_div(logit_s, logit_t, target)
            else:
                loss_div = criterion_div(logit_s, logit_t)

            # other kd beyond KL divergence
            if opt.distill == 'kd':
                loss_kd = torch.tensor(0.0, device=input.device)
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
                f_s = feat_s[-1]
                f_t = feat_t[-1]
                loss_kd_crd = criterion_kd1(f_s, f_t, index, contrast_idx)

                g_s = feat_s[:-1]
                g_t = feat_t[:-1]
                loss_group = criterion_kd(g_s, g_t)
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
            elif opt.distill == 'pursuhint_bsat':
                f_s = feat_s
                f_t = feat_t
                loss_kd, proj_cp = criterion_kd(
                    [f.float() for f in f_s],
                    [f.float() for f in f_t]
                )
                if dual:
                    feat_s2, logit_s2 = model_s2(input, is_feat=True, preact=preact)
                    feat_s2 = [feat_s2[int(i)] for i in opt.s_points.split(',')]
                    loss_cls_2 = criterion_cls_2(logit_s2, target)
                    loss_div_2 = criterion_div_2(logit_s2, logit_t)
                    loss_kd_2, proj_tucker = criterion_kd_2(
                        [f.float() for f in feat_s2],
                        [f.float() for f in feat_t]
                    )
                    cw = getattr(criterion_kd_2, 'coupling_weight', 1.0)
                    for P_cp_i, P_tuck_i in zip(proj_cp, proj_tucker):
                        loss_kd   = loss_kd   + cw * F.mse_loss(P_cp_i, P_tuck_i.detach())
                        loss_kd_2 = loss_kd_2 + cw * F.mse_loss(P_tuck_i, P_cp_i.detach())
            else:
                raise NotImplementedError(opt.distill)

            if loss_weighter is not None and opt.distill != 'kd':
                loss = loss_weighter(loss_cls, loss_div, loss_kd)
            else:
                loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        losses_cls.update(loss_cls.item(), input.size(0))
        losses_div.update(loss_div.item(), input.size(0))
        losses_kd.update(loss_kd.item() if hasattr(loss_kd, 'item') else loss_kd, input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward CP=====================
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # ===================forward+backward Tucker student (dual mode)=====================
        if dual:
            if opt.distill != 'pursuhint_bsat':
                with torch.amp.autocast('cuda', enabled=scaler_2 is not None):
                    feat_s2, logit_s2 = model_s2(input, is_feat=True, preact=preact)
                    feat_s2 = [feat_s2[int(i)] for i in opt.s_points.split(',')]

                    loss_cls_2 = criterion_cls_2(logit_s2, target)
                    loss_div_2 = criterion_div_2(logit_s2, logit_t, target) if opt.distill == 'WSL_att' else criterion_div_2(logit_s2, logit_t)

                    if opt.distill == 'kd':
                        loss_kd_2 = torch.tensor(0.0, device=input.device)
                    elif opt.distill == 'hint':
                        f_s2 = [module_list_2[1+i](feat_s2[i]) for i in range(len(feat_s2))]
                        loss_kd_2 = criterion_kd_2(f_s2, feat_t)
                    elif opt.distill in ['attention', 'WSL_att']:
                        loss_kd_2 = sum(criterion_kd_2(feat_s2, feat_t))
                    elif opt.distill == 'vid':
                        loss_kd_2 = sum([c(f_s, f_t) for f_s, f_t, c in zip(feat_s2, feat_t, criterion_kd_2)])

                    if loss_weighter_2 is not None and opt.distill != 'kd':
                        loss_2 = loss_weighter_2(loss_cls_2, loss_div_2, loss_kd_2)
                    else:
                        loss_2 = opt.gamma * loss_cls_2 + opt.alpha * loss_div_2 + opt.beta * loss_kd_2
            else:
                # pursuhint_bsat: Tucker forward + bidirectional coupling already done inside the
                # CP autocast block above; loss_cls_2, loss_div_2, loss_kd_2 are already set.
                if loss_weighter_2 is not None:
                    loss_2 = loss_weighter_2(loss_cls_2, loss_div_2, loss_kd_2)
                else:
                    loss_2 = opt.gamma * loss_cls_2 + opt.alpha * loss_div_2 + opt.beta * loss_kd_2

            acc1_2, acc5_2 = accuracy(logit_s2, target, topk=(1, 5))
            losses_2.update(loss_2.item(), input.size(0))
            losses_cls_2.update(loss_cls_2.item(), input.size(0))
            losses_div_2.update(loss_div_2.item(), input.size(0))
            losses_kd_2.update(loss_kd_2.item() if hasattr(loss_kd_2, 'item') else loss_kd_2, input.size(0))
            top1_2.update(acc1_2[0], input.size(0))
            top5_2.update(acc5_2[0], input.size(0))

            optimizer_2.zero_grad()
            if scaler_2 is not None:
                scaler_2.scale(loss_2).backward()
                scaler_2.unscale_(optimizer_2)
                for group in optimizer_2.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                scaler_2.step(optimizer_2)
                scaler_2.update()
            else:
                loss_2.backward()
                optimizer_2.step()

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

    if dual:
        return (top1.avg, top5.avg, losses.avg, losses_cls.avg, losses_div.avg, losses_kd.avg,
                top1_2.avg, top5_2.avg, losses_2.avg, losses_cls_2.avg, losses_div_2.avg, losses_kd_2.avg)
    return top1.avg, top5.avg, losses.avg, losses_cls.avg, losses_div.avg, losses_kd.avg


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
