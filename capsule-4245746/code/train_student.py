"""
the general training framework for CIFAR experiments
"""

from __future__ import print_function

import os
import argparse
import time
import csv
import json
import copy

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import tensorboard_logger as tb_logger

import math
import numpy as np
import random
import psutil

from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.cifar10 import get_cifar10_dataloaders, get_cifar10_dataloaders_sample
from dataset.imagenet import get_imagenet_dataloader

from distiller_zoo.FitNet import HintLoss
from distiller_zoo.KD import DistillKLD
from distiller_zoo.AT import Attention
from distiller_zoo.VID import VIDLoss
from distiller_zoo.WSL import WSLLoss
from distiller_zoo.CMTF import CoupledTensorLoss

from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init
from helper.util import adjust_learning_rate_with_warmup
from helper.uncertainty_weighter import DynamicLossWeighter

from for_init import remove_module, add_module

from decomposition import decompose_model

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10', 'imagenet'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    parser.add_argument('--path_s', type=str, default=None, help='student model snapshot')
    parser.add_argument('--model_t', type=str, default=None, help='teacher model')
    

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention',
                                                                      'vid', 'crd',
                                                                      'WSL_att', 'WSL_crd','ATT_crd', 'pursuhint_cmtf'])
    parser.add_argument('--dual_cmtf', action='store_true', help='Instantiate and train both a CP and Tucker composed student simultaneously on the same Teacher cache')
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--cp_rank_ratio', type=float, default=0.5, help='compression ratio for CP')
    parser.add_argument('--tucker_rank_ratio', type=float, default=0.5, help='compression ratio for Tucker')
    parser.add_argument('--use_vbmf', action='store_true',
                        help='auto-select decomposition rank per layer via EVBMF instead of global ratio')
    parser.add_argument('--cmtf_rank', type=int, default=8,
                        help='SVD rank R for batch-subspace alignment in CMTF loss')
    parser.add_argument('--cmtf_coupling_weight', type=float, default=1.0,
                        help='weight for the Tucker←CP coupling term in dual CMTF mode')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')
    parser.add_argument('--beta2', type=float, default=1.0, help='weight balance between CRD and ATT')

    # hint point:
    parser.add_argument('--hint_points', type=str, default='15,37,53')

    #s_points are the last layers of blocks, unless it is specified:
    #NOTE: s_points should be specified by args for ATT+CRD experiments.
    parser.add_argument('--s_points', type=str, default=None)

    #for preAct:
    parser.add_argument('--preact', type=bool, default=False)

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # PyTorch model compile optimization
    parser.add_argument('--torch_compile', action='store_true')

    # Dynamic loss weighting (Kendall et al. CVPR 2018)
    parser.add_argument('--dynamic_loss_weights', action='store_true',
                        help='Learn per-task loss weights via homoscedastic uncertainty '
                             '(Kendall et al. CVPR 2018). Replaces fixed gamma/alpha/beta.')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    opt.model_path = './save/student_model'


    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.dual_cmtf and opt.distill != 'pursuhint_cmtf':
        raise ValueError("--dual_cmtf requires --distill pursuhint_cmtf")

    if not opt.model_t:
        opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S-{}_T-{}_{}_{}_r-{}_a-{}_b-{}_b2-{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.beta2, opt.trial)


    if opt.preact:
        opt.save_folder = os.path.join(opt.model_path, opt.dataset,'preact_'+str(opt.hint_points), opt.model_name)
    else:
        opt.save_folder = os.path.join(opt.model_path, opt.dataset, str(opt.hint_points), opt.model_name)

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.tb_folder = os.path.join('./save/tensorboard', opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls, model_t):
    print('==> loading teacher model')

    model = model_dict[model_t](num_classes=n_cls)    
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False)['model'])
    except Exception:
        model.load_state_dict(remove_module(torch.load(model_path, map_location='cpu', weights_only=False)['model']))

    print('==> done')
    return model


def main():
    best_acc = 0
    best_acc_2 = 0

    opt = parse_option()


    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd'] or opt.distill == 'WSL_crd' or opt.distill == 'ATT_crd':
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        n_cls = 100
        opt.nce_t = 0.1

    elif opt.dataset == 'cifar10':
        if opt.distill in ['crd'] or opt.distill == 'WSL_crd' or opt.distill == 'ATT_crd':
            train_loader, val_loader, n_data = get_cifar10_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_cifar10_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        n_cls = 10
        opt.nce_t = 0.1

    elif opt.dataset == 'imagenet':
        if opt.distill in ['crd'] or opt.distill == 'WSL_crd' or opt.distill == 'ATT_crd':

            raise NotImplementedError

        else:
            train_loader, val_loader, n_data = get_imagenet_dataloader(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        n_cls = 1000

    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = load_teacher(opt.path_t, n_cls, opt.model_t)
    model_t.eval()

    model_s = model_dict[opt.model_s](num_classes=n_cls)
    if opt.dual_cmtf:
        model_s2 = copy.deepcopy(model_s)

    if opt.distill == 'pursuhint_cmtf':
        _teacher_for_vbmf = model_t if opt.use_vbmf else None
        rank_desc = "teacher VBMF" if opt.use_vbmf else f"ratio={opt.cp_rank_ratio}"
        print(f"==> Decomposing student model with CP factorization ({rank_desc})...")
        model_s = decompose_model(model_s, method='cp', cp_rank_ratio=opt.cp_rank_ratio,
                                  use_vbmf=opt.use_vbmf, teacher_model=_teacher_for_vbmf)
        if opt.dual_cmtf:
            t_rank_desc = "teacher VBMF" if opt.use_vbmf else f"ratio={opt.tucker_rank_ratio}"
            print(f"==> Decomposing parallel student model with Tucker ({t_rank_desc})...")
            model_s2 = decompose_model(model_s2, method='tucker',
                                       tucker_rank_ratio=opt.tucker_rank_ratio,
                                       use_vbmf=opt.use_vbmf, teacher_model=_teacher_for_vbmf)
    
    # ---- Wrap in DataParallel FIRST ----
    model_s = nn.DataParallel(model_s).cuda()
    model_t = nn.DataParallel(model_t).cuda()
    if opt.dual_cmtf:
        model_s2 = nn.DataParallel(model_s2).cuda()

    # ---- Compile AFTER DataParallel ----
    # torch.compile + nn.DataParallel is broken on multi-GPU: DataParallel.replicate()
    # creates shallow copies of the OptimizedModule that lose the module tree, causing
    # AttributeError on any attribute access during the compiled forward. Single-GPU
    # DataParallel uses a fast path (no replicate) so compile works there.
    if opt.torch_compile:
        if torch.cuda.device_count() > 1:
            print("Warning: --torch_compile is incompatible with nn.DataParallel on "
                  f"{torch.cuda.device_count()} GPUs. Compilation skipped. "
                  "Use DistributedDataParallel (torchrun) for multi-GPU + compile.")
        else:
            model_s = torch.compile(model_s, dynamic=True)
            model_t = torch.compile(model_t, dynamic=True)
            if opt.dual_cmtf:
                model_s2 = torch.compile(model_s2, dynamic=True)
    
    if opt.path_s:
        try:
            model_s.load_state_dict(add_module(torch.load(opt.path_s, map_location='cpu', weights_only=False)['model']))
        except Exception:
            model_s.load_state_dict((torch.load(opt.path_s, map_location='cpu', weights_only=False)['model']))

    # Determine s_points (student hint positions) BEFORE probing, so we only extract what we need
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
        elif opt.model_s == 'ShuffleV2':
            s_points = '3,10,13'
        elif opt.model_s == 'wrn_16_2':
            s_points = '2,4,6'
        elif opt.model_s in ['vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19']:
            s_points = opt.hint_points
        else:
            raise NotImplementedError
    # Persist so loops.py doesn't recompute every batch
    opt.s_points = s_points

    # Probe feature shapes (needed for ConvReg, VID, etc.)
    if opt.dataset in ['cifar10', 'cifar100']:
        data = torch.randn(2, 3, 32, 32).cuda()
    elif opt.dataset == 'imagenet':
        data = torch.randn(2, 3, 224, 224).cuda()

    model_t.eval()
    model_s.eval()
    with torch.no_grad():
        feat_t, _ = model_t(data, is_feat=True)
        feat_s, _ = model_s(data, is_feat=True)

    feat_t = [feat_t[int(i)] for i in opt.hint_points.split(',')]
    feat_s = [feat_s[int(i)] for i in s_points.split(',')]

    if opt.dual_cmtf:
        model_s2.eval()
        with torch.no_grad():
            feat_s2, _ = model_s2(data, is_feat=True)
        feat_s2 = [feat_s2[int(i)] for i in s_points.split(',')]


    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)
    
    if opt.dual_cmtf:
        module_list_2 = nn.ModuleList([model_s2])
        trainable_list_2 = nn.ModuleList([model_s2])

    criterion_cls = nn.CrossEntropyLoss()

    
    if opt.distill == 'WSL_att' or opt.distill == 'WSL_crd':
        criterion_div = WSLLoss(opt.kd_T)
    else:
        criterion_div = DistillKLD(opt.kd_T)


    if opt.distill == 'kd':
        criterion_kd = DistillKLD(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()

        s_shapes = [f.shape for f in feat_s]
        t_shapes = [f.shape for f in feat_t]

        for i in range(len(t_shapes)):
            regress_s = ConvReg(s_shapes[i], t_shapes[i])
            module_list.append(regress_s)
            trainable_list.append(regress_s)


    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'ATT_crd':
        # 4+1(CRD) points should be chosen for s_points for ATT+CRD experiments:
        opt.s_dim = feat_s[-1].shape[1] 
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = Attention()
        criterion_kd1 = CRDLoss(opt)

        module_list.append(criterion_kd1.embed_s)
        module_list.append(criterion_kd1.embed_t)
        trainable_list.append(criterion_kd1.embed_s)
        trainable_list.append(criterion_kd1.embed_t)


    elif opt.distill == 'attention':
        criterion_kd = Attention()

    elif opt.distill == 'WSL_att':
        criterion_kd = Attention()
    elif opt.distill == 'WSL_crd':

        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)


    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s]
        t_n = [f.shape[1] for f in feat_t]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'pursuhint_cmtf':
        criterion_kd = CoupledTensorLoss(rank=opt.cmtf_rank,
                                         coupling_weight=opt.cmtf_coupling_weight)
        if opt.dual_cmtf:
            criterion_kd_2 = CoupledTensorLoss(rank=opt.cmtf_rank,
                                               coupling_weight=opt.cmtf_coupling_weight)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss
    criterion_list.append(criterion_kd)     # CMTF loss for CP model
    if opt.dual_cmtf:
        criterion_list_2 = nn.ModuleList([])
        criterion_list_2.append(criterion_cls)
        criterion_list_2.append(criterion_div)
        criterion_list_2.append(criterion_kd_2) # CMTF loss for Tucker model

    if opt.distill == 'ATT_crd':
        criterion_list.append(criterion_kd1) #In this case, criterion_list consists of 4 components.

    # Dynamic loss weighters (Kendall et al. CVPR 2018)
    # log_vars use a separate param group with weight_decay=0 to avoid biasing toward equal weights.
    loss_weighter = None
    loss_weighter_2 = None
    if opt.dynamic_loss_weights:
        if opt.gamma != 1.0 or opt.alpha != 1.0 or opt.beta != 1.0:
            print(f"WARNING: --dynamic_loss_weights is ON; --gamma/--alpha/--beta ({opt.gamma}/{opt.alpha}/{opt.beta}) are ignored.")
        loss_weighter = DynamicLossWeighter(num_losses=3)
        if opt.dual_cmtf:
            loss_weighter_2 = DynamicLossWeighter(num_losses=3)

    # optimizer
    if opt.dynamic_loss_weights:
        optimizer = optim.SGD([
            {'params': trainable_list.parameters(), 'weight_decay': opt.weight_decay},
            {'params': loss_weighter.parameters(), 'weight_decay': 0.0},
        ], lr=opt.learning_rate, momentum=opt.momentum)
        if opt.dual_cmtf:
            optimizer_2 = optim.SGD([
                {'params': trainable_list_2.parameters(), 'weight_decay': opt.weight_decay},
                {'params': loss_weighter_2.parameters(), 'weight_decay': 0.0},
            ], lr=opt.learning_rate, momentum=opt.momentum)
    else:
        optimizer = optim.SGD(trainable_list.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
        if opt.dual_cmtf:
            optimizer_2 = optim.SGD(trainable_list_2.parameters(),
                                lr=opt.learning_rate,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)
    if opt.dual_cmtf:
        module_list_2.append(model_t)

    if torch.cuda.is_available():
        module_list = module_list.cuda()
        criterion_list.cuda()
        if opt.dual_cmtf:
            module_list_2 = module_list_2.cuda()
            criterion_list_2.cuda()
        if loss_weighter is not None:
            loss_weighter = loss_weighter.cuda()
        if loss_weighter_2 is not None:
            loss_weighter_2 = loss_weighter_2.cuda()
        cudnn.benchmark = False
        cudnn.deterministic = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    # validate student accuracy before training/with initial weights:
    student_acc, _, _ = validate(val_loader, model_s, criterion_cls, opt)
    print('student CP accuracy: ', student_acc)
    
    if opt.dual_cmtf:
        student2_acc, _, _ = validate(val_loader, model_s2, criterion_cls, opt)
        print('student Tucker accuracy: ', student2_acc)

    print("==> Precomputing teacher outputs to save compute...")

    precompute_loader = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=opt.batch_size,
        shuffle=False, num_workers=opt.num_workers, drop_last=False
    )

    n_data = len(train_loader.dataset)
    hint_indices = [int(i) for i in opt.hint_points.split(',')]

    # Estimate CPU RAM needed before committing: probe teacher with one sample
    with torch.no_grad():
        if opt.dataset in ['cifar10', 'cifar100']:
            _probe = torch.randn(1, 3, 32, 32).cuda()
        else:
            _probe = torch.randn(1, 3, 224, 224).cuda()
        _feats, _ = model_t(_probe, is_feat=True, preact=opt.preact)
        _feat_bytes = sum(n_data * _feats[hi].numel() * 4 for hi in hint_indices)
        del _probe, _feats

    _total_cache_gb = (_feat_bytes + n_data * n_cls * 4) / 1024 ** 3
    _available_gb = psutil.virtual_memory().available / 1024 ** 3
    _CACHE_LIMIT_GB = _available_gb * 0.75

    if _total_cache_gb > _CACHE_LIMIT_GB:
        print(f"==> Cache would need {_total_cache_gb:.1f} GB but only "
              f"{_available_gb:.1f} GB RAM available; "
              f"teacher features computed on-the-fly (no RAM overhead).")
        teacher_cache = None
    else:
        teacher_logits = torch.zeros(n_data, n_cls)
        teacher_feats = None

        model_t.eval()
        with torch.no_grad():
            for idx, data in enumerate(precompute_loader):
                if opt.distill in ['crd', 'WSL_crd', 'ATT_crd']:
                    input_t, _, index_t, _ = data
                else:
                    input_t, _, index_t = data
                input_t = input_t.float().cuda()
                with torch.amp.autocast('cuda', enabled=True):
                    feat_t_batch, logit_t_batch = model_t(input_t, is_feat=True, preact=opt.preact)
                    feat_t_selected = [feat_t_batch[hi].detach().cpu() for hi in hint_indices]
                    logit_t_batch = logit_t_batch.detach().cpu()

                if teacher_feats is None:
                    teacher_feats = [torch.zeros(n_data, *f.shape[1:]) for f in feat_t_selected]

                teacher_logits[index_t] = logit_t_batch.float()
                for hp_idx, f in enumerate(feat_t_selected):
                    teacher_feats[hp_idx][index_t] = f.float()

        teacher_cache = (teacher_logits, teacher_feats)
        print(f'==> Teacher cache ready: logits {teacher_logits.shape}, '
              f'{len(teacher_feats)} feat tensors')

    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    scaler = torch.amp.GradScaler('cuda')
    if opt.dual_cmtf:
        scaler_2 = torch.amp.GradScaler('cuda')
    # routine

    print('the number of teacher model parameters: {}'.format(sum([p.data.nelement() for p in model_t.parameters()])))
    print('the number of student model parameters: {}'.format(sum([p.data.nelement() for p in model_s.parameters()])))

    # --- CSV Logger Setup ---
    csv_path_cp = os.path.join(opt.save_folder, 'training_log_cp.csv')
    csv_file_cp = open(csv_path_cp, 'w', newline='')
    csv_writer_cp = csv.writer(csv_file_cp)
    csv_writer_cp.writerow(['epoch', 'lr', 'epoch_time',
                            'train_acc', 'train_acc_top5', 'train_loss',
                            'train_loss_cls', 'train_loss_div', 'train_loss_kd',
                            'test_acc', 'test_acc_top5', 'test_loss', 'best_acc',
                            'w_cls', 'w_div', 'w_kd'])
    print(f'CP CSV log: {csv_path_cp}')

    if opt.dual_cmtf:
        csv_path_tk = os.path.join(opt.save_folder, 'training_log_tucker.csv')
        csv_file_tk = open(csv_path_tk, 'w', newline='')
        csv_writer_tk = csv.writer(csv_file_tk)
        csv_writer_tk.writerow(['epoch', 'lr', 'epoch_time',
                            'train_acc', 'train_acc_top5', 'train_loss',
                            'train_loss_cls', 'train_loss_div', 'train_loss_kd',
                            'test_acc', 'test_acc_top5', 'test_loss', 'best_acc',
                            'w_cls', 'w_div', 'w_kd'])
        print(f'Tucker CSV log: {csv_path_tk}')

    def _f(v):
        return v.item() if hasattr(v, 'item') else float(v)

    for epoch in range(opt.start_epoch, opt.epochs + 1):

        adjust_learning_rate_with_warmup(epoch, opt, optimizer, warmup_epochs=5)
        if opt.dual_cmtf:
            adjust_learning_rate_with_warmup(epoch, opt, optimizer_2, warmup_epochs=5)

        print("==> training...")
        print('hint positions: ', opt.hint_points)

        time1 = time.time()
        if opt.dual_cmtf:
            (train_acc, train_acc_top5, train_loss, train_loss_cls, train_loss_div, train_loss_kd,
             train_acc_2, train_acc_top5_2, train_loss_2, train_loss_cls_2, train_loss_div_2, train_loss_kd_2) = train(
                epoch, train_loader, module_list, criterion_list, optimizer, opt,
                scaler=scaler, teacher_cache=teacher_cache,
                module_list_2=module_list_2, criterion_list_2=criterion_list_2,
                optimizer_2=optimizer_2, scaler_2=scaler_2,
                loss_weighter=loss_weighter, loss_weighter_2=loss_weighter_2)
        else:
            train_acc, train_acc_top5, train_loss, train_loss_cls, train_loss_div, train_loss_kd = train(
                epoch, train_loader, module_list, criterion_list, optimizer, opt,
                scaler=scaler, teacher_cache=teacher_cache,
                loss_weighter=loss_weighter)
        time2 = time.time()
        epoch_time = time2 - time1

        print('Epoch {} | Total Time {:.2f}'.format(epoch, epoch_time))

        print('train_acc CP:', train_acc)
        print('train_loss CP:', train_loss)
        if opt.dual_cmtf:
            print('train_acc Tucker:', train_acc_2)
            print('train_loss Tucker:', train_loss_2)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)
        if opt.dual_cmtf:
            test_acc_2, test_acc_top5_2, test_loss_2 = validate(val_loader, model_s2, criterion_cls, opt)

        logger.log_value('train_acc_cp', train_acc, epoch)
        logger.log_value('train_acc_top5_cp', train_acc_top5, epoch)
        logger.log_value('train_loss_cp', train_loss, epoch)
        logger.log_value('train_loss_cls_cp', train_loss_cls, epoch)
        logger.log_value('train_loss_div_cp', train_loss_div, epoch)
        logger.log_value('train_loss_kd_cp', train_loss_kd, epoch)
        logger.log_value('test_acc_cp', test_acc, epoch)
        logger.log_value('test_acc_top5_cp', test_acc_top5, epoch)
        logger.log_value('test_loss_cp', test_loss, epoch)
        
        if opt.dual_cmtf:
            logger.log_value('train_acc_tucker', train_acc_2, epoch)
            logger.log_value('train_acc_top5_tucker', train_acc_top5_2, epoch)
            logger.log_value('train_loss_tucker', train_loss_2, epoch)
            logger.log_value('train_loss_cls_tucker', train_loss_cls_2, epoch)
            logger.log_value('train_loss_div_tucker', train_loss_div_2, epoch)
            logger.log_value('train_loss_kd_tucker', train_loss_kd_2, epoch)
            logger.log_value('test_acc_tucker', test_acc_2, epoch)
            logger.log_value('test_acc_top5_tucker', test_acc_top5_2, epoch)
            logger.log_value('test_loss_tucker', test_loss_2, epoch)

        print('test_acc CP:', test_acc)
        print('test_loss CP:', test_loss)
        print('test_acc_top5 CP:', test_acc_top5)

        if loss_weighter is not None:
            w = loss_weighter.effective_weights()
            print(f'  [DynW CP]  cls={w[0]:.4f}  div={w[1]:.4f}  kd={w[2]:.4f}')
            logger.log_value('dyn_w_cls_cp', w[0], epoch)
            logger.log_value('dyn_w_div_cp', w[1], epoch)
            logger.log_value('dyn_w_kd_cp',  w[2], epoch)

        if opt.dual_cmtf:
            print('test_acc Tucker:', test_acc_2)
            print('test_loss Tucker:', test_loss_2)
            print('test_acc_top5 Tucker:', test_acc_top5_2)
            if loss_weighter_2 is not None:
                w2 = loss_weighter_2.effective_weights()
                print(f'  [DynW Tucker]  cls={w2[0]:.4f}  div={w2[1]:.4f}  kd={w2[2]:.4f}')
                logger.log_value('dyn_w_cls_tucker', w2[0], epoch)
                logger.log_value('dyn_w_div_tucker', w2[1], epoch)
                logger.log_value('dyn_w_kd_tucker',  w2[2], epoch)

        current_lr = optimizer.param_groups[0]['lr']

        # save the best CP model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best_cp.pth'.format(opt.model_s))
            print('saving the best CP model!')
            torch.save(state, save_file)

        # save the best Tucker model
        if opt.dual_cmtf:
            if test_acc_2 > best_acc_2:
                best_acc_2 = test_acc_2
                state2 = {
                    'epoch': epoch,
                    'model': model_s2.state_dict(),
                    'best_acc': best_acc_2,
                }
                save_file2 = os.path.join(opt.save_folder, '{}_best_tucker.pth'.format(opt.model_s))
                print('saving the best Tucker model!')
                torch.save(state2, save_file2)

        # --- CSV: write epoch rows (after best_acc update so the column is accurate) ---
        if loss_weighter is not None:
            ew = loss_weighter.effective_weights()
        else:
            ew = [
                opt.gamma if opt.gamma is not None else 1.0,
                opt.alpha if opt.alpha is not None else 0.0,
                opt.beta  if opt.beta  is not None else 0.0,
            ]
        csv_writer_cp.writerow([epoch, f'{current_lr:.6f}', f'{epoch_time:.2f}',
                                f'{_f(train_acc):.4f}', f'{_f(train_acc_top5):.4f}',
                                f'{_f(train_loss):.4f}', f'{_f(train_loss_cls):.4f}',
                                f'{_f(train_loss_div):.4f}', f'{_f(train_loss_kd):.4f}',
                                f'{_f(test_acc):.4f}', f'{_f(test_acc_top5):.4f}',
                                f'{_f(test_loss):.4f}', f'{_f(best_acc):.4f}',
                                f'{ew[0]:.6f}', f'{ew[1]:.6f}', f'{ew[2]:.6f}'])
        csv_file_cp.flush()

        if opt.dual_cmtf:
            if loss_weighter_2 is not None:
                ew2 = loss_weighter_2.effective_weights()
            else:
                ew2 = [
                    opt.gamma if opt.gamma is not None else 1.0,
                    opt.alpha if opt.alpha is not None else 0.0,
                    opt.beta  if opt.beta  is not None else 0.0,
                ]
            csv_writer_tk.writerow([epoch, f'{current_lr:.6f}', f'{epoch_time:.2f}',
                                   f'{_f(train_acc_2):.4f}', f'{_f(train_acc_top5_2):.4f}',
                                   f'{_f(train_loss_2):.4f}', f'{_f(train_loss_cls_2):.4f}',
                                   f'{_f(train_loss_div_2):.4f}', f'{_f(train_loss_kd_2):.4f}',
                                   f'{_f(test_acc_2):.4f}', f'{_f(test_acc_top5_2):.4f}',
                                   f'{_f(test_loss_2):.4f}', f'{_f(best_acc_2):.4f}',
                                   f'{ew2[0]:.6f}', f'{ew2[1]:.6f}', f'{ew2[2]:.6f}'])
            csv_file_tk.flush()

    # --- Close CSVs ---
    csv_file_cp.close()
    print(f'CP training log saved to: {csv_path_cp}')
    if opt.dual_cmtf:
        csv_file_tk.close()
        print(f'Tucker training log saved to: {csv_path_tk}')

    # The results compared with results in CRD study are from the last epoch. 
    print('best CP accuracy:', best_acc)
    if opt.dual_cmtf:
        print('best Tucker accuracy:', best_acc_2)

    # --- Save experiment summary JSON ---
    summary = {
        'teacher': opt.model_t,
        'student': opt.model_s,
        'dataset': opt.dataset,
        'distill': opt.distill,
        'epochs': opt.epochs,
        'learning_rate': opt.learning_rate,
        'lr_decay_epochs': opt.lr_decay_epochs,
        'weight_decay': opt.weight_decay,
        'batch_size': opt.batch_size,
        'hint_points': opt.hint_points,
        'gamma': opt.gamma,
        'alpha': opt.alpha,
        'beta': opt.beta,
        'cp_rank_ratio': opt.cp_rank_ratio,
        'use_vbmf': opt.use_vbmf,
        'best_acc_cp': round(_f(best_acc), 4),
    }
    if opt.dynamic_loss_weights and loss_weighter is not None:
        summary['dynamic_loss_weights'] = True
        summary['final_weights_cp'] = {
            'cls': round(loss_weighter.effective_weights()[0], 6),
            'div': round(loss_weighter.effective_weights()[1], 6),
            'kd':  round(loss_weighter.effective_weights()[2], 6),
        }
    if opt.dual_cmtf:
        summary['tucker_rank_ratio'] = opt.tucker_rank_ratio
        summary['best_acc_tucker'] = round(_f(best_acc_2), 4)
        if opt.dynamic_loss_weights and loss_weighter_2 is not None:
            summary['final_weights_tucker'] = {
                'cls': round(loss_weighter_2.effective_weights()[0], 6),
                'div': round(loss_weighter_2.effective_weights()[1], 6),
                'kd':  round(loss_weighter_2.effective_weights()[2], 6),
            }
    summary_path = os.path.join(opt.save_folder, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f'Experiment summary saved to: {summary_path}')

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last_cp.pth'.format(opt.model_s))
    torch.save(state, save_file)
    
    if opt.dual_cmtf:
        state2 = {
            'opt': opt,
            'model': model_s2.state_dict(),
        }
        save_file2 = os.path.join(opt.save_folder, '{}_last_tucker.pth'.format(opt.model_s))
        torch.save(state2, save_file2)


if __name__ == '__main__':
    main()