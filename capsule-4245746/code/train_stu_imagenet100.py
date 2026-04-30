"""
The general training framework for ImageNet distillation experiments.
Supports multiple distillation methods (kd, hint, attention, vid, WSL_att, pursuhint_cmtf),
model decomposition (CP/Tucker), dual CMTF training, DALI/standard DataLoader backends,
CSV/JSON logging, and configurable teacher/student architectures (ResNet18, ResNet34).

CRD-based methods are NOT supported on ImageNet (require per-sample indices not available
from DALI, and ImageNet-scale precomputation is impractical).

Usage examples:
  # Attention distillation with DALI:
  python -m torch.distributed.launch --nproc_per_node=1 train_stu_imagenet.py \
      --distill attention --alpha 0.5 --beta 0.01 --gamma 1 --lr 0.1 \
      --hint_points 2,6,11,15 /path/to/imagenet

  # PURSUhInT CMTF with standard DataLoader (local debugging):
  python train_stu_imagenet.py --distill pursuhint_cmtf --no_dali \
      --hint_points 2,6,11,15 /path/to/imagenet

  # Vanilla student training (no distillation):
  python train_stu_imagenet.py /path/to/imagenet
"""

from __future__ import print_function

import argparse
import os
import shutil
import time
import math
import csv
import json
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

import numpy as np
import sys

from models import model_dict
from models.util import ConvReg
from for_init import remove_module, add_module
from decomposition import decompose_model

from helper.uncertainty_weighter import DynamicLossWeighter
from distiller_zoo.FitNet import HintLoss
from distiller_zoo.KD import DistillKLD
from distiller_zoo.AT import Attention
from distiller_zoo.VID import VIDLoss
from distiller_zoo.WSL import WSLLoss
from distiller_zoo.CMTF import CoupledTensorLoss

# ---------- DALI (optional) ----------
DALI_AVAILABLE = False
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
    DALI_AVAILABLE = True
except ImportError:
    pass

# Student hint points: last layer of each residual block group
# feats = [f0] + f1_act + f2_act + f3_act + f4_act + [f5]
S_POINTS_DICT = {
    'ResNet18': '2,4,6,8',    # [2,2,2,2] BasicBlocks
    'ResNet34': '3,7,13,16',  # [3,4,6,3] BasicBlocks
}


def parse():
    parser = argparse.ArgumentParser(description='ImageNet Knowledge Distillation Training')

    # Data
    parser.add_argument('data', metavar='DIR', nargs='*',
                        help='path(s) to dataset (single dir with train/val subdirs, '
                             'or two paths for train and val)')

    # Model
    parser.add_argument('--model_s', type=str, default='ResNet18',
                        choices=['ResNet18', 'ResNet34'],
                        help='student model architecture')
    parser.add_argument('--model_t', type=str, default='ResNet34',
                        choices=['ResNet18', 'ResNet34'],
                        help='teacher model architecture')
    parser.add_argument('--path_t', type=str,
                        default='save/models/ResNet34_imagenet/ResNet34_333f7ec4.pth',
                        help='path to teacher model checkpoint')

    # Training hyper-parameters
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size per process (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                        help='Initial learning rate. Scaled by global_batch/256.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    # Distillation
    parser.add_argument('--distill', type=str, default=None,
                        choices=['kd', 'hint', 'attention', 'vid',
                                 'WSL_att', 'pursuhint_cmtf'],
                        help='distillation method (omit for vanilla student training)')
    parser.add_argument('--dual_cmtf', action='store_true',
                        help='Train both CP and Tucker decomposed students simultaneously')
    parser.add_argument('--cp_rank_ratio', type=float, default=0.5,
                        help='compression ratio for CP decomposition')
    parser.add_argument('--tucker_rank_ratio', type=float, default=0.5,
                        help='compression ratio for Tucker decomposition')
    parser.add_argument('--cmtf_rank', type=int, default=8,
                        help='rank for Coupled Tensor Loss')

    # Loss weights
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='weight for KL divergence loss')
    parser.add_argument('--beta', type=float, default=50.0,
                        help='weight for distillation-specific loss')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='weight for classification loss')
    parser.add_argument('--kd_T', type=float, default=4.0,
                        help='temperature for KD distillation')

    # Hint points
    parser.add_argument('--hint_points', type=str, default='3,7,13,16',
                        help='teacher hint positions (comma-separated, 1-indexed)')
    parser.add_argument('--s_points', type=str, default=None,
                        help='student hint positions (auto-detected if not specified)')
    parser.add_argument('--preact', type=bool, default=False,
                        help='use pre-activation features')

    # Data loading backend
    parser.add_argument('--no_dali', action='store_true',
                        help='Use standard PyTorch DataLoader instead of DALI (for debugging)')
    parser.add_argument('--dali_cpu', action='store_true',
                        help='Runs CPU based version of DALI pipeline')

    # DDP / Apex
    parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN')
    parser.add_argument('--opt-level', type=str, default=None)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--channels-last', type=bool, default=False)

    # Profiling / test mode
    parser.add_argument('--prof', default=-1, type=int,
                        help='Only run N iterations for profiling')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('-t', '--test', action='store_true',
                        help='Launch test mode with preset arguments')

    # PyTorch Compile Optimization
    parser.add_argument('--torch_compile', action='store_true')

    # Dynamic loss weighting (Kendall et al. CVPR 2018)
    parser.add_argument('--dynamic_loss_weights', action='store_true',
                        help='Learn per-task loss weights via homoscedastic uncertainty '
                             '(Kendall et al. CVPR 2018). Replaces fixed gamma/alpha/beta.')

    args = parser.parse_args()

    # ---- Determine DALI usage ----
    args.use_dali = not args.no_dali
    if args.use_dali and not DALI_AVAILABLE:
        print("Warning: DALI not available. Falling back to standard DataLoader.")
        args.use_dali = False

    # ---- Auto-detect student hint points ----
    if args.s_points is None and args.distill is not None:
        if args.model_s in S_POINTS_DICT:
            args.s_points = S_POINTS_DICT[args.model_s]
        else:
            raise ValueError(f"No default s_points for {args.model_s}. "
                             f"Please specify --s_points.")

    # ---- Save paths ----
    distill_tag = args.distill or 'vanilla'
    args.model_name = 'S-{}_T-{}_imagenet_{}_r-{}_a-{}_b-{}_{}'.format(
        args.model_s, args.model_t, distill_tag,
        args.gamma, args.alpha, args.beta, args.trial)
    args.model_path = './save/student_model'

    if args.distill:
        args.save_folder = os.path.join(args.model_path, 'imagenet100',
                                        str(args.hint_points), args.model_name)
    else:
        args.save_folder = os.path.join(args.model_path, 'imagenet100', args.model_name)

    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    args.tb_folder = os.path.join('./save/tensorboard', args.model_name)
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    return args


# ---------- Utility ----------------------------------------------------------------
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


# ---------- DALI Pipeline ----------------------------------------------------------
if DALI_AVAILABLE:
    @pipeline_def
    def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards,
                             dali_cpu=False, is_training=True):
        images, labels = fn.readers.file(file_root=data_dir,
                                         shard_id=shard_id,
                                         num_shards=num_shards,
                                         random_shuffle=is_training,
                                         pad_last_batch=True,
                                         name="Reader")
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
        preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
        if is_training:
            images = fn.decoders.image_random_crop(
                images, device=decoder_device, output_type=types.RGB,
                device_memory_padding=device_memory_padding,
                host_memory_padding=host_memory_padding,
                preallocate_width_hint=preallocate_width_hint,
                preallocate_height_hint=preallocate_height_hint,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[0.1, 1.0],
                num_attempts=100)
            images = fn.resize(images, device=dali_device,
                               resize_x=crop, resize_y=crop,
                               interp_type=types.INTERP_TRIANGULAR)
            mirror = fn.random.coin_flip(probability=0.5)
        else:
            images = fn.decoders.image(images, device=decoder_device,
                                       output_type=types.RGB)
            images = fn.resize(images, device=dali_device,
                               size=size, mode="not_smaller",
                               interp_type=types.INTERP_TRIANGULAR)
            mirror = False

        images = fn.crop_mirror_normalize(
            images.gpu(), dtype=types.FLOAT, output_layout="CHW",
            crop=(crop, crop),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=mirror)
        labels = labels.gpu()
        return images, labels


# ---------- Standard PyTorch DataLoader (fallback) ----------------------------------
def get_standard_imagenet_loaders(traindir, valdir, batch_size=256, num_workers=4):
    """Standard PyTorch DataLoaders for ImageNet (for local debugging)."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# ---------- Teacher loading ---------------------------------------------------------
def load_teacher(model_path, n_cls, model_t='ResNet34'):
    """Load a pre-trained teacher model, handling various checkpoint formats."""
    print('==> loading teacher model')
    model = model_dict[model_t](num_classes=n_cls)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state = checkpoint['model']
    else:
        state = checkpoint

    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(remove_module(state))

    print('==> done')
    return model


# ---------- Batch extraction helper -------------------------------------------------
def extract_batch(data, use_dali):
    """Unified batch extraction from DALI or standard DataLoader."""
    if use_dali:
        inp = data[0]["data"]
        tgt = data[0]["label"].squeeze(-1).long()
    else:
        inp, tgt = data
        inp = inp.cuda()
        tgt = tgt.cuda()
    return inp, tgt


def get_loader_len(loader, batch_size, use_dali):
    if use_dali:
        return int(math.ceil(loader._size / batch_size))
    else:
        return len(loader)


# ==================================================================================
#  MAIN
# ==================================================================================
def main():
    global best_acc1, best_acc1_2, args
    best_acc1 = 0
    best_acc1_2 = 0
    args = parse()

    # ---- Test mode ----
    if args.test:
        args.opt_level = None
        args.epochs = 1
        args.start_epoch = 0
        args.batch_size = 64
        args.data = []
        args.sync_bn = False
        args.data.append('/data/imagenet/train-jpeg/')
        args.data.append('/data/imagenet/val-jpeg/')
        print("Test mode - no DDP, no apex, 10 iterations")

    if not len(args.data):
        raise Exception("error: No data set provided")

    # ---- Distributed setup ----
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.opt_level is not None or args.distributed or args.sync_bn:
        try:
            global DDP, amp, optimizers, parallel
            from apex.parallel import DistributedDataParallel as DDP
            from apex import amp, optimizers, parallel
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex")

    print("opt_level = {}".format(args.opt_level))
    print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32),
          type(args.keep_batchnorm_fp32))
    print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))
    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    args.total_batch_size = args.world_size * args.batch_size
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    n_cls = 100

    # ---- Create student model ----
    model_s = model_dict[args.model_s](num_classes=n_cls)

    if args.dual_cmtf:
        model_s2 = copy.deepcopy(model_s)

    # ---- Model decomposition for pursuhint_cmtf ----
    if args.distill == 'pursuhint_cmtf':
        print(f"==> Decomposing student with CP factorization (ratio: {args.cp_rank_ratio})...")
        model_s = decompose_model(model_s, method='cp', cp_rank_ratio=args.cp_rank_ratio)
        if args.dual_cmtf:
            print(f"==> Decomposing parallel student with Tucker (ratio: {args.tucker_rank_ratio})...")
            model_s2 = decompose_model(model_s2, method='tucker',
                                       tucker_rank_ratio=args.tucker_rank_ratio)

    if args.sync_bn:
        print("using apex synced BN")
        model_s = parallel.convert_syncbn_model(model_s)

    if hasattr(torch, 'channels_last') and hasattr(torch, 'contiguous_format'):
        memory_format = torch.channels_last if args.channels_last else torch.contiguous_format
        model_s = model_s.cuda().to(memory_format=memory_format)
    else:
        model_s = model_s.cuda()

    if args.dual_cmtf:
        model_s2 = model_s2.cuda()

    # ---- Load teacher (only if doing distillation) ----
    teacher = None
    if args.distill is not None:
        teacher = load_teacher(args.path_t, n_cls, args.model_t).cuda()
        teacher.eval()

    # ---- Build module_list / trainable_list / criterion_list ----
    module_list = nn.ModuleList([model_s])
    trainable_list = nn.ModuleList([model_s])

    if args.dual_cmtf:
        module_list_2 = nn.ModuleList([model_s2])
        trainable_list_2 = nn.ModuleList([model_s2])

    criterion_cls = nn.CrossEntropyLoss().cuda()

    if args.distill is not None:
        # -- Divergence criterion --
        if args.distill == 'WSL_att':
            criterion_div = WSLLoss(args.kd_T)
        else:
            criterion_div = DistillKLD(args.kd_T)

        # -- Probe feature shapes for hint / vid / cmtf --
        if args.distill in ['hint', 'vid', 'pursuhint_cmtf']:
            probe_data = torch.randn(2, 3, 224, 224).cuda()
            model_s.eval()
            teacher.eval()
            with torch.no_grad():
                feat_s_probe, _ = model_s(probe_data, is_feat=True, preact=args.preact,
                                          hint_points=args.s_points)
                feat_t_probe, _ = teacher(probe_data, is_feat=True, preact=args.preact,
                                          hint_points=args.hint_points)

        # -- Method-specific criterion --
        if args.distill == 'kd':
            criterion_kd = DistillKLD(args.kd_T)

        elif args.distill == 'hint':
            criterion_kd = HintLoss()
            s_shapes = [f.shape for f in feat_s_probe]
            t_shapes = [f.shape for f in feat_t_probe]
            for i in range(len(t_shapes)):
                regress_s = ConvReg(s_shapes[i], t_shapes[i])
                module_list.append(regress_s)
                trainable_list.append(regress_s)

        elif args.distill == 'attention':
            criterion_kd = Attention()

        elif args.distill == 'vid':
            s_n = [f.shape[1] for f in feat_s_probe]
            t_n = [f.shape[1] for f in feat_t_probe]
            criterion_kd = nn.ModuleList(
                [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
            )
            trainable_list.append(criterion_kd)

        elif args.distill == 'WSL_att':
            criterion_kd = Attention()

        elif args.distill == 'pursuhint_cmtf':
            criterion_kd = CoupledTensorLoss(model=model_s, rank=args.cmtf_rank, iter_max=10)
            if args.dual_cmtf:
                criterion_kd_2 = CoupledTensorLoss(model=model_s2, rank=args.cmtf_rank, iter_max=10)

        else:
            raise NotImplementedError(args.distill)

        criterion_list = nn.ModuleList([criterion_cls, criterion_div, criterion_kd])

        if args.dual_cmtf:
            criterion_list_2 = nn.ModuleList([
                criterion_cls, criterion_div, criterion_kd_2])
    else:
        criterion_list = None

    # ---- Optimizer ----
    # Scale learning rate based on global batch size
    args.lr = args.lr * float(args.batch_size * args.world_size) / 256.

    # Dynamic loss weighters (Kendall et al. CVPR 2018).
    # Created before optimizers so log_vars get a zero-weight-decay param group.
    loss_weighter = None
    loss_weighter_2 = None
    if args.dynamic_loss_weights:
        loss_weighter = DynamicLossWeighter(num_losses=3).cuda()
        if args.dual_cmtf:
            loss_weighter_2 = DynamicLossWeighter(num_losses=3).cuda()

    if args.dynamic_loss_weights:
        optimizer = torch.optim.SGD([
            {'params': trainable_list.parameters(), 'weight_decay': args.weight_decay},
            {'params': loss_weighter.parameters(), 'weight_decay': 0.0},
        ], args.lr, momentum=args.momentum)
        if args.dual_cmtf:
            optimizer_2 = torch.optim.SGD([
                {'params': trainable_list_2.parameters(), 'weight_decay': args.weight_decay},
                {'params': loss_weighter_2.parameters(), 'weight_decay': 0.0},
            ], args.lr, momentum=args.momentum)
    else:
        optimizer = torch.optim.SGD(trainable_list.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        if args.dual_cmtf:
            optimizer_2 = torch.optim.SGD(trainable_list_2.parameters(), args.lr,
                                          momentum=args.momentum,
                                          weight_decay=args.weight_decay)

    # Append teacher AFTER optimizer creation (avoid weight decay on teacher)
    if teacher is not None:
        module_list.append(teacher)
        if args.dual_cmtf:
            module_list_2.append(teacher)

    # ---- Apex AMP ----
    if args.opt_level is not None:
        model_s, optimizer = amp.initialize(model_s, optimizer,
                                            opt_level=args.opt_level,
                                            keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                            loss_scale=args.loss_scale)
        if args.dual_cmtf:
            model_s2, optimizer_2 = amp.initialize(model_s2, optimizer_2,
                                                   opt_level=args.opt_level,
                                                   keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                                   loss_scale=args.loss_scale)
        if teacher is not None:
            teacher = amp.initialize(teacher, opt_level=args.opt_level,
                                     keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                     loss_scale=args.loss_scale)

    # ---- DDP ----
    if args.distributed:
        print('distributed training..')
        model_s = DDP(model_s, delay_allreduce=True)
        if args.dual_cmtf:
            model_s2 = DDP(model_s2, delay_allreduce=True)

    # ---- Compile student models ----
    if args.torch_compile:
        model_s = torch.compile(model_s, dynamic=True)
        if args.dual_cmtf:
            model_s2 = torch.compile(model_s2, dynamic=True)
        if teacher is not None:
            teacher = torch.compile(teacher, dynamic=True)
        
        # Update the module lists so the training loop uses the compiled versions!
        module_list[0] = model_s
        if teacher is not None:
            module_list[-1] = teacher
            
        if args.dual_cmtf:
            module_list_2[0] = model_s2
            if teacher is not None:
                module_list_2[-1] = teacher

    # ---- Resume ----
    if args.resume:
        if os.path.isdir(args.resume):
            cp_path = os.path.join(args.resume, 'checkpoint_cp.pth.tar')
            tk_path = os.path.join(args.resume, 'checkpoint_tucker.pth.tar')
            
            # 1. Load CP Student
            if os.path.isfile(cp_path):
                print("=> loading CP checkpoint '{}'".format(cp_path))
                ckpt_cp = torch.load(cp_path, map_location=lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = ckpt_cp['epoch']
                best_acc1 = ckpt_cp.get('best_acc1', 0.0)
                model_s.load_state_dict(ckpt_cp['state_dict'])
                optimizer.load_state_dict(ckpt_cp['optimizer'])
                if loss_weighter is not None and 'loss_weighter' in ckpt_cp:
                    loss_weighter.load_state_dict(ckpt_cp['loss_weighter'])
                print("=> loaded CP checkpoint (epoch {})".format(ckpt_cp['epoch']))
            else:
                print("=> no CP checkpoint found at '{}'".format(cp_path))

            # 2. Load Tucker Student
            if args.dual_cmtf:
                if os.path.isfile(tk_path):
                    print("=> loading Tucker checkpoint '{}'".format(tk_path))
                    ckpt_tk = torch.load(tk_path, map_location=lambda storage, loc: storage.cuda(args.gpu))
                    best_acc1_2 = ckpt_tk.get('best_acc1', 0.0)
                    model_s2.load_state_dict(ckpt_tk['state_dict'])
                    optimizer_2.load_state_dict(ckpt_tk['optimizer'])
                    if loss_weighter_2 is not None and 'loss_weighter' in ckpt_tk:
                        loss_weighter_2.load_state_dict(ckpt_tk['loss_weighter'])
                    print("=> loaded Tucker checkpoint (epoch {})".format(ckpt_tk['epoch']))
                else:
                    print("=> no Tucker checkpoint found at '{}'".format(tk_path))
                    
        elif os.path.isfile(args.resume):
            # Fallback for standard single-student distillation
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint.get('best_acc1', 0.0)
            model_s.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    # ---- Data loading ----
    if len(args.data) == 1:
        traindir = os.path.join(args.data[0], 'train')
        valdir = os.path.join(args.data[0], 'val')
    else:
        traindir = args.data[0]
        valdir = args.data[1]

    crop_size = 224
    val_size = 256

    if args.use_dali:
        pipe = create_dali_pipeline(batch_size=args.batch_size,
                                    num_threads=args.workers,
                                    device_id=args.local_rank,
                                    seed=12 + args.local_rank,
                                    data_dir=traindir,
                                    crop=crop_size, size=val_size,
                                    dali_cpu=args.dali_cpu,
                                    shard_id=args.local_rank,
                                    num_shards=args.world_size,
                                    is_training=True)
        pipe.build()
        train_loader = DALIClassificationIterator(
            pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

        pipe = create_dali_pipeline(batch_size=args.batch_size,
                                    num_threads=args.workers,
                                    device_id=args.local_rank,
                                    seed=12 + args.local_rank,
                                    data_dir=valdir,
                                    crop=crop_size, size=val_size,
                                    dali_cpu=args.dali_cpu,
                                    shard_id=args.local_rank,
                                    num_shards=args.world_size,
                                    is_training=False)
        pipe.build()
        val_loader = DALIClassificationIterator(
            pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    else:
        print("Using standard PyTorch DataLoader (local debugging mode)")
        train_loader, val_loader = get_standard_imagenet_loaders(
            traindir, valdir, batch_size=args.batch_size, num_workers=args.workers)

    if args.evaluate:
        if args.dual_cmtf:
            validate(val_loader, model_s, criterion_cls, model_s2=model_s2)
        else:
            validate(val_loader, model_s, criterion_cls)
        return

    # ---- Print model info ----
    if teacher is not None:
        print('Teacher ({}) parameters: {}'.format(
            args.model_t, sum(p.numel() for p in teacher.parameters())))
    print('Student ({}) parameters: {}'.format(
        args.model_s, sum(p.numel() for p in model_s.parameters())))
    if args.dual_cmtf:
        print('Student2 (Tucker) parameters: {}'.format(
            sum(p.numel() for p in model_s2.parameters())))

    # ---- CSV Logger Setup ----
    file_mode = 'a' if args.resume else 'w'

    csv_path_cp = os.path.join(args.save_folder, 'training_log_cp.csv')
    write_header_cp = not os.path.exists(csv_path_cp) or not args.resume
    csv_file_cp = open(csv_path_cp, file_mode, newline='')
    csv_writer_cp = csv.writer(csv_file_cp)
    if write_header_cp:
        csv_writer_cp.writerow(['epoch', 'lr', 'train_loss', 'train_acc1', 'train_acc5',
                            'val_acc1', 'val_acc5', 'best_acc1',
                            'w_cls', 'w_div', 'w_kd'])
    print(f'CSV log: {csv_path_cp}')

    if args.dual_cmtf:
        csv_path_tk = os.path.join(args.save_folder, 'training_log_tucker.csv')
        write_header_tk = not os.path.exists(csv_path_tk) or not args.resume
        csv_file_tk = open(csv_path_tk, file_mode, newline='')
        csv_writer_tk = csv.writer(csv_file_tk)
        if write_header_tk:
            csv_writer_tk.writerow(['epoch', 'lr', 'train_loss', 'train_acc1', 'train_acc5',
                                    'val_acc1', 'val_acc5', 'best_acc1',
                                    'w_cls', 'w_div', 'w_kd'])
        print(f'Tucker CSV log: {csv_path_tk}')

    # ==================== Training Loop ====================
    total_time = AverageMeter()

    for epoch in range(args.start_epoch, args.epochs):
        # ---- Train ----
        if args.distill is not None:
            if args.dual_cmtf:
                avg_train_time, train_loss, train_p1, train_p5, train_loss_2, train_p1_2, train_p5_2 = train_kd(
                    train_loader, module_list, criterion_list, optimizer, epoch,
                    module_list_2=module_list_2, criterion_list_2=criterion_list_2, optimizer_2=optimizer_2,
                    loss_weighter=loss_weighter, loss_weighter_2=loss_weighter_2)
            else:
                avg_train_time, train_loss, train_p1, train_p5 = train_kd(
                    train_loader, module_list, criterion_list, optimizer, epoch,
                    loss_weighter=loss_weighter)
        else:
            avg_train_time, train_loss, train_p1, train_p5 = train(
                train_loader, model_s, criterion_cls, optimizer, epoch)
            
        total_time.update(avg_train_time)
        if args.test:
            break

        # ---- Validate ----
        if args.dual_cmtf:
            acc1, acc5, acc1_2, acc5_2 = validate(val_loader, model_s, criterion_cls, model_s2=model_s2)
        else:
            acc1, acc5 = validate(val_loader, model_s, criterion_cls)
        
        # ---- Save best / checkpoint ----
        if args.local_rank == 0:
            current_lr = optimizer.param_groups[0]['lr']

            # CP / primary student
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            cp_ckpt = {
                'epoch': epoch + 1,
                'model_s': args.model_s,
                'state_dict': model_s.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }
            if loss_weighter is not None:
                cp_ckpt['loss_weighter'] = loss_weighter.state_dict()
            save_checkpoint(cp_ckpt, is_best, args.save_folder, tag='cp')

            print('CP  => Acc@1 {:.3f}  Acc@5 {:.3f}  Best {:.3f}'.format(
                acc1, acc5, best_acc1))
            if loss_weighter is not None:
                ew = loss_weighter.effective_weights()
            else:
                ew = [args.gamma, args.alpha, args.beta]

            # ---- CSV row ----
            csv_writer_cp.writerow([
                epoch, f'{current_lr:.6f}',
                f'{train_loss:.4f}', f'{train_p1:.3f}', f'{train_p5:.3f}',
                f'{acc1:.3f}', f'{acc5:.3f}', f'{best_acc1:.3f}',
                f'{ew[0]:.6f}', f'{ew[1]:.6f}', f'{ew[2]:.6f}'])
            csv_file_cp.flush()

            # Tucker / secondary student
            if args.dual_cmtf:
                is_best_2 = acc1_2 > best_acc1_2
                best_acc1_2 = max(acc1_2, best_acc1_2)

                tk_ckpt = {
                    'epoch': epoch + 1,
                    'model_s': args.model_s,
                    'state_dict': model_s2.state_dict(),
                    'best_acc1': best_acc1_2,
                    'optimizer': optimizer_2.state_dict(),
                }
                if loss_weighter_2 is not None:
                    tk_ckpt['loss_weighter'] = loss_weighter_2.state_dict()
                save_checkpoint(tk_ckpt, is_best_2, args.save_folder, tag='tucker')
                if loss_weighter_2 is not None:
                    ew2 = loss_weighter_2.effective_weights()
                else:
                    ew2 = [args.gamma, args.alpha, args.beta]

                print('Tucker => Acc@1 {:.3f}  Acc@5 {:.3f}  Best {:.3f}'.format(
                    acc1_2, acc5_2, best_acc1_2))

                csv_writer_tk.writerow([
                    epoch, f'{current_lr:.6f}',
                    f'{train_loss_2:.4f}', f'{train_p1_2:.3f}', f'{train_p5_2:.3f}',
                    f'{acc1_2:.3f}', f'{acc5_2:.3f}', f'{best_acc1_2:.3f}',
                    f'{ew2[0]:.6f}', f'{ew2[1]:.6f}', f'{ew2[2]:.6f}'])
                csv_file_tk.flush()

            if epoch == args.epochs - 1:
                print('##Top-1 {0}\n##Top-5 {1}\n##Perf  {2}'.format(
                    acc1, acc5,
                    args.total_batch_size / total_time.avg))

        # ---- Reset DALI loaders ----
        if args.use_dali:
            train_loader.reset()
            val_loader.reset()

    # ---- Close CSVs ----
    csv_file_cp.close()
    print(f'CP training log saved to: {csv_path_cp}')
    if args.dual_cmtf:
        csv_file_tk.close()
        print(f'Tucker training log saved to: {csv_path_tk}')

    # ---- Save last model ----
    if args.local_rank == 0:
        torch.save({'opt': vars(args), 'model': model_s.state_dict()},
                    os.path.join(args.save_folder, f'{args.model_s}_last_cp.pth'))
        if args.dual_cmtf:
            torch.save({'opt': vars(args), 'model': model_s2.state_dict()},
                        os.path.join(args.save_folder, f'{args.model_s}_last_tucker.pth'))

    # ---- Save experiment summary JSON ----
    summary = {
        'teacher': args.model_t,
        'student': args.model_s,
        'dataset': 'imagenet100',
        'distill': args.distill,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'hint_points': args.hint_points,
        'gamma': args.gamma,
        'alpha': args.alpha,
        'beta': args.beta,
        'best_acc1_cp': round(best_acc1, 4),
    }
    if args.distill == 'pursuhint_cmtf':
        summary['cp_rank_ratio'] = args.cp_rank_ratio
    if args.dynamic_loss_weights and loss_weighter is not None:
        summary['dynamic_loss_weights'] = True
        w = loss_weighter.effective_weights()
        summary['final_weights_cp'] = {'cls': round(w[0], 6), 'div': round(w[1], 6), 'kd': round(w[2], 6)}
    if args.dual_cmtf:
        summary['tucker_rank_ratio'] = args.tucker_rank_ratio
        summary['best_acc1_tucker'] = round(best_acc1_2, 4)
        if args.dynamic_loss_weights and loss_weighter_2 is not None:
            w2 = loss_weighter_2.effective_weights()
            summary['final_weights_tucker'] = {'cls': round(w2[0], 6), 'div': round(w2[1], 6), 'kd': round(w2[2], 6)}
    summary_path = os.path.join(args.save_folder, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f'Experiment summary saved to: {summary_path}')

    print('best CP accuracy:', best_acc1)
    if args.dual_cmtf:
        print('best Tucker accuracy:', best_acc1_2)


# ==================================================================================
#  TRAINING FUNCTIONS
# ==================================================================================

def train(train_loader, model, criterion, optimizer, epoch):
    """Vanilla training (no distillation)."""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()
    train_loader_len = get_loader_len(train_loader, args.batch_size, args.use_dali)

    for i, data in enumerate(train_loader):
        inp, target = extract_batch(data, args.use_dali)

        if args.prof >= 0 and i == args.prof:
            torch.cuda.cudart().cudaProfilerStart()
        if args.prof >= 0:
            torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

        adjust_learning_rate(optimizer, epoch, i, train_loader_len)
        if args.test and i > 10:
            break

        _, logit_s = model(inp, is_feat=True, preact=args.preact)
        loss = criterion(logit_s, target)

        optimizer.zero_grad()
        if args.opt_level is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            acc1, acc5 = accuracy(logit_s.data, target, topk=(1, 5))
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                acc1 = reduce_tensor(acc1)
                acc5 = reduce_tensor(acc5)
            else:
                reduced_loss = loss.data

            losses.update(to_python_float(reduced_loss), inp.size(0))
            top1.update(to_python_float(acc1), inp.size(0))
            top5.update(to_python_float(acc5), inp.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if args.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, train_loader_len,
                       args.world_size * args.batch_size / batch_time.val,
                       args.world_size * args.batch_size / batch_time.avg,
                       batch_time=batch_time, loss=losses, top1=top1, top5=top5))

        if args.prof >= 0:
            torch.cuda.nvtx.range_pop()
        if args.prof >= 0 and i == args.prof + 10:
            torch.cuda.cudart().cudaProfilerStop()
            quit()

    return batch_time.avg, losses.avg, top1.avg, top5.avg


def train_kd(train_loader, module_list, criterion_list, optimizer, epoch,
             module_list_2=None, criterion_list_2=None, optimizer_2=None,
             loss_weighter=None, loss_weighter_2=None):
    """One epoch of knowledge distillation training (Unified Dual-Student)."""
    dual = module_list_2 is not None

    for module in module_list: module.train()
    module_list[-1].eval()  # teacher is always last
    model_s, model_t = module_list[0], module_list[-1]
    criterion_cls, criterion_div, criterion_kd = criterion_list[0], criterion_list[1], criterion_list[2]

    if dual:
        for module in module_list_2: module.train()
        module_list_2[-1].eval()
        model_s2 = module_list_2[0]
        criterion_cls_2, criterion_div_2, criterion_kd_2 = criterion_list_2[0], criterion_list_2[1], criterion_list_2[2]
        losses_2, top1_2, top5_2 = AverageMeter(), AverageMeter(), AverageMeter()

    batch_time, losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    end = time.time()
    train_loader_len = get_loader_len(train_loader, args.batch_size, args.use_dali)

    for i, data in enumerate(train_loader):
        inp, target = extract_batch(data, args.use_dali)

        if args.prof >= 0 and i == args.prof: torch.cuda.cudart().cudaProfilerStart()
        if args.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

        adjust_learning_rate(optimizer, epoch, i, train_loader_len)
        if dual: adjust_learning_rate(optimizer_2, epoch, i, train_loader_len)
        if args.test and i > 10: break

        # ---- Forward: Teacher ----
        with torch.no_grad():
            feat_t, logit_t = model_t(inp, is_feat=True, preact=args.preact, hint_points=args.hint_points)
            feat_t = [f.detach() for f in feat_t]

        # ---- Process CP Student ----
        feat_s, logit_s = model_s(inp, is_feat=True, preact=args.preact, hint_points=args.s_points)
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t, target) if args.distill == 'WSL_att' else criterion_div(logit_s, logit_t)

        if args.distill == 'kd': loss_kd = torch.tensor(0.0, device=inp.device)
        elif args.distill == 'hint':
            f_s = [module_list[1 + j](feat_s[j]) for j in range(len(feat_s))]
            loss_kd = criterion_kd(f_s, feat_t)
        elif args.distill in ['attention', 'WSL_att']: loss_kd = sum(criterion_kd(feat_s, feat_t))
        elif args.distill == 'vid': loss_kd = sum([c(f_s, f_t) for f_s, f_t, c in zip(feat_s, feat_t, criterion_kd)])
        elif args.distill == 'pursuhint_cmtf': loss_kd = criterion_kd(feat_s, feat_t)
        else: raise NotImplementedError(args.distill)

        if loss_weighter is not None:
            loss = loss_weighter(loss_cls, loss_div, loss_kd)
        else:
            loss = args.gamma * loss_cls + args.alpha * loss_div + args.beta * loss_kd

        optimizer.zero_grad()
        if args.opt_level is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss: scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # ---- Process Tucker Student ----
        if dual:
            feat_s2, logit_s2 = model_s2(inp, is_feat=True, preact=args.preact, hint_points=args.s_points)
            loss_cls_2 = criterion_cls_2(logit_s2, target)
            loss_div_2 = criterion_div_2(logit_s2, logit_t, target) if args.distill == 'WSL_att' else criterion_div_2(logit_s2, logit_t)
            
            # Tucker is currently only designed to be used with pursuhint_cmtf
            loss_kd_2 = criterion_kd_2(feat_s2, feat_t) 
            if loss_weighter_2 is not None:
                loss_2 = loss_weighter_2(loss_cls_2, loss_div_2, loss_kd_2)
            else:
                loss_2 = args.gamma * loss_cls_2 + args.alpha * loss_div_2 + args.beta * loss_kd_2

            optimizer_2.zero_grad()
            if args.opt_level is not None:
                with amp.scale_loss(loss_2, optimizer_2) as scaled_loss_2: scaled_loss_2.backward()
            else:
                loss_2.backward()
            optimizer_2.step()

        # ---- Logging & Metrics ----
        if i % args.print_freq == 0:
            acc1, acc5 = accuracy(logit_s.data, target, topk=(1, 5))
            if dual: acc1_2, acc5_2 = accuracy(logit_s2.data, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                acc1, acc5 = reduce_tensor(acc1), reduce_tensor(acc5)
                if dual:
                    reduced_loss_2 = reduce_tensor(loss_2.data)
                    acc1_2, acc5_2 = reduce_tensor(acc1_2), reduce_tensor(acc5_2)
            else:
                reduced_loss = loss.data
                if dual: reduced_loss_2 = loss_2.data

            losses.update(to_python_float(reduced_loss), inp.size(0))
            top1.update(to_python_float(acc1), inp.size(0))
            top5.update(to_python_float(acc5), inp.size(0))

            if dual:
                losses_2.update(to_python_float(reduced_loss_2), inp.size(0))
                top1_2.update(to_python_float(acc1_2), inp.size(0))
                top5_2.update(to_python_float(acc5_2), inp.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if args.local_rank == 0:
                print('Epoch [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       epoch, i, train_loader_len, batch_time=batch_time, loss=losses, top1=top1))
                if dual:
                    print('  [Tucker] \t\tLoss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses_2, top1=top1_2))

        if args.prof >= 0: torch.cuda.nvtx.range_pop()
        if args.prof >= 0 and i == args.prof + 10:
            torch.cuda.cudart().cudaProfilerStop()
            quit()

    if dual:
        return batch_time.avg, losses.avg, top1.avg, top5.avg, losses_2.avg, top1_2.avg, top5_2.avg
    return batch_time.avg, losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, model_s2=None):
    """Evaluate on validation set (Unified Dual-Student)."""
    dual = model_s2 is not None

    batch_time, losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    if dual:
        losses_2, top1_2, top5_2 = AverageMeter(), AverageMeter(), AverageMeter()

    model.eval()
    if dual: model_s2.eval()
    
    end = time.time()
    val_loader_len = get_loader_len(val_loader, args.batch_size, args.use_dali)

    for i, data in enumerate(val_loader):
        inp, target = extract_batch(data, args.use_dali)

        with torch.no_grad():
            _, logit_s = model(inp, is_feat=True, preact=args.preact)
            loss = criterion(logit_s, target)
            if dual:
                _, logit_s2 = model_s2(inp, is_feat=True, preact=args.preact)
                loss_2 = criterion(logit_s2, target)

        acc1, acc5 = accuracy(logit_s.data, target, topk=(1, 5))
        if dual: acc1_2, acc5_2 = accuracy(logit_s2.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            acc1, acc5 = reduce_tensor(acc1), reduce_tensor(acc5)
            if dual:
                reduced_loss_2 = reduce_tensor(loss_2.data)
                acc1_2, acc5_2 = reduce_tensor(acc1_2), reduce_tensor(acc5_2)
        else:
            reduced_loss = loss.data
            if dual: reduced_loss_2 = loss_2.data

        losses.update(to_python_float(reduced_loss), inp.size(0))
        top1.update(to_python_float(acc1), inp.size(0))
        top5.update(to_python_float(acc5), inp.size(0))

        if dual:
            losses_2.update(to_python_float(reduced_loss_2), inp.size(0))
            top1_2.update(to_python_float(acc1_2), inp.size(0))
            top5_2.update(to_python_float(acc5_2), inp.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   i, val_loader_len, batch_time=batch_time, loss=losses, top1=top1))
            if dual:
                print('  [Tucker] \tLoss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses_2, top1=top1_2))

    print(' * [CP] Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    if dual:
        print(' * [Tucker] Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1_2, top5=top5_2))
        return top1.avg, top5.avg, top1_2.avg, top5_2.avg

    return top1.avg, top5.avg


# ==================================================================================
#  UTILITIES
# ==================================================================================

def save_checkpoint(state, is_best, save_folder, tag='cp'):
    """Save checkpoint to save_folder with tag (cp or tucker)."""
    filename = os.path.join(save_folder, f'checkpoint_{tag}.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_path = os.path.join(save_folder, f'{state["model_s"]}_best_{tag}.pth')
        shutil.copyfile(filename, best_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30
    if epoch >= 80:
        factor = factor + 1
    lr = args.lr * (0.1 ** factor)

    # Warmup
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt


if __name__ == '__main__':
    main()