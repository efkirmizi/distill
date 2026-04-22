"""
Teacher model training script for ImageNet-100.

Uses standard PyTorch DataLoaders (no DALI dependency) with the project's own
ResNet architectures from models/resnetv2.py so that the resulting checkpoint
is directly usable by store_hints_imagenet100.py and train_stu_imagenet100.py.

ImageNet-100 is a 100-class subset of ImageNet-1K defined by dataset/imagenet100.txt.
The dataset folder must have the standard ImageFolder layout:
    <data_folder>/train/<class_id>/...
    <data_folder>/val/<class_id>/...

Usage:
    python train_teacher_imagenet100.py \
        --model ResNet34 \
        --data_folder /path/to/imagenet100 \
        --epochs 100 \
        --batch_size 256 \
        --learning_rate 0.1 \
        --lr_decay_epochs 30,60,90
"""

from __future__ import print_function

import os
import argparse
import time
import csv
import json
import math

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import numpy as np
import random
import sys

from models import model_dict

# ---- Reproducibility ----
torch.manual_seed(0)
cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(0)
random.seed(0)

# Number of classes in ImageNet-100
N_CLS = 100


def parse_option():
    parser = argparse.ArgumentParser('Train a teacher model on ImageNet-100')

    # Logging / checkpointing
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency (batches)')
    parser.add_argument('--save_freq', type=int, default=20, help='checkpoint save frequency (epochs)')

    # Training
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='dataloader workers')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')

    # Optimisation
    parser.add_argument('--learning_rate', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,60,90',
                        help='comma-separated epoch milestones for LR decay')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='LR decay factor')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='linear warmup epochs (0 to disable)')

    # Model
    parser.add_argument('--model', type=str, default='ResNet34',
                        choices=['ResNet18', 'ResNet34'],
                        help='teacher model architecture (from models/resnetv2.py)')

    # Data
    parser.add_argument('--data_folder', type=str, required=True,
                        help='path to ImageNet-100 root (must have train/ and val/ subdirs)')

    # Experiment
    parser.add_argument('--trial', type=int, default=0, help='experiment id')

    opt = parser.parse_args()

    # Parse LR decay milestones
    opt.lr_decay_epochs = [int(x) for x in opt.lr_decay_epochs.split(',')]

    # Build output paths
    opt.model_name = '{}_imagenet100_lr_{}_decay_{}_trial_{}'.format(
        opt.model, opt.learning_rate, opt.weight_decay, opt.trial)

    opt.save_folder = os.path.join('./save', 'models', opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    opt.tb_folder = os.path.join('./save', 'tensorboard', opt.model_name)
    os.makedirs(opt.tb_folder, exist_ok=True)

    return opt


# ==================================================================================
#  Data loading
# ==================================================================================

def get_imagenet100_dataloaders(data_folder, batch_size=256, num_workers=8):
    """Standard PyTorch DataLoaders for ImageNet-100 (train + val)."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_folder = os.path.join(data_folder, 'train')
    val_folder = os.path.join(data_folder, 'val')

    train_set = datasets.ImageFolder(train_folder, transform=train_transform)
    val_set = datasets.ImageFolder(val_folder, transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    print(f'ImageNet-100 loaded: {len(train_set)} train / {len(val_set)} val images, '
          f'{len(train_set.classes)} classes')

    return train_loader, val_loader


# ==================================================================================
#  LR schedule
# ==================================================================================

def adjust_learning_rate(epoch, opt, optimizer):
    """Step decay with optional linear warmup."""
    if epoch <= opt.warmup_epochs and opt.warmup_epochs > 0:
        lr = opt.learning_rate * (epoch / opt.warmup_epochs)
    else:
        steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
        lr = opt.learning_rate * (opt.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# ==================================================================================
#  Training / Validation loops
# ==================================================================================

class AverageMeter:
    """Tracks running mean and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0

    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
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


def train_one_epoch(epoch, train_loader, model, criterion, optimizer, opt, scaler):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (images, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # Forward
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            output = model(images)
            loss = criterion(output, targets)

        # Metrics
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        # Backward
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            print(f'Epoch: [{epoch}][{idx}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')
            sys.stdout.flush()

    print(f' * Train Acc@1 {top1.avg:.3f}  Acc@5 {top5.avg:.3f}')
    return top1.avg, losses.avg


@torch.no_grad()
def validate(val_loader, model, criterion, opt):
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (images, targets) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        output = model(images)
        loss = criterion(output, targets)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            print(f'  Val: [{idx}/{len(val_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')

    print(f' * Val  Acc@1 {top1.avg:.3f}  Acc@5 {top5.avg:.3f}')
    return top1.avg, top5.avg, losses.avg


# ==================================================================================
#  Main
# ==================================================================================

def main():
    best_acc = 0.0
    opt = parse_option()

    print('=' * 60)
    print(f'Training {opt.model} on ImageNet-100  ({N_CLS} classes)')
    print(f'  LR={opt.learning_rate}  decay@{opt.lr_decay_epochs}  '
          f'warmup={opt.warmup_epochs}  epochs={opt.epochs}')
    print(f'  batch={opt.batch_size}  wd={opt.weight_decay}  momentum={opt.momentum}')
    print(f'  save -> {opt.save_folder}')
    print('=' * 60)

    # ---- Data ----
    train_loader, val_loader = get_imagenet100_dataloaders(
        data_folder=opt.data_folder,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers)

    # ---- Model ----
    model = model_dict[opt.model](num_classes=N_CLS)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    # ---- Optimizer ----
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    # ---- Tensorboard (optional) ----
    tb_logger = None
    try:
        import tensorboard_logger
        tb_logger = tensorboard_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    except ImportError:
        print('tensorboard_logger not installed — skipping TB logging')

    # ---- CSV Logger ----
    csv_path = os.path.join(opt.save_folder, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'lr', 'train_acc', 'train_loss',
                         'val_acc', 'val_acc_top5', 'val_loss', 'best_acc'])
    print(f'CSV log: {csv_path}')

    # ==================== Training Loop ====================
    for epoch in range(1, opt.epochs + 1):
        lr = adjust_learning_rate(epoch, opt, optimizer)

        print(f'\n==> Epoch {epoch}/{opt.epochs}  lr={lr:.6f}')

        t0 = time.time()
        train_acc, train_loss = train_one_epoch(
            epoch, train_loader, model, criterion, optimizer, opt, scaler)
        t1 = time.time()
        print(f'Epoch time: {t1 - t0:.1f}s')

        val_acc, val_acc5, val_loss = validate(val_loader, model, criterion, opt)

        # ---- Tensorboard ----
        if tb_logger is not None:
            tb_logger.log_value('train_acc', train_acc, epoch)
            tb_logger.log_value('train_loss', train_loss, epoch)
            tb_logger.log_value('val_acc', val_acc, epoch)
            tb_logger.log_value('val_acc_top5', val_acc5, epoch)
            tb_logger.log_value('val_loss', val_loss, epoch)

        # ---- Save best ----
        if val_acc > best_acc:
            best_acc = val_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            best_path = os.path.join(opt.save_folder,
                                     f'{opt.model}_best_{best_acc:.2f}.pth')
            torch.save(state, best_path)
            print(f'New best! Acc@1={best_acc:.3f}  saved -> {best_path}')

        # ---- CSV row ----
        csv_writer.writerow([
            epoch, f'{lr:.6f}',
            f'{train_acc:.4f}', f'{train_loss:.4f}',
            f'{val_acc:.4f}', f'{val_acc5:.4f}',
            f'{val_loss:.4f}', f'{best_acc:.4f}'])
        csv_file.flush()

        # ---- Periodic checkpoint ----
        if epoch % opt.save_freq == 0:
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': val_acc,
                'optimizer': optimizer.state_dict(),
            }
            ckpt_path = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
            torch.save(state, ckpt_path)
            print(f'  Checkpoint saved -> {ckpt_path}')

    # ---- Close CSV ----
    csv_file.close()
    print(f'\nTraining log saved to: {csv_path}')
    print(f'Best accuracy: {best_acc:.3f}')

    # ---- Save last model ----
    state = {
        'opt': vars(opt),
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    last_path = os.path.join(opt.save_folder, f'{opt.model}_last.pth')
    torch.save(state, last_path)
    print(f'Last model saved to: {last_path}')

    # ---- Experiment summary JSON ----
    summary = {
        'model': opt.model,
        'dataset': 'imagenet100',
        'num_classes': N_CLS,
        'epochs': opt.epochs,
        'learning_rate': opt.learning_rate,
        'lr_decay_epochs': opt.lr_decay_epochs,
        'lr_decay_rate': opt.lr_decay_rate,
        'weight_decay': opt.weight_decay,
        'batch_size': opt.batch_size,
        'warmup_epochs': opt.warmup_epochs,
        'best_accuracy': round(best_acc, 4),
        'total_parameters': sum(p.numel() for p in model.parameters()),
    }
    summary_path = os.path.join(opt.save_folder, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f'Experiment summary saved to: {summary_path}')


if __name__ == '__main__':
    main()
