from __future__ import print_function

import os
import argparse
import socket
import time
import csv
import json

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders
from dataset.cifar10 import get_cifar10_dataloaders

from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_vanilla as train, validate

import math
import numpy as np
import random

torch.manual_seed(0)
cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(0)
random.seed(0)


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'wrn_40_2', 'wrn_16_2', 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar100', 'cifar10'], help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    # PyTorch model compile optimization
    parser.add_argument('--torch_compile', action='store_true')

    opt = parser.parse_args()
    

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './save/models'
        opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():
    best_acc = 0

    opt = parse_option()

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
    elif opt.dataset == 'cifar10':
        train_loader, val_loader = get_cifar10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 10
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = model_dict[opt.model](num_classes=n_cls)

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

        if opt.torch_compile:
            model = torch.compile(model, dynamic=True)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    # --- CSV Logger Setup ---
    csv_path = os.path.join(opt.save_folder, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'lr', 'epoch_time','train_acc', 'train_acc_top5', 'train_loss',
                         'test_acc', 'test_acc_top5', 'test_loss', 'best_acc'])
    print(f'CSV log will be saved to: {csv_path}')

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_acc_top5, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt, scaler=scaler)
        time2 = time.time()
        epoch_time = time2 - time1
        print('epoch {}, total time {:.2f}'.format(epoch, epoch_time))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_acc_top5', train_acc_top5, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best teacher model!')
            torch.save(state, save_file)

        # --- CSV: write epoch row ---
        current_lr = optimizer.param_groups[0]['lr']
        _ta = train_acc.item() if hasattr(train_acc, 'item') else float(train_acc)
        _t5 = train_acc_top5.item() if hasattr(train_acc_top5, 'item') else float(train_acc_top5)
        _va = test_acc.item() if hasattr(test_acc, 'item') else float(test_acc)
        _v5 = test_acc_top5.item() if hasattr(test_acc_top5, 'item') else float(test_acc_top5)
        _best = best_acc.item() if hasattr(best_acc, 'item') else float(best_acc)
        
        csv_writer.writerow([epoch, f'{current_lr:.6f}', f'{epoch_time:.2f}',
                             f'{_ta:.4f}', f'{_t5:.4f}', f'{train_loss:.4f}',
                             f'{_va:.4f}', f'{_v5:.4f}',
                             f'{test_loss:.4f}', f'{_best:.4f}'])
        csv_file.flush()

    # Close CSV
    csv_file.close()
    print(f'Training log saved to: {csv_path}')

    print('best accuracy:', best_acc)

    # Save experiment summary JSON
    _best = best_acc.item() if hasattr(best_acc, 'item') else float(best_acc)
    summary = {
        'model': opt.model,
        'dataset': opt.dataset,
        'epochs': opt.epochs,
        'learning_rate': opt.learning_rate,
        'lr_decay_epochs': opt.lr_decay_epochs,
        'weight_decay': opt.weight_decay,
        'batch_size': opt.batch_size,
        'best_accuracy': round(_best, 4),
    }
    summary_path = os.path.join(opt.save_folder, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f'Experiment summary saved to: {summary_path}')

    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()