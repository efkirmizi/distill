"""
Step 1 of the PURSUhInT pipeline:
Extracts N x C layer representations from sub-blocks of a pre-trained Teacher model.
Supports CIFAR-10 and CIFAR-100. Output is stored as hint{i}.pt files.
"""

import torchvision.transforms as transforms
import torchvision
import torch
import os
import sys
import argparse
import numpy as np

from models import model_dict
from for_init import remove_module


def parse_args():
    parser = argparse.ArgumentParser(description='Store teacher layer representations for PURSUhInT clustering.')
    parser.add_argument('--g', type=int, required=True, help='Sub-block index to extract (1-indexed).')
    parser.add_argument('--t_path', type=str, required=True, help='Path to the pre-trained teacher model checkpoint.')
    parser.add_argument('--model_t', type=str, required=True, help='Teacher model architecture name.')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'], help='Dataset to use for extracting representations.')
    parser.add_argument('--preact', type=bool, default=False, help='Extract pre-activation hints.')
    parser.add_argument('--hints_path', type=str, default=None, help='Output directory for hint .pt files. Defaults to ./save/hints/<model_name>.')
    parser.add_argument('--data_folder', type=str, default='../data', help='Path to the dataset root directory.')
    return parser.parse_args()


def load_teacher(model_path, n_cls, model_t):
    """Robustly loads a teacher model, handling wrappers and compile/parallel prefixes."""
    print('==> Loading teacher model for hint extraction...')
    model = model_dict[model_t](num_classes=n_cls)
    
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.load_state_dict(remove_module(state_dict))
        
    print('==> Done.')
    return model


def main():
    opt = parse_args()

    # Dataset config
    if opt.dataset == 'cifar10':
        n_cls = 10
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)
        # Number of sub-blocks (feature map outputs) for each supported teacher
        ch_dicts = {
            'wrn_40_2': [32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128],
            'resnet110': [16]*18 + [32]*18 + [64]*18,
            'resnet32x4': [64]*5 + [128]*5 + [256]*5,
            'vgg8':  [128, 256, 512, 512],
            'vgg11': [128, 256, 512, 512], 
            'vgg13': [128, 256, 512, 512],
            'vgg16': [128, 256, 512, 512],
            'vgg19': [128, 256, 512, 512]
        }
    elif opt.dataset == 'cifar100':
        n_cls = 100
        mean = (0.5071, 0.4867, 0.4408)
        std  = (0.2675, 0.2565, 0.2761)
        ch_dicts = {
            'wrn_40_2':  [32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128],
            'resnet110': [16]*18 + [32]*18 + [64]*18,
            'resnet32x4': [64]*5 + [128]*5 + [256]*5,
            'resnet56':  [16]*9 + [32]*9 + [64]*9,
            'vgg8':  [128, 256, 512, 512],
            'vgg11': [128, 256, 512, 512], 
            'vgg13': [128, 256, 512, 512],
            'vgg16': [128, 256, 512, 512],
            'vgg19': [128, 256, 512, 512]
        }
    else:
        raise ValueError(f"Unknown dataset: {opt.dataset}")

    if opt.model_t not in ch_dicts:
        raise NotImplementedError(f"Channel dict not defined for model: {opt.model_t}. Please add it to store_hints.py.")

    ch_dict = ch_dicts[opt.model_t]

    # Output path
    if opt.hints_path is None:
        model_basename = os.path.splitext(os.path.basename(opt.t_path))[0]
        suffix = '_preact' if opt.preact else ''
        opt.hints_path = os.path.join('./save', 'hints', model_basename + suffix)

    os.makedirs(opt.hints_path, exist_ok=True)

    print(f'==> Saving hint {opt.g} to: {opt.hints_path}')

    # Data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if opt.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=opt.data_folder, train=True, download=True, transform=transform)
    else:
        trainset = torchvision.datasets.CIFAR100(root=opt.data_folder, train=True, download=True, transform=transform)

    # Use shuffle=False so indices are deterministic; num_workers=0 on Windows (no fork)
    hint_workers = 0 if sys.platform == 'win32' else 4
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=hint_workers)

    # Load teacher
    t_net = load_teacher(opt.t_path, n_cls, opt.model_t)
    t_net.eval()
    if torch.cuda.is_available():
        t_net = t_net.cuda()

    # Extract hint for sub-block opt.g
    i = opt.g
    hint_name = os.path.join(opt.hints_path, f'hint{i}.pt')
    all_hints = []

    use_cuda = torch.cuda.is_available()

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(trainloader):
            # Sample 1/5 of all batches (~10,000 samples total), as the paper specifies N=10^4
            if batch_idx % 5 == 0:
                if use_cuda:
                    inputs = inputs.cuda()
                hints, _ = t_net(inputs, is_feat=True, preact=opt.preact)
                # Get the desired sub-block output, spatially average H×W dims -> (batch, C)
                hint_ = hints[int(i)]
                hint_ = torch.mean(hint_, dim=(2, 3)).cpu().detach().numpy()
                all_hints.append(hint_)

    hint_curr = np.concatenate(all_hints, axis=0)
    torch.save(hint_curr, hint_name)
    print(f'==> Saved hint {i}: shape={hint_curr.shape} -> {hint_name}')


if __name__ == '__main__':
    main()
