"""
Step 1 of the PURSUhInT pipeline (ImageNet):
Extracts N x C layer representations from sub-blocks of a pre-trained Teacher model.
Supports ImageNet with standard PyTorch DataLoader. Output is stored as hint{i}.pt files.

Usage (single hint):
    python store_hints_imagenet.py --g 5 --model_t ResNet34 \
        --t_path save/models/ResNet34_imagenet/ResNet34_333f7ec4.pth \
        --data_folder /path/to/imagenet

Called in a loop from the run script to extract all hints:
    for i in {1..16}; do
        python store_hints_imagenet.py --g $i --model_t ResNet34 --t_path $TEACHER_PATH ...
    done
"""

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

import torch
import os
import argparse
import numpy as np

from models import model_dict
from for_init import remove_module


def parse_args():
    parser = argparse.ArgumentParser(
        description='Store teacher layer representations for PURSUhInT clustering (ImageNet).')
    parser.add_argument('--g', type=int, required=True,
                        help='Sub-block index to extract (1-indexed).')
    parser.add_argument('--t_path', type=str, required=True,
                        help='Path to the pre-trained teacher model checkpoint.')
    parser.add_argument('--model_t', type=str, default='ResNet34',
                        choices=['ResNet18', 'ResNet34'],
                        help='Teacher model architecture name.')
    parser.add_argument('--preact', type=bool, default=False,
                        help='Extract pre-activation hints.')
    parser.add_argument('--hints_path', type=str, default=None,
                        help='Output directory for hint .pt files. '
                             'Defaults to ./save/hints/<model_basename>.')
    parser.add_argument('--data_folder', type=str, default='data/imagenet100',
                        help='Path to the ImageNet dataset root directory.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for data loading.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers.')
    return parser.parse_args()


def load_teacher(model_path, n_cls, model_t='ResNet34'):
    """Load a pre-trained teacher model, handling various checkpoint formats."""
    print('==> Loading teacher model for hint extraction...')
    model = model_dict[model_t](num_classes=n_cls)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats:
    #   1) raw state_dict  2) {'model': state_dict}  3) either with 'module.' prefix
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state = checkpoint['model']
    else:
        state = checkpoint

    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(remove_module(state))

    print('==> Done.')
    return model


def get_imagenet_dataloader(data_folder, batch_size=32, num_workers=4):
    """Standard PyTorch DataLoader for ImageNet training set (no augmentation beyond resize)."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_folder = os.path.join(data_folder, 'train')
    train_set = datasets.ImageFolder(train_folder, transform=train_transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True)
    return train_loader


def main():
    opt = parse_args()
    n_cls = 100  # ImageNet100

    # ---- Per-model channel dictionaries ----
    # These map sub-block index i (1-indexed) to the channel count of feats[i]
    # from resnetv2.py forward: feats = [f0] + f1_act + f2_act + f3_act + f4_act + [f5]
    # Only BasicBlock-based models (ResNet18/34) support feature collection.
    ch_dicts = {
        # ResNet18: [2,2,2,2] BasicBlocks -> 8 sub-blocks
        'ResNet18': [64, 64, 128, 128, 256, 256, 512, 512],
        # ResNet34: [3,4,6,3] BasicBlocks -> 16 sub-blocks
        'ResNet34': [64, 64, 64, 128, 128, 128, 128,
                     256, 256, 256, 256, 256, 256, 512, 512, 512],
    }

    if opt.model_t not in ch_dicts:
        raise NotImplementedError(
            f"Channel dict not defined for model: {opt.model_t}. "
            f"Supported ImageNet teachers: {list(ch_dicts.keys())}. "
            f"Please add it to store_hints_imagenet100.py."
        )

    ch_dict = ch_dicts[opt.model_t]

    # Validate sub-block index
    num_layers = len(ch_dict)
    if opt.g < 1 or opt.g > num_layers:
        raise ValueError(
            f"Sub-block index --g={opt.g} out of range [1, {num_layers}] "
            f"for model {opt.model_t}.")

    # ---- Output path ----
    if opt.hints_path is None:
        model_basename = os.path.splitext(os.path.basename(opt.t_path))[0]
        suffix = '_preact' if opt.preact else ''
        opt.hints_path = os.path.join('./save', 'hints', model_basename + suffix)

    os.makedirs(opt.hints_path, exist_ok=True)
    print(f'==> Saving hint {opt.g} to: {opt.hints_path}')

    # ---- Data loader ----
    trainloader = get_imagenet_dataloader(data_folder=opt.data_folder,
                                          batch_size=opt.batch_size,
                                          num_workers=opt.num_workers)

    # ---- Load teacher ----
    t_net = load_teacher(opt.t_path, n_cls, opt.model_t)
    t_net.eval()
    if torch.cuda.is_available():
        t_net = t_net.cuda()

    # ---- Extract hint for sub-block opt.g ----
    i = opt.g
    hint_name = os.path.join(opt.hints_path, f'hint{i}.pt')
    all_hints = []

    use_cuda = torch.cuda.is_available()

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(trainloader):
            # Sample 1/12 of all batches (~10k samples total for ImageNet100)
            if batch_idx % 12 == 0:
                if use_cuda:
                    inputs = inputs.cuda()
                # hint_points=str(i) makes model return [feats[i]]
                hints, _ = t_net(inputs, is_feat=True, preact=opt.preact,
                                 hint_points=str(i))
                # Spatially average H x W dims -> (batch, C)
                hint_ = hints[0]
                hint_ = torch.mean(hint_, dim=(2, 3)).cpu().detach().numpy()
                all_hints.append(hint_)

                del inputs, hint_

    hint_curr = np.concatenate(all_hints, axis=0)
    torch.save(hint_curr, hint_name)
    print(f'==> Saved hint {i}: shape={hint_curr.shape} -> {hint_name}')


if __name__ == '__main__':
    main()
