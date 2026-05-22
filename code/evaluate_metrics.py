"""
Evaluate teacher and student models: top-1/top-5 accuracy, parameters, FLOPs, latency.
Usage:
    python evaluate_metrics.py --dataset cifar10 \
        --model_t wrn_40_2 --path_t ./save/models/.../wrn_40_2_best.pth \
        --model_s wrn_16_2 --path_s_cp ./save/student_model/.../wrn_16_2_best_cp.pth \
                           --path_s_tucker ./save/student_model/.../wrn_16_2_best_tucker.pth
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import argparse
import sys
import json
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from thop import profile, clever_format
from models import model_dict
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.cifar10 import get_cifar10_dataloaders
from dataset.imagenet import get_imagenet_dataloader
from helper.loops import validate
from helper.util import AverageMeter, accuracy
from for_init import remove_module, add_module
from decomposition import decompose_model


def parse_option():
    parser = argparse.ArgumentParser('Evaluate teacher & student models')

    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar100', 'cifar10', 'imagenet100'],
                        help='dataset used for evaluation')
    parser.add_argument('--data_folder', type=str, default=None,
                        help='path to dataset root (required for imagenet100; must contain val/ subdir)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')

    # Teacher
    parser.add_argument('--model_t', type=str, default='wrn_40_2', help='teacher architecture')
    parser.add_argument('--path_t', type=str, required=True, help='path to teacher checkpoint')

    # Student (CP)
    parser.add_argument('--model_s', type=str, default='wrn_16_2', help='student architecture')
    parser.add_argument('--path_s_cp', type=str, default=None,
                        help='path to CP-decomposed student checkpoint')
    parser.add_argument('--cp_rank_ratio', type=float, default=0.5,
                        help='CP rank ratio used during training')

    # Student (Tucker) — optional
    parser.add_argument('--path_s_tucker', type=str, default=None,
                        help='path to Tucker-decomposed student checkpoint')
    parser.add_argument('--tucker_rank_ratio', type=float, default=0.5,
                        help='Tucker rank ratio used during training')
    parser.add_argument('--use_vbmf', action='store_true',
                        help='reconstruct decomposed students using teacher-VBMF ranks (must match training)')

    # PyTorch model compile optimization
    parser.add_argument('--torch_compile', action='store_true')

    # Skip decomposition (for KD / AT / FitNet / WSL baselines whose checkpoints
    # are plain undecomposed students, not CP- or Tucker-factorised weights)
    parser.add_argument('--no_decompose', action='store_true',
                        help='Load student as plain architecture without decomposition '
                             '(use for KD, AT, FitNet, WSL distillation methods)')

    # Output
    parser.add_argument('--save_path', type=str,
                        default=None,
                        help='where to save JSON results (default: same dir as --path_s_cp or --path_s_tucker)')

    opt = parser.parse_args()

    if opt.save_path is None:
        ref = opt.path_s_cp or opt.path_s_tucker
        if ref:
            opt.save_path = os.path.join(os.path.dirname(ref), 'evaluation_results.json')
        else:
            opt.save_path = './save/evaluation_results.json'

    return opt


def measure_model(model, dataset='cifar10', device='cpu'):
    """Measure parameter count, FLOPs, and inference latency."""
    model.eval()

    if 'imagenet' in dataset:
        input_size = (1, 3, 224, 224)
    else:
        input_size = (1, 3, 32, 32)

    data = torch.randn(input_size).to(device)
    model.to(device)

    # Params & FLOPs
    macs, params = profile(model, inputs=(data,), verbose=False)

    # Inference latency (warmup + timed)
    with torch.no_grad():
        for _ in range(10):
            _ = model(data)

    num_runs = 50
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(data)
    end = time.time()
    avg_latency = (end - start) / num_runs * 1000  # ms

    return params, macs, avg_latency


def load_checkpoint(model, checkpoint_path):
    """Load model weights from a checkpoint file, handling DataParallel prefixes."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt 
        
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        try:
            from for_init import remove_module, add_module
            model.load_state_dict(remove_module(state_dict))
        except RuntimeError:
            model.load_state_dict(add_module(state_dict))
    return model


def evaluate_model(model, model_name, val_loader, criterion, opt, device):
    """Run full evaluation on a single model: accuracy + efficiency metrics."""
    
    # --- 1. Efficiency (on CPU, UNCOMPILED) ---
    # We MUST run THOP on a clean, uncompiled CPU model, otherwise thop.profile will crash.
    model.cpu()
    model.eval()
    params, macs, latency = measure_model(model, dataset=opt.dataset, device='cpu')
    macs_str, params_str = clever_format([macs, params], "%.3f")

    # --- 2. Compile for Speed ---
    model.to(device)
    if opt.torch_compile and device == 'cuda':
        model = torch.compile(model, dynamic=True)
    model.eval()

    # --- 3. Accuracy Evaluation ---
    top1_acc, top5_acc, val_loss = validate(val_loader, model, criterion, opt)

    # Convert tensor to float if needed
    if hasattr(top1_acc, 'item'):
        top1_acc = top1_acc.item()
    if hasattr(top5_acc, 'item'):
        top5_acc = top5_acc.item()

    result = {
        'model': model_name,
        'top1_accuracy': round(top1_acc, 2),
        'top5_accuracy': round(top5_acc, 2),
        'val_loss': round(val_loss, 4),
        'parameters': params_str,
        'parameters_raw': int(params),
        'flops': macs_str,
        'flops_raw': int(macs),
        'latency_ms': round(latency, 2),
    }
    return result


def print_result_table(results):
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 80)
    print(f"{'MODEL EVALUATION RESULTS':^80}")
    print("=" * 80)

    header = f"{'Model':<30} {'Top-1 %':>8} {'Top-5 %':>8} {'Params':>10} {'FLOPs':>10} {'Lat.(ms)':>10}"
    print(header)
    print("-" * 80)

    for r in results:
        row = f"{r['model']:<30} {r['top1_accuracy']:>8.2f} {r['top5_accuracy']:>8.2f} {r['parameters']:>10} {r['flops']:>10} {r['latency_ms']:>10.2f}"
        print(row)

    print("=" * 80)


def evaluate():
    opt = parse_option()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Evaluating on {opt.dataset.upper()} | Device: {device}")

    # ----- Dataset -----
    if opt.dataset == 'cifar100':
        _, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                 num_workers=opt.num_workers)
        n_cls = 100
    elif opt.dataset == 'cifar10':
        _, val_loader = get_cifar10_dataloaders(batch_size=opt.batch_size,
                                                 num_workers=opt.num_workers)
        n_cls = 10
    elif opt.dataset == 'imagenet100':
        if opt.data_folder is None:
            raise ValueError("--data_folder is required for imagenet100 evaluation")
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_set = datasets.ImageFolder(os.path.join(opt.data_folder, 'val'), transform=val_transform)
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    results = []

    # ===== 1. Teacher Model =====
    print(f"\n--- Evaluating Teacher: {opt.model_t} ---")
    teacher = model_dict[opt.model_t](num_classes=n_cls)
    teacher = load_checkpoint(teacher, opt.path_t)
    if torch.cuda.is_available():
        teacher = teacher.cuda()

    teacher_result = evaluate_model(teacher, f"Teacher ({opt.model_t})",
                                     val_loader, criterion, opt, device)
    results.append(teacher_result)

    # ===== 2. Student Model — CP Decomposed (or plain for non-BSAT baselines) =====
    if opt.path_s_cp:
        if opt.no_decompose:
            print(f"\n--- Evaluating Student: {opt.model_s} ---")
            student_cp = model_dict[opt.model_s](num_classes=n_cls)
            student_label = f"Student ({opt.model_s})"
        else:
            print(f"\n--- Evaluating Student (CP): {opt.model_s} ---")
            student_cp = model_dict[opt.model_s](num_classes=n_cls)
            student_cp = decompose_model(student_cp, method='cp', cp_rank_ratio=opt.cp_rank_ratio,
                                         use_vbmf=opt.use_vbmf,
                                         teacher_model=teacher if opt.use_vbmf else None)
            rank_desc = "VBMF" if opt.use_vbmf else f"r={opt.cp_rank_ratio}"
            student_label = f"Student CP ({opt.model_s}, {rank_desc})"

        student_cp = load_checkpoint(student_cp, opt.path_s_cp)
        if torch.cuda.is_available():
            student_cp = student_cp.cuda()

        cp_result = evaluate_model(student_cp, student_label, val_loader, criterion, opt, device)
        results.append(cp_result)

    # ===== 3. Student Model — Tucker Decomposed =====
    if opt.path_s_tucker:
        print(f"\n--- Evaluating Student (Tucker): {opt.model_s} ---")
        student_tk = model_dict[opt.model_s](num_classes=n_cls)
        if not opt.no_decompose:
            student_tk = decompose_model(student_tk, method='tucker', tucker_rank_ratio=opt.tucker_rank_ratio,
                                         use_vbmf=opt.use_vbmf,
                                         teacher_model=teacher if opt.use_vbmf else None)
        student_tk = load_checkpoint(student_tk, opt.path_s_tucker)
        if torch.cuda.is_available():
            student_tk = student_tk.cuda()

        t_rank_desc = "VBMF" if opt.use_vbmf else f"r={opt.tucker_rank_ratio}"
        tk_result = evaluate_model(student_tk,
                                    f"Student Tucker ({opt.model_s}, {t_rank_desc})",
                                    val_loader, criterion, opt, device)
        results.append(tk_result)

    # ===== Print & Save =====
    print_result_table(results)

    # Compression ratios (relative to teacher)
    t_params = results[0]['parameters_raw']
    t_flops = results[0]['flops_raw']
    print("\nCompression Ratios (vs. Teacher):")
    for r in results[1:]:
        param_ratio = t_params / r['parameters_raw'] if r['parameters_raw'] > 0 else float('inf')
        flop_ratio = t_flops / r['flops_raw'] if r['flops_raw'] > 0 else float('inf')
        print(f"  {r['model']}: {param_ratio:.2f}x param reduction, {flop_ratio:.2f}x FLOP reduction")

    # Save JSON
    save_dir = os.path.dirname(opt.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(opt.save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {opt.save_path}")


if __name__ == "__main__":
    evaluate()