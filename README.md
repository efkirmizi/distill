# Joint Knowledge Distillation and Tensor-Based Compression of Deep Neural Networks

An extension of the **PURSUhInT** knowledge distillation framework that introduces a novel **Coupled Matrix-Tensor Factorization (CMTF) distillation loss** and integrates **CP and Tucker tensor decomposition** directly into the student training pipeline. Two structurally different compressed student models — one CP-decomposed, one Tucker-decomposed — are trained simultaneously under a shared teacher, enforcing a common semantic latent space via bidirectional batch-subspace coupling.

Supports **CIFAR-10**, **CIFAR-100**, and **ImageNet-100**.

---

## Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Quick Start](#quick-start)
  - [CIFAR Full Pipeline](#cifar-full-pipeline)
  - [ImageNet-100 Full Pipeline](#imagenet-100-full-pipeline)
  - [Manual Step-by-Step](#manual-step-by-step)
- [Method Details](#method-details)
  - [PURSUhInT Hint Selection](#pursuhint-hint-selection)
  - [Tensor Decomposition](#tensor-decomposition)
  - [CMTF Loss](#cmtf-loss)
  - [Dual-Student Training](#dual-student-training)
  - [Dynamic Loss Weighting](#dynamic-loss-weighting)
- [Supported Models](#supported-models)
- [Supported Distillation Methods](#supported-distillation-methods)
- [Training Arguments](#training-arguments)
  - [train_teacher.py (CIFAR)](#train_teacherpy-cifar)
  - [train_teacher_imagenet100.py](#train_teacher_imagenet100py)
  - [train_student.py (CIFAR)](#train_studentpy-cifar)
  - [train_stu_imagenet100.py (ImageNet-100, DDP)](#train_stu_imagenet100py-imagenet-100-ddp)
- [Output Structure](#output-structure)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)

---

## Overview

This work builds on **PURSUhInT** — a prior method for automatic hint-point selection in knowledge distillation — and extends it with three original contributions:

1. **What to distill into**: The student model's convolutional layers are structurally compressed via CP or Tucker tensor decomposition *before* training begins, not as a post-processing step. The decomposed model is trained end-to-end from scratch.

2. **How to distill**: A novel CMTF loss aligns the student's batch subspace with the teacher's via truncated SVD projectors, combining spatial attention matching with sign-invariant subspace alignment.

3. **Dual-student coupling**: A CP-decomposed and a Tucker-decomposed student are trained simultaneously under the same teacher, with a bidirectional coupling term that forces them to share a common semantic latent space.

PURSUhInT itself provides the hint-point selection mechanism: it clusters all teacher layers using representation similarity (R²-CCA or CKA) and identifies the most representative layers automatically, eliminating the need for manual layer selection. This component is used as-is from the base method.

---

## Key Contributions

> **Note**: PURSUhInT (automatic hint-point selection via representation clustering) is prior work. The contributions of this project are the CMTF loss, CP/Tucker decomposition integration, dual-student training, VBMF-guided rank selection, and the bidirectional coupling mechanism described below.

### Base Method: PURSUhInT
PURSUhInT selects which teacher layers to use as distillation targets by running K-Means clustering over the teacher's layer representations, using R²-CCA or CKA as the distance measure. This project uses PURSUhInT's hint-point selection pipeline directly and builds the CMTF loss and decomposition framework on top of it.

### CP Decomposition
Each eligible `Conv2d(out, in, kH, kW)` is replaced by four sequential operations:

```
x → pointwise_in(in→R, 1×1) → depthwise_h(R→R, kH×1) → depthwise_w(R→R, 1×kW) → pointwise_out(R→out, 1×1)
```

The rank `R` controls the compression ratio. Ranks are either set as a fixed global fraction (`--cp_rank_ratio`) or estimated per-layer by running Empirical VBMF on the corresponding teacher layer's weights and scaling the resulting fraction to the student's dimensions.

### Tucker Decomposition
Each eligible `Conv2d` is replaced by three operations:

```
x → pointwise_in(in→rank_in, 1×1) → core_conv(rank_in→rank_out, kH×kW) → pointwise_out(rank_out→out, 1×1)
```

The spatial kernel is preserved in the core convolution, which makes Tucker compression particularly effective for layers where the dominant cost is the spatial filtering rather than the channel mixing.

### CMTF Loss — Coupled Matrix-Tensor Factorization
A two-part loss applied at each PURSUhInT hint point:

**Part 1 — Spatial Attention Matching**: Squares each feature map element, averages across the channel dimension, L2-normalizes, and computes MSE between the student and teacher maps. This is a standard AT-style loss that is sign-invariant.

**Part 2 — Batch-Subspace Alignment**: Reshapes the feature batch `B×C×H×W` into `B×(C·H·W)`, computes the top-R left singular vectors `U` via `torch.svd_lowrank`, and forms the orthogonal projector `P = U Uᵀ` (a `B×B` matrix). MSE between `P_student` and `P_teacher` aligns the student's batch-mode subspace with the teacher's. Because the projector is unique (up to sign in the full SVD, but projectors are sign-invariant), gradients flow cleanly through the SVD.

**Coupling Term**: When two students are trained together, the CP student's projectors are detached and passed to the Tucker student as an additional regression target. This enforces that the two architecturally different students compress the input into the same semantic subspace.

### VBMF-Guided Rank Selection
Empirical Variational Bayes Matrix Factorization (Nakajima et al., JMLR 2013) estimates the intrinsic rank of a weight matrix by thresholding singular values against the Marchenko-Pastur noise distribution. When applied to teacher weights, it captures how much capacity each teacher layer actually uses. That fraction is then transferred to the student's differently-sized layer via positional interpolation across the layer list.

---

## Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│  [1/5] Train Teacher                                             │
│         SGD + step-decay LR (+ warmup on ImageNet)              │
│         → save/models/{model_name}/{model}_best.pth             │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│  [2/5] PURSUhInT Step 1: Extract Layer Representations          │
│         Forward teacher on training set, spatial-average each    │
│         layer's activations → hint{i}.pt  (N × C arrays)        │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│  [3/5] PURSUhInT Step 2: K-Means Clustering                    │
│         Distance metric: 1 − R²_CCA  (or CKA)                  │
│         K centroids → selected hint points                       │
│         → {model}_{K}clusters_{metric}_centroids.txt            │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│  [4/5] Train Dual Students (CP + Tucker)                        │
│         Both students share teacher; per-batch bidirectional     │
│         coupling via CMTF projectors                             │
│         Loss = γ·CE  +  α·KLDiv  +  β·CMTF                     │
│         CIFAR: teacher outputs cached to RAM (if sufficient)     │
│         → {model}_best_cp.pth, {model}_best_tucker.pth          │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│  [5/5] Evaluate                                                  │
│         Top-1/5 accuracy, parameters, FLOPs, latency            │
│         Compression ratios vs. teacher                           │
│         → evaluation_results.json                               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Installation

**Python ≥ 3.9**, **PyTorch ≥ 2.0**, CUDA recommended.

```bash
pip install torch torchvision
pip install tensorly
pip install thop
pip install scipy numpy pillow tqdm tensorboard tensorboard_logger psutil
pip install kaggle
```

Optional — NVIDIA DALI (faster GPU data loading for ImageNet student training):
```bash
pip install nvidia-dali-cuda120   # match your CUDA version
```

Optional — NVIDIA Apex (mixed-precision and sync-BN for ImageNet DDP training):
```bash
# Follow instructions at https://github.com/NVIDIA/apex
```

---

## Dataset Preparation

### CIFAR-10 / CIFAR-100

Downloaded automatically on first run via `torchvision.datasets`. Data is saved to `./data/`.

### ImageNet-100

ImageNet-100 is a 100-class subset of ImageNet available on Kaggle. A helper script downloads and restructures it:

```bash
# Requires ~/.kaggle/kaggle.json (Kaggle API key)
bash code/prepare_imagenet100.sh
```

This creates `./data/imagenet100/train/<class>/` and `./data/imagenet100/val/<class>/`.

Alternatively, create the directory structure manually from your own ImageNet copy by selecting 100 classes.

---

## Quick Start

### CIFAR Full Pipeline

Edit the configuration block at the top of the script, then run:

```bash
cd code
bash run_pipeline_cifar.sh
```

Key settings in `run_pipeline_cifar.sh`:

| Variable | Default | Description |
|---|---|---|
| `MODEL_T` | `wrn_40_2` | Teacher architecture |
| `MODEL_S` | `wrn_16_2` | Student architecture |
| `DATASET` | `cifar100` | `cifar10` or `cifar100` |
| `NUM_LAYERS` | `18` | Total teacher layers to extract hints for |
| `NUM_CLUSTERS` | `3` | Number of hint points to select (K) |
| `METRIC` | `r2` | Clustering distance: `r2`, `cka_linear`, `cka_rbf` |
| `TEACHER_EPOCHS` | `240` | Epochs for teacher training |
| `STUDENT_EPOCHS` | `240` | Epochs for student training |
| `BATCH` | `64` | Batch size |
| `LR` | `0.05` | Initial learning rate |
| `CP_RANK_RATIO` | `0.5` | CP global rank ratio (ignored when `USE_VBMF=1`) |
| `TUCKER_RANK_RATIO` | `0.25` | Tucker global rank ratio (ignored when `USE_VBMF=1`) |
| `CMTF_RANK` | `8` | SVD rank R for batch-subspace projectors |
| `CMTF_COUPLING_WEIGHT` | `1.0` | Weight of Tucker←CP coupling term |
| `USE_VBMF` | `1` | `1` = VBMF ranks from teacher, `0` = fixed ratios |
| `ENABLE_TORCH_COMPILE` | `1` | `torch.compile` for faster training |
| `ENABLE_DYNAMIC_LOSS_WEIGHTS` | `1` | Kendall et al. uncertainty weighting |
| `GAMMA / ALPHA / BETA` | `1.0 / 4.0 / 25.0` | CE / KL-div / CMTF loss weights |

Individual pipeline stages can be skipped by setting flags to `0`:

```bash
RUN_TEACHER=1      # teacher training
RUN_HINTS=1        # hint extraction
RUN_CLUSTERING=1   # clustering
RUN_TRAINING=1     # train students
RUN_EVALUATION=1   # run evaluation
```

### ImageNet-100 Full Pipeline

```bash
cd code
bash run_pipeline_imagenet.sh
```

Key differences from CIFAR:
- Student training uses `torchrun --nproc_per_node=N` (DDP). Set `NUM_GPUS` in the script.
- `IMAGENET_DIR` must point to the directory containing `train/` and `val/`.
- `NUM_LAYERS=16` (ResNet34 has 16 BasicBlock sub-blocks).
- `NUM_CLUSTERS=4` hint points.
- DALI GPU data pipeline is used by default; disable with `USE_NO_DALI=1` for local debugging.
- LR is automatically scaled by `batch_size × world_size / 256`.

### Manual Step-by-Step

**Step 1 — Train Teacher (CIFAR)**
```bash
python train_teacher.py \
    --model wrn_40_2 \
    --dataset cifar100 \
    --epochs 240 \
    --learning_rate 0.05 \
    --lr_decay_epochs 150,180,210 \
    --batch_size 64 \
    --trial 0
```

**Step 1 — Train Teacher (ImageNet-100)**
```bash
python train_teacher_imagenet100.py \
    --model ResNet34 \
    --data_folder ./data/imagenet100 \
    --epochs 100 \
    --learning_rate 0.1 \
    --lr_decay_epochs 30,60,90 \
    --batch_size 256 \
    --warmup_epochs 5 \
    --trial 0
```

**Step 2 — Extract Layer Representations (CIFAR)**
```bash
for i in $(seq 1 18); do
    python store_hints.py \
        --g $i \
        --model_t wrn_40_2 \
        --t_path ./save/models/wrn_40_2_cifar100_lr_0.05_decay_0.0005_trial_0/wrn_40_2_best.pth \
        --dataset cifar100 \
        --hints_path ./save/hints/wrn_40_2_cifar100_best
done
```

**Step 3 — Cluster and Select Hint Points**
```bash
python k_means.py \
    --hints_dir ./save/hints/wrn_40_2_cifar100_best \
    --num_clusters 3 \
    --num_layers 18 \
    --metric_name r2 \
    --output_dir ./save/hints \
    --model_name wrn_40_2
# Output: ./save/hints/wrn_40_2_3clusters_r2_centroids.txt  (e.g. "5,10,16")
```

**Step 4 — Train Dual Students (CIFAR)**
```bash
python train_student.py \
    --dataset cifar100 \
    --model_s wrn_16_2 \
    --model_t wrn_40_2 \
    --path_t ./save/models/wrn_40_2_cifar100_lr_0.05_decay_0.0005_trial_0/wrn_40_2_best.pth \
    --distill pursuhint_cmtf \
    --dual_cmtf \
    --hint_points 5,10,16 \
    --cp_rank_ratio 0.5 \
    --tucker_rank_ratio 0.25 \
    --use_vbmf \
    --cmtf_rank 8 \
    --cmtf_coupling_weight 1.0 \
    --epochs 240 \
    --learning_rate 0.05 \
    --lr_decay_epochs 150,180,210 \
    --batch_size 64 \
    -r 1.0 -a 4.0 -b 25.0 \
    --dynamic_loss_weights \
    --torch_compile
```

**Step 4 — Train Dual Students (ImageNet-100)**
```bash
torchrun --nproc_per_node=1 --master_port 9200 train_stu_imagenet100.py \
    --model_s ResNet18 \
    --model_t ResNet34 \
    --path_t ./save/models/ResNet34_imagenet100_lr_0.1_decay_0.0001_trial_0/ResNet34_best.pth \
    --distill pursuhint_cmtf \
    --dual_cmtf \
    --hint_points 3,7,13,16 \
    --cp_rank_ratio 0.5 \
    --tucker_rank_ratio 0.25 \
    --use_vbmf \
    --cmtf_rank 8 \
    --epochs 100 \
    --lr 0.1 \
    --weight-decay 1e-4 \
    --batch-size 256 \
    --workers 8 \
    --gamma 1.0 --alpha 4.0 --beta 25.0 \
    --dynamic_loss_weights \
    ./data/imagenet100
```

**Step 5 — Evaluate**
```bash
python evaluate_metrics.py \
    --dataset cifar100 \
    --model_t wrn_40_2 \
    --path_t ./save/models/wrn_40_2_cifar100_lr_0.05_decay_0.0005_trial_0/wrn_40_2_best.pth \
    --model_s wrn_16_2 \
    --path_s_cp  ./save/student_model/cifar100/5,10,16/S-wrn_16_2_T-wrn_40_2_cifar100_pursuhint_cmtf_.../wrn_16_2_best_cp.pth \
    --path_s_tucker ./save/student_model/cifar100/5,10,16/S-wrn_16_2_T-wrn_40_2_cifar100_pursuhint_cmtf_.../wrn_16_2_best_tucker.pth \
    --use_vbmf
```

---

## Method Details

### PURSUhInT Hint Selection (Prior Work)

This stage is adopted from PURSUhInT. It is used unchanged as the hint-point selection front-end for the CMTF distillation pipeline described below.

**Representation extraction** (`store_hints.py` / `store_hints_imagenet100.py`):  
For each teacher layer `i`, the script runs the teacher on a random subset of the training set (1/5 on CIFAR, 1/12 on ImageNet) with `is_feat=True` to collect intermediate feature maps. Spatial dimensions are collapsed by average pooling, giving an `N × C` matrix. This is saved as `hint{i}.pt`.

**Clustering** (`k_means.py`):  
Pairwise distances between all layer representations are computed using one of three metrics:

| Metric | Formula | Interpretation |
|---|---|---|
| `r2` | `1 − R²_CCA(X_i, X_j)` | 1 minus the proportion of variance explained by the best linear mapping between representations |
| `cka_linear` | `1 − CKA(X_i X_iᵀ, X_j X_jᵀ)` | 1 minus centered kernel alignment with linear kernel |
| `cka_rbf` | `1 − CKA(K_rbf(X_i), K_rbf(X_j))` | 1 minus CKA with RBF kernel |

K-Means runs on the layer index space, using these distances as the dissimilarity measure. Initialization uses the first, middle, and last layers. The centroid of each cluster is the layer whose index is closest to the cluster's mean index.

The resulting K centroid indices (1-indexed) are written to `{model}_{K}clusters_{metric}_centroids.txt` and passed as `--hint_points` to the student training script.

### Tensor Decomposition

Decomposition is applied to all `Conv2d` layers that are not 1×1 convolutions and are not depthwise/grouped convolutions. The architecture is replaced in-place before training starts and the decomposed model is trained from scratch (the initial PARAFAC/Tucker factors are discarded; only the layer structure is kept).

**CP Decomposition** decomposes a `(out, in, kH, kW)` weight tensor into four factor matrices. The convolution is replaced by four sequential `Conv2d` operations totaling `R(in + kH + kW + out)` parameters instead of `in · out · kH · kW`.

**Tucker Decomposition** decomposes into a compact core and four factor matrices. The implementation contracts the spatial factor matrices into the core before training, leaving three `Conv2d` operations with `rank_in · in + rank_in · rank_out · kH · kW + rank_out · out` parameters.

**VBMF rank selection** (`--use_vbmf`): For each student layer `i`, the corresponding teacher layer `j = round(i × (T−1) / (S−1))` is identified via positional interpolation. EVBMF is run on the teacher layer's mode-0 (output channel) unfolding to estimate its intrinsic rank. That rank is expressed as a fraction of the teacher's channel count and scaled to the student's channel dimensions. This transfers the teacher's capacity utilization pattern, layer by layer, to the structurally smaller student.

### CMTF Loss

Applied at each PURSUhInT hint point. The total loss per batch is:

```
L_total = γ · L_CE  +  α · L_KLDiv  +  β · L_CMTF
```

`L_CMTF` sums contributions from all hint points, each consisting of two terms:

```
L_CMTF = Σ_k [ MSE(att_s^k, att_t^k)  +  MSE(P_s^k, P_t^k) ]
```

where `att^k = normalize(feat^k.pow(2).mean(channels))` is the spatial attention map and `P^k = U_k U_k^T` is the rank-R orthogonal projector of the batch-unfolded feature matrix.

The teacher's projectors are computed with `detach()` so no gradient flows back into the teacher. The student's projectors are returned for use as coupling targets.

**Coupling** (dual-student mode): After the CP student's forward pass, its projectors `{P_cp^k}` are detached and supplied to the Tucker student. Both students receive coupling losses symmetrically:

```
L_kd_cp     += coupling_weight · Σ_k MSE(P_cp^k, P_tucker^k.detach())
L_kd_tucker += coupling_weight · Σ_k MSE(P_tucker^k, P_cp^k.detach())
```

If student and teacher feature maps have different spatial dimensions, `adaptive_avg_pool2d` is applied to the student before computing the attention map.

### Dual-Student Training

Both students are instantiated, decomposed, and optimized within a single training loop. They share:
- The same teacher (cached to RAM on CIFAR, on-the-fly on ImageNet)
- The same data loader and batch
- The same loss hyperparameters (γ, α, β)

They maintain separate:
- Optimizers and learning rate schedulers
- GradScaler (AMP) instances
- Best-checkpoint tracking (top-1 accuracy)
- Training log CSV files and TensorBoard event files

On CIFAR, teacher logits and hint-point features are precomputed over the entire training set into RAM before the first epoch. The precompute step first probes how much RAM the cache would need; if it exceeds 75% of available memory, the teacher runs on-the-fly per batch instead.

**Resuming**: pass `--resume /path/to/student_save_dir` to continue from the most recent `checkpoint_cp.pth` and `checkpoint_tucker.pth` written at the end of each epoch. If the CP and Tucker checkpoints are at different epochs (e.g. due to a crash between the two saves), both are rolled back to the earlier epoch automatically.

### Dynamic Loss Weighting

With `--dynamic_loss_weights`, the three loss terms (CE, KLDiv, CMTF) are combined using Kendall et al.'s homoscedastic uncertainty method (CVPR 2018). A learnable log-variance parameter `s_i` is maintained for each loss component:

```
L_total = Σ_i [ exp(−s_i) · L_i  +  s_i ]
```

`exp(−s_i)` acts as a learned inverse weight; `s_i` is the regularization term. All `s_i` are initialized to 0 (effective weight 1). They are placed in a separate parameter group with no weight decay to prevent them from being driven to infinity. When this flag is active, the fixed `--gamma/--alpha/--beta` weights are ignored.

---

## Supported Models

### CIFAR Architectures

| Model | Description | Feature List Length |
|---|---|---|
| `resnet8` | 3-stage, n=1 | 4 |
| `resnet14` | 3-stage, n=2 | 7 |
| `resnet20` | 3-stage, n=3 | 10 |
| `resnet32` | 3-stage, n=5 | 16 |
| `resnet44` | 3-stage, n=7 | 22 |
| `resnet56` | 3-stage, n=9 | 28 |
| `resnet110` | 3-stage, n=18 | 55 |
| `resnet8x4` | Wide variant of resnet8 | 4 |
| `resnet32x4` | Wide variant of resnet32 | 16 |
| `wrn_16_1` | WideResNet depth=16, width=1 | 8 |
| `wrn_16_2` | WideResNet depth=16, width=2 | 8 |
| `wrn_40_1` | WideResNet depth=40, width=1 | 20 |
| `wrn_40_2` | WideResNet depth=40, width=2 | 20 |
| `vgg8` through `vgg19` | VGG with batch norm | 6 (fixed) |
| `MobileNetV2` | MobileNet V2 | 6 (fixed) |
| `mobile_half` | MobileNet V2 half-width | 6 (fixed) |
| `ShuffleV1` | ShuffleNet V1 (4-8-4 blocks) | 18 |
| `ShuffleV2` | ShuffleNet V2 | 5 (fixed) |
| `ResNet50` | Standard ResNet50 | 18 |

**Teacher-only CIFAR models** (available in `train_teacher.py` but not `train_student.py`):  
`resnet8`, `resnet14`, `resnet20`, `resnet32`, `resnet44`, `resnet56`, `resnet110`, `wrn_40_2`, `wrn_16_2`, `vgg8` through `vgg19`.

### ImageNet Architectures

| Model | Description | Default `s_points` |
|---|---|---|
| `ResNet18` | Standard ResNet18 | `2,4,6,8` |
| `ResNet34` | Standard ResNet34 | `3,7,13,16` |

> `ResNet50` and `ResNet101` are present in `models/resnetv2.py` but are not wired into `train_stu_imagenet100.py`'s argument choices.

All models accept `forward(x, is_feat=True, preact=False)` to return intermediate feature maps alongside logits.

### Default Student Hint Points (`s_points`)

When `--s_points` is not passed, the student training scripts auto-detect the correct layer indices per architecture. To override, pass `--s_points i,j,k` (must have the same count as `--hint_points`).

---

## Supported Distillation Methods

| `--distill` | CIFAR | ImageNet | Description |
|---|---|---|---|
| `kd` | ✓ | ✓ | Hinton et al. KL-divergence on softened logits |
| `hint` | ✓ | ✓ | FitNet intermediate feature regression with ConvReg adapters |
| `attention` | ✓ | ✓ | Attention Transfer — spatial attention map MSE |
| `vid` | ✓ | ✓ | Variational Information Distillation — per-channel Gaussian NLL |
| `WSL_att` | ✓ | ✓ | Weak Soft Labels combined with Attention Transfer |
| `crd` | ✓ | ✗ | Contrastive Representation Distillation — NCE on feature embeddings |
| `WSL_crd` | ✓ | ✗ | Weak Soft Labels combined with CRD |
| `ATT_crd` | ✓ | ✗ | Attention Transfer combined with CRD |
| `pursuhint_cmtf` | ✓ | ✓ | **This work** — PURSUhInT hint selection + CMTF loss |

CRD-based methods (`crd`, `WSL_crd`, `ATT_crd`) are **not supported on ImageNet** because they require per-sample contrastive indices that are unavailable from the DALI pipeline, and ImageNet-scale precomputation of the memory bank is impractical.

---

## Training Arguments

### `train_teacher.py` (CIFAR)

Trains a teacher model on CIFAR-10 or CIFAR-100 with standard SGD and step-decay LR.

| Argument | Default | Description |
|---|---|---|
| `--model` | `resnet110` | Architecture: `resnet8/14/20/32/44/56/110`, `wrn_40_2`, `wrn_16_2`, `vgg8/11/13/16/19` |
| `--dataset` | `cifar10` | `cifar10` or `cifar100` |
| `--epochs` | `240` | Total training epochs |
| `--learning_rate` | `0.05` | Initial learning rate |
| `--lr_decay_epochs` | `150,180,210` | Epochs at which LR is multiplied by `lr_decay_rate` |
| `--lr_decay_rate` | `0.1` | LR decay factor |
| `--weight_decay` | `5e-4` | SGD weight decay |
| `--momentum` | `0.9` | SGD momentum |
| `--batch_size` | `64` | Batch size |
| `--num_workers` | `8` | DataLoader worker count |
| `--trial` | `0` | Experiment index (used in output directory naming) |
| `--torch_compile` | off | Apply `torch.compile` for faster training |
| `--print_freq` | `100` | Batch log interval |

**Output**: `./save/models/{model}_{dataset}_lr_{lr}_decay_{wd}_trial_{n}/`  
Saves `{model}_best.pth`, `{model}_last.pth`, `training_log.csv`, `experiment_summary.json`.

---

### `train_teacher_imagenet100.py`

Trains a teacher model on ImageNet-100 with linear warmup followed by step-decay LR. Uses AMP (`torch.amp.GradScaler`) and optionally `DataParallel` on multi-GPU machines.

| Argument | Default | Description |
|---|---|---|
| `--model` | `ResNet34` | Architecture: `ResNet18` or `ResNet34` |
| `--data_folder` | — (required) | Path to ImageNet-100 root with `train/` and `val/` subdirs |
| `--epochs` | `100` | Total training epochs |
| `--learning_rate` | `0.1` | Initial learning rate |
| `--lr_decay_epochs` | `30,60,90` | Epoch milestones for step-decay |
| `--lr_decay_rate` | `0.1` | LR decay factor |
| `--weight_decay` | `1e-4` | SGD weight decay |
| `--momentum` | `0.9` | SGD momentum |
| `--warmup_epochs` | `5` | Linear warmup epochs (set to `0` to disable) |
| `--batch_size` | `256` | Batch size |
| `--num_workers` | `8` | DataLoader worker count |
| `--trial` | `0` | Experiment index |
| `--torch_compile` | off | Apply `torch.compile` (compiled before `DataParallel`) |
| `--print_freq` | `100` | Batch log interval |

**Output**: `./save/models/{model}_imagenet100_lr_{lr}_decay_{wd}_trial_{n}/`  
Saves `{model}_best.pth`, `{model}_last.pth`, `training_log.csv`, `experiment_summary.json`.

---

### `train_student.py` (CIFAR)

Main student training script for CIFAR-10/100. Supports all distillation methods. Uses `nn.DataParallel` for multi-GPU, AMP (`torch.amp.GradScaler`), and LR warmup (5 epochs). Teacher outputs and hint-point features are precomputed into RAM before training begins (with automatic fallback to on-the-fly inference if RAM is insufficient).

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `cifar100` | `cifar10` or `cifar100` |
| `--model_s` | `resnet8` | Student architecture |
| `--model_t` | — | Teacher architecture (inferred from checkpoint path if not set) |
| `--path_t` | — | Path to teacher `.pth` checkpoint |
| `--path_s` | — | Optional: path to a pre-trained student checkpoint to load |
| `--distill` | `kd` | Distillation method (see table above) |
| `--dual_cmtf` | off | Train CP + Tucker students simultaneously (requires `pursuhint_cmtf`) |
| `--hint_points` | `15,37,53` | Teacher layer indices for distillation (comma-separated, 0-indexed) |
| `--s_points` | auto | Student layer indices; auto-detected per architecture if not specified |
| `--preact` | off | Use pre-activation features (post-BN, pre-ReLU) |
| `--cp_rank_ratio` | `0.5` | CP global rank as fraction of `max(out, in)` channels |
| `--tucker_rank_ratio` | `0.5` | Tucker rank as fraction of channel count |
| `--use_vbmf` | off | Estimate per-layer ranks from teacher weights via EVBMF |
| `--cmtf_rank` | `8` | SVD truncation rank R for batch-subspace projectors |
| `--cmtf_coupling_weight` | `1.0` | Weight of the Tucker←CP coupling term in dual CMTF mode |
| `--epochs` | `240` | Training epochs |
| `--start_epoch` | `1` | Starting epoch (set automatically when resuming) |
| `--batch_size` | `64` | Batch size |
| `--learning_rate` | `0.05` | Initial SGD LR (`0.01` for MobileNetV2, ShuffleV1, ShuffleV2) |
| `--lr_decay_epochs` | `150,180,210` | Epochs at which LR is stepped by `lr_decay_rate` |
| `--lr_decay_rate` | `0.1` | LR decay factor |
| `--weight_decay` | `5e-4` | SGD weight decay |
| `--momentum` | `0.9` | SGD momentum |
| `-r / --gamma` | `1.0` | Cross-entropy loss weight |
| `-a / --alpha` | `None` | KL-divergence loss weight (must be set explicitly) |
| `-b / --beta` | `None` | Distillation loss weight (must be set explicitly) |
| `--beta2` | `1.0` | Balance weight between AT and CRD in `ATT_crd` mode |
| `--kd_T` | `4.0` | Softmax temperature for KL-divergence |
| `--dynamic_loss_weights` | off | Kendall et al. homoscedastic uncertainty weighting (overrides γ/α/β) |
| `--torch_compile` | off | Apply `torch.compile` (single-GPU only; skipped on multi-GPU DataParallel) |
| `--resume` | — | Path to a previous student save directory to resume from |
| `--init_epochs` | `30` | Pretraining epochs for two-stage hint methods (FitNet) |
| `--trial` | `1` | Trial index for output directory naming |
| `--num_workers` | `8` | DataLoader worker count |
| `--print_freq` | `100` | Batch log interval |
| `--feat_dim` | `128` | CRD embedding dimension |
| `--nce_k` | `16384` | Number of negative samples for NCE (CRD) |
| `--nce_t` | `0.07` | NCE temperature (auto-set to `0.1` for CIFAR) |
| `--nce_m` | `0.5` | Momentum for non-parametric updates (CRD) |
| `--mode` | `exact` | CRD sampling mode: `exact` or `relax` |

**Output directory**: `./save/student_model/{dataset}/{hint_points}/S-{model_s}_T-{model_t}_{dataset}_{distill}_r-{γ}_a-{α}_b-{β}_b2-{β2}_{trial}/`

---

### `train_stu_imagenet100.py` (ImageNet-100, DDP)

Student training script for ImageNet-100. Launched via `torchrun`. Supports DDP via `torch.distributed` (NCCL backend) and optionally Apex for sync-BN and AMP. DALI GPU pipeline is the default data backend; fall back to standard PyTorch DataLoader with `--no_dali`. Teacher runs on-the-fly per batch (no caching).

**LR scaling**: the base `--lr` is automatically multiplied by `batch_size × world_size / 256` at startup. The schedule is linear warmup for 5 epochs followed by a step decay by 10× every 30 epochs, with an extra decay step after epoch 80.

| Argument | Default | Description |
|---|---|---|
| `DATA_DIR` | positional | Path to ImageNet-100 root (`train/` and `val/` subdirs) |
| `--model_s` | `ResNet18` | Student architecture: `ResNet18` or `ResNet34` |
| `--model_t` | `ResNet34` | Teacher architecture: `ResNet18` or `ResNet34` |
| `--path_t` | — | Path to teacher checkpoint |
| `--distill` | — | Distillation method (omit for vanilla student training) |
| `--dual_cmtf` | off | Train CP + Tucker students simultaneously (requires `pursuhint_cmtf`) |
| `--hint_points` | `3,7,13,16` | Teacher layer indices (comma-separated, 0-indexed) |
| `--s_points` | auto | Student layer indices; auto-detected from `S_POINTS_DICT` |
| `--preact` | off | Use pre-activation features |
| `--cp_rank_ratio` | `0.5` | CP global rank ratio |
| `--tucker_rank_ratio` | `0.5` | Tucker global rank ratio |
| `--use_vbmf` | off | EVBMF-from-teacher rank selection |
| `--cmtf_rank` | `8` | SVD rank R for batch-subspace projectors |
| `--cmtf_coupling_weight` | `1.0` | Tucker←CP coupling weight |
| `--epochs` | `100` | Total training epochs |
| `--lr` | `0.1` | Base learning rate (scaled by global batch / 256) |
| `--weight-decay` | `1e-4` | SGD weight decay |
| `--momentum` | `0.9` | SGD momentum |
| `--batch-size` | `256` | Per-GPU batch size |
| `--workers` | `4` | DataLoader workers per GPU |
| `--gamma` | `0.5` | Cross-entropy loss weight |
| `--alpha` | `0.9` | KL-divergence loss weight |
| `--beta` | `50.0` | Distillation loss weight |
| `--kd_T` | `4.0` | Softmax temperature for KD |
| `--dynamic_loss_weights` | off | Kendall et al. uncertainty weighting |
| `--torch_compile` | off | Apply `torch.compile` |
| `--no_dali` | off | Use standard PyTorch DataLoader instead of DALI |
| `--dali_cpu` | off | Run DALI pipeline on CPU |
| `--channels-last` | off | Use channels-last memory layout |
| `--resume` | — | Path to student save directory or checkpoint file |
| `--trial` | `1` | Trial index |
| `--local_rank` | `0` | Local rank (set automatically by `torchrun`) |
| `--sync_bn` | off | Use Apex sync BatchNorm across GPUs |
| `--opt-level` | — | Apex AMP optimization level (e.g. `O1`, `O2`) |
| `--loss-scale` | — | Apex AMP loss scale |
| `--evaluate` | off | Run validation only (no training) |
| `--deterministic` | off | Enable cuDNN deterministic mode |
| `--prof` | `-1` | Run N iterations for NVTX profiling then exit |
| `--print-freq` | `10` | Batch log interval |

**Output directory**: `./save/student_model/imagenet100/{hint_points}/S-{model_s}_T-{model_t}_imagenet_{distill}_r-{γ}_a-{α}_b-{β}_{trial}/`

---

## Output Structure

```
save/
├── models/
│   ├── {model}_{dataset}_lr_{lr}_decay_{wd}_trial_{n}/      # CIFAR teacher
│   │   ├── {model}_best.pth
│   │   ├── {model}_last.pth
│   │   ├── training_log.csv
│   │   └── experiment_summary.json
│   └── {model}_imagenet100_lr_{lr}_decay_{wd}_trial_{n}/    # ImageNet teacher
│       ├── {model}_best.pth
│       ├── {model}_last.pth
│       ├── training_log.csv
│       └── experiment_summary.json
│
├── hints/
│   ├── {model_t}_{dataset}_best/
│   │   └── hint1.pt, hint2.pt, ...    (N × C arrays, one per teacher layer)
│   ├── {model_t}_{K}clusters_{metric}_centroids.txt   (selected hint points)
│   └── {model_t}_{K}clusters_{metric}_clusters.txt   (cluster membership)
│
├── student_model/
│   ├── {dataset}/{hint_points}/                              # CIFAR students
│   │   └── S-{model_s}_T-{model_t}_{dataset}_{distill}_r-{γ}_a-{α}_b-{β}_b2-{β2}_{trial}/
│   │       ├── {model_s}_best_cp.pth          # best CP student weights
│   │       ├── {model_s}_best_tucker.pth      # best Tucker student weights
│   │       ├── {model_s}_last_cp.pth          # final epoch CP weights
│   │       ├── {model_s}_last_tucker.pth      # final epoch Tucker weights
│   │       ├── checkpoint_cp.pth              # rolling resume checkpoint (CP)
│   │       ├── checkpoint_tucker.pth          # rolling resume checkpoint (Tucker)
│   │       ├── training_log_cp.csv            # per-epoch metrics (CP)
│   │       ├── training_log_tucker.csv        # per-epoch metrics (Tucker)
│   │       └── experiment_summary.json
│   └── imagenet100/{hint_points}/                            # ImageNet students
│       └── S-{model_s}_T-{model_t}_imagenet_{distill}_r-{γ}_a-{α}_b-{β}_{trial}/
│           ├── {model_s}_best_cp.pth
│           ├── {model_s}_best_tucker.pth
│           ├── {model_s}_last_cp.pth
│           ├── {model_s}_last_tucker.pth
│           ├── checkpoint_cp.pth.tar          # rolling resume checkpoint (CP)
│           ├── checkpoint_tucker.pth.tar      # rolling resume checkpoint (Tucker)
│           ├── training_log_cp.csv
│           ├── training_log_tucker.csv
│           └── experiment_summary.json
│
├── logs/
│   └── {model_t}_{dataset}_pipeline.log      # stdout from pipeline shell script
│
└── tensorboard/
    └── {experiment_name}/                     # TensorBoard event files
```

Each training CSV contains columns: `epoch, lr, epoch_time, train_acc, train_acc_top5, train_loss, train_loss_cls, train_loss_div, train_loss_kd, test_acc, test_acc_top5, test_loss, best_acc, w_cls, w_div, w_kd` — where `w_*` are the effective loss weights (dynamic if `--dynamic_loss_weights`, otherwise the fixed γ/α/β values).

---

## Evaluation

`evaluate_metrics.py` loads and benchmarks teacher, CP student, and Tucker student, and reports:

- **Top-1 / Top-5 accuracy** on the validation set
- **Parameter count** (via THOP)
- **FLOPs** (multiply-accumulate operations, via THOP)
- **Inference latency** (ms per image, CPU, single-image batch, 50 timed runs after 10 warmup)
- **Compression ratios** vs. teacher (parameter reduction × and FLOPs reduction ×)

Efficiency metrics (parameters, FLOPs, latency) are measured on CPU with an uncompiled model — THOP does not support compiled or CUDA models. Accuracy is then measured on GPU with optional `torch.compile`.

Results are printed as a formatted table and saved to `evaluation_results.json` next to the student checkpoint.

```bash
python evaluate_metrics.py \
    --dataset cifar100 \
    --model_t wrn_40_2 \
    --path_t ./save/models/.../wrn_40_2_best.pth \
    --model_s wrn_16_2 \
    --path_s_cp    ./save/student_model/.../wrn_16_2_best_cp.pth \
    --path_s_tucker ./save/student_model/.../wrn_16_2_best_tucker.pth \
    --cp_rank_ratio 0.5 \
    --tucker_rank_ratio 0.25 \
    --use_vbmf
```

---

## Project Structure

```
code/
├── train_teacher.py              # Teacher training (CIFAR-10/100)
├── train_teacher_imagenet100.py  # Teacher training (ImageNet-100, warmup + AMP)
├── train_student.py              # Student training (CIFAR, DataParallel + AMP + teacher cache)
├── train_stu_imagenet100.py      # Student training (ImageNet-100, DDP + DALI + Apex)
├── store_hints.py                # Extract teacher layer representations (CIFAR)
├── store_hints_imagenet100.py    # Extract teacher layer representations (ImageNet-100)
├── k_means.py                    # Cluster representations, select hint points
├── evaluate_metrics.py           # Evaluate accuracy + efficiency metrics
├── decomposition.py              # CP/Tucker decomposition, EVBMF rank estimation
├── for_init.py                   # DataParallel / torch.compile prefix handling
├── run_pipeline_cifar.sh         # End-to-end CIFAR pipeline script
├── run_pipeline_imagenet.sh      # End-to-end ImageNet-100 pipeline script
├── prepare_imagenet100.sh        # Download + restructure ImageNet-100 from Kaggle
│
├── distiller_zoo/
│   ├── CMTF.py                   # Coupled Matrix-Tensor Factorization loss (novel)
│   ├── KD.py                     # Hinton et al. KL-divergence distillation
│   ├── FitNet.py                 # Hint-based feature regression (FitNet)
│   ├── AT.py                     # Attention Transfer
│   ├── VID.py                    # Variational Information Distillation
│   └── WSL.py                    # Weak Soft Labels
│
├── models/
│   ├── __init__.py               # model_dict registry
│   ├── resnet.py                 # CIFAR ResNets (8/14/20/32/44/56/110, x4 wide variants)
│   ├── wrn.py                    # WideResNet (depth 16/40, width 1/2)
│   ├── resnetv2.py               # ImageNet ResNets (18/34/50/101)
│   ├── vgg.py                    # VGG with batch norm (8–19)
│   ├── mobilenetv2.py            # MobileNet V2 and mobile_half
│   ├── ShuffleNetv1.py           # ShuffleNet V1
│   ├── ShuffleNetv2.py           # ShuffleNet V2
│   └── util.py                   # ConvReg, Embed, LinearEmbed adapter modules
│
├── helper/
│   ├── loops.py                  # train_vanilla, train_distill, validate
│   ├── util.py                   # AverageMeter, accuracy, LR schedulers
│   ├── uncertainty_weighter.py   # Kendall et al. dynamic loss weighting
│   └── pretrain.py               # FitNet hint-layer pretraining (non-CMTF methods)
│
├── dataset/
│   ├── cifar100.py               # CIFAR-100 loaders (standard, instance, sample)
│   ├── cifar10.py                # CIFAR-10 loaders
│   └── imagenet.py               # ImageNet loaders (standard, instance, sample)
│
├── clustering/
│   └── utils/
│       ├── cka.py                # gram_linear, gram_rbf, cka(), r2_cca()
│       ├── logger.py             # logging
│
└── crd/
    └── criterion.py              # CRD contrastive loss (CIFAR only)
```

---

## References

- Keser, R. K., Ayanzadeh, A., Aghdam, O. A., Kilcioglu, C., Toreyin, B. U., & Ure, N. K. (2023). Pursuhint: In search of informative hint points based on layer clustering for knowledge distillation. Expert Systems with Applications, 213, 119040.
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.
- Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y. (2014). FitNets: hints for thin deep nets (2014). arXiv preprint arXiv:1412.6550, 3.
- Zagoruyko, S., & Komodakis, N. (2016). Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. arXiv preprint arXiv:1612.03928.
- Tian, Y., Krishnan, D., & Isola, P. (2019). Contrastive representation distillation. arXiv preprint arXiv:1910.10699.
- Kim, Y. D., Park, E., Yoo, S., Choi, T., Yang, L., & Shin, D. (2015). Compression of deep convolutional neural networks for fast and low power mobile applications. arXiv preprint arXiv:1511.06530.
- Nakajima, S., Sugiyama, M., Babacan, S. D., & Tomioka, R. (2013). Global analytic solution of fully-observed variational Bayesian matrix factorization. The Journal of Machine Learning Research, 14(1), 1-37.
- Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7482-7491).
- Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019, May). Similarity of neural network representations revisited. In International conference on machine learning (pp. 3519-3529). PMlR.
