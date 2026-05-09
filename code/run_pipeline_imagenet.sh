#!/bin/bash
set -e  # Exit on any error

# Suppress TensorFlow/TensorBoard C++ and oneDNN log noise
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

mkdir -p ./save/logs
mkdir -p ./save/hints
mkdir -p ./save/models
mkdir -p ./save/student_model

echo "=========================================================================="
echo "Starting FULL PURSUhInT + CMTF Pipeline for ImageNet-100"
echo "  (Teacher -> Hints -> Clustering -> Dual Student Training -> Evaluation)"
echo "=========================================================================="

# ==============================================================================
# Configuration  (edit these to match your environment)
# ==============================================================================
PYTHON="python3"                     # or: /path/to/venv/bin/python
DATASET="imagenet100"
MODEL_T="ResNet50"
MODEL_S="ResNet18"
TRIAL=0
NUM_LAYERS=16                        # ResNet50 has 16 Bottleneck sub-blocks ([3,4,6,3])
NUM_CLUSTERS=4                       # 4 hint points for ImageNet experiments
METRIC="r2"
TEACHER_EPOCHS=100
STUDENT_EPOCHS=100
BATCH=256                            # Lowered from 256 to prevent 24GB VRAM OOM
NUM_WORKERS=8
LR=0.1
WEIGHT_DECAY=0.0001

# Paths
IMAGENET_DIR="data/imagenet100"  # must contain train/ and val/ subdirs

# Teacher output path (auto-derived from train_teacher_imagenet100.py naming convention)
TEACHER_SAVE_DIR="./save/models/${MODEL_T}_imagenet100_lr_${LR}_decay_${WEIGHT_DECAY}_trial_${TRIAL}"
TEACHER_PATH="${TEACHER_SAVE_DIR}/${MODEL_T}_best.pth"
HINTS_DIR="./save/hints/${MODEL_T}_${DATASET}_best"

# Resume from a previous run (leave empty to start fresh)
# Set to the student save directory, e.g.:
# RESUME_DIR="./save/student_model/imagenet100/3,8,11,16/S-ResNet18_T-ResNet34_imagenet_pursuhint_cmtf_r-1.0_a-4.0_b-25.0_0"
RESUME_DIR=""

# Student training loss weights
GAMMA=1.0
ALPHA=4.0
BETA=25.0

# CP / Tucker Rank Ratios and CMTF Rank
# (used when USE_VBMF=0; ignored for rank selection when USE_VBMF=1)
CP_RANK_RATIO=0.5
TUCKER_RANK_RATIO=0.25
CMTF_RANK=8
CMTF_COUPLING_WEIGHT=1.0          # weight for Tucker←CP coupling term in dual CMTF

# VBMF automatic rank selection (uses teacher weight spectrum; recommended over fixed ratios)
USE_VBMF=1
if [ "$USE_VBMF" -eq 1 ]; then
    VBMF_FLAG="--use_vbmf"
else
    VBMF_FLAG=""
fi

# PyTorch Compile Optimization ON/OFF (disabled by default for DDP)
ENABLE_TORCH_COMPILE=1  # decomposed models have 3-4x more layers; compile can OOMs on CPU RAM
if [ "$ENABLE_TORCH_COMPILE" -eq 1 ]; then
    TORCH_COMPILE="--torch_compile"
else
    TORCH_COMPILE=""
fi

# Dynamic loss weighting (Kendall et al. CVPR 2018) ON/OFF
ENABLE_DYNAMIC_LOSS_WEIGHTS=1
if [ "$ENABLE_DYNAMIC_LOSS_WEIGHTS" -eq 1 ]; then
    DYNAMIC_LOSS_WEIGHTS="--dynamic_loss_weights"
else
    DYNAMIC_LOSS_WEIGHTS=""
fi

# Log file
LOG_DIR="./save/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/${MODEL_T}_${DATASET}_pipeline.log"
echo "Log file: ${LOG}"

# ==============================================================================
# Flags  (set to 1 to enable, 0 to skip a step)
# ==============================================================================
RUN_TEACHER=1
RUN_HINTS=1
RUN_CLUSTERING=1
RUN_TRAINING=1
RUN_EVALUATION=1

# Use standard DataLoader instead of DALI? (set to 1 for local debugging)
USE_NO_DALI=0

# ==============================================================================
# [1/5] Train Teacher
# ==============================================================================
if [ "$RUN_TEACHER" -eq 1 ]; then
    echo "[1/5] Training Teacher (${MODEL_T}) on ${DATASET}..."
    echo "[1/5] Training Teacher (${MODEL_T}) on ${DATASET}..." >> "${LOG}" 2>&1

    ${PYTHON} train_teacher_imagenet100.py \
        --model ${MODEL_T} \
        --data_folder "${IMAGENET_DIR}" \
        --epochs ${TEACHER_EPOCHS} \
        --batch_size ${BATCH} \
        --learning_rate ${LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --lr_decay_epochs 30,60,90 \
        --num_workers ${NUM_WORKERS} \
        --trial ${TRIAL} \
        ${TORCH_COMPILE} >> "${LOG}" 2>&1 || { echo "ERROR: Teacher training failed. Check ${LOG}."; exit 1; }

    echo "Teacher training complete." >> "${LOG}"
else
    echo "[1/5] SKIPPED (RUN_TEACHER=0)"
fi

# Verify teacher checkpoint exists before proceeding
if [ ! -f "${TEACHER_PATH}" ]; then
    echo "ERROR: Teacher checkpoint not found: ${TEACHER_PATH}"
    echo "  Either run teacher training (RUN_TEACHER=1) or update TEACHER_PATH."
    exit 1
fi

# ==============================================================================
# [2/5] PURSUhInT Step 1: Extract layer representations
# ==============================================================================
if [ "$RUN_HINTS" -eq 1 ]; then
    echo "[2/5] Storing ${NUM_LAYERS} teacher layer representations..."
    echo "[2/5] Storing teacher layer representations (${NUM_LAYERS} sub-blocks)..." >> "${LOG}" 2>&1

    for i in $(seq 1 ${NUM_LAYERS}); do
        echo "  Extracting hint ${i} / ${NUM_LAYERS}..."
        ${PYTHON} store_hints_imagenet100.py \
            --g ${i} \
            --model_t ${MODEL_T} \
            --t_path "${TEACHER_PATH}" \
            --hints_path "${HINTS_DIR}" \
            --data_folder "${IMAGENET_DIR}" \
            --batch_size ${BATCH} \
            --num_workers ${NUM_WORKERS} >> "${LOG}" 2>&1 || { echo "ERROR: store_hints_imagenet100.py failed on layer ${i}. Check ${LOG}."; exit 1; }
    done
    echo "Layer representation extraction complete." >> "${LOG}"
else
    echo "[2/5] SKIPPED (RUN_HINTS=0)"
fi

# ==============================================================================
# [3/5] PURSUhInT Step 2: Cluster representations with K-Means
# ==============================================================================
if [ "$RUN_CLUSTERING" -eq 1 ]; then
    echo "[3/5] Clustering layer representations with ${METRIC} (${NUM_CLUSTERS} clusters)..."
    echo "[3/5] Running K-Means clustering..." >> "${LOG}" 2>&1

    ${PYTHON} k_means.py \
        --hints_dir "${HINTS_DIR}" \
        --num_clusters ${NUM_CLUSTERS} \
        --num_layers ${NUM_LAYERS} \
        --metric_name ${METRIC} \
        --output_dir ./save/hints \
        --model_name ${MODEL_T}_${DATASET} >> "${LOG}" 2>&1 || { echo "ERROR: k_means.py clustering failed. Check ${LOG}."; exit 1; }

    # Read the centroid file -> HINT_POINTS variable
    CENTROID_FILE="./save/hints/${MODEL_T}_${DATASET}_${NUM_CLUSTERS}clusters_${METRIC}_centroids.txt"
    if [ ! -f "${CENTROID_FILE}" ]; then
        echo "ERROR: Centroid file not found: ${CENTROID_FILE}"
        exit 1
    fi

    HINT_POINTS=$(cat "${CENTROID_FILE}")
    echo "PURSUhInT selected hint points: ${HINT_POINTS}" | tee -a "${LOG}"
else
    echo "[3/5] SKIPPED (RUN_CLUSTERING=0)"
    # If skipping clustering, provide hint points manually here:
    HINT_POINTS="3,7,13,16"
    echo "Using default hint points: ${HINT_POINTS}"
fi

# ==============================================================================
# [4/5] Train Dual Students (CP + Tucker) using discovered hint points
# ==============================================================================
if [ "$RUN_TRAINING" -eq 1 ]; then
    echo "[4/5] Training Dual Students (${MODEL_S}) with hint points: ${HINT_POINTS}..."
    echo "[4/5] Training students with hint_points=${HINT_POINTS}..." >> "${LOG}" 2>&1

    DALI_FLAG=""
    if [ "$USE_NO_DALI" -eq 1 ]; then
        DALI_FLAG="--no_dali"
        echo "  Using standard PyTorch DataLoader (no DALI)"
    fi

    RESUME_FLAG=""
    if [ -n "$RESUME_DIR" ]; then
        RESUME_FLAG="--resume ${RESUME_DIR}"
        echo "  Resuming from: ${RESUME_DIR}"
    fi

    # Number of GPUs to use for training
    NUM_GPUS=1

    torchrun --nproc_per_node=${NUM_GPUS} --master_port 9200 train_stu_imagenet100.py \
        --model_s ${MODEL_S} \
        --model_t ${MODEL_T} \
        --path_t "${TEACHER_PATH}" \
        --distill pursuhint_cmtf \
        --dual_cmtf \
        --trial ${TRIAL} \
        --cp_rank_ratio ${CP_RANK_RATIO} \
        --tucker_rank_ratio ${TUCKER_RANK_RATIO} \
        --cmtf_rank ${CMTF_RANK} \
        --cmtf_coupling_weight ${CMTF_COUPLING_WEIGHT} \
        --epochs ${STUDENT_EPOCHS} \
        --lr ${LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --batch-size ${BATCH} \
        --workers ${NUM_WORKERS} \
        --hint_points "${HINT_POINTS}" \
        --gamma ${GAMMA} \
        --alpha ${ALPHA} \
        --beta ${BETA} \
        ${VBMF_FLAG} ${DALI_FLAG} \
        ${DYNAMIC_LOSS_WEIGHTS} \
        ${TORCH_COMPILE} \
        ${RESUME_FLAG} \
        "${IMAGENET_DIR}" >> "${LOG}" 2>&1 || { echo "ERROR: Student training failed. Check ${LOG}."; exit 1; }

    echo "Student training complete." >> "${LOG}"
else
    echo "[4/5] SKIPPED (RUN_TRAINING=0)"
fi

# ==============================================================================
# [5/5] Evaluation
# ==============================================================================
if [ "$RUN_EVALUATION" -eq 1 ]; then
    echo "[5/5] Running Evaluation Metrics..."
    echo "[5/5] Running evaluation..." >> "${LOG}" 2>&1

    STUDENT_DIR="./save/student_model/imagenet100/${HINT_POINTS}/S-${MODEL_S}_T-${MODEL_T}_imagenet_pursuhint_cmtf_r-${GAMMA}_a-${ALPHA}_b-${BETA}_${TRIAL}"

    ${PYTHON} evaluate_metrics.py \
        --dataset ${DATASET} \
        --data_folder "${IMAGENET_DIR}" \
        --model_t ${MODEL_T} \
        --path_t "${TEACHER_PATH}" \
        --model_s ${MODEL_S} \
        --path_s_cp "${STUDENT_DIR}/${MODEL_S}_best_cp.pth" \
        --path_s_tucker "${STUDENT_DIR}/${MODEL_S}_best_tucker.pth" \
        --cp_rank_ratio ${CP_RANK_RATIO} \
        --tucker_rank_ratio ${TUCKER_RANK_RATIO} \
        ${VBMF_FLAG} ${TORCH_COMPILE} >> "${LOG}" 2>&1 || { echo "WARNING: Evaluation failed. Check ${LOG}."; }
else
    echo "[5/5] SKIPPED (RUN_EVALUATION=0)"
fi

# ==============================================================================
# Done
# ==============================================================================
echo ""
echo "============================================================"
echo "Pipeline complete! Results logged to: ${LOG}"
echo "PURSUhInT hint points used: ${HINT_POINTS}"
echo "============================================================"