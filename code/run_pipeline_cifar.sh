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
echo "Starting FULL PURSUhInT + BSAT Pipeline for CIFAR-100"
echo "  (Teacher -> Hints -> Clustering -> Dual Student Training -> Evaluation)"
echo "=========================================================================="

# ==============================================================================
# Configuration  (edit these to match your environment)
# ==============================================================================
PYTHON="python3"
DATASET="cifar100"
MODEL_T="wrn_40_2"
MODEL_S="wrn_16_2"
TRIAL=0

# Distillation method — change this to switch methods:
#   pursuhint_bsat  GAMMA=1.0  ALPHA=4.0  BETA=25.0
#   kd              GAMMA=1.0  ALPHA=4.0  BETA=0.0
#   attention       GAMMA=1.0  ALPHA=4.0  BETA=1000.0
#   hint            GAMMA=1.0  ALPHA=4.0  BETA=1000.0
#   WSL_att         GAMMA=1.0  ALPHA=4.0  BETA=1000.0
#   vid             GAMMA=1.0  ALPHA=4.0  BETA=1000.0
DISTILL_METHOD="kd"
NUM_LAYERS=18                         # vgg13 has 4 pooling-separated blocks
NUM_CLUSTERS=3                       # must match vgg8 student's s_points (1,2,3)
METRIC="r2"
EPOCHS=1
LR_DECAY_EPOCHS="150,180,210"
BATCH=64
NUM_WORKERS=4
LR=0.05
WEIGHT_DECAY=0.0005                  # Standard CIFAR WD

# Paths
TEACHER_SAVE_DIR="./save/models/${MODEL_T}_${DATASET}_lr_${LR}_decay_${WEIGHT_DECAY}_trial_${TRIAL}"
TEACHER_PATH="${TEACHER_SAVE_DIR}/${MODEL_T}_best.pth"
HINTS_DIR="./save/hints/${MODEL_T}_${DATASET}_best"

# Student training loss weights (Optimized 65/15/20 Split)
GAMMA=1.0
ALPHA=4.0
BETA=0.0

# CP / Tucker Rank Ratios and BSAT Rank
# (used when USE_VBMF=0; ignored for rank selection when USE_VBMF=1)
CP_RANK_RATIO=0.5
TUCKER_RANK_RATIO=0.25
BSAT_RANK=8
BSAT_COUPLING_WEIGHT=1.0          # weight for Tucker←CP coupling term in dual BSAT

# BSAT CIFAR-stability knobs (defaults reproduce the original paper behavior)
BSAT_ALIGN_MODE="projector"       # 'projector' (original, eigh) or 'gram' (eigh-free, recommended for CIFAR)
BSAT_PROJ_STABLE=0                # 1 = float64 eigh + relative jitter + NaN-safe (projector mode only)
BSAT_SPLIT_LOSSES=0               # 1 = give the BSA term its own dynamic loss weight
BSAT_SUBSPACE_WARMUP=0            # epochs to linearly ramp BSA + coupling from 0 (0 = off)

BSAT_EXTRA_FLAGS="--bsat_align_mode ${BSAT_ALIGN_MODE} --bsat_subspace_warmup ${BSAT_SUBSPACE_WARMUP}"
if [ "$BSAT_PROJ_STABLE" -eq 1 ]; then BSAT_EXTRA_FLAGS="${BSAT_EXTRA_FLAGS} --bsat_proj_stable"; fi
if [ "$BSAT_SPLIT_LOSSES" -eq 1 ]; then BSAT_EXTRA_FLAGS="${BSAT_EXTRA_FLAGS} --bsat_split_losses"; fi

# VBMF automatic rank selection (uses teacher weight spectrum; recommended over fixed ratios)
USE_VBMF=1
if [ "$USE_VBMF" -eq 1 ]; then
    VBMF_FLAG="--use_vbmf"
else
    VBMF_FLAG=""
fi

# PyTorch Compile Optimization ON/OFF
ENABLE_TORCH_COMPILE=0
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

# Skip teacher output precomputation — compute on-the-fly each batch.
# Use when running multiple parallel jobs to avoid RAM contention (~10-15% slower).
NO_TEACHER_CACHE=1
if [ "$NO_TEACHER_CACHE" -eq 1 ]; then
    NO_TEACHER_CACHE_FLAG="--no_teacher_cache"
else
    NO_TEACHER_CACHE_FLAG=""
fi

# Log file
LOG_DIR="./save/logs"
LOG="${LOG_DIR}/${MODEL_T}_${MODEL_S}_${DATASET}_${DISTILL_METHOD}_pipeline.log"
echo "Log file: ${LOG}"

# ==============================================================================
# Flags  (set to 1 to enable, 0 to skip a step)
# ==============================================================================
RUN_TEACHER=0
RUN_HINTS=0
RUN_CLUSTERING=0
RUN_TRAINING=1
RUN_EVALUATION=1

# ==============================================================================
# [1/5] Train Teacher
# ==============================================================================
if [ "$RUN_TEACHER" -eq 1 ]; then
    echo "[1/5] Training Teacher (${MODEL_T}) on ${DATASET}..."
    echo "[1/5] Training Teacher (${MODEL_T}) on ${DATASET}..." >> "${LOG}" 2>&1

    ${PYTHON} train_teacher.py \
        --dataset ${DATASET} \
        --model ${MODEL_T} \
        --epochs ${EPOCHS} \
        --learning_rate ${LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --lr_decay_epochs ${LR_DECAY_EPOCHS} \
        --batch_size ${BATCH} \
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
        ${PYTHON} store_hints.py \
            --g ${i} \
            --model_t ${MODEL_T} \
            --t_path "${TEACHER_PATH}" \
            --dataset ${DATASET} \
            --hints_path "${HINTS_DIR}" >> "${LOG}" 2>&1 || { echo "ERROR: store_hints.py failed on layer ${i}. Check ${LOG}."; exit 1; }
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
    # Fallback to standard CRD hint points if skipping clustering
    HINT_POINTS="5,10,16"
    echo "Using default hint points: ${HINT_POINTS}"
fi

# ==============================================================================
# [4/5] Train Dual Students (CP + Tucker)
# ==============================================================================
if [ "$RUN_TRAINING" -eq 1 ]; then
    echo "[4/5] Training Dual Students (${MODEL_S}) with hint points: ${HINT_POINTS}..."
    echo "[4/5] Training students with hint_points=${HINT_POINTS}..." >> "${LOG}" 2>&1

    ${PYTHON} train_student.py \
        --dataset ${DATASET} \
        --model_s ${MODEL_S} \
        --model_t ${MODEL_T} \
        --path_t "${TEACHER_PATH}" \
        --distill ${DISTILL_METHOD} \
        --dual_bsat \
        --trial ${TRIAL} \
        --cp_rank_ratio ${CP_RANK_RATIO} \
        --tucker_rank_ratio ${TUCKER_RANK_RATIO} \
        --bsat_rank ${BSAT_RANK} \
        --bsat_coupling_weight ${BSAT_COUPLING_WEIGHT} \
        --epochs ${EPOCHS} \
        --learning_rate ${LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --lr_decay_epochs ${LR_DECAY_EPOCHS} \
        --batch_size ${BATCH} \
        --num_workers ${NUM_WORKERS} \
        --hint_points "${HINT_POINTS}" \
        -r ${GAMMA} -a ${ALPHA} -b ${BETA} \
        ${BSAT_EXTRA_FLAGS} \
        ${VBMF_FLAG} ${TORCH_COMPILE} ${DYNAMIC_LOSS_WEIGHTS} ${NO_TEACHER_CACHE_FLAG} >> "${LOG}" 2>&1 || { echo "ERROR: Student training failed. Check ${LOG}."; exit 1; }

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

    # Format matches train_student.py output exactly
    STUDENT_DIR="./save/student_model/${DATASET}/${HINT_POINTS}/S-${MODEL_S}_T-${MODEL_T}_${DATASET}_${DISTILL_METHOD}_r-${GAMMA}_a-${ALPHA}_b-${BETA}_b2-1.0_${TRIAL}"

    ${PYTHON} evaluate_metrics.py \
        --dataset ${DATASET} \
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