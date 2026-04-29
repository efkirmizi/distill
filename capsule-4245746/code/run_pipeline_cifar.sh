#!/bin/bash
set -e  # Exit on any error

mkdir -p ./save/logs
mkdir -p ./save/hints
mkdir -p ./save/models
mkdir -p ./save/student_model

echo "=========================================================================="
echo "Starting FULL PURSUhInT + CMTF Pipeline for CIFAR-100"
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
NUM_LAYERS=18                        # WideResNet-40 has 18 sub-blocks
NUM_CLUSTERS=3                       # 3 hint points for CIFAR experiments
METRIC="r2"
EPOCHS=1
LR_DECAY_EPOCHS="60,120,160"
BATCH=64
NUM_WORKERS=4
LR=0.1
WEIGHT_DECAY=0.0005                  # Standard CIFAR WD

# Paths
TEACHER_SAVE_DIR="./save/models/${MODEL_T}_${DATASET}_lr_${LR}_decay_${WEIGHT_DECAY}_trial_${TRIAL}"
TEACHER_PATH="${TEACHER_SAVE_DIR}/${MODEL_T}_last.pth"
HINTS_DIR="./save/hints/${MODEL_T}_${DATASET}_last"

# Student training loss weights (Optimized 65/15/20 Split)
GAMMA=1.0
ALPHA=4.0
BETA=25.0

# CP / Tucker Rank Ratios and CMTF Rank
CP_RANK_RATIO=0.5
TUCKER_RANK_RATIO=0.25
CMTF_RANK=8

# PyTorch Compile Optimization ON/OFF
ENABLE_TORCH_COMPILE=1
if [ "$ENABLE_TORCH_COMPILE" -eq 1 ]; then
    TORCH_COMPILE="--torch_compile"
else
    TORCH_COMPILE=""
fi

# Log file
LOG_DIR="./save/logs"
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
        --model_name ${MODEL_T} >> "${LOG}" 2>&1 || { echo "ERROR: k_means.py clustering failed. Check ${LOG}."; exit 1; }

    # Read the centroid file -> HINT_POINTS variable
    CENTROID_FILE="./save/hints/${MODEL_T}_${NUM_CLUSTERS}clusters_${METRIC}_centroids.txt"
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
        --distill pursuhint_cmtf \
        --dual_cmtf \
        --trial ${TRIAL} \
        --cp_rank_ratio ${CP_RANK_RATIO} \
        --tucker_rank_ratio ${TUCKER_RANK_RATIO} \
        --cmtf_rank ${CMTF_RANK} \
        --epochs ${EPOCHS} \
        --learning_rate ${LR} \
        --lr_decay_epochs ${LR_DECAY_EPOCHS} \
        --batch_size ${BATCH} \
        --num_workers ${NUM_WORKERS} \
        --hint_points "${HINT_POINTS}" \
        -r ${GAMMA} -a ${ALPHA} -b ${BETA} \
        ${TORCH_COMPILE} >> "${LOG}" 2>&1 || { echo "ERROR: Student training failed. Check ${LOG}."; exit 1; }

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
    STUDENT_DIR="./save/student_model/${DATASET}/${HINT_POINTS}/S-${MODEL_S}_T-${MODEL_T}_${DATASET}_pursuhint_cmtf_r-${GAMMA}_a-${ALPHA}_b-${BETA}_b2-1.0_${TRIAL}"

    ${PYTHON} evaluate_metrics.py \
        --dataset ${DATASET} \
        --model_t ${MODEL_T} \
        --path_t "${TEACHER_PATH}" \
        --model_s ${MODEL_S} \
        --path_s_cp "${STUDENT_DIR}/${MODEL_S}_best_cp.pth" \
        --path_s_tucker "${STUDENT_DIR}/${MODEL_S}_best_tucker.pth" \
        --cp_rank_ratio ${CP_RANK_RATIO} \
        --tucker_rank_ratio ${TUCKER_RANK_RATIO} \
        ${TORCH_COMPILE} >> "${LOG}" 2>&1 || { echo "WARNING: Evaluation failed. Check ${LOG}."; }
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