#!/bin/bash
set -e  # Exit on any error

echo "=========================================================================="
echo "Starting FULL PURSUhInT + CMTF Pipeline for ImageNet"
echo "  (Hints -> Clustering -> Dual Student Training -> Evaluation)"
echo "=========================================================================="

# ==============================================================================
# Configuration  (edit these to match your environment)
# ==============================================================================
PYTHON="python3"                     # or: /path/to/venv/bin/python
DATASET="imagenet"
MODEL_T="ResNet34"
MODEL_S="ResNet18"
NUM_LAYERS=16                        # ResNet34 has 16 BasicBlock sub-blocks
NUM_CLUSTERS=4                       # 4 hint points for ImageNet experiments
METRIC="r2"
EPOCHS=100
BATCH=128                            # Lowered from 256 to prevent 24GB VRAM OOM
NUM_WORKERS=4
LR=0.1

# Paths
IMAGENET_DIR="/path/to/imagenet"     # must contain train/ and val/ subdirs
TEACHER_PATH="./save/models/ResNet34_imagenet/ResNet34_333f7ec4.pth" # Ensure this matches your actual file
HINTS_DIR="./save/hints/ResNet34_imagenet"

# Student training loss weights
GAMMA=2.5
ALPHA=0.75
BETA=250.0
TRIAL=1

# Log file
LOG_DIR="./save/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/${MODEL_T}_${DATASET}_pipeline.log"
echo "Log file: ${LOG}"

# ==============================================================================
# Flags  (set to 1 to enable, 0 to skip a step)
# ==============================================================================
RUN_HINTS=1
RUN_CLUSTERING=1
RUN_TRAINING=1
RUN_EVALUATION=1

# Use standard DataLoader instead of DALI? (set to 1 for local debugging)
USE_NO_DALI=0

# ==============================================================================
# [1/4] PURSUhInT Step 1: Extract layer representations
# ==============================================================================
if [ "$RUN_HINTS" -eq 1 ]; then
    echo "[1/4] Storing ${NUM_LAYERS} teacher layer representations..."
    echo "[1/4] Storing teacher layer representations (${NUM_LAYERS} sub-blocks)..." >> "${LOG}" 2>&1

    for i in $(seq 1 ${NUM_LAYERS}); do
        echo "  Extracting hint ${i} / ${NUM_LAYERS}..."
        ${PYTHON} store_hints_imagenet.py \
            --g ${i} \
            --model_t ${MODEL_T} \
            --t_path "${TEACHER_PATH}" \
            --hints_path "${HINTS_DIR}" \
            --data_folder "${IMAGENET_DIR}" \
            --batch_size ${BATCH} \
            --num_workers ${NUM_WORKERS} >> "${LOG}" 2>&1 || { echo "ERROR: store_hints_imagenet.py failed on layer ${i}. Check ${LOG}."; exit 1; }
    done
    echo "Layer representation extraction complete." >> "${LOG}"
else
    echo "[1/4] SKIPPED (RUN_HINTS=0)"
fi

# ==============================================================================
# [2/4] PURSUhInT Step 2: Cluster representations with K-Means
# ==============================================================================
if [ "$RUN_CLUSTERING" -eq 1 ]; then
    echo "[2/4] Clustering layer representations with ${METRIC} (${NUM_CLUSTERS} clusters)..."
    echo "[2/4] Running K-Means clustering..." >> "${LOG}" 2>&1

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
    echo "[2/4] SKIPPED (RUN_CLUSTERING=0)"
    # If skipping clustering, provide hint points manually here:
    HINT_POINTS="3,7,13,16"
    echo "Using default hint points: ${HINT_POINTS}"
fi

# ==============================================================================
# [3/4] Train Dual Students (CP + Tucker) using discovered hint points
# ==============================================================================
if [ "$RUN_TRAINING" -eq 1 ]; then
    echo "[3/4] Training Dual Students (${MODEL_S}) with hint points: ${HINT_POINTS}..."
    echo "[3/4] Training students with hint_points=${HINT_POINTS}..." >> "${LOG}" 2>&1

    DALI_FLAG=""
    if [ "$USE_NO_DALI" -eq 1 ]; then
        DALI_FLAG="--no_dali"
        echo "  Using standard PyTorch DataLoader (no DALI)"
    fi

    # Number of GPUs to use for training
    NUM_GPUS=1

    ${PYTHON} -m torch.distributed.launch --master_port 9200 --nproc_per_node=${NUM_GPUS} train_stu_imagenet.py \
        --model_s ${MODEL_S} \
        --model_t ${MODEL_T} \
        --path_t "${TEACHER_PATH}" \
        --distill pursuhint_cmtf \
        --dual_cmtf \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --batch-size ${BATCH} \
        --workers ${NUM_WORKERS} \
        --hint_points "${HINT_POINTS}" \
        --gamma ${GAMMA} \
        --alpha ${ALPHA} \
        --beta ${BETA} \
        --trial ${TRIAL} \
        ${DALI_FLAG} \
        "${IMAGENET_DIR}" >> "${LOG}" 2>&1 || { echo "ERROR: Student training failed. Check ${LOG}."; exit 1; }

    echo "Student training complete." >> "${LOG}"
else
    echo "[3/4] SKIPPED (RUN_TRAINING=0)"
fi

# ==============================================================================
# [4/4] Evaluation
# ==============================================================================
if [ "$RUN_EVALUATION" -eq 1 ]; then
    echo "[4/4] Running Evaluation Metrics..."
    echo "[4/4] Running evaluation..." >> "${LOG}" 2>&1

    STUDENT_DIR="./save/student_model/imagenet/${HINT_POINTS}/S-${MODEL_S}_T-${MODEL_T}_imagenet_pursuhint_cmtf_r-${GAMMA}_a-${ALPHA}_b-${BETA}_${TRIAL}"

    ${PYTHON} evaluate_metrics.py \
        --dataset ${DATASET} \
        --model_t ${MODEL_T} \
        --path_t "${TEACHER_PATH}" \
        --model_s ${MODEL_S} \
        --path_s_cp "${STUDENT_DIR}/${MODEL_S}_last_cp.pth" \
        --path_s_tucker "${STUDENT_DIR}/${MODEL_S}_last_tucker.pth" >> "${LOG}" 2>&1 || { echo "WARNING: Evaluation failed. Check ${LOG}."; }
else
    echo "[4/4] SKIPPED (RUN_EVALUATION=0)"
fi

# ==============================================================================
# Done
# ==============================================================================
echo ""
echo "============================================================"
echo "Pipeline complete! Results logged to: ${LOG}"
echo "PURSUhInT hint points used: ${HINT_POINTS}"
echo "============================================================"