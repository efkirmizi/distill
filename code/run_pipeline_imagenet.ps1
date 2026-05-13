# ==============================================================================
# Starting FULL PURSUhInT + BSAT Pipeline for ImageNet-100 (Windows Version)
# ==============================================================================
$ErrorActionPreference = "Continue" # Let $LASTEXITCODE handle actual crashes

# Suppress TensorFlow/TensorBoard C++ warnings
$env:TF_CPP_MIN_LOG_LEVEL = 3
$env:TF_ENABLE_ONEDNN_OPTS = 0

# Disable Linux-specific networking for PyTorch DDP on Windows
$env:USE_LIBUV = 0

# Create directories
New-Item -ItemType Directory -Force -Path ".\save" | Out-Null
New-Item -ItemType Directory -Force -Path ".\save\logs" | Out-Null
New-Item -ItemType Directory -Force -Path ".\save\hints" | Out-Null
New-Item -ItemType Directory -Force -Path ".\save\models" | Out-Null
New-Item -ItemType Directory -Force -Path ".\save\student_model" | Out-Null

Write-Host "=========================================================================="
Write-Host "Starting FULL PURSUhInT + BSAT Pipeline for ImageNet-100"
Write-Host "  (Teacher -> Hints -> Clustering -> Dual Student Training -> Evaluation)"
Write-Host "=========================================================================="

# ==============================================================================
# Configuration  (edit these to match your environment)
# ==============================================================================
# Note: On Windows, the command is usually 'python' instead of 'python3'
$PYTHON = "python" 
$DATASET = "imagenet100"
$MODEL_T = "ResNet50"
$MODEL_S = "ResNet18"
$TRIAL = "1"
$NUM_LAYERS = 16
$NUM_CLUSTERS = 4
$METRIC = "r2"
$TEACHER_EPOCHS = 1
$STUDENT_EPOCHS = 1
$BATCH = 4
$NUM_WORKERS = 0 # Note: Windows PyTorch may require lower workers if you hit IPC errors
$LR = 0.1
$WEIGHT_DECAY = 0.0001

# Paths
$IMAGENET_DIR = "data\imagenet_tiny"

$TEACHER_SAVE_DIR = ".\save\models\${MODEL_T}_imagenet100_lr_${LR}_decay_${WEIGHT_DECAY}_trial_${TRIAL}"
$TEACHER_PATH = "${TEACHER_SAVE_DIR}\${MODEL_T}_best.pth"
$HINTS_DIR = ".\save\hints\${MODEL_T}_${DATASET}_best"

# Student training loss weights
$GAMMA = "1.0"
$ALPHA = "4.0"
$BETA = "25.0"

# CP / Tucker Rank Ratios and BSAT Rank
# (used when USE_VBMF=0; ignored for rank selection when USE_VBMF=1)
$CP_RANK_RATIO = 0.5
$TUCKER_RANK_RATIO = 0.25
$BSAT_RANK = 8
$BSAT_COUPLING_WEIGHT = 1.0       # weight for Tucker←CP coupling term in dual BSAT

# VBMF automatic rank selection (uses teacher weight spectrum; recommended over fixed ratios)
$USE_VBMF = 1

# Log file
$LOG_DIR = ".\save\logs"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$LOG = "${LOG_DIR}\${MODEL_T}_${DATASET}_pipeline.log"
Write-Host "Log file: $LOG"

# ==============================================================================
# Flags  (set to 1 to enable, 0 to skip a step)
# ==============================================================================
$RUN_TEACHER = 1
$RUN_HINTS = 1
$RUN_CLUSTERING = 1
$RUN_TRAINING = 1
$RUN_EVALUATION = 1

$USE_NO_DALI = 0

# Dynamic loss weighting (Kendall et al. CVPR 2018) ON/OFF
$ENABLE_DYNAMIC_LOSS_WEIGHTS = 1

# ==============================================================================
# [1/5] Train Teacher
# ==============================================================================
if ($RUN_TEACHER -eq 1) {
    Write-Host "[1/5] Training Teacher (${MODEL_T}) on ${DATASET}..."
    "[1/5] Training Teacher (${MODEL_T}) on ${DATASET}..." | Out-File -FilePath $LOG -Append

    # Note: *>> is PowerShell's equivalent to >> file.log 2>&1
    & $PYTHON train_teacher_imagenet100.py `
        --model $MODEL_T `
        --data_folder $IMAGENET_DIR `
        --epochs $TEACHER_EPOCHS `
        --batch_size $BATCH `
        --learning_rate $LR `
        --weight_decay $WEIGHT_DECAY `
        --lr_decay_epochs "30,60,90" `
        --num_workers $NUM_WORKERS `
        --trial $TRIAL *>> $LOG

    if ($LASTEXITCODE -ne 0) {
        Write-Error "ERROR: Teacher training failed. Check $LOG."
        exit 1
    }
    "Teacher training complete." | Out-File -FilePath $LOG -Append
}
else {
    Write-Host "[1/5] SKIPPED (RUN_TEACHER=0)"
}

# Verify teacher checkpoint
if (-Not (Test-Path $TEACHER_PATH)) {
    Write-Error "ERROR: Teacher checkpoint not found: $TEACHER_PATH`nEither run teacher training (RUN_TEACHER=1) or update TEACHER_PATH."
    exit 1
}

# ==============================================================================
# [2/5] PURSUhInT Step 1: Extract layer representations
# ==============================================================================
if ($RUN_HINTS -eq 1) {
    Write-Host "[2/5] Storing $NUM_LAYERS teacher layer representations..."
    "[2/5] Storing teacher layer representations (${NUM_LAYERS} sub-blocks)..." | Out-File -FilePath $LOG -Append

    for ($i = 1; $i -le $NUM_LAYERS; $i++) {
        Write-Host "  Extracting hint $i / $NUM_LAYERS..."
        & $PYTHON store_hints_imagenet100.py `
            --g $i `
            --model_t $MODEL_T `
            --t_path $TEACHER_PATH `
            --hints_path $HINTS_DIR `
            --data_folder $IMAGENET_DIR `
            --batch_size $BATCH `
            --num_workers $NUM_WORKERS *>> $LOG
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "ERROR: store_hints_imagenet100.py failed on layer ${i}. Check $LOG."
            exit 1
        }
    }
    "Layer representation extraction complete." | Out-File -FilePath $LOG -Append
}
else {
    Write-Host "[2/5] SKIPPED (RUN_HINTS=0)"
}

# ==============================================================================
# [3/5] PURSUhInT Step 2: Cluster representations with K-Means
# ==============================================================================
if ($RUN_CLUSTERING -eq 1) {
    Write-Host "[3/5] Clustering layer representations with $METRIC ($NUM_CLUSTERS clusters)..."
    "[3/5] Running K-Means clustering..." | Out-File -FilePath $LOG -Append

    & $PYTHON k_means.py `
        --hints_dir $HINTS_DIR `
        --num_clusters $NUM_CLUSTERS `
        --num_layers $NUM_LAYERS `
        --metric_name $METRIC `
        --output_dir .\save\hints `
        --model_name $MODEL_T *>> $LOG

    if ($LASTEXITCODE -ne 0) {
        Write-Error "ERROR: k_means.py clustering failed. Check $LOG."
        exit 1
    }

    $CENTROID_FILE = ".\save\hints\${MODEL_T}_${NUM_CLUSTERS}clusters_${METRIC}_centroids.txt"
    if (-Not (Test-Path $CENTROID_FILE)) {
        Write-Error "ERROR: Centroid file not found: $CENTROID_FILE"
        exit 1
    }

    $HINT_POINTS = (Get-Content $CENTROID_FILE -Raw).Trim()
    Write-Host "PURSUhInT selected hint points: $HINT_POINTS"
    "PURSUhInT selected hint points: $HINT_POINTS" | Out-File -FilePath $LOG -Append
}
else {
    Write-Host "[3/5] SKIPPED (RUN_CLUSTERING=0)"
    $HINT_POINTS = "8,11,12,16"
    Write-Host "Using default hint points: $HINT_POINTS"
}

# ==============================================================================
# [4/5] Train Dual Students (CP + Tucker) using discovered hint points
# ==============================================================================
if ($RUN_TRAINING -eq 1) {
    Write-Host "[4/5] Training Dual Students ($MODEL_S) with hint points: $HINT_POINTS..."
    "[4/5] Training students with hint_points=${HINT_POINTS}..." | Out-File -FilePath $LOG -Append

    # In PowerShell, using an array for complex command arguments prevents string parsing bugs
    $TRAIN_ARGS = @(
        "train_stu_imagenet100.py",
        "--model_s", $MODEL_S,
        "--model_t", $MODEL_T,
        "--path_t", $TEACHER_PATH,
        "--distill", "pursuhint_bsat",
        "--dual_bsat",
        "--trial", $TRIAL,
        "--cp_rank_ratio", $CP_RANK_RATIO,
        "--tucker_rank_ratio", $TUCKER_RANK_RATIO,
        "--bsat_rank", $BSAT_RANK,
        "--bsat_coupling_weight", $BSAT_COUPLING_WEIGHT,
        "--epochs", $STUDENT_EPOCHS,
        "--lr", $LR,
        "--weight-decay", $WEIGHT_DECAY,
        "--batch-size", $BATCH,
        "--workers", $NUM_WORKERS,
        "--hint_points", $HINT_POINTS,
        "--gamma", $GAMMA,
        "--alpha", $ALPHA,
        "--beta", $BETA
    )

    if ($USE_NO_DALI -eq 1) {
        $TRAIN_ARGS += "--no_dali"
        Write-Host "  Using standard PyTorch DataLoader (no DALI)"
    }

    if ($ENABLE_DYNAMIC_LOSS_WEIGHTS -eq 1) {
        $TRAIN_ARGS += "--dynamic_loss_weights"
        Write-Host "  Dynamic loss weighting enabled (Kendall et al.)"
    }

    if ($USE_VBMF -eq 1) {
        $TRAIN_ARGS += "--use_vbmf"
        Write-Host "  VBMF rank selection enabled (teacher weight spectrum)"
    }

    $TRAIN_ARGS += $IMAGENET_DIR

    & $PYTHON $TRAIN_ARGS *>> $LOG

    if ($LASTEXITCODE -ne 0) {
        Write-Error "ERROR: Student training failed. Check $LOG."
        exit 1
    }

    "Student training complete." | Out-File -FilePath $LOG -Append
}
else {
    Write-Host "[4/5] SKIPPED (RUN_TRAINING=0)"
}

# ==============================================================================
# [5/5] Evaluation
# ==============================================================================
if ($RUN_EVALUATION -eq 1) {
    Write-Host "[5/5] Running Evaluation Metrics..."
    "[5/5] Running evaluation..." | Out-File -FilePath $LOG -Append

    $STUDENT_DIR = ".\save\student_model\imagenet100\${HINT_POINTS}\S-${MODEL_S}_T-${MODEL_T}_imagenet_pursuhint_bsat_r-${GAMMA}_a-${ALPHA}_b-${BETA}_${TRIAL}"

    $EVAL_ARGS = @(
        "evaluate_metrics.py",
        "--dataset", $DATASET,
        "--data_folder", $IMAGENET_DIR,
        "--model_t", $MODEL_T,
        "--path_t", $TEACHER_PATH,
        "--model_s", $MODEL_S,
        "--path_s_cp", "$STUDENT_DIR\${MODEL_S}_best_cp.pth",
        "--path_s_tucker", "$STUDENT_DIR\${MODEL_S}_best_tucker.pth",
        "--cp_rank_ratio", $CP_RANK_RATIO,
        "--tucker_rank_ratio", $TUCKER_RANK_RATIO
    )
    if ($USE_VBMF -eq 1) { $EVAL_ARGS += "--use_vbmf" }

    & $PYTHON $EVAL_ARGS *>> $LOG

    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Evaluation failed. Check $LOG." -ForegroundColor Yellow
    }
}
else {
    Write-Host "[5/5] SKIPPED (RUN_EVALUATION=0)"
}

# ==============================================================================
# Done
# ==============================================================================
Write-Host ""
Write-Host "============================================================"
Write-Host "Pipeline complete! Results logged to: $LOG"
Write-Host "PURSUhInT hint points used: $HINT_POINTS"
Write-Host "============================================================"
