@echo off
setlocal enabledelayedexpansion

rem Suppress TensorFlow/TensorBoard C++ and oneDNN log noise
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0

echo ==========================================================================
echo Starting FULL PURSUhInT + CMTF Pipeline (Teacher - Hints - Student - Eval)
echo ==========================================================================

if not exist ".\save\logs" mkdir ".\save\logs"
if not exist ".\save\hints" mkdir ".\save\hints"
if not exist ".\save\models" mkdir ".\save\models"
if not exist ".\save\student_model" mkdir ".\save\student_model"

set PYTHON=..\..\.venv\Scripts\python.exe
set DATASET=cifar100
set MODEL_T=wrn_40_2
set MODEL_S=wrn_16_2
set LOG=.\save\logs\%MODEL_T%_%DATASET%_log.txt
set TRIAL=0
set NUM_LAYERS=18
set NUM_CLUSTERS=3
set METRIC=r2
set EPOCHS=1
set LEARNING_RATE=0.05
set WEIGHT_DECAY=0.0005
set LR_DECAY_EPOCHS=60,120,160
set BATCH=64
set NUM_WORKERS=4

set HINTS_DIR=.\save\hints\%MODEL_T%_%DATASET%_best
set TEACHER_PATH=.\save\models\%MODEL_T%_%DATASET%_lr_%LEARNING_RATE%_decay_%WEIGHT_DECAY%_trial_%TRIAL%\%MODEL_T%_best.pth

REM ============================================================
REM Student training loss weights
REM ============================================================
set GAMMA=1.0
set ALPHA=4.0
set BETA=25.0

REM ============================================================
REM CP / Tucker Rank Ratios and CMTF Rank
REM (used when ENABLE_VBMF=0; ignored for rank selection when ENABLE_VBMF=1)
REM ============================================================
set CP_RANK_RATIO=0.5
set TUCKER_RANK_RATIO=0.25
set CMTF_RANK=8
set CMTF_COUPLING_WEIGHT=1.0

REM ============================================================
REM VBMF automatic rank selection (1=on, 0=off)
REM ============================================================
set ENABLE_VBMF=1

REM ============================================================
REM Dynamic loss weighting (Kendall et al. CVPR 2018) ON/OFF
REM ============================================================
set ENABLE_DYNAMIC_LOSS_WEIGHTS=1

REM ============================================================
REM PyTorch Compile Optimization (1=on, 0=off)
REM ============================================================
set ENABLE_TORCH_COMPILE=0

REM ---- Derived flag strings (do not edit below this line) ----
set TEACHER_FLAGS=
if %ENABLE_TORCH_COMPILE% == 1 set TEACHER_FLAGS=--torch_compile

set EXTRA_FLAGS=
if %ENABLE_DYNAMIC_LOSS_WEIGHTS% == 1 set EXTRA_FLAGS=%EXTRA_FLAGS% --dynamic_loss_weights
if %ENABLE_VBMF% == 1 set EXTRA_FLAGS=%EXTRA_FLAGS% --use_vbmf
if %ENABLE_TORCH_COMPILE% == 1 set EXTRA_FLAGS=%EXTRA_FLAGS% --torch_compile

set EVAL_FLAGS=
if %ENABLE_VBMF% == 1 set EVAL_FLAGS=%EVAL_FLAGS% --use_vbmf
if %ENABLE_TORCH_COMPILE% == 1 set EVAL_FLAGS=%EVAL_FLAGS% --torch_compile

REM ============================================================
REM [1/5] Train Teacher
REM ============================================================
echo [1/5] Training Teacher (%MODEL_T%) on %DATASET%...
%PYTHON% train_teacher.py ^
    --dataset %DATASET% ^
    --model %MODEL_T% ^
    --epochs %EPOCHS% ^
    --learning_rate %LEARNING_RATE% ^
    --weight_decay %WEIGHT_DECAY% ^
    --lr_decay_epochs %LR_DECAY_EPOCHS% ^
    --batch_size %BATCH% ^
    --num_workers %NUM_WORKERS% ^
    --trial %TRIAL% %TEACHER_FLAGS% >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERROR: Teacher training failed. Check %LOG%.
    goto :error
)
echo Teacher training complete. >> %LOG%

REM ============================================================
REM [2/5] PURSUhInT Step 1: Extract layer representations
REM        Run store_hints.py for every sub-block (1..NUM_LAYERS)
REM ============================================================
echo [2/5] PURSUhInT Step 1 - Storing %NUM_LAYERS% teacher layer representations...
echo [2/5] Storing teacher layer representations (%NUM_LAYERS% sub-blocks)... >> %LOG% 2>&1

for /L %%i in (1, 1, %NUM_LAYERS%) do (
    echo   Extracting hint %%i / %NUM_LAYERS%...
    %PYTHON% store_hints.py ^
        --g %%i ^
        --model_t %MODEL_T% ^
        --t_path %TEACHER_PATH% ^
        --dataset %DATASET% ^
        --hints_path %HINTS_DIR% >> %LOG% 2>&1
    if errorlevel 1 (
        echo ERROR: store_hints.py failed on layer %%i. Check %LOG%.
        goto :error
    )
)
echo Layer representation extraction complete. >> %LOG%

REM ============================================================
REM [3/5] PURSUhInT Step 2: Cluster representations with R2_CCA
REM        Output: centroid indices as comma-separated string
REM ============================================================
echo [3/5] PURSUhInT Step 2 - Clustering layer representations with %METRIC%...
echo [3/5] Running K-Means clustering (%NUM_CLUSTERS% clusters, metric=%METRIC%)... >> %LOG% 2>&1

%PYTHON% k_means.py ^
    --hints_dir %HINTS_DIR% ^
    --num_clusters %NUM_CLUSTERS% ^
    --num_layers %NUM_LAYERS% ^
    --metric_name %METRIC% ^
    --output_dir .\save\hints ^
    --model_name %MODEL_T% >> %LOG% 2>&1

if errorlevel 1 (
    echo ERROR: k_means.py clustering failed. Check %LOG%.
    goto :error
)

REM Read the centroid file -> HINT_POINTS variable
set CENTROID_FILE=.\save\hints\%MODEL_T%_%NUM_CLUSTERS%clusters_%METRIC%_centroids.txt
if not exist %CENTROID_FILE% (
    echo ERROR: Centroid file not found: %CENTROID_FILE%
    goto :error
)

set /p HINT_POINTS=<%CENTROID_FILE%
echo PURSUhInT selected hint points: %HINT_POINTS% >> %LOG%
echo PURSUhInT selected hint points: %HINT_POINTS%

REM ============================================================
REM [4/5] Train Dual Students (CP + Tucker) using discovered hint points
REM ============================================================
echo [4/5] Training Dual Students (%MODEL_S%) with PURSUhInT hint points: %HINT_POINTS%...
echo [4/5] Training students with hint_points=%HINT_POINTS%... >> %LOG% 2>&1

set TRAIN_CMD=%PYTHON% train_student.py ^
    --dataset %DATASET% ^
    --model_s %MODEL_S% ^
    --model_t %MODEL_T% ^
    --path_t %TEACHER_PATH% ^
    --distill pursuhint_cmtf ^
    --dual_cmtf ^
    --trial %TRIAL% ^
    --cp_rank_ratio %CP_RANK_RATIO% ^
    --tucker_rank_ratio %TUCKER_RANK_RATIO% ^
    --cmtf_rank %CMTF_RANK% ^
    --cmtf_coupling_weight %CMTF_COUPLING_WEIGHT% ^
    --epochs %EPOCHS% ^
    --learning_rate %LEARNING_RATE% ^
    --weight_decay %WEIGHT_DECAY% ^
    --lr_decay_epochs %LR_DECAY_EPOCHS% ^
    --batch_size %BATCH% ^
    --hint_points %HINT_POINTS% ^
    --num_workers %NUM_WORKERS% ^
    -r %GAMMA% -a %ALPHA% -b %BETA%

%TRAIN_CMD% %EXTRA_FLAGS% >> %LOG% 2>&1

if errorlevel 1 (
    echo ERROR: Student training failed. Check %LOG%.
    goto :error
)

REM ============================================================
REM [5/5] Evaluation
REM ============================================================
echo [5/5] Running Evaluation Metrics...

REM We must use delayed expansion (!HINT_POINTS!) because the hint variable is loaded dynamically
set STUDENT_DIR=.\save\student_model\%DATASET%\!HINT_POINTS!\S-%MODEL_S%_T-%MODEL_T%_%DATASET%_pursuhint_cmtf_r-%GAMMA%_a-%ALPHA%_b-%BETA%_b2-1.0_%TRIAL%

%PYTHON% evaluate_metrics.py ^
    --dataset %DATASET% ^
    --model_t %MODEL_T% ^
    --path_t "%TEACHER_PATH%" ^
    --model_s %MODEL_S% ^
    --path_s_cp "!STUDENT_DIR!\%MODEL_S%_best_cp.pth" ^
    --path_s_tucker "!STUDENT_DIR!\%MODEL_S%_best_tucker.pth" ^
    --cp_rank_ratio %CP_RANK_RATIO% ^
    --tucker_rank_ratio %TUCKER_RANK_RATIO% %EVAL_FLAGS% >> %LOG% 2>&1

if errorlevel 1 (
    echo WARNING: Evaluation failed. Check %LOG%.
)

echo.
echo ============================================================
echo Pipeline complete! Results logged to: %LOG%
echo PURSUhInT hint points used: %HINT_POINTS%
echo ============================================================
goto :eof

:error
echo.
echo ============================================================
echo CRITICAL ERROR: The pipeline terminated early.
echo Please check %LOG% for details.
echo ============================================================
exit /b 1
