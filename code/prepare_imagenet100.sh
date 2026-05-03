#!/usr/bin/env bash
# Downloads and prepares the ImageNet100 dataset from Kaggle.
# Output: ./data/imagenet100/{train,val}/<class_folders>  +  Labels.json
#
# Requirements: kaggle CLI configured (~/.kaggle/kaggle.json with your API key)
# Usage: bash prepare_imagenet100.sh

set -euo pipefail

DATASET_SLUG="ambityga/imagenet100"
TARGET_DIR="./data/imagenet100"
TMP_DIR="./data/_imagenet100_tmp"
ZIP_FILE="./data/imagenet100.zip"

# ── 1. Check prerequisites ────────────────────────────────────────────────────
if ! command -v kaggle &>/dev/null; then
    echo "ERROR: kaggle CLI not found. Install it with: pip install kaggle"
    echo "       Then place your API key at ~/.kaggle/kaggle.json"
    exit 1
fi

if ! command -v unzip &>/dev/null; then
    echo "ERROR: 'unzip' command not found. Please install it (e.g., apt install unzip)."
    exit 1
fi

mkdir -p "./data"

# ── 2. Download and Extract ───────────────────────────────────────────────────
if [ ! -f "$ZIP_FILE" ]; then
    echo "==> Downloading ImageNet100 from Kaggle..."
    kaggle datasets download -d "$DATASET_SLUG" -p "./data"
else
    echo "==> Zip file already exists on disk. Skipping download!"
fi

if [ -f "$ZIP_FILE" ]; then
    echo "==> Extracting archive directly into RAM (/dev/shm)..."
    unzip -q "$ZIP_FILE" -d "$TMP_DIR"
    # Notice we removed `rm "$ZIP_FILE"` so it stays on your disk for next time!
else
    echo "ERROR: Download failed, $ZIP_FILE not found."
    exit 1
fi

# ── 3. Locate the extracted root (handles one extra nesting level) ────────────
EXTRACTED_ROOT="$TMP_DIR"
if [ ! -d "$EXTRACTED_ROOT/val.X" ] && [ ! -d "$EXTRACTED_ROOT/train.X1" ]; then
    # Look one level deeper, strictly ignoring the parent directory itself
    INNER=$(find "$TMP_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)
    
    if [ -n "$INNER" ] && { [ -d "$INNER/val.X" ] || [ -d "$INNER/train.X1" ]; }; then
        EXTRACTED_ROOT="$INNER"
    else
        echo "ERROR: Could not locate train.X1 or val.X inside the downloaded archive."
        echo "       Contents of $TMP_DIR:"
        ls -la "$TMP_DIR"
        exit 1
    fi
fi

# ── 4. Create target layout ───────────────────────────────────────────────────
echo "==> Setting up $TARGET_DIR ..."
mkdir -p "$TARGET_DIR/train"
mkdir -p "$TARGET_DIR/val"

# ── 5. Merge train.X1 … train.X4 → train/ ─────────────────────────────────────
echo "==> Merging train splits..."
for split in train.X1 train.X2 train.X3 train.X4; do
    SPLIT_DIR="$EXTRACTED_ROOT/$split"
    if [ ! -d "$SPLIT_DIR" ]; then
        echo "WARNING: $split not found, skipping."
        continue
    fi
    echo "    Moving $split ..."
    
    # Each split contains class-named subdirectories; move each one
    for class_dir in "$SPLIT_DIR"/*/; do
        class_name=$(basename "$class_dir")
        if [ -d "$TARGET_DIR/train/$class_name" ]; then
            # Move contents if the directory already exists
            mv "$class_dir"* "$TARGET_DIR/train/$class_name/" 2>/dev/null || true
        else
            # Move the whole directory if it doesn't exist yet
            mv "$class_dir" "$TARGET_DIR/train/"
        fi
    done
done

# ── 6. Move val.X → val/ ──────────────────────────────────────────────────────
echo "==> Moving validation set..."
VAL_SRC="$EXTRACTED_ROOT/val.X"
if [ -d "$VAL_SRC" ]; then
    for class_dir in "$VAL_SRC"/*/; do
        mv "$class_dir" "$TARGET_DIR/val/"
    done
else
    echo "WARNING: val.X not found."
fi

# ── 7. Copy Labels.json ───────────────────────────────────────────────────────
LABELS_SRC="$EXTRACTED_ROOT/Labels.json"
if [ -f "$LABELS_SRC" ]; then
    cp "$LABELS_SRC" "$TARGET_DIR/Labels.json"
    echo "==> Copied Labels.json"
else
    echo "WARNING: Labels.json not found in extracted archive."
fi

# ── 8. Clean up temp directory ────────────────────────────────────────────────
if [ -d "$TMP_DIR" ]; then
    echo "==> Cleaning up temporary files..."
    rm -rf "$TMP_DIR"
fi

# ── 9. Summary ────────────────────────────────────────────────────────────────
TRAIN_CLASSES=$(find "$TARGET_DIR/train" -mindepth 1 -maxdepth 1 -type d | wc -l)
VAL_CLASSES=$(find "$TARGET_DIR/val"   -mindepth 1 -maxdepth 1 -type d | wc -l)
echo ""
echo "Done. Dataset ready at: $TARGET_DIR"
echo "  train classes : $TRAIN_CLASSES"
echo "  val   classes : $VAL_CLASSES"