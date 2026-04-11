import os
import shutil

# Paths
src_dir = '/data/imagenet'      # full 150GB dataset
dest_dir = '/data/imagenet100'  # 15GB subset
list_path = 'imagenet100.txt'

# Load the 100 official classes
with open(list_path, 'r') as f:
    classes = [line.strip() for line in f.readlines() if line.strip()]

# Copy the folders
for split in ['train', 'val']:
    os.makedirs(os.path.join(dest_dir, split), exist_ok=True)
    
    print(f"Extracting {split} split...")
    for c in classes:
        src_class = os.path.join(src_dir, split, c)
        dest_class = os.path.join(dest_dir, split, c)
        
        if os.path.exists(src_class):
            shutil.copytree(src_class, dest_class)
        else:
            print(f"Warning: Could not find {src_class}")

print("ImageNet-100 extraction complete!")
