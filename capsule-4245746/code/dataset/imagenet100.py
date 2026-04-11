"""
Utility script to extract ImageNet-100 (a 100-class subset) from the full ImageNet dataset.

The 100 class WordNet IDs are read from imagenet100.txt (same directory as this script).
Both train/ and val/ splits are copied.

Usage:
    python dataset/imagenet100.py --src_dir /data/imagenet --dest_dir /data/imagenet100
"""

import os
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser(description='Extract ImageNet-100 subset from full ImageNet')
    parser.add_argument('--src_dir', type=str, default='/data/imagenet',
                        help='Path to the full ImageNet dataset (must have train/ and val/ subdirs)')
    parser.add_argument('--dest_dir', type=str, default='/data/imagenet100',
                        help='Destination path for the ImageNet-100 subset')
    opt = parser.parse_args()

    # Resolve imagenet100.txt path relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    list_path = os.path.join(script_dir, 'imagenet100.txt')

    if not os.path.isfile(list_path):
        raise FileNotFoundError(f"Class list file not found: {list_path}")

    # Load the 100 official classes
    with open(list_path, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Loaded {len(classes)} classes from {list_path}")
    print(f"Source: {opt.src_dir}")
    print(f"Destination: {opt.dest_dir}")

    # Copy the folders
    for split in ['train', 'val']:
        split_dest = os.path.join(opt.dest_dir, split)
        os.makedirs(split_dest, exist_ok=True)

        print(f"\nExtracting {split} split...")
        for c in classes:
            src_class = os.path.join(opt.src_dir, split, c)
            dest_class = os.path.join(opt.dest_dir, split, c)

            if os.path.exists(dest_class):
                print(f"  Skipping {c} (already exists)")
                continue

            if os.path.exists(src_class):
                shutil.copytree(src_class, dest_class)
            else:
                print(f"  Warning: Could not find {src_class}")

    print("\nImageNet-100 extraction complete!")
    print(f"  {opt.dest_dir}/train/ and {opt.dest_dir}/val/ are ready.")


if __name__ == '__main__':
    main()
