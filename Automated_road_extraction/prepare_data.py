import os
import shutil
import glob
import random
from pathlib import Path

def prepare_data(source_dir, output_dir, split_ratio=0.2):
    """
    Scans source_dir for image/mask pairs and splits them into train/test folders.
    structure:
        source_dir/
            *_sat.jpg
            *_mask.png
    """
    
    # 1. Setup Directories
    for split in ['train', 'test']:
        for type_ in ['images', 'masks']:
            os.makedirs(os.path.join(output_dir, split, type_), exist_ok=True)
            
    print(f"Scanning {source_dir}...")
    
    # 2. Find Pairs
    # Kaggle dataset convention seems to be: {id}_sat.jpg and {id}_mask.png
    all_files = os.listdir(source_dir)
    sat_images = [f for f in all_files if f.endswith('_sat.jpg')]
    
    pairs = []
    for sat_img in sat_images:
        # Construct corresponding mask name
        img_id = sat_img.replace('_sat.jpg', '')
        mask_name = f"{img_id}_mask.png"
        
        if mask_name in all_files:
            pairs.append((sat_img, mask_name))
            
    print(f"Found {len(pairs)} valid image/mask pairs.")
    
    if len(pairs) == 0:
        print("No pairs found! Check directory structure.")
        return

    # 3. Shuffle and Split
    random.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - split_ratio))
    
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]
    
    print(f"Splitting: {len(train_pairs)} Train, {len(test_pairs)} Test")
    
    # 4. Copy Files
    def copy_set(pair_list, split_name):
        print(f"Copying {split_name} data...")
        for sat, mask in pair_list:
            # Copy Image
            src_img = os.path.join(source_dir, sat)
            dst_img = os.path.join(output_dir, split_name, 'images', sat)
            shutil.copy2(src_img, dst_img)
            
            # Copy Mask
            src_mask = os.path.join(source_dir, mask)
            dst_mask = os.path.join(output_dir, split_name, 'masks', mask)
            shutil.copy2(src_mask, dst_mask)
            
    copy_set(train_pairs, 'train')
    copy_set(test_pairs, 'test')
    
    print("\n[DONE] Data preparation complete!")
    print(f"Train Images: {os.path.abspath(os.path.join(output_dir, 'train', 'images'))}")
    print(f"Train Masks:  {os.path.abspath(os.path.join(output_dir, 'train', 'masks'))}")
    print(f"Test Images:  {os.path.abspath(os.path.join(output_dir, 'test', 'images'))}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for training/testing")
    parser.add_argument("--source", type=str, default="raw_data", help="Source directory containing raw images and masks")
    parser.add_argument("--output", type=str, default="data", help="Output directory for processed data")
    
    args = parser.parse_args()
    
    SOURCE = os.path.abspath(args.source)
    OUTPUT = os.path.abspath(args.output)
    
    if not os.path.exists(SOURCE):
        print(f"Error: Source directory {SOURCE} not found.")
        print("Please provide a valid source directory using --source or create a 'raw_data' folder.")
    else:
        prepare_data(SOURCE, OUTPUT)
