import json
import os
import shutil
import random

source_dir = 'data/temp_dfdc'
metadata_file = os.path.join(source_dir, 'metadata.json')

with open(metadata_file, 'r') as f:
    metadata = json.load(f)

train_real_dir = 'data/raw_videos/train/real'
train_fake_dir = 'data/raw_videos/train/fake'
val_real_dir = 'data/raw_videos/val/real'
val_fake_dir = 'data/raw_videos/val/fake'

os.makedirs(train_real_dir, exist_ok=True)
os.makedirs(train_fake_dir, exist_ok=True)
os.makedirs(val_real_dir, exist_ok=True)
os.makedirs(val_fake_dir, exist_ok=True)

# Process files
for filename, info in metadata.items():
    src_path = os.path.join(source_dir, filename)
    if not os.path.exists(src_path):
        continue
        
    label = info.get('label', '').upper()
    
    # 80/20 train/val split since DFDC train subset metadata usually says "train" for all
    is_train = random.random() < 0.8
    
    if label == 'REAL':
        dest_dir = train_real_dir if is_train else val_real_dir
    else:
        dest_dir = train_fake_dir if is_train else val_fake_dir
        
    dest_path = os.path.join(dest_dir, filename)
    shutil.move(src_path, dest_path)

print("Finished sorting videos.")
