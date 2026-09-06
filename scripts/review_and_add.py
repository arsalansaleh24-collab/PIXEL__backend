import os
import json
import shutil
from pathlib import Path

COLLECTED_DIR = Path("data/collected")
METADATA_FILE = COLLECTED_DIR / "metadata.jsonl"

RAW_TRAIN_REAL = Path("data/raw/train/real")
RAW_TRAIN_FAKE = Path("data/raw/train/fake")
RAW_TRAIN_VIDEOS_REAL = Path("data/raw/train_videos/real")
RAW_TRAIN_VIDEOS_FAKE = Path("data/raw/train_videos/fake")

def ensure_dirs():
    for d in [RAW_TRAIN_REAL, RAW_TRAIN_FAKE, RAW_TRAIN_VIDEOS_REAL, RAW_TRAIN_VIDEOS_FAKE]:
        d.mkdir(parents=True, exist_ok=True)

def review_collected():
    if not METADATA_FILE.exists():
        print("No collected data found.")
        return

    with open(METADATA_FILE, "r") as f:
        lines = f.readlines()
        
    if not lines:
        print("No collected data found.")
        return

    print(f"Found {len(lines)} collected uploads to review.")
    
    remaining_lines = []
    processed_count = 0
    
    ensure_dirs()
    
    for line in lines:
        entry = json.loads(line.strip())
        file_id = entry['id']
        is_video = entry['type'] == 'video'
        
        # determine extension from the dir
        dir_path = COLLECTED_DIR / ("videos" if is_video else "images")
        
        # find the actual file (we don't know the extension from the metadata alone)
        files = list(dir_path.glob(f"{file_id}.*"))
        if not files:
            print(f"Warning: file for id {file_id} not found on disk. Skipping.")
            continue
            
        source_file = files[0]
        
        print("\n" + "="*50)
        print(f"File: {entry['original_filename']} ({source_file.name})")
        print(f"Type: {entry['type']}")
        print(f"Model Predicted: {'FAKE' if entry['predicted_fake'] else 'REAL'} (Confidence: {entry['confidence']:.2f})")
        print("="*50)
        
        # ask human to review
        while True:
            action = input("Is this [r]eal, [f]ake, [s]kip, or [q]uit? ").lower()
            if action in ['r', 'f', 's', 'q']:
                break
                
        if action == 'q':
            # put this line back in the list to save
            remaining_lines.append(line)
            # append the rest of the lines we didn't get to
            idx = lines.index(line) + 1
            remaining_lines.extend(lines[idx:])
            break
            
        if action == 's':
            # keep it in the log for next time
            remaining_lines.append(line)
            continue
            
        # move the file
        if is_video:
            dest_dir = RAW_TRAIN_VIDEOS_FAKE if action == 'f' else RAW_TRAIN_VIDEOS_REAL
        else:
            dest_dir = RAW_TRAIN_FAKE if action == 'f' else RAW_TRAIN_REAL
            
        dest_file = dest_dir / source_file.name
        
        # handle name collisions
        counter = 1
        while dest_file.exists():
            dest_file = dest_dir / f"{dest_file.stem}_{counter}{dest_file.suffix}"
            counter += 1
            
        shutil.move(str(source_file), str(dest_file))
        print(f"-> Moved to {dest_file}")
        processed_count += 1

    # rewrite the metadata file with only the unprocessed lines
    with open(METADATA_FILE, "w") as f:
        for line in remaining_lines:
            f.write(line)
            
    print(f"\nDone! Processed and moved {processed_count} files into the training dataset.")
    print("You can now re-run `python data/preprocess.py` and `python data/train.py` to train on the new data.")

if __name__ == "__main__":
    review_collected()
