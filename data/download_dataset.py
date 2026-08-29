import os
import sys
import shutil
import subprocess
from pathlib import Path

# pulls the 140k dataset and sets up the folder structure
# needs kaggle.json in ~/.kaggle/ to work

DATA_ID = "xhlulu/140k-real-and-fake-faces"
TEMP_DIR = "kaggle_dl_temp"

def setup():
    # check if key exists so it doesn't just crash on download
    key_path = Path.home() / ".kaggle" / "kaggle.json"
    if not key_path.exists():
        print("missing kaggle API key!")
        print("1. go to kaggle.com -> settings -> create new token")
        print("2. put kaggle.json in ~/.kaggle/ (mac/linux) or C:\\Users\\<User>\\.kaggle\\ (windows)")
        return False
        
    # make sure the cli is actually installed
    try:
        import kaggle
    except ImportError:
        print("kaggle cli missing, installing real quick...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])
        
    return True

def main():
    if not setup():
        sys.exit(1)
        
    print(f"downloading {DATA_ID} (this might take a minute)...")
    subprocess.run([
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", DATA_ID, "-p", TEMP_DIR, "--unzip"
    ], check=True)
    
    # the zip extracts into a really deep nested folder structure
    # so we just walk the tree to find the train directory dynamically
    train_path = None
    for root, dirs, _ in os.walk(TEMP_DIR):
        if "train" in dirs:
            # verify it's the right train folder by checking inside
            if "real" in os.listdir(Path(root) / "train"):
                train_path = Path(root) / "train"
                break
                
    if not train_path:
        print("error: couldn't find train/real folders in the downloaded zip")
        sys.exit(1)
        
    src_real = train_path / "real"
    src_fake = train_path / "fake"
    
    dst_real = Path("data/raw/train/real")
    dst_fake = Path("data/raw/train/fake")
    
    dst_real.mkdir(parents=True, exist_ok=True)
    dst_fake.mkdir(parents=True, exist_ok=True)
    
    # count files for a sanity check
    real_files = os.listdir(src_real)
    fake_files = os.listdir(src_fake)
    
    print(f"found {len(real_files)} real and {len(fake_files)} fake faces.")
    print("moving files over...")
    
    # use move instead of copy so we aren't duplicating 10GB of data
    for f in real_files:
        shutil.move(src_real / f, dst_real / f)
    for f in fake_files:
        shutil.move(src_fake / f, dst_fake / f)
        
    print("cleaning up temp folders...")
    shutil.rmtree(TEMP_DIR)
    
    print("dataset is ready to go!")

if __name__ == "__main__":
    main()