import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

# set up mtcnn ok
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
extractor = MTCNN(image_size=224, margin=20, keep_all=False, select_largest=True, device=device)

def process_folder(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    valid_exts = ('.png', '.jpg', '.jpeg')
    files = [f for f in os.listdir(in_dir) if f.lower().endswith(valid_exts)]
    
    print(f"-> processing {len(files)} files in {in_dir}")
    
    for f in tqdm(files):
        in_path = os.path.join(in_dir, f)
        out_path = os.path.join(out_dir, f)

        if os.path.exists(out_path):
            continue
        
        try:
            img = Image.open(in_path).convert('RGB')
            extractor(img, save_path=out_path)
        except Exception:
            # skip can add something later in here 
            pass

if __name__ == "__main__":
    dirs = [
        ('data/raw/train/real', 'data/cropped/train/real'),
        ('data/raw/train/fake', 'data/cropped/train/fake')
    ]
    
    for in_path, out_path in dirs:
        process_folder(in_path, out_path)