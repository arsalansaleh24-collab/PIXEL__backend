from pathlib import Path

UP_DIR = Path('uploads')
UP_DIR.mkdir(exist_ok=True)

# mlops dirs
COLLECT_DIR = Path('data/collected')
IMG_DIR = COLLECT_DIR / 'images'
VID_DIR = COLLECT_DIR / 'videos'
META_FILE = COLLECT_DIR / 'metadata.jsonl'

IMG_DIR.mkdir(parents=True, exist_ok=True)
VID_DIR.mkdir(parents=True, exist_ok=True)

img_types = ['image/jpeg', 'image/png', 'image/webp']
vid_types = ['video/mp4', 'video/quicktime', 'video/webm']
