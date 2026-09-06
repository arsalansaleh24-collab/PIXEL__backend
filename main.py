import os
import uuid
import shutil
import json
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models.image_effort import analyze_image
from models.video_temporal import analyze_video

app = FastAPI(title="Deepfake API")

# frontend stuff
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # TODO: change this in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def build_resp(res, is_vid):
    conf = res['confidence']
    conf_pct = round(conf * 100, 1)
    
    # gray zone
    if 0.4 <= conf <= 0.6 and not res['is_fake']:
        return {
            "status": "UNCERTAIN",
            "confidence_score": f"{conf_pct}%",
            "details": {
                "message": "not sure tbh",
                "detector": res.get('detector', 'idk'),
                "frames_analyzed": res.get('frames_analyzed', None),
                "face_detected": "Yes" if res.get('face_detected') else "No",
                "temporal_consistency": "Inconclusive" if is_vid else None
            }
        }
    
    class_str = "DEEPFAKE" if res['is_fake'] else "AUTHENTIC"
    
    msg = "fake face found" if is_vid and res['is_fake'] else "ai pic detected" if res['is_fake'] else "looks real"
    temp = "Suspicious" if is_vid and res['is_fake'] else "Consistent" if is_vid else None
    
    out = {
        "status": "ANALYSIS COMPLETE",
        "classification": class_str,
        "confidence_score": f"{conf_pct}%",
        "details": {
            "message": msg,
            "detector": res.get('detector', 'idk'),
            "face_detected": "Yes" if res.get('face_detected') else "No",
        }
    }
    
    if is_vid:
        out["details"]["frames_analyzed"] = res.get('frames_analyzed', 0)
        out["details"]["temporal_consistency"] = temp
    
    return out

def log_meta(fid, orig_name, is_vid, res):
    # save data for flywheel
    row = {
        "id": fid,
        "timestamp": datetime.utcnow().isoformat(),
        "original_filename": orig_name,
        "type": "video" if is_vid else "image",
        "predicted_fake": res['is_fake'],
        "confidence": res['confidence'],
        "detector": res.get('detector', 'idk'),
    }
    
    with open(META_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    ctype = file.content_type or ""
    
    is_img = ctype in img_types
    is_vid = ctype in vid_types
    
    if not is_img and not is_vid:
        # print("bad file type:", ctype)
        raise HTTPException(status_code=400, detail="bad file type")
    
    fid = str(uuid.uuid4())
    ext = Path(file.filename).suffix if file.filename else ".tmp"
    tmp_path = UP_DIR / f"{fid}{ext}"
    
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        if is_img:
            res = analyze_image(str(tmp_path))
            dest = IMG_DIR
        else:
            res = analyze_video(str(tmp_path))
            dest = VID_DIR
            
        log_meta(fid, file.filename or "unknown", is_vid, res)
        
        # move to collected folder
        final_path = dest / f"{fid}{ext}"
        shutil.move(str(tmp_path), str(final_path))
        
        return build_resp(res, is_vid)
    
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

