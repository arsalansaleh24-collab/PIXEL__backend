import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

from models.image_effort import analyze_image
from models.video_temporal import analyze_video
from api.config import UP_DIR, IMG_DIR, VID_DIR, img_types, vid_types
from api.utils import build_resp, log_meta

router = APIRouter()

@router.post("/analyze")
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

@router.get("/health")
async def health():
    return {"status": "ok"}
