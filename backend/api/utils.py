import json
from datetime import datetime
from api.config import META_FILE

def build_resp(res, is_vid):
    conf = res['confidence']
    conf_pct = round(conf * 100, 1)
    
    # gray zone

    
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
