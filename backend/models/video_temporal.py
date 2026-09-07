import cv2
import torch
import torch.nn as nn
import numpy as np
import timm
from pathlib import Path
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from models import DEVICE

from models.video_model import DeepfakeVideoDetector

VIDEO_WEIGHTS_PATH = Path(__file__).parent / "best_video_lstm.pth"
VIDEO_CHECKPOINT_PATH = Path(__file__).parent / "best_video_lstm_checkpoint.pth"

_video_model = None
_mtcnn = None
_best_threshold = 0.5  # default, overridden by checkpoint if available

def _get_video_model():
    global _video_model, _best_threshold
    if _video_model is None:
        _video_model = DeepfakeVideoDetector()
        
        if VIDEO_WEIGHTS_PATH.exists():
            state = torch.load(VIDEO_WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
            _video_model.load_state_dict(state)
            print(f"loaded weights from {VIDEO_WEIGHTS_PATH}")
        else:
            print(f"warning: {VIDEO_WEIGHTS_PATH} not found, using untrained model for video")
        
        # Load the optimal decision threshold from the training checkpoint
        if VIDEO_CHECKPOINT_PATH.exists():
            try:
                ckpt = torch.load(VIDEO_CHECKPOINT_PATH, map_location='cpu', weights_only=False)
                _best_threshold = ckpt.get('best_threshold', 0.5)
                print(f"loaded best threshold from checkpoint: {_best_threshold:.4f}")
            except Exception as e:
                print(f"warning: couldn't load threshold from checkpoint: {e}")
            
        _video_model = _video_model.to(DEVICE)
        _video_model.eval()
    return _video_model

def _get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        # use cpu for mtcnn even if we have gpu, must match training args exactly
        _mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, select_largest=True, post_process=False, device='cpu')
    return _mtcnn

_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def _sample_frames(video_path, num_frames=8):
    """grab evenly spaced frames from the video"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total <= 0:
        cap.release()
        return []
    
    # pick frame indices evenly across the video
    if total >= num_frames:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
    else:
        # pad short vids with the last frame
        indices = list(range(total)) + [total - 1] * (num_frames - total)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # opencv gives bgr, pil wants rgb
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames

def analyze_video(video_path):
    """
    extract faces, pass seq to cnn+lstm
    """
    model = _get_video_model()
    mtcnn = _get_mtcnn()
    
    frames = _sample_frames(video_path, num_frames=8)
    print(f"[DEBUG] Sampled {len(frames)} frames from video")
    
    if not frames:
        return {
            'confidence': 0.5,
            'is_fake': False,
            'face_detected': False,
            'detector': 'Video Temporal Pipeline',
            'frames_analyzed': 0
        }
    
    face_tensors = []
    faces_found = 0
    
    for i, frame in enumerate(frames):
        # try to get a face out of this frame
        try:
            face = mtcnn(frame)
        except Exception:
            face = None
        
        if face is not None:
            faces_found += 1
            # mtcnn outputs 0-255 roughly (post_process=False), scale to 0-1
            print(f"[DEBUG] Frame {i}: face found, raw range [{face.min().item():.2f}, {face.max().item():.2f}]")
            face = face / 255.0
            tensor = _transform(face)
            print(f"[DEBUG] Frame {i}: after norm range [{tensor.min().item():.2f}, {tensor.max().item():.2f}]")
            face_tensors.append(tensor)
        else:
            # pad with zeros to ensure sequence length remains exactly 8 (like in training)
            print(f"[DEBUG] Frame {i}: no face, padding with zeros")
            face_tensors.append(torch.zeros(3, 224, 224))
    
    print(f"[DEBUG] Total faces found: {faces_found}/{len(frames)}")
    
    if faces_found == 0:
        return {
            'confidence': 0.5,
            'is_fake': False,
            'face_detected': False,
            'detector': 'Video Temporal Pipeline',
            'frames_analyzed': len(frames)
        }
        
    # batch it up (batch size 1)
    batch_tensor = torch.stack(face_tensors).unsqueeze(0).to(DEVICE)
    print(f"[DEBUG] Input tensor shape: {batch_tensor.shape}")
    
    with torch.no_grad():
        output = model(batch_tensor)
        probs = torch.softmax(output, dim=1)
    
    print(f"[DEBUG] Raw logits: {output[0].tolist()}")
    print(f"[DEBUG] Softmax probs: {probs[0].tolist()}")
        
    # class 0 = real, class 1 = fake
    real_prob = probs[0][0].item()
    fake_prob = probs[0][1].item()
    
    # Use the optimal threshold from training instead of a naive 0.5
    # The training script grid-searched 99 thresholds on the validation set
    # to find the one that maximizes F1 score
    is_fake = fake_prob >= _best_threshold
    confidence = fake_prob if is_fake else real_prob
    
    print(f"[DEBUG] real_prob={real_prob:.4f}, fake_prob={fake_prob:.4f}, "
          f"threshold={_best_threshold:.4f}, is_fake={is_fake}, confidence={confidence:.4f}")
    
    return {
        'confidence': confidence,
        'is_fake': is_fake,
        'face_detected': True,
        'detector': 'Video Temporal Pipeline',
        'frames_analyzed': len(frames),
        'faces_found': faces_found
    }

