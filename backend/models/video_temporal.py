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

# cnn + lstm architecture
class VideoTemporalModel(nn.Module):
    def __init__(self, cnn_backbone='efficientnet_b4', hidden_dim=256, num_layers=1, num_classes=2):
        super().__init__()
        # load cnn backbone but drop the classification head so it outputs raw features
        self.cnn = timm.create_model(cnn_backbone, pretrained=True, num_classes=0)
        cnn_out_dim = self.cnn.num_features
        
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # expects shape: (batch, frames, c, h, w)
        batch_size, seq_length, c, h, w = x.size()
        
        # flatten batch and seq so we can pass everything through cnn at once
        x = x.view(batch_size * seq_length, c, h, w)
        features = self.cnn(x)
        
        # put the sequence dimension back
        features = features.view(batch_size, seq_length, -1)
        
        # run through lstm
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # just grab the final hidden state to classify the whole vid
        last_hidden = h_n[-1] # shape: (Batch, hidden_dim)
        
        out = self.fc(last_hidden)
        return out

VIDEO_WEIGHTS_PATH = Path("deepfake_video_lstm.pth")

_video_model = None
_mtcnn = None

def _get_video_model():
    global _video_model
    if _video_model is None:
        _video_model = VideoTemporalModel()
        
        if VIDEO_WEIGHTS_PATH.exists():
            state = torch.load(VIDEO_WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
            _video_model.load_state_dict(state)
            print(f"loaded weights from {VIDEO_WEIGHTS_PATH}")
        else:
            print(f"warning: {VIDEO_WEIGHTS_PATH} not found, using untrained model for video")
            
        _video_model = _video_model.to(DEVICE)
        _video_model.eval()
    return _video_model

def _get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        # use cpu for mtcnn even if we have gpu
        _mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, select_largest=True, device='cpu')
    return _mtcnn

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def _sample_frames(video_path, num_frames=32):
    """grab evenly spaced frames from the video"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total <= 0:
        cap.release()
        return []
    
    # pick frame indices evenly across the video
    num_frames = min(num_frames, total)
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    
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
    
    frames = _sample_frames(video_path, num_frames=32)
    
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
    
    for frame in frames:
        # try to get a face out of this frame
        try:
            face = mtcnn(frame)
        except Exception:
            continue
        
        if face is None:
            continue
        
        faces_found += 1
        
        # mtcnn returns tensor in [-1, 1], we need [0, 1] for our transforms
        # re-normalize to imagenet stats
        face_pil = transforms.ToPILImage()((face + 1) / 2)  # back to [0,1] then to pil
        tensor = _transform(face_pil)
        face_tensors.append(tensor)
    
    if not face_tensors:
        return {
            'confidence': 0.5,
            'is_fake': False,
            'face_detected': False,
            'detector': 'Video Temporal Pipeline',
            'frames_analyzed': len(frames)
        }
        
    # batch it up (batch size 1)
    batch_tensor = torch.stack(face_tensors).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(batch_tensor)
        probs = torch.softmax(output, dim=1)
        
    # class 0 = fake, class 1 = real
    fake_prob = probs[0][0].item()
    real_prob = probs[0][1].item()
    
    is_fake = fake_prob > real_prob
    confidence = fake_prob if is_fake else real_prob
    
    return {
        'confidence': confidence,
        'is_fake': is_fake,
        'face_detected': True,
        'detector': 'Video Temporal Pipeline',
        'frames_analyzed': len(frames),
        'faces_found': faces_found
    }
