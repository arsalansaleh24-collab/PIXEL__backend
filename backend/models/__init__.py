import torch
import timm
from pathlib import Path

# figure out what hardware we got
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

WEIGHTS_PATH = Path("deepfake_detector_b4.pth")

def load_model():
    """loads the efficientnet weights if they exist, otherwise returns None"""
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=2)
    
    if WEIGHTS_PATH.exists():
        state = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        print(f"loaded weights from {WEIGHTS_PATH}")
    else:
        # no trained model yet, use pretrained imagenet weights
        # this wont give real results but at least the api wont crash
        model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=2)
        print(f"warning: {WEIGHTS_PATH} not found, using untrained model")
    
    model = model.to(DEVICE)
    model.eval()
    return model
