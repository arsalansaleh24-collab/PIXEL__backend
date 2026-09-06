import torch
from PIL import Image
from torchvision import transforms
from models import DEVICE, load_model

# load once when the module is imported
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model

# same normalization as training
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def analyze_image(image_path):
    """
    runs the image through efficientnet and returns confidence + metadata.
    returns dict with 'confidence', 'is_fake', 'face_detected'
    """
    model = _get_model()
    
    img = Image.open(image_path).convert('RGB')
    tensor = _transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
    
    # class 0 = fake, class 1 = real (from ImageFolder alphabetical ordering)
    fake_prob = probs[0][0].item()
    real_prob = probs[0][1].item()
    
    is_fake = fake_prob > real_prob
    confidence = fake_prob if is_fake else real_prob
    
    return {
        'confidence': confidence,
        'is_fake': is_fake,
        'face_detected': True,  # for images we just run the whole thing
        'detector': 'Image AIGI Pipeline'
    }
