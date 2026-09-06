import torch
import torch.nn as nn
import timm
import os

class DeepfakeVideoDetector(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden_size=512, lstm_layers=2):
        super(DeepfakeVideoDetector, self).__init__()
        
        # cnn backbone (drop the classifier head)
        self.backbone = timm.create_model('efficientnet_b4', pretrained=False, num_classes=0)
        
        # load weights if we have them
        weight_path = "models/deepfake_detector_b4.pth"
        if not os.path.exists(weight_path):
            weight_path = "deepfake_detector_b4.pth" # Fallback to root
            
        if os.path.exists(weight_path):
            print(f"Loading pretrained backbone weights from {weight_path}...")
            # strict=False so we don't crash on missing classifier weights
            self.backbone.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)
        else:
            print(f"WARNING: Could not find pretrained weights at {weight_path}. Using random weights.")
            
        # freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # temporal head (lstm) - b4 outputs 1792 dims
        self.lstm = nn.LSTM(
            input_size=1792,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Classifier Head
        # Bidirectional means output size is 2 * hidden_size
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_classes)
        
    def forward(self, x):
        batch_size, frames, c, h, w = x.size()
        
        # fold frames into batch dim for cnn
        x = x.view(batch_size * frames, c, h, w)
        
        with torch.no_grad():
            features = self.backbone(x) 
            
        # unfold back to seq
        features = features.view(batch_size, frames, -1)
        
        # pass to lstm
        lstm_out, _ = self.lstm(features)
        
        # mean pool across time
        pooled = torch.mean(lstm_out, dim=1)
        
        output = self.classifier(pooled)
        return output

if __name__ == '__main__':
    # Quick diagnostic test
    print("Testing VideoDeepfakeDetector architecture...")
    model = DeepfakeVideoDetector()
    dummy_input = torch.randn(2, 16, 3, 224, 224) # Batch size 2, 16 frames
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 2), "Output shape should be (batch_size, 2)"
    print("Architecture looks good!")
