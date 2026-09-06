import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os

from models.video_model import DeepfakeVideoDetector
from dataset_video import VideoDeepfakeDataset
from torch.cuda.amp import autocast, GradScaler

def main():
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
        
    print(f"Using device: {device}")
    
    # mtcnn already crops to 224x224, just need to normalize it
    # standard imagenet norm since b4 expects it
    val_transforms = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_dir = 'data/raw_videos/train'
    val_dir = 'data/raw_videos/val'
    
    # make sure dirs exist so we dont crash
    os.makedirs(os.path.join(train_dir, 'real'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'fake'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'real'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'fake'), exist_ok=True)
    
    train_dataset = VideoDeepfakeDataset(train_dir, num_frames=8, transform=train_transforms)
    val_dataset = VideoDeepfakeDataset(val_dir, num_frames=8, transform=val_transforms)
    
    # 0 workers on windows otherwise it deadlocks
    nw = 0 if os.name == 'nt' or device == 'mps' else 4
    
    batch_size = 4 # fits on 8gb gpu
    
    if len(train_dataset) == 0:
        print("Error: No training data found! Please download the Kaggle DFDC dataset and extract it to data/raw_videos/train/real and data/raw_videos/train/fake.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw) if len(val_dataset) > 0 else []
    
    model = DeepfakeVideoDetector().to(device)
    
    # freeze cnn, only train lstm/head
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    
    use_amp = device == 'cuda'
    scaler = GradScaler() if use_amp else None
    
    epochs = 5
    best_acc = 0.0
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        if len(train_loader) > 0:
            pbar = tqdm(train_loader, desc=f"Train Ep {epoch+1}/{epochs}")
            for videos, labels in pbar:
                videos, labels = videos.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                if use_amp:
                    with autocast():
                        outputs = model(videos)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct/total:.1f}%"})
        else:
            print("No training data found.")
            
        # val loop
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        if len(val_loader) > 0:
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Val Ep {epoch+1}/{epochs}")
                for videos, labels in val_pbar:
                    videos, labels = videos.to(device), labels.to(device)
                    
                    if use_amp:
                        with autocast():
                            outputs = model(videos)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(videos)
                        loss = criterion(outputs, labels)
                        
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
            val_acc = 100. * val_correct / val_total
            print(f"Epoch {epoch+1} Val Acc: {val_acc:.2f}% | Val Loss: {val_loss/len(val_loader):.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "best_video_lstm.pth")
                print(f"-> Saved new best model with accuracy {best_acc:.2f}%")
        else:
            print("No validation data found.")
            # save anyway if we have no val data
            torch.save(model.state_dict(), "best_video_lstm.pth")

if __name__ == '__main__':
    main()
