import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='run a quick demo training loop for CI/CD')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
        
    # for demo, just run 1 epoch with small batch to finish in seconds
    batch_size = 4 if args.demo else 16 
    epochs = 1 if args.demo else 10
    lr = 1e-4

    print(f"using {device}")
    if args.demo:
        print("DEMO MODE: Running 1 fast epoch for CI/CD pipeline.")

    # basic augs fr trans
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),#fr flipping img
        transforms.RandomRotation(10),#for rotating it
        transforms.ColorJitter(brightness=0.1, contrast=0.1),#changing in brig&contrast
        transforms.ToTensor(),#converting fr 0.0 to 1,0 insted of 0 to 255 this make it faster i guess
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#got the formula from chatgpt dont know what this do fr now  but it normalize the values fr the model i guess
    ])

    train_dataset = datasets.ImageFolder('data/cropped/train', transform=train_transforms)
    # mps doesnt play nice with multiprocessing workers
    nw = 0 if device == 'mps' else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

    print(f"classes: {train_dataset.class_to_idx}")
    print(f"imgs: {len(train_dataset)}")
    
    # using pre-trained model (can make it from stracth but laptop limitted fr now)
    model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # mixed precision only works properly on cuda
    use_amp = device == 'cuda'
    scaler = GradScaler(device) if use_amp else None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"ep {epoch+1}/{epochs}")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast(device):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'acc': f"{100.*correct/total:.1f}%"
            })
            
            # demo mode: only process one batch to prove it runs
            if args.demo:
                break
            
        epoch_acc = 100. * correct / total
        print(f"ep {epoch+1} | loss: {total_loss/len(train_loader):.4f} | acc: {epoch_acc:.2f}%")
        
    torch.save(model.state_dict(), "deepfake_detector_b4.pth")
    print("saved to deepfake_detector_b4.pth")

if __name__ == '__main__':
    main()