import os
import cv2
import torch
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms

class VideoDeepfakeDataset(Dataset):
    def __init__(self, data_dir, num_frames=16, transform=None):
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.transform = transform
        
        # init mtcnn lazily so windows multiprocessing doesnt crash
        self.mtcnn = None 

        
        self.samples = [] # List of tuples: (video_path, label)
        
        # label mapping: real -> 0, fake -> 1
        class_to_idx = {'real': 0, 'fake': 1}
        
        for class_name, label in class_to_idx.items():
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory not found: {class_dir}")
                continue
                
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(class_dir, filename)
                    self.samples.append((video_path, label))
                    
        print(f"Loaded {len(self.samples)} videos from {data_dir}")

    def _get_mtcnn(self):
        if self.mtcnn is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, post_process=False, device=device)
        return self.mtcnn

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        # grab frames
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # skip dead vids
        if total_frames == 0:
            cap.release()
            return torch.zeros((self.num_frames, 3, 224, 224)), label

        # sample evenly
        if total_frames >= self.num_frames:
            step = total_frames / self.num_frames
            frame_indices = [int(i * step) for i in range(self.num_frames)]
        else:
            # pad short vids with the last frame
            frame_indices = list(range(total_frames)) + [total_frames - 1] * (self.num_frames - total_frames)

        frames = []
        mtcnn = self._get_mtcnn()
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                # If reading fails, pad with zeros
                frames.append(torch.zeros(3, 224, 224))
                continue
                
            # Convert BGR (OpenCV) to RGB (PIL/PyTorch)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # extract face
            try:
                face = mtcnn(pil_img)
                if face is not None:
                    # mtcnn outputs 0-255 roughly, scale to 0-1
                    face = face / 255.0
                    if self.transform:
                        face = self.transform(face)
                    frames.append(face)
                else:
                    frames.append(torch.zeros(3, 224, 224))
            except Exception as e:
                # Catch any MTCNN errors
                frames.append(torch.zeros(3, 224, 224))
                
        cap.release()
        
        # Stack into (16, 3, 224, 224)
        video_tensor = torch.stack(frames)
        
        return video_tensor, label

if __name__ == '__main__':
    # Quick diagnostic test
    print("Testing VideoDataset setup...")
    os.makedirs("data/raw_videos/train/real", exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = VideoDeepfakeDataset("data/raw_videos/train", transform=transform)
    print(f"Dataset length: {len(dataset)}")
