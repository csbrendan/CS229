import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from config import config as cfg
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.common import detect_all, grab_all_frames
from test_tools.utils import get_crop_box
from utils.plugin_loader import PluginLoader

import random
from PIL import Image



class SimpleVideoAugmenter:
    def __init__(self, p=0.5):
        self.p = p
        
    def resolution_degradation(self, image, scale_range=(0.5, 0.8)):
        """Simulate low bandwidth resolution"""
        if random.random() > self.p:
            return image
            
        h, w = image.shape[:2]
        scale = random.uniform(*scale_range)
        small_h, small_w = int(h * scale), int(w * scale)
        
        # Downscale and upscale to simulate low resolution
        small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    def lighting_adjustment(self, image, brightness_range=(-30, 30)):
        """Simulate varying lighting conditions"""
        if random.random() > self.p:
            return image
            
        brightness = random.uniform(*brightness_range)
        adjusted = image.astype(np.float32) + brightness
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def background_blur(self, image, kernel_range=(3, 7)):
        """Simulate Zoom-like background blur"""
        if random.random() > self.p:
            return image
            
        kernel_size = random.randrange(*kernel_range) * 2 + 1  # Ensure odd
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def apply_to_clip(self, frames):
        """Apply augmentations to a sequence of frames consistently"""
        # Randomly choose which augmentations to apply for this clip
        use_resolution = random.random() < self.p
        use_lighting = random.random() < self.p
        use_blur = random.random() < self.p
        
        augmented_frames = []
        for frame in frames:
            augmented = frame.copy()
            
            # Apply chosen augmentations consistently across frames
            if use_resolution:
                augmented = self.resolution_degradation(augmented)
            if use_lighting:
                augmented = self.lighting_adjustment(augmented)
            if use_blur:
                augmented = self.background_blur(augmented)
                
            augmented_frames.append(augmented)
            
        return augmented_frames







class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, clip_size=16, max_videos=10, apply_augmentation=False):
        self.augmenter = SimpleVideoAugmenter(p=0.5)
        self.apply_augmentation = apply_augmentation

        self.clip_size = clip_size
        self.videos = []
        self.labels = []
        
        # Collect paths
        real_videos = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.mp4')][:max_videos//2]
        fake_videos = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.mp4')][:max_videos//2]
        
        self.videos = real_videos + fake_videos
        self.labels = [torch.tensor(0, dtype=torch.float32) for _ in real_videos] + \
                     [torch.tensor(1, dtype=torch.float32) for _ in fake_videos]
        
        print(f"\nProcessing {len(self.videos)} videos:")
        self.processed_data = []
        
        # Initialize face alignment
        self.crop_align_func = FasterCropAlignXRay(cfg.imsize)
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1, 1)
        
        for idx, video_path in enumerate(self.videos):
            try:
                print(f"\nProcessing video {idx+1}/{len(self.videos)}: {os.path.basename(video_path)}")
                
                # Use the exact same processing pipeline as bpm_demo.py
                cache_file = f"{video_path}_{self.clip_size}.pth"
                
                if os.path.exists(cache_file):
                    detect_res, all_lm68 = torch.load(cache_file)
                    frames = grab_all_frames(video_path, max_size=self.clip_size, cvt=True)
                    print("Loaded from cache")
                else:
                    print("Detecting faces")
                    detect_res, all_lm68, frames = detect_all(video_path, return_frames=True, max_size=self.clip_size)
                    torch.save((detect_res, all_lm68), cache_file)
                
                # Process detection results exactly as in bpm_demo.py
                all_detect_res = []
                for faces, faces_lm68 in zip(detect_res, all_lm68):
                    new_faces = []
                    for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
                        new_face = (box, lm5, face_lm68, score)
                        new_faces.append(new_face)
                    all_detect_res.append(new_faces)
                
                detect_res = all_detect_res
                
                # For single-face videos, treat the whole video as one track
                tracks = [detect_res]
                tuples = [(0, len(detect_res))]
                
                # Process data exactly as in bpm_demo.py
                data_storage = {}
                frame_boxes = {}
                
                for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
                    for face, frame_idx, j in zip(track, range(start, end), range(len(track))):
                        if not face:  # Skip if no face detected
                            continue
                            
                        box, lm5, lm68 = face[0][:3]
                        big_box = get_crop_box(frames[0].shape[:2], box, scale=0.5)
                        
                        top_left = big_box[:2][None, :]
                        new_lm5 = lm5 - top_left
                        new_lm68 = lm68 - top_left
                        new_box = (box.reshape(2, 2) - top_left).reshape(-1)
                        info = (new_box, new_lm5, new_lm68, big_box)
                        
                        x1, y1, x2, y2 = big_box
                        cropped = frames[frame_idx][y1:y2, x1:x2]
                        base_key = f"{track_i}_{j}_"
                        data_storage[f"{base_key}img"] = cropped
                        data_storage[f"{base_key}ldm"] = info
                        data_storage[f"{base_key}idx"] = frame_idx
                        frame_boxes[frame_idx] = np.rint(box).astype(int)
                
                # Create clip
                if len(data_storage) < self.clip_size:
                    print(f"Skipping video - not enough faces detected ({len(data_storage)} < {self.clip_size})")
                    continue
                
                # Get frames and landmarks
                frames = []
                landmarks = []
                
                for i in range(min(self.clip_size, len(track))):
                    base_key = f"0_{i}_"  # Single track, so track_i is always 0
                    if f"{base_key}img" in data_storage and f"{base_key}ldm" in data_storage:
                        frames.append(data_storage[f"{base_key}img"])
                        landmarks.append(data_storage[f"{base_key}ldm"])
                
                if len(frames) < self.clip_size:
                    print("Not enough valid frames")
                    continue
                
                # Align faces using same function as inference
                _, aligned_frames = self.crop_align_func(landmarks, frames)
                
                # Convert to tensor and normalize
                clip = torch.as_tensor(aligned_frames, dtype=torch.float32)
                clip = clip.permute(3, 0, 1, 2)  # [C, T, H, W]
                clip = clip.sub(self.mean.cpu()).div(self.std.cpu())
                
                # Remove extra dimension if present
                print(f"Clip shape before squeeze: {clip.shape}")
                if clip.dim() == 5 and clip.size(0) == 1:
                    clip = clip.squeeze(0)
                print(f"Final clip shape: {clip.shape}")
                
                self.processed_data.append((clip, self.labels[idx]))
                
            except Exception as e:
                print(f"Error processing video {video_path}: {str(e)}")
                import traceback
                print("Full traceback:")
                traceback.print_exc()

    def __len__(self):
        return len(self.processed_data)


    def __getitem__(self, idx):
        clip, label = self.processed_data[idx]
        
        if self.apply_augmentation:
            # Convert tensor to NumPy array for augmentation
            clip_np = clip.permute(1, 2, 3, 0).numpy()  # [T, H, W, C]
            
            # Apply augmentations
            augmented_frames = self.augmenter.apply_to_clip(clip_np)
            
            # Convert the list of frames to a NumPy array
            clip_np = np.stack(augmented_frames, axis=0)  # [T, H, W, C]
            
            # Convert back to tensor
            clip = torch.from_numpy(clip_np).permute(3, 0, 1, 2)  # [C, T, H, W]
        
        return clip, label


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for clips, labels in dataloader:
            clips = clips.to(device)
            labels = labels.to(device)

            outputs = model(clips)
            loss = criterion(outputs['final_output'].squeeze(), labels)
            total_loss += loss.item()

            scores = torch.sigmoid(outputs['final_output'].squeeze()).cpu().numpy()
            preds = (scores > 0.5).astype(float)

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def train_model(cfg_path, ckpt_path, real_dir, fake_dir, output_dir, num_epochs=5):
    cfg.init_with_yaml()
    cfg.update_with_yaml(cfg_path)
    cfg.freeze()

    model = PluginLoader.get_classifier(cfg.classifier_type)()
    model.to('cuda')
    model.load(ckpt_path)

    # Freeze layers except the last layer
    for name, param in model.named_parameters():
        param.requires_grad = "head.projection" in name

    train_dataset = DeepfakeDataset(real_dir, fake_dir, clip_size=cfg.clip_size, max_videos=100, apply_augmentation=True)
    val_dataset = DeepfakeDataset(real_dir, fake_dir, clip_size=cfg.clip_size, max_videos=20, apply_augmentation=False)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_acc = 0
    training_history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for clips, labels in tqdm(train_dataloader):
            clips, labels = clips.cuda(), labels.float().cuda()
            optimizer.zero_grad()

            outputs = model(clips)
            loss = criterion(outputs['final_output'].squeeze(), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(outputs['final_output'].squeeze()) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_metrics = evaluate_model(model, val_dataloader, criterion, 'cuda')

        print(f"\nEpoch {epoch + 1}/{num_epochs} Summary:")
        print(f"Training Loss: {train_loss / len(train_dataloader):.4f}, Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        print("Confusion Matrix:", val_metrics['confusion_matrix'])

        history_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_dataloader),
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1']
        }
        training_history.append(history_entry)

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f"New best model saved with accuracy: {best_val_acc:.4f}")

        scheduler.step(val_metrics['accuracy'])

    return model, training_history

if __name__ == "__main__":
    cfg_path = "i3d_ori.yaml"
    ckpt_path = "checkpoints/model.pth"
    real_dir = "/home/azureuser/datasets/ff/original_sequences/youtube/c23/videos"
    fake_dir = "/home/azureuser/datasets/ff/manipulated_sequences/FaceSwap/c23/videos"
    output_dir = "checkpoints"

    os.makedirs(output_dir, exist_ok=True)
    trained_model, history = train_model(cfg_path, ckpt_path, real_dir, fake_dir, output_dir)

    print("Training completed. Model and history saved.")
