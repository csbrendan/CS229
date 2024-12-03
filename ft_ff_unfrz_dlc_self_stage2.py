import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

from config import config as cfg
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.common import detect_all, grab_all_frames
from test_tools.utils import get_crop_box
from utils.plugin_loader import PluginLoader


class ZoomAugmenter:
    def __init__(self):
        self.target_height = 720
        self.target_width = 1280
    
    def apply_zoom_effects(self, frame):
        """Apply Zoom-like effects to a frame"""
        frame = self._resize_frame(frame)
        frame = self._apply_quality_effects(frame)
        frame = self._apply_lighting_variation(frame)
        return frame
        
    def _resize_frame(self, frame):
        """Resize to typical Zoom resolution"""
        return cv2.resize(frame, (self.target_width, self.target_height))
    
    def _apply_quality_effects(self, frame):
        """Add compression artifacts and blur"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  #at 40 may degrade too much
        _, encoded = cv2.imencode('.jpg', frame, encode_param)
        frame = cv2.imdecode(encoded, 1)

        #minimal noise to sim complression - killed it too much dist
        #noise = np.random.normal(0, 2, frame.shape).astype(np.uint8)
        #frame = cv2.add(frame, noise)

        return cv2.GaussianBlur(frame, (5, 5), 1.5) #7,7, 3.0 degrades too much
    
    def _apply_lighting_variation(self, frame):
        """Simulate webcam lighting variations"""
        brightness = np.random.uniform(0.9, 1.1)
        return cv2.convertScaleAbs(frame, alpha=brightness, beta=0)


    def _apply_screen_lighting(self, frame):
        # Create gradient to simulate screen glow
        height, width = frame.shape[:2]
        gradient = np.zeros_like(frame)
        center = (width//2, height//2)
        
        # Create radial gradient
        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x-center[0])**2 + (y-center[1])**2)
                intensity = 1 - min(1, distance/(width/2))
                gradient[y,x] = [intensity*30]*3  # Screen glow intensity
                
        # Apply varying intensity
        intensity = np.random.uniform(0.8, 1.4)
        screen_effect = cv2.addWeighted(frame, 1, gradient, intensity, 0)
        
        return screen_effect


    def _apply_lighting_variations(self, frame):
        # Backlight simulation
        backlight = np.random.choice([True, False], p=[0.3, 0.7])
        if backlight:
            mask = np.zeros_like(frame)
            center_x = frame.shape[1] // 2
            center_y = frame.shape[0] // 2
            cv2.circle(mask, (center_x, center_y), int(frame.shape[1]*0.4), (1,1,1), -1)
            mask = cv2.GaussianBlur(mask, (99,99), 30)
            darkening = np.random.uniform(0.5, 0.8)
            frame = frame * (1 - mask * darkening)

        # Side lighting
        if np.random.random() < 0.3:
            gradient = np.linspace(0.7, 1.3, frame.shape[1])
            frame = frame * gradient.reshape(1, -1, 1)

        # Poor lighting
        if np.random.random() < 0.4:
            darkness = np.random.uniform(0.6, 0.9)
            frame = frame * darkness

        return frame.astype(np.uint8)




class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, clip_size=16, max_videos=200, save_samples=False, sample_dir=None):
        self.clip_size = clip_size
        self.videos = []
        self.labels = []

        self.save_samples = save_samples
        self.sample_dir = sample_dir
        self.augmenter = ZoomAugmenter()

        if save_samples and sample_dir:
            os.makedirs(sample_dir, exist_ok=True)


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


                # NO AUGMENTATION FOR STAGE 2
                # Apply Zoom augmentations to frames
                #video_name = os.path.basename(video_path)
                #for i, frame in enumerate(frames):
                #    frames[i] = self.process_frame(frame, video_name, i)

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
        return self.processed_data[idx]




# Define the function to freeze/unfreeze layers
def set_trainable_layers(model, layers_to_unfreeze):
    """
    Freeze all layers except those specified in layers_to_unfreeze.
    """
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_unfreeze):
            param.requires_grad = True
        else:
            param.requires_grad = False

# Define the gradual unfreezing strategy
def gradual_unfreezing(epoch):
    """
    Returns a list of layers to unfreeze based on the current epoch.
    """
    if epoch <= 3:
        return ["head"]
    elif epoch <= 7:
        return ["head", "layer4"]
    else:
        return ["head", "layer4", "layer3"]

# Training loop
def train_model_with_gradual_unfreezing(cfg_path, ckpt_path, real_dir, fake_dir, output_dir, total_epochs=10):
    cfg.init_with_yaml()
    cfg.update_with_yaml(cfg_path)
    cfg.freeze()

    # Create the model and load checkpoint
    model = PluginLoader.get_classifier(cfg.classifier_type)()
    model.to('cuda')
    model.load(ckpt_path)

    # Datasets and dataloaders
    train_dataset = DeepfakeDataset(real_dir, fake_dir, clip_size=cfg.clip_size, max_videos=100)
    val_dataset = DeepfakeDataset(real_dir, fake_dir, clip_size=cfg.clip_size, max_videos=20)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0
    training_history = []

    for epoch in range(1, total_epochs + 1):
        # Update trainable layers based on epoch
        layers_to_unfreeze = gradual_unfreezing(epoch)
        set_trainable_layers(model, layers_to_unfreeze)
        print(f"\nEpoch {epoch}: Training layers {layers_to_unfreeze}")

        # Training step
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

        # Print epoch summary
        print(f"Epoch {epoch}/{total_epochs} Summary:")
        print(f"Training Loss: {train_loss / len(train_dataloader):.4f}, Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        print(f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print("Confusion Matrix:", val_metrics['confusion_matrix'])

        # Save history
        history_entry = {
            'epoch': epoch,
            'train_loss': train_loss / len(train_dataloader),
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_auc': val_metrics['auc'],
        }
        training_history.append(history_entry)

        # Save the best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), os.path.join(output_dir, 'stage_2_best_model.pth'))
            print(f"New best model saved with accuracy: {best_val_acc:.4f}")

        scheduler.step(val_metrics['accuracy'])

    print("Training completed.")

    return model, training_history





def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_scores = []  # To store raw probabilities for AUC calculation


    with torch.no_grad():
        for clips, labels in dataloader:
            clips = clips.to(device)
            labels = labels.to(device)

            outputs = model(clips)
            loss = criterion(outputs['final_output'].squeeze(), labels)
            total_loss += loss.item()

            scores = torch.sigmoid(outputs['final_output'].squeeze()).cpu().numpy()
            preds = (scores > 0.1).astype(float)  #thresold 0.5?

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores)


    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    # Calculate AUC
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except ValueError:
        auc = None  # Handle case where AUC can't be calculated due to class imbalance

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }

def train_model(cfg_path, ckpt_path, real_dir, fake_dir, output_dir, num_epochs=20):
    cfg.init_with_yaml()
    cfg.update_with_yaml(cfg_path)
    cfg.freeze()

    # Create sample directory for augmentations
    sample_dir = os.path.join(output_dir, 'augmentation_samples')
    os.makedirs(sample_dir, exist_ok=True)

    model = PluginLoader.get_classifier(cfg.classifier_type)()
    model.to('cuda')
    model.load(ckpt_path)

    # Freeze layers except the last layer
    for name, param in model.named_parameters():
        param.requires_grad = "head.projection" in name

    train_dataset = DeepfakeDataset(real_dir, fake_dir, clip_size=cfg.clip_size, max_videos=94, save_samples=False, sample_dir=sample_dir)
    val_dataset = DeepfakeDataset(real_dir, fake_dir, clip_size=cfg.clip_size, max_videos=20, save_samples=False)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # attempt to reduce false positives - it didnt work
    #pos_weight = torch.tensor([3.0]).cuda()  # Higher weight for real class
    #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.000005, weight_decay=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0
    training_history = []


    # Save initial augmentation samples before training
    #print("\nSaved augmented samples in:", sample_dir)


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
        print(f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        print(f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print("Confusion Matrix:", val_metrics['confusion_matrix'])


        history_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_dataloader),
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_auc': val_metrics['auc']
        }
        training_history.append(history_entry)

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            # SAVE BEST MODEL : STAGE 2
            torch.save(model.state_dict(), os.path.join(output_dir, 'stage_2_self_best_model.pth'))
            print(f"New best model saved with accuracy: {best_val_acc:.4f}")

        if epoch % 2 == 0:
            checkpoint_path = os.path.join(output_dir, f'stage2_self_checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, checkpoint_path)

        scheduler.step(val_metrics['accuracy'])

    return model, training_history

if __name__ == "__main__":
    cfg_path = "i3d_ori.yaml"
    ckpt_path = "checkpoints/model.pth"
    real_dir = "/home/azureuser/AltFreezing/examples/dlc_self_ff/dlc_self_trimmed"
    fake_dir = "/home/azureuser/AltFreezing/examples/dlc_self_ff/dlc_self_processed"
    output_dir = "checkpoints"

    os.makedirs(output_dir, exist_ok=True)
    trained_model, history = train_model_with_gradual_unfreezing(cfg_path, ckpt_path, real_dir, fake_dir, output_dir)

    print("Training completed. Model and history saved.")
