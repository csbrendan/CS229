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


class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, clip_size=32, max_videos=50, save_samples=False, sample_dir=None):
        self.clip_size = clip_size
        self.videos = []
        self.labels = []
        self.processed_data = []

        if save_samples and sample_dir:
            os.makedirs(sample_dir, exist_ok=True)

        # Collect paths
        real_videos = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.mp4')][:max_videos//2]
        fake_videos = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.mp4')][:max_videos//2]

        self.videos = real_videos + fake_videos
        self.labels = [0 for _ in real_videos] + [1 for _ in fake_videos]
        
        print(f"\nProcessing {len(self.videos)} videos:")
        
        # Initialize face alignment
        self.crop_align_func = FasterCropAlignXRay(cfg.imsize)
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1, 1)
        
        for idx, video_path in enumerate(self.videos):
            try:
                print(f"\nProcessing video {idx+1}/{len(self.videos)}: {os.path.basename(video_path)}")
                
                # Process all frames without max_size limit
                cache_file = f"{video_path}_all.pth"

                if os.path.exists(cache_file):
                    detect_res, all_lm68 = torch.load(cache_file)
                    frames = grab_all_frames(video_path, max_size=None, cvt=True)
                    print("Loaded from cache")
                else:
                    print("Detecting faces")
                    detect_res, all_lm68, frames = detect_all(video_path, return_frames=True, max_size=None)
                    torch.save((detect_res, all_lm68), cache_file)

                # Process detection results
                all_detect_res = []
                for faces, faces_lm68 in zip(detect_res, all_lm68):
                    new_faces = []
                    for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
                        new_face = (box, lm5, face_lm68, score)
                        new_faces.append(new_face)
                    all_detect_res.append(new_faces)
                
                detect_res = all_detect_res
                tracks = [detect_res]
                tuples = [(0, len(detect_res))]
                
                # Process data
                data_storage = {}
                frame_boxes = {}
                
                for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
                    for face, frame_idx, j in zip(track, range(start, end), range(len(track))):
                        if not face:
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
                
                if len(data_storage) < self.clip_size:
                    print(f"Skipping video - not enough faces detected ({len(data_storage)} < {self.clip_size})")
                    continue
                
                num_frames = len(track)
                num_clips = num_frames // self.clip_size

                for clip_idx in range(num_clips):
                    start_idx = clip_idx * self.clip_size
                    end_idx = start_idx + self.clip_size
                    
                    frames = []
                    landmarks = []
                    
                    for i in range(start_idx, end_idx):
                        base_key = f"0_{i}_"
                        if f"{base_key}img" in data_storage and f"{base_key}ldm" in data_storage:
                            frames.append(data_storage[f"{base_key}img"])
                            landmarks.append(data_storage[f"{base_key}ldm"])
                    
                    if len(frames) < self.clip_size:
                        continue
                    
                    _, aligned_frames = self.crop_align_func(landmarks, frames)
                    clip = torch.as_tensor(aligned_frames, dtype=torch.float32)
                    clip = clip.permute(3, 0, 1, 2)
                    clip = clip.sub(self.mean.cpu()).div(self.std.cpu())
                    
                    if clip.dim() == 5 and clip.size(0) == 1:
                        clip = clip.squeeze(0)
                    
                    self.processed_data.append((clip, torch.tensor(self.labels[idx], dtype=torch.float32)))
                
            except Exception as e:
                print(f"Error processing video {video_path}: {str(e)}")
                import traceback
                print("Full traceback:")
                traceback.print_exc()

        print(f"Total number of clips processed: {len(self.processed_data)}")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


def split_parameters(model):
    """
    Split model parameters into spatial and temporal groups
    Both groups include shared parameters (1x1x1 conv, linear, bn)
    """
    spatial_params = []
    temporal_params = []
    shared_params = []

    for name, param in model.named_parameters():
        if any(x in name.lower() for x in ['conv.weight']):
            if 'time' in name.lower() or 'temporal' in name.lower():
                temporal_params.append(param)
            else:
                spatial_params.append(param)
        else:
            shared_params.append(param)
    
    spatial_params.extend(shared_params)
    temporal_params.extend(shared_params)
    
    return spatial_params, temporal_params


def create_optimizers(model, lr=1e-4):
    """Create separate optimizers for spatial and temporal parameters"""
    spatial_params, temporal_params = split_parameters(model)
    optim_spatial = optim.Adam(spatial_params, lr=lr)
    optim_temporal = optim.Adam(temporal_params, lr=lr)
    return optim_spatial, optim_temporal


def should_update_spatial(iteration, Is=20, It=1):
    """Determine whether to update spatial parameters using 20:1 ratio"""
    cycle_length = Is + It
    return (iteration % cycle_length) < Is


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_scores = []

    eval_pbar = tqdm(dataloader, desc='Evaluating')

    with torch.no_grad():
        for clips, labels in eval_pbar:
            clips = clips.to(device)
            labels = labels.to(device)

            outputs = model(clips)
            loss = criterion(outputs['final_output'].view(-1), labels)

            total_loss += loss.item()
            scores = torch.sigmoid(outputs['final_output'].view(-1)).cpu().numpy()
            preds = (scores > 0.5).astype(float)

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores)

            eval_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_scores)
    except ValueError:
        auc = None

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }


def train_model_with_alt_freezing(cfg_path, ckpt_path, real_dir, fake_dir, output_dir, total_epochs=10):
    cfg.init_with_yaml()
    cfg.update_with_yaml(cfg_path)
    cfg.freeze()

    model = PluginLoader.get_classifier(cfg.classifier_type)()
    model.to('cuda')
    model.load(ckpt_path)

    batch_size = 4
    num_workers = 4
    train_dataset = DeepfakeDataset(real_dir, fake_dir, clip_size=32, max_videos=90) # quick for test 4/2
    val_dataset = DeepfakeDataset(real_dir, fake_dir, clip_size=32, max_videos=24)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    optim_spatial, optim_temporal = create_optimizers(model)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Training on {len(train_dataset)} clips, validating on {len(val_dataset)} clips")

    best_val_acc = 0
    training_history = []
    iteration = 0

    for epoch in range(1, total_epochs + 1):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch}/{total_epochs} [Train]')
        
        for clips, labels in train_pbar:
            clips, labels = clips.cuda(), labels.float().cuda()
            
            optim_spatial.zero_grad()
            optim_temporal.zero_grad()
            
            outputs = model(clips)
            loss = criterion(outputs['final_output'].view(-1), labels)
            loss.backward()
            
            if should_update_spatial(iteration):
                optim_spatial.step()
                current_phase = "spatial"
            else:
                optim_temporal.step()
                current_phase = "temporal"
            
            iteration += 1
            
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs['final_output'].view(-1)) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}',
                'phase': current_phase
            })

        train_acc = correct / total
        val_metrics = evaluate_model(model, val_dataloader, criterion, 'cuda')

        print(f"\nEpoch {epoch}/{total_epochs} Summary:")
        print(f"Training Loss: {train_loss / len(train_dataloader):.4f}, Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        print(f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")

        history_entry = {
            'epoch': epoch,
            'train_loss': train_loss / len(train_dataloader),
            'train_acc': train_acc,
            'val_metrics': val_metrics
        }
        training_history.append(history_entry)

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            model_save_path = os.path.join(output_dir, f'best_altfreeze_ffdlc_model_acc_{best_val_acc:.4f}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'spatial_optimizer_state_dict': optim_spatial.state_dict(),
                'temporal_optimizer_state_dict': optim_temporal.state_dict(),
                'val_accuracy': best_val_acc,
                'history': training_history,
                'iteration': iteration
            }, model_save_path)
            print(f"New best model saved with accuracy: {best_val_acc:.4f}")

    print("Training completed.")
    return model, training_history


if __name__ == "__main__":
    cfg_path = "i3d_ori.yaml"
    ckpt_path = "checkpoints/model.pth"
    real_dir = "/home/azureuser/AltFreezing/examples/dlc_cross_ff/dlc_cross_trimmed"
    fake_dir = "/home/azureuser/AltFreezing/examples/dlc_cross_ff/dlc_cross_processed"
    output_dir = "checkpoints"

    os.makedirs(output_dir, exist_ok=True)
    trained_model, history = train_model_with_alt_freezing(cfg_path, ckpt_path, real_dir, fake_dir, output_dir)

    # Save final training history
    history_path = os.path.join(output_dir, 'altfreeze_training_history.pth')
    torch.save(history, history_path)
    print(f"Training history saved to {history_path}")
    print("Training completed. Model and history saved.")
