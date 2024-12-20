import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import glob
from config import config as cfg
from test_tools.common import detect_all, grab_all_frames
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.supply_writer import SupplyWriter
from test_tools.utils import get_crop_box
from utils.plugin_loader import PluginLoader
import json
from datetime import datetime
import random
import os
from sklearn.metrics import roc_auc_score


# Function to convert numpy types to Python native types
def convert_to_serializable(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Constants for image normalization
mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1, 1)
std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1, 1)

# Configuration variables
max_frame = 300

#max_videos = 10

real_dir = "/home/azureuser/CelebDF/Celeb-real"
fake_dir = "/home/azureuser/CelebDF/Celeb-synthesis"

out_dir = "prediction"
cfg_path = "i3d_ori.yaml"

ckpt_path = "checkpoints/model.pth"
#ckpt_path = "checkpoints/best_zoom_model_acc_0.8376.pth"
#ckpt_path = "checkpoints/best_ffdlc_model_acc_0.9792.pth"
#ckpt_path = "checkpoints/stage_2_best_model.pth"
optimal_threshold = 0.1

def process_video(video_path, ground_truth_label):
    print(f"\nProcessing video: {os.path.basename(video_path)}")
    
    # Prepare output file
    basename = f"{os.path.splitext(os.path.basename(video_path))[0]}.avi"
    out_file = os.path.join(out_dir, basename)

    # Cache file for storing detection results
    cache_file = f"{video_path}_{max_frame}.pth"

    # Perform face detection or load from cache
    if os.path.exists(cache_file):
        detect_res, all_lm68 = torch.load(cache_file)
        frames = grab_all_frames(video_path, max_size=max_frame, cvt=True)
        print("Detection result loaded from cache")
    else:
        print("Detecting faces")
        detect_res, all_lm68, frames = detect_all(video_path, return_frames=True, max_size=max_frame)
        torch.save((detect_res, all_lm68), cache_file)
        print("Detection finished")

    if len(frames) == 0:
        print("Error: No frames were extracted from the video. Please check:")
        print(f"1. Video path exists: {os.path.exists(video_path)}")
        print(f"2. Video file size: {os.path.getsize(video_path)} bytes")
        
        # Try to get video info using cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            print(f"3. Video properties:")
            print(f"   - Frame count: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
            print(f"   - FPS: {cap.get(cv2.CAP_PROP_FPS)}")
            print(f"   - Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        cap.release()
        return None

    print("Number of frames:", len(frames))

    # Add debug info after face detection
    print("\nFace detection summary:")
    no_faces_count = 0
    faces_detected_count = 0
    # Process frames
    for i, (faces, faces_lm68) in enumerate(zip(detect_res, all_lm68)):
        if not faces:
            no_faces_count += 1
        else:
            faces_detected_count += 1
    print(f"{no_faces_count}: Frames with no face detected")
    print(f"{faces_detected_count}: Frames with a face detected")



    # Process detection results with improved error handling
    all_detect_res = []
    for frame_idx, (faces, faces_lm68) in enumerate(zip(detect_res, all_lm68)):
        new_faces = []
        if not faces:
            all_detect_res.append([])
            continue
            
        try:
            for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
                new_face = (box, lm5, face_lm68, score)
                new_faces.append(new_face)
            all_detect_res.append(new_faces)
        except Exception as e:
            print(f"Error processing detection results for frame {frame_idx}:")
            print(f"Error details: {str(e)}")
            all_detect_res.append([])

    detect_res = all_detect_res
    tracks = [detect_res]
    tuples = [(0, len(detect_res))]

    print("Processing video")

    # Prepare data storage
    data_storage = {}
    frame_boxes = {}
    valid_frames = set()
    super_clips = [len(detect_res)]

    # Process each frame
    for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
        print(f"Processing frames {start} to {end}")
        
        for face, frame_idx, j in zip(track, range(start, end), range(len(track))):
            if not face:
                print(f"Warning: No faces detected in frame {frame_idx}")
                continue
                
            try:
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
                valid_frames.add((track_i, j))
            except Exception as e:
                print(f"Error processing frame {frame_idx}:")
                print(f"Error details: {str(e)}")
                continue

    print("Sampling clips")

    # Prepare clips for processing
    clips_for_video = []
    clip_size = 32 #cfg.clip_size
    #pad_length = clip_size - 1

    # Create clips from the video
    # Create sequential clips from valid frames
    for super_clip_idx, super_clip_size in enumerate(super_clips):
        valid_indices = [i for i in range(super_clip_size) if (super_clip_idx, i) in valid_frames]
        
        if len(valid_indices) < clip_size:
            print(f"Warning: Not enough valid frames ({len(valid_indices)}) to create clips")
            continue
        
        # Create sequential clips without padding
        for i in range(0, len(valid_indices) - clip_size + 1):
            current_indices = valid_indices[i:i + clip_size]
            clip = [(super_clip_idx, t) for t in current_indices]
            clips_for_video.append(clip)



    preds = []
    frame_res = {}






    print("Running predictions")

    for clip in tqdm(clips_for_video, desc="Testing"):
        try:
            images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
            landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
            frame_ids = [data_storage[f"{i}_{j}_idx"] for i, j in clip]
            _, images_align = crop_align_func(landmarks, images)
            
            images = torch.as_tensor(images_align, dtype=torch.float32).cuda().permute(3, 0, 1, 2)
            images = images.unsqueeze(0).sub(mean).div(std)

            with torch.no_grad():
                output = classifier(images)
            pred = float(F.sigmoid(output["final_output"]))
            
            for f_id in frame_ids:
                if f_id not in frame_res:
                    frame_res[f_id] = []
                frame_res[f_id].append(pred)
            preds.append(pred)
        except KeyError as e:
            print(f"Warning: Skipping clip due to missing frame data: {e}")
            continue
        except Exception as e:
            print(f"Error processing clip: {e}")
            continue

    if not preds:
        print("Error: No valid predictions were made")
        return None

    # Prepare final results
    boxes = []
    scores = []

    for frame_idx in range(len(frames)):
        if frame_idx in frame_res:
            pred_prob = np.mean(frame_res[frame_idx])
            rect = frame_boxes[frame_idx]
        else:
            pred_prob = None
            rect = None
        scores.append(pred_prob)
        boxes.append(rect)

    # Generate output video
    SupplyWriter(video_path, out_file, optimal_threshold).run(frames, scores, boxes)

    overall_mean_pred = np.mean(preds)
    classification_error = abs(optimal_threshold - overall_mean_pred)
    predicted_fake = overall_mean_pred > optimal_threshold

    # Print immediate results
    print("\n--- Immediate Results ---")
    print(f"Mean prediction: {overall_mean_pred:.4f}")
    print(f"Classification error: {classification_error:.4f}")
    print(f"Predicted class: {'FAKE' if predicted_fake else 'REAL'}")
    print("-" * 25)


    result = {
        'filename': os.path.basename(video_path),
        'mean_prediction': overall_mean_pred,
        'classification_error': classification_error,
        'ground_truth': bool(ground_truth_label),  # Ground truth label from input
        'predicted_fake': predicted_fake,
        'correct_prediction': (predicted_fake == bool(ground_truth_label))
    }

    return result





if __name__ == "__main__":
    # Initialize and load configuration
    cfg.init_with_yaml()
    cfg.update_with_yaml(cfg_path)
    cfg.freeze()


    ###################

    # Initialize and load the classifier
    classifier = PluginLoader.get_classifier(cfg.classifier_type)()
    classifier.cuda()
    classifier.eval()
    classifier.load(ckpt_path)


    #############


    # Initialize the face alignment function
    crop_align_func = FasterCropAlignXRay(cfg.imsize)

    # Prepare output directory
    os.makedirs(out_dir, exist_ok=True)

    # Load all videos from both directories
    real_videos = sorted(glob.glob(os.path.join(real_dir, "*.mp4")))  #for quick testing append [:2], [:1]
    fake_videos = sorted(glob.glob(os.path.join(fake_dir, "*.mp4")))

    # Set the maximum number of samples from each directory while testing, for time saving
    max_samples_per_class = 50  #
    real_videos = real_videos[:max_samples_per_class]
    fake_videos = fake_videos[:max_samples_per_class]
    #################################################

    video_files = real_videos + fake_videos
    ground_truth = [0] * len(real_videos) + [1] * len(fake_videos)

    print(f"\nTotal Real Videos: {len(real_videos)}")
    print(f"Total Fake Videos: {len(fake_videos)}")
    print(f"Total Videos for Processing: {len(video_files)}\n")

    # Ensure at least some videos are loaded
    if not video_files:
        print("No .mp4 files found in the specified directories.")
        exit(1)

    # Process all videos and collect results
    results = []
    for i, (video_path, ground_truth_label) in enumerate(zip(video_files, ground_truth), 1):
        print(f"\nProcessing video {i}/{len(video_files)}: {os.path.basename(video_path)}")
        result = process_video(video_path, ground_truth_label)  # Pass ground truth label
        if result:
            results.append(result)


    # Calculate AUC
    all_labels = []
    all_preds = []
    for result in results:
        all_labels.append(int(result['ground_truth']))
        all_preds.append(result['mean_prediction'])

    auc_score = roc_auc_score(all_labels, all_preds)

    # Convert all numpy values to Python native types
    processed_results = []
    for result in results:
        processed_result = {k: convert_to_serializable(v) for k, v in result.items()}
        processed_results.append(processed_result)

    # Create final results dictionary
    final_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'threshold': float(optimal_threshold),
            'total_videos': len(video_files),
            'successful_videos': len(results),
            'real_directory': real_dir,
            'fake_directory': fake_dir
        },
        'video_results': processed_results,
        'summary': {
            'average_error': float(np.mean([r['classification_error'] for r in results])),
            'average_prediction': float(np.mean([r['mean_prediction'] for r in results])),
            'auc_score': float(auc_score)
        }
    }

    # Save results to JSON file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_filename = os.path.join(out_dir, f'results_eval_base_cdf_thresh_pt1_{timestamp}.json')

    try:
        with open(json_filename, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"\nResults successfully saved to: {json_filename}")
    except Exception as e:
        print(f"\nError saving results to JSON: {str(e)}")
        print("Continuing with summary display...")

    # Print summary table
    print("\n=== Processing Summary ===")
    print(f"Processed {len(results)}/{len(video_files)} videos successfully")
    print(f"Results saved to: {json_filename}")
    print("\nDetailed Results:")
    print("-" * 80)
    print(f"{'Filename':<40} {'Mean Pred':<10} {'Class Error':<12} {'Predicted':<8}")
    print("-" * 80)

    for result in sorted(results, key=lambda x: x['classification_error']):
        print(f"{result['filename']:<40} {result['mean_prediction']:.4f}    {result['classification_error']:.4f}      {'FAKE' if result['ground_truth'] else 'REAL'}")

    print("-" * 80)

    # Calculate and print average error
    avg_error = np.mean([r['classification_error'] for r in results])
    avg_prediction = np.mean([r['mean_prediction'] for r in results])
    print(f"\nAverage classification error: {avg_error:.4f}")
    print(f"Average prediction: {avg_prediction:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
