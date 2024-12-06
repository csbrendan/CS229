import os
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Feature extraction: Advanced features from each frame
def extract_frame_features(frame):
    # Resize frame for consistency
    frame_resized = cv2.resize(frame, (64, 64))
    
    # Grayscale conversion
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    # Histogram of pixel intensities
    hist = cv2.calcHist([gray_frame], [0], None, [32], [0, 256]).flatten()
    
    # Edge features using Sobel filter
    sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2).flatten()
    
    # Combine features
    return np.concatenate([hist, edge_magnitude[:256]])  # Limit edge features to 256 values

# Extract video-level features
def extract_video_features(video_path, clip_size=32):
    cap = cv2.VideoCapture(video_path)
    frame_features = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract features for the frame
        features = extract_frame_features(frame)
        frame_features.append(features)
        frame_count += 1

        if frame_count >= clip_size:
            break
    
    cap.release()

    # Aggregate frame features into video-level features
    if len(frame_features) < clip_size:
        frame_features.extend([np.zeros_like(frame_features[0])] * (clip_size - len(frame_features)))
    frame_features = np.array(frame_features[:clip_size])

    # Compute video-level features (e.g., mean and variance across frames)
    video_features = np.concatenate([frame_features.mean(axis=0), frame_features.std(axis=0)])
    return video_features

# Prepare dataset
def prepare_advanced_dataset(real_dir, fake_dir, max_videos=50, clip_size=32):
    X = []
    y = []

    # Process real videos
    real_videos = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.mp4')][:max_videos // 2]
    for video in tqdm(real_videos, desc="Processing Real Videos"):
        video_features = extract_video_features(video, clip_size=clip_size)
        X.append(video_features)
        y.append(0)  # Label 0 for real videos

import os
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Feature extraction: Advanced features from each frame
def extract_frame_features(frame):
    # Resize frame for consistency
    frame_resized = cv2.resize(frame, (64, 64))
    
    # Grayscale conversion
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    # Histogram of pixel intensities
    hist = cv2.calcHist([gray_frame], [0], None, [32], [0, 256]).flatten()
    
    # Edge features using Sobel filter
    sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2).flatten()
    
    # Combine features
    return np.concatenate([hist, edge_magnitude[:256]])  # Limit edge features to 256 values

# Extract video-level features
def extract_video_features(video_path, clip_size=32):
    cap = cv2.VideoCapture(video_path)
    frame_features = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract features for the frame
        features = extract_frame_features(frame)
        frame_features.append(features)
        frame_count += 1

        if frame_count >= clip_size:
            break
    
    cap.release()

    # Aggregate frame features into video-level features
    if len(frame_features) < clip_size:
        frame_features.extend([np.zeros_like(frame_features[0])] * (clip_size - len(frame_features)))
    frame_features = np.array(frame_features[:clip_size])

    # Compute video-level features (e.g., mean and variance across frames)
    video_features = np.concatenate([frame_features.mean(axis=0), frame_features.std(axis=0)])
    return video_features

# Prepare dataset
def prepare_advanced_dataset(real_dir, fake_dir, max_videos=50, clip_size=32):
    X = []
    y = []

    # Process real videos
    real_videos = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.mp4')][:max_videos // 2]
    for video in tqdm(real_videos, desc="Processing Real Videos"):
        video_features = extract_video_features(video, clip_size=clip_size)
        X.append(video_features)
        y.append(0)  # Label 0 for real videos

    # Process fake videos
    fake_videos = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.mp4')][:max_videos // 2]
    for video in tqdm(fake_videos, desc="Processing Fake Videos"):
        video_features = extract_video_features(video, clip_size=clip_size)
        X.append(video_features)
        y.append(1)  # Label 1 for fake videos

    return np.array(X), np.array(y)

# Train logistic regression and evaluate
def train_and_evaluate_advanced(real_dir, fake_dir):
    print("Preparing dataset...")
    X, y = prepare_advanced_dataset(real_dir, fake_dir, max_videos=50, clip_size=32)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training logistic regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]  # Probability of the positive class (fake)
    
    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_scores)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

# Main script
if __name__ == "__main__":
    real_dir = "/home/azureuser/AltFreezing/examples/kaggle_zoom/real"
    fake_dir = "/home/azureuser/AltFreezing/examples/kaggle_zoom/fake"
    train_and_evaluate_advanced(real_dir, fake_dir)
