import os
import subprocess

# Base directories
VIDEO_DIR = "/Users/brendanmurphy/Desktop/CS229ML/PROJECT/selfie_vid_dataset/trimmed/MALE/"  # Directory containing target videos
FACE_DIR = "/Users/brendanmurphy/Desktop/CS229ML/PROJECT/selfie_vid_dataset/trimmed/MALE/face_pic/"  # Directory containing face images
OUTPUT_DIR = "/Users/brendanmurphy/Desktop/CS229ML/PROJECT/selfie_vid_dataset/trimmed/MALE/processed_vids"  # Directory to save processed videos

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mappings
# Define image and video filenames for mapping
image_to_video_mapping = {
    "3_8_extracted_frame.jpg": "1_7_trimmed.mp4",
    "3_7_extracted_frame.jpg": "1_8_trimmed.mp4",
    "5_7_extracted_frame.jpg": "brendan_R_trimmed.mp4",
    "5_8_extracted_frame.jpg": "paul_R_trimmed.mp4",
    "1_8_extracted_frame.jpg": "5_7_trimmed.mp4",
    "1_7_extracted_frame.jpg": "5_8_trimmed.mp4",
    "7_7_extracted_frame.jpg": "3_8_trimmed.mp4",
    "7_8_extracted_frame.jpg": "3_7_trimmed.mp4",
    "9_8_extracted_frame.jpg": "7_8_trimmed.mp4",
    "9_7_extracted_frame.jpg": "7_7_trimmed.mp4",
    "brendan_R_extracted_frame.jpg": "9_7_trimmed.mp4",
    "paul_R_extracted_frame.jpg": "9_8_trimmed.mp4"
}

# Process each mapping
for face_image, target_video in image_to_video_mapping.items():
    # Define paths
    source_face_path = os.path.join(FACE_DIR, face_image)
    target_video_path = os.path.join(VIDEO_DIR, target_video)
    trimmed_video_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(target_video)[0]}_trimmed.mp4")
    output_video_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(target_video)[0]}_dlc.mp4")

    # Step 1: Trim the target video to the first 10 seconds
    print(f"Trimming the first 10 seconds of video: {target_video}")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", target_video_path,
                "-t", "00:00:10",
                "-c", "copy",
                trimmed_video_path
            ],
            check=True
        )
        print(f"Trimmed video successfully saved to: {trimmed_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error trimming video {target_video}: {e}")
        continue

    # Step 2: Process the trimmed video using the face image
    print(f"Processing trimmed video with face image {source_face_path}: {trimmed_video_path}")
    try:
        subprocess.run(
            [
                "python", "/Users/brendanmurphy/Desktop/CS229ML/PROJECT/deeplivecam/Deep-Live-Cam/run.py",
                "--source", source_face_path,
                "--target", trimmed_video_path,
                "--output", output_video_path,
            ],
            check=True
        )
        print(f"Successfully processed video and saved to: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing video {target_video}: {e}")
