import os
import subprocess

# Define the index-to-sex mapping
index_to_sex = {
    903: "M", 904: "F", 905: "F", 907: "F", 913: "F", 915: "M", 917: "F", 918: "F", 919: "M", 920: "M",
    924: "F", 925: "M", 928: "M", 929: "F", 932: "M", 933: "M", 934: "F", 938: "M", 939: "M", 942: "M",
    943: "M", 944: "F", 945: "M", 946: "F", 947: "F", 948: "F", 950: "F", 951: "F", 952: "F", 953: "M",
    957: "F", 959: "F", 960: "F", 962: "F", 965: "F", 966: "M", 968: "F", 969: "F", 971: "M", 974: "M",
    977: "F", 980: "F", 981: "F", 982: "M", 985: "F", 986: "F", 987: "M", 988: "M", 989: "F", 990: "F",
    991: "M", 992: "F", 993: "F", 994: "F", 996: "M", 998: "F", 999: "F"
}

# Create male and female lists
male = [index for index, sex in index_to_sex.items() if sex == "M"]
female = [index for index, sex in index_to_sex.items() if sex == "F"]

# Shift each list by one
shifted_male = male[1:] + male[:1]
shifted_female = female[1:] + female[:1]

# Map each target video index to a new source frame index
male_mapping = dict(zip(male, shifted_male))
female_mapping = dict(zip(female, shifted_female))

# Base directories
TARGET_DIR = "/Users/brendanmurphy/Desktop/CS229ML/PROJECT/deeplivecam/my_dataset/ff/"  # Directory containing target videos
FRAME_DIR = "/Users/brendanmurphy/Desktop/CS229ML/PROJECT/deeplivecam/my_dataset/dlc_extracted_frames/"  # Directory containing extracted frames
OUTPUT_DIR = "/Users/brendanmurphy/Desktop/CS229ML/PROJECT/deeplivecam/my_dataset/dlc_cross_processed/"  # Directory to save processed videos

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get the list of target video files
target_videos = [file for file in os.listdir(TARGET_DIR) if file.endswith(".mp4")]

# Process each target video
for target_video in target_videos:
    # Extract the video index from the file name
    video_index = int(os.path.splitext(target_video)[0])

    # Determine the source frame based on gender
    if video_index in male_mapping:
        source_frame_index = male_mapping[video_index]
    elif video_index in female_mapping:
        source_frame_index = female_mapping[video_index]
    else:
        print(f"Skipping video {target_video}: no matching source frame found.")
        continue

    # Define paths
    target_video_path = os.path.join(TARGET_DIR, target_video)
    source_frame_path = os.path.join(FRAME_DIR, f"{source_frame_index}_extracted_frame.jpg")
    trimmed_video_path = os.path.join(OUTPUT_DIR, f"{video_index}_trimmed.mp4")
    output_video_path = os.path.join(OUTPUT_DIR, f"{video_index}_dlc.mp4")

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

    # Step 2: Process the trimmed video using the mapped source frame
    print(f"Processing trimmed video with source frame {source_frame_path}: {trimmed_video_path}")
    try:
        subprocess.run(
            [
                "python", "/Users/brendanmurphy/Desktop/CS229ML/PROJECT/deeplivecam/Deep-Live-Cam/run.py",
                "--source", source_frame_path,
                "--target", trimmed_video_path,
                "--output", output_video_path,
            ],
            check=True
        )
        print(f"Successfully processed video and saved to: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing video {target_video}: {e}")
