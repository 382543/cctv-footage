import cv2
import os
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import csv  # Import the CSV module

# Paths
DATA_FOLDER = r"D:\DV Lab\FaceRecognitionProject\data"
OUTPUT_FOLDER = r"D:\DV Lab\FaceRecognitionProject\output"
TARGET_IMAGE_PATH = os.path.join(DATA_FOLDER, "target.jpeg")
CAM_VIDEOS = [f"cam{i}.mp4" for i in range(1, 8)]  # cam1.mp4 to cam7.mp4

# Ensure the output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize MTCNN for face detection and InceptionResnetV1 for embeddings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --- Process the target face ---
target_img_cv = cv2.imread(TARGET_IMAGE_PATH)
if target_img_cv is None:
    raise FileNotFoundError(f"Target image not found at: {TARGET_IMAGE_PATH}")
target_img_rgb = cv2.cvtColor(target_img_cv, cv2.COLOR_BGR2RGB)
target_pil = Image.fromarray(target_img_rgb)

# Get the aligned face from the target image using MTCNN
target_face = mtcnn(target_pil)
if target_face is None:
    raise ValueError("No face detected in the target image.")

# Ensure target_face has 3 channels (if it's grayscale, repeat the channel)
if target_face.ndim == 3 and target_face.shape[0] == 1:
    target_face = target_face.repeat(3, 1, 1)
with torch.no_grad():
    target_embedding = resnet(target_face.unsqueeze(0).to(device)).cpu().numpy()

# --- Video Processing Function ---
def process_video(video_name):
    """Processes a single video to extract frames with the target person and generates a report."""
    video_path = os.path.join(DATA_FOLDER, video_name)
    output_path = os.path.join(OUTPUT_FOLDER, video_name.replace(".mp4", "_processed.mp4"))
    report_path = os.path.join(OUTPUT_FOLDER, video_name.replace(".mp4", "_report.csv"))  # CSV report path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize VideoWriter (MP4 format)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    camera_label = video_name.replace(".mp4", "")  # e.g., cam1, cam2, etc.
    frame_number = 0
    written_frames = 0  # Count of frames written to output
    threshold = 0.95  # Euclidean distance threshold for a match (adjust as needed)

    # Initialize CSV writer for the report
    with open(report_path, 'w', newline='') as csvfile:
        report_writer = csv.writer(csvfile)
        report_writer.writerow(["Frame Number", "Timestamp (s)", "Face Detected", "Match Found", 
                                 "Face X1", "Face Y1", "Face X2", "Face Y2", "Distance"])  # Header row

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)

            # Detect faces using MTCNN
            boxes, probs = mtcnn.detect(pil_frame)
            match_found = False
            x1, y1, x2, y2 = 0, 0, 0, 0  # Initialize face coordinates
            dist = float('inf')  # Initialize distance

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    face_tensor_list = mtcnn.extract(pil_frame, np.array([box]), None)
                    if face_tensor_list is None or len(face_tensor_list) == 0:
                        continue

                    face_tensor = face_tensor_list[0].to(device)
                    # Ensure the face tensor has 3 channels
                    if face_tensor.ndim == 3 and face_tensor.shape[0] == 1:
                        face_tensor = face_tensor.repeat(3, 1, 1)
                    elif face_tensor.ndim == 2:
                        # If shape is (H, W), add a channel dimension and then repeat it
                        face_tensor = face_tensor.unsqueeze(0).repeat(3, 1, 1)

                    with torch.no_grad():
                        # Add batch dimension and compute embedding
                        input_tensor = face_tensor.unsqueeze(0)
                        face_embedding = resnet(input_tensor).cpu().numpy()

                    # Compare embeddings using Euclidean distance
                    dist = np.linalg.norm(face_embedding - target_embedding)
                    if dist < threshold:
                        match_found = True
                        break  # Stop after first match

            if match_found:
                timestamp = frame_number / fps
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"Camera: {camera_label}, Time: {timestamp:.2f}s"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                out.write(frame)
                written_frames += 1

            # Write row to CSV report
            report_writer.writerow([frame_number, frame_number / fps, boxes is not None, match_found,
                                     x1, y1, x2, y2, dist])

            frame_number += 1

    cap.release()
    out.release()

    if written_frames == 0:
        # Delete the output file if no matching frames were written
        if os.path.exists(output_path):
            os.remove(output_path)
        print(f"No matching frames found in {video_name}. Output file removed.")
    else:
        print(f"Processed {video_name} → {output_path} ({written_frames} frames written)")
        print(f"Report generated: {report_path}")

# Process all videos
for cam in CAM_VIDEOS:
    process_video(cam)

import os
import glob
import subprocess
import shlex

# Specify the full path to ffmpeg.exe if not in your PATH
FFMPEG_EXECUTABLE = r"C:\ffmpeg\bin\ffmpeg.exe"
OUTPUT_FOLDER = r"D:\DV Lab\FaceRecognitionProject\output"
FINAL_OUTPUT = os.path.join(OUTPUT_FOLDER, "final_summary.mp4")
TEMP_FOLDER = os.path.join(OUTPUT_FOLDER, "temp_videos")

# Ensure temp folder exists
os.makedirs(TEMP_FOLDER, exist_ok=True)

def reencode_videos():
    """Re-encodes all videos to ensure consistent format."""
    video_files = sorted(glob.glob(os.path.join(OUTPUT_FOLDER, "*_processed.mp4")))
    if not video_files:
        print("No processed video files found!")
        return []

    reencoded_videos = []
    for idx, video in enumerate(video_files):
        output_video = os.path.join(TEMP_FOLDER, f"reencoded_{idx}.mp4")
        ffmpeg_command = [
            FFMPEG_EXECUTABLE,
            "-i", video,
            "-vf", "scale=1280:720",  # Resize to 1280x720 (adjust as needed)
            "-r", "30",  # Force 30 FPS
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-an",  # Remove audio
            "-y", output_video
        ]
        print(f"Re-encoding: {video}")
        subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        reencoded_videos.append(output_video)

    return reencoded_videos

def merge_videos_ffmpeg():
    """Merges re-encoded videos."""
    reencoded_videos = reencode_videos()
    if not reencoded_videos:
        print("No videos to merge!")
        return

    file_list_path = os.path.join(TEMP_FOLDER, "file_list.txt")
    with open(file_list_path, "w") as f:
        for video in reencoded_videos:
            f.write(f"file '{video}'\n")

    ffmpeg_command = [
        FFMPEG_EXECUTABLE,
        "-f", "concat",
        "-safe", "0",
        "-i", file_list_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",  # Remove audio
        FINAL_OUTPUT
    ]

    print("Running FFmpeg command:")
    print(" ".join(shlex.quote(arg) for arg in ffmpeg_command))

    result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("FFmpeg output:", result.stdout)
    print("FFmpeg errors:", result.stderr)

    if result.returncode == 0:
        print(f"Final summary video saved: {FINAL_OUTPUT}")
    else:
        print("Error merging videos using FFmpeg.")

if _name_ == "_main_":
    merge_videos_ffmpeg()