import cv2
import os

VIDEO_PATH = "data/raw_video/cctv_video.mp4"
OUTPUT_FOLDER = "data/frames"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():
    raise ValueError(
        "Unable to open CCTV Video"
    )

# Read Video info.

fps = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

print("=" * 40)

print(f"FPS : {fps}")

print(f"Width : {width}")

print(f"Height : {height}")

print(f"Frames : {total_frames}")

print("=" * 40)

# Extracting frames via openCV:

frame_number = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_name = os.path.join(
        OUTPUT_FOLDER,
        f"frame_{frame_number:04d}.jpg"
    )

    cv2.imwrite(
        frame_name,
        frame
    )

    frame_number += 1

cap.release()
print("=" * 40)
print(f"Frames Extracted : {frame_number}")
print("Extraction Complete")
print("=" * 40)