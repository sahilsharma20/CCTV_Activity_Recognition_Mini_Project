import cv2
import numpy as np

WIDTH = 640
HEIGHT = 580
FPS = 30
DURATION = 5

TOTAL_FRAMES = FPS * DURATION

output_path = "data/raw_video/cctv_video.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    output_path,
    fourcc,
    FPS,
    (WIDTH, HEIGHT)
)

# np.full() will creates an array and fills every element with the same value.
for frame_number in range(TOTAL_FRAMES):
    frame = np.full((HEIGHT, WIDTH, 3), 220, dtype=np.uint8)
    
    cv2.putText(
        frame,
        "CCTV CAMERA",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Frame : {frame_number}",
        (15, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    writer.write(frame)

writer.release()

print("Video Created Succesfully")