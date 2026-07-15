"""
A CNN cannot directly use them.
We need to preprocess every frame before training.

Extracted Frame
      │
      ▼
Resize (48 * 48)
      │
      ▼
BGR → RGB
      │
      ▼
Normalize (0 to 1)
      │
      ▼
Convert to Tensor
      │
      ▼
    CNN
"""

import cv2
import numpy as np
import os

# ==========================
# Paths
# ==========================
FRAME_FOLDER = "data/frames"
OUTPUT_FOLDER = "data/preprocessed"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================
# Preprocessing Function
# ==========================
def preprocess(frame):
    """
    Resize image to 48x48,
    convert BGR to RGB,
    normalize pixel values (0-1)
    """

    # Resize image
    resized = cv2.resize(frame, (48, 48))

    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize
    normalized = rgb.astype(np.float32) / 255.0

    return normalized


# ==========================
# Read All Frames
# ==========================
frame_files = sorted(os.listdir(FRAME_FOLDER))

print("=" * 40)
print(f"Total Frames Found : {len(frame_files)}")
print("=" * 40)


# ==========================
# Process Each Frame
# ==========================
for file_name in frame_files:

    image_path = os.path.join(FRAME_FOLDER, file_name)

    image = cv2.imread(image_path)

    if image is None:
        print(f"Skipping {file_name}")
        continue

    processed = preprocess(image)

    # Convert back to uint8 for saving
    processed_uint8 = (processed * 255).astype(np.uint8)

    # Convert RGB back to BGR (OpenCV saves in BGR)
    processed_bgr = cv2.cvtColor(processed_uint8, cv2.COLOR_RGB2BGR)

    output_path = os.path.join(OUTPUT_FOLDER, file_name)

    cv2.imwrite(output_path, processed_bgr)


print("=" * 40)
print(f"Frames Processed Successfully : {len(frame_files)}")
print(f"Saved To : {OUTPUT_FOLDER}")
print("Preprocessing Complete")
print("=" * 40)
