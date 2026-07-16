"""
Input CCTV Video
        │
        ▼
OpenCV VideoCapture
        │
        ▼
Read One Frame
        │
        ▼
Resize (48*48)
        │
        ▼
Normalize (÷255)
        │
        ▼
Convert to Tensor
        │
        ▼
CNN Model
        │
        ▼
Softmax Probabilities
        │
        ▼
argmax()
        │
        ▼
Predicted Activity
        │
        ▼
Write Activity on Frame
        │
        ▼
Save Output Video + CSV
"""


import os
import csv
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ===============================
# Loading the CNN Model 
# ===============================

model = load_model("models/activity_model.keras")

# ===============================
# Define class labels  
# ===============================

class_labels = {
    0 : "Fighting",
    1 : "Running",
    2 : "Sitting",
    3 : "Walking"
}

# ===============================
# Video Path
# ===============================

VIDEO_PATH = "data/raw_video/cctv_video.mp4"

# ===============================
# Open Video
# ===============================

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError("Could not find the video")

# ===============================
# Video Properties: 
# ===============================

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# ===============================
# Creating Output Folder
# ===============================

os.makedirs("output", exist_ok=True)

# ===============================
# Creating Output Video via VideoWriter 
# ===============================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    "output/predicted_video.mp4",
    fourcc,
    fps,
    (frame_width, frame_height)
)

if not out.isOpened():
    raise ValueError("Could not create output video.")


# ===============================
# Creating CSV Files
# ===============================

csv_file = open(
    "output/predictions.csv",
    mode="w",
    newline=""
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "Frame Number",
    "Activity",
    "Confidence (%)"
])

# =====================================================
# Frame Counter
# =====================================================

frame_number = 0


# ===============================
# Reading Frames
# ===============================
# This Loop will process for 150 times 

while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame_number += 1
    if frame_number % 30 == 0:
        print(
            f"Processed {frame_number} frames..."
        )

    # ===============================
    # Preprocess Frame
    # Resize -> Normalize -> Tensor
    # ===============================

    img = cv2.resize(frame, (48, 48))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # ===============================
    # CNN Model Prediction 
    # ===============================

    prediction = model.predict(img, verbose=0)

    # ===============================
    # Convert Activity Index to Label (Softmax + argmax())
    # ===============================

    activity_index = np.argmax(prediction)
    activity = class_labels[activity_index]
    confidence = prediction[0][activity_index] * 100

    # ===============================
    # Save Prediction to CSV
    # ===============================

    csv_writer.writerow([
        frame_number,
        activity,
        f"{confidence:.2f}"
    ])


    # ===============================
    # Display Activity
    # ===============================

    cv2.putText(
        frame,
        f"Activity : {activity}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # ===============================
    # Display Confidence
    # ===============================

    cv2.putText(
        frame,
        f"Confidence : {confidence:.2f}%",
        (20,90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255,0,0),
        2
    )
    # ==============================================
    # Display Frame Number
    # ==============================================

    cv2.putText(
        frame,
        f"Frame : {frame_number}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    # ==============================================
    # Display FPS
    # ==============================================

    cv2.putText(
        frame,
        f"FPS : {int(fps)}",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )


    # ==============================================
    # Save Frame to Output Video
    # ==============================================

    out.write(frame)

# =====================================================
# Release Resources
# =====================================================

cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

# =====================================================
# Success Message
# =====================================================

print("=" * 50)
print("CCTV Activity Recognition Completed")
print(f"Frames Processed : {frame_number}")
print("Output Video     : output/predicted_video.mp4")
print("Prediction CSV   : output/predictions.csv")
print("=" * 50)