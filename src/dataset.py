"""
Dataset Creation and Organization:

1. Target Pipeline:
CCTV Video
      │
      ▼
Extract Frames
      │
      ▼
Preprocess Images
      │
      ▼
Organized Dataset
      │
      ▼
TensorFlow Dataset Loader
      │
      ▼
    CNN

2. Updated Foleder Structure :

data/
│
├── frames/
│
├── preprocessed/
│
└── dataset/
    │
    ├── train/
    │   ├── Walking/
    │   ├── Sitting/
    │   ├── Fighting/
    │   └── Running/
    │
    └── test/
        ├── Walking/
        ├── Sitting/
        ├── Fighting/
        └── Running/

3. Why do we need this structure?
TensorFlow ImageDataGenerator

train/
│
├── Walking/
│     img1.jpg
│     img2.jpg
│
├── Sitting/
│
├── Fighting/
│
└── Running/

It automatically creates:
Walking  → 0
Sitting  → 1
Fighting → 2
Running  → 3
"""



import os       # os → Work with folders and files.
import shutil   # shutil → Copy images into new folders.

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PREPROCESSED_FOLDER = "data/preprocessed"
DATASET_FOLDER = "data/dataset"

TRAIN_FOLDER = os.path.join(DATASET_FOLDER, "train")
TEST_FOLDER = os.path.join(DATASET_FOLDER, "test")

ACTIVITIES = [
    "Walking",
    "Sitting",
    "Fighting",
    "Running"
]

TRAIN_IMAGES = 30

def get_activity(frame_number):
    if frame_number < 38:
        return "Walking"
    elif frame_number < 75:
        return "Sitting"
    elif frame_number < 113:
        return "Fighting"
    else:
        return "Running"
# Since we created the video, we already know which activity each frame belongs to. This is called ground truth labeling.


for folder in [TRAIN_FOLDER, TEST_FOLDER]:
    for activity in ACTIVITIES:
        os.makedirs(    
            os.path.join(folder, activity),
            exist_ok=True
        )

frame_files = sorted(
    [
        file
        for file in os.listdir(PREPROCESSED_FOLDER)
        if file.endswith(".jpg")
    ]
)

print("=" * 40)
print(f"Total images : {len(frame_files)}")
print("=" * 40)

# Creating Counters : We need to know how many images we've already placed into each activity
activity_count = {
    "Walking": 0,
    "Sitting": 0,
    "Fighting":0,
    "Running":0
}

# Preprocess every image:
for file_name in frame_files:
    frame_number  = int(
        file_name.split("_")[1].split(".")[0]
    )
    activity = get_activity(frame_number).strip()

    activity_count[activity] += 1


    # Decide Train or Test:
    if activity_count[activity] <= TRAIN_IMAGES:
        destination_folder = os.path.join(
            TRAIN_FOLDER,
            activity
        )
    else :
        destination_folder = os.path.join(
            TEST_FOLDER,
            activity
        )

    # Why 30 : coz activity has 38 frames , 30 for tarining and 8 for testing fo each activity

    # Copy the image :
    source = os.path.join(
        PREPROCESSED_FOLDER,
        file_name
    )
    destination_path = os.path.join(
        destination_folder,
        file_name
    )
    shutil.copy(
        source,
        destination_path
    )


# Print Summary report:
print("=" * 50)
print("Dataset Created Succesfully.")
print("=" * 50)


total_train = 0
total_test = 0
for activity in ACTIVITIES:
    train_count = len(
        os.listdir(
            os.path.join(TRAIN_FOLDER, activity)
        )
    )
    test_count = len(
        os.listdir(
            os.path.join(
                TEST_FOLDER, activity
            )
        )
    )
    total_train += train_count
    total_test += test_count

    print(f"{activity}")
    print(f" Train : {train_count}")
    print(f" Test : {test_count}")

print("-" * 50)
print(f"Total Train Images : {total_train}")
print(f"Total Test Images  : {total_test}")
print(f"Grand Total        : {total_train + total_test}")
print("=" * 50)