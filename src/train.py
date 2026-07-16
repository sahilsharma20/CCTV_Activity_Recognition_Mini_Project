import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnn_model import model

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "output",
    exist_ok=True
)

# ==================================
# Dataset Paths
# ==================================

TRAIN_PATH = "data/dataset/train"
TEST_PATH = "data/dataset/test"

# =====================================================
# Image Generators
# =====================================================

train_generator = ImageDataGenerator(
    rescale=1./255
)

test_generator = ImageDataGenerator(
    rescale=1./255
)

# =====================================================
# Load Training Dataset
# =====================================================

train_dataset = train_generator.flow_from_directory(
    TRAIN_PATH,
    target_size = (48, 48),
    batch_size = 16,
    class_mode = "categorical"
)

# =====================================================
# Load Testing Dataset
# =====================================================

test_dataset = test_generator.flow_from_directory(
    TEST_PATH,
    target_size=(48,48),
    batch_size=16,
    class_mode="categorical",
    shuffle = False
)


# =====================================================
# Dataset Info.
# =====================================================

print("=" * 50)
print("Class Labels")
print("=" * 50)
print(train_dataset.class_indices)

print()

print(f"Training Images : {train_dataset.samples}")
print(f"Testing Images  : {test_dataset.samples}")

print("="*50)
print("Training Started...")
print("="*50)


# =====================================================
# Train CNN
# =====================================================

history = model.fit(
    train_dataset,
    validation_data = test_dataset,
    epochs=20
)


# =====================================================
# Training Accuracy
# =====================================================

train_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]

print("="*50)
print(f"Final Training Accuracy : {train_acc:.4f}")
print(f"Final Validation Accuracy : {val_acc:.4f}")
print("="*50)

# =====================================================
# Save Model
# =====================================================

model.save(
    "models/activity_model.keras"
)

# =====================================================
# Accuracy graph
# =====================================================
plt.figure(figsize=(8,5))
plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)
plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig(
    "output/training_accuracy.png"
)
plt.show()


# =====================================================
# Loss Graph
# =====================================================
plt.figure(figsize=(8,5))
plt.plot(
    history.history["loss"],
    label="Training Loss"
)
plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(
    "output/training_loss.png"
)
plt.show()

# =====================================================
# Success Message:
# =====================================================

print("="*50)
print("Training Complete")
print("Model Saved Successfully")
print("="*50)