from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

# Create CNN Model
model = Sequential()

# Input image for layers
model.add(
    Input(shape=(48, 48, 3))
)

# First Convolution Layer
model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
    )
)

# First Pooling Layer
# Pooling reduces computation while preserving important features.
model.add(
    MaxPooling2D(
        pool_size = (2, 2)
    )
)

# Second Convolution Layer
model.add(
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    )
)

# Second Pooling Layer
model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)

# Flatten Feature Maps
model.add(
    Flatten()
)

# Hidden Layer
model.add(
    Dense(
        128,
        activation="relu"
    )
)

# Prevent Overfitting
model.add(
    Dropout(
        0.5
    )
)

# Output Layer (4 Classes)
model.add(
    Dense(
        4, 
        activation="softmax"
    )
)

# Compile Model:
model.compile(
    optimizer= "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)

# Print Model Summary
model.summary()
