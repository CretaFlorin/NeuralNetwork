import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

import numpy as np

# Load the Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]

# One hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# NN
model = models.Sequential(
    [
        # --- Old Architecture ---
        # layers.Flatten(input_shape=(28, 28, 1)),
        # layers.Dense(784, activation="relu"),
        # layers.Dense(64, activation="relu"),
        # layers.Dense(10, activation="softmax"),
        # --- New Architecture ---
        layers.Conv2D(
            32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)
        ),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        #
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        #
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)


# Predict
loss, acc = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {acc} - Loss: {loss}")
