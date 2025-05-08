# ensemble_cancer_detection_and_data_augmentation.py

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.applications import EfficientNetB0  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
from torch.utils.data import DataLoader, Dataset  # type: ignore
import flask  # type: ignore
from flask import Flask, request, jsonify  # type: ignore
from datasets import load_dataset  # type: ignore
import shap  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

# ============================
# 1. DATA LOADING & PREPROCESSING
# ============================

# For demonstration, we load a cancer histopathology dataset from Hugging Face
dataset: dict = load_dataset("cancer_histopathology", split="train")


def preprocess_images(data: dict) -> tuple:
    """
    Convert raw dataset to normalized numpy arrays.
    :param data: Input dataset containing 'image' and 'label' fields.
    :return: Tuple of (images, labels)
    """
    images: np.ndarray = np.array([np.array(img) for img in data["image"]])
    labels: np.ndarray = np.array(data["label"])
    images = images.astype(np.float32) / 255.0  # Normalize images to [0, 1]
    return images, labels


X: np.ndarray; y: np.ndarray = preprocess_images(dataset)
# Split dataset into train and test (80/20 split)
split_index: int = int(0.8 * len(X))
X_train: np.ndarray = X[:split_index]
y_train: np.ndarray = y[:split_index]
X_test: np.ndarray = X[split_index:]
y_test: np.ndarray = y[split_index:]


# ============================
# 2. GAN-BASED DATA AUGMENTATION (SIMPLIFIED EXAMPLE)
# ============================
def build_generator() -> tf.keras.Model:
    """Builds a simple generator model."""
    noise_dim: int = 100
    generator: tf.keras.Model = models.Sequential([
        layers.Dense(128, activation="relu", input_shape=(noise_dim,)),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(np.prod((64, 64, 3)), activation="tanh"),
        layers.Reshape((64, 64, 3))
    ])
    return generator


generator: tf.keras.Model = build_generator()
# Generate synthetic images (for demo, generate 100 synthetic images)
noise: np.ndarray = np.random.normal(0, 1, (100, 100)).astype(np.float32)
synthetic_images: np.ndarray = generator.predict(noise)
# Concatenate synthetic data to training images (simple augmentation)
X_train_aug: np.ndarray = np.concatenate((X_train, synthetic_images), axis=0)
y_train_aug: np.ndarray = np.concatenate((y_train, np.full((100,), fill_value=1)), axis=0)  # assuming label '1' for cancerous

# ============================
# 3. ENSEMBLE MODEL CREATION
# ============================


# Define a simple custom CNN using TensorFlow
def build_custom_cnn(input_shape: tuple) -> tf.keras.Model:
    model: tf.keras.Model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    return model


input_shape: tuple = (64, 64, 3)  # Adjusted shape for synthetic images
custom_cnn: tf.keras.Model = build_custom_cnn(input_shape)
custom_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load a pretrained EfficientNet model and fine-tune
efficient_net: tf.keras.Model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
efficient_net.trainable = False
x: tf.Tensor = layers.Flatten()(efficient_net.output)
output: tf.Tensor = layers.Dense(2, activation='softmax')(x)
efficient_model: tf.keras.Model = tf.keras.Model(inputs=efficient_net.input, outputs=output)
efficient_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train both models (simplified training, using ImageDataGenerator for augmentation)
datagen: ImageDataGenerator = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2)
train_generator = datagen.flow(X_train_aug, y_train_aug, batch_size=32)

# For brevity, we train for only a few epochs
custom_cnn.fit(train_generator, epochs=3, validation_split=0.1)
efficient_model.fit(train_generator, epochs=3, validation_split=0.1)


# ============================
# 4. ENSEMBLE PREDICTION FUNCTION
# ============================
def ensemble_predict(image: np.ndarray) -> dict:
    """
    Generates ensemble prediction from both models.
    :param image: Preprocessed image array.
    :return: Dictionary with ensemble prediction and confidence.
    """
    image_expanded: np.ndarray = np.expand_dims(image, axis=0)
    pred1: np.ndarray = custom_cnn.predict(image_expanded)
    pred2: np.ndarray = efficient_model.predict(image_expanded)
    # Average the predictions
    ensemble_prob: np.ndarray = (pred1 + pred2) / 2.0
    predicted_label: int = int(np.argmax(ensemble_prob))
    confidence: float = float(np.max(ensemble_prob) * 100)
    return {"prediction": predicted_label, "confidence": confidence}


# ============================
# 5. MODEL EXPLAINABILITY USING SHAP (for custom CNN)
# ============================
explainer: shap.Explainer = shap.DeepExplainer(custom_cnn, X_train_aug[:50])
shap_values: list = explainer.shap_values(X_test[:10])
shap.summary_plot(shap_values, X_test[:10])

# ============================
# 6. FLASK API DEPLOYMENT
# ============================
app: Flask = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_api() -> flask.Response:
    """
    API endpoint to predict cancer from input image.
    Expects JSON payload with key 'image' containing pixel data.
    """
    input_data: dict = request.get_json()
    image_data: list = input_data.get('image', [])
    image_array: np.ndarray = np.array(image_data, dtype=np.float32)
    image_array /= 255.0  # Normalize
    result: dict = ensemble_predict(image_array)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
