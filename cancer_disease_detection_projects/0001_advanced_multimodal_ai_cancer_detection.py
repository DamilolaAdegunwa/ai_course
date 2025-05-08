# Project Title: Advanced Multimodal AI for Cancer Disease Detection (project #1)

# File Name: advanced_multimodal_ai_cancer_detection.py

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, Dataset
import flask
from flask import Flask, request, jsonify

# Load Dataset from Hugging Face (Example Dataset for Demonstration)
from datasets import load_dataset

dataset = load_dataset("medical-cancer-detection")


# Data Preprocessing
def preprocess_data(dataset):
    images = np.array([np.array(img) for img in dataset['image']])
    labels = np.array(dataset['label'])
    images = images / 255.0  # Normalize images
    return images, labels


X, y = preprocess_data(dataset['train'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define PyTorch Custom Dataset
class CancerDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose([transforms.ToTensor()])
train_dataset = CancerDataset(X_train, y_train, transform=transform)
test_dataset = CancerDataset(X_test, y_test, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define Model (Using ResNet)
class CancerDetectionModel(torch.nn.Module):
    def __init__(self):
        super(CancerDetectionModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(512, 2)

    def forward(self, x):
        return self.model(x)


model = CancerDetectionModel()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")


train_model(model, train_loader, criterion, optimizer)


# Model Evaluation
def evaluate_model(model, test_loader):
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.numpy())
            y_true.extend(labels.numpy())
    print(classification_report(y_true, y_pred))


evaluate_model(model, test_loader)

# Explainability using SHAP
explainer = shap.Explainer(model)
shap_values = explainer.shap_values(X_test[:10])
shap.summary_plot(shap_values, X_test[:10])

# Deploy as a Flask API
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    data = np.array(data) / 255.0  # Normalize
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    output = model(data)
    _, predicted = torch.max(output, 1)
    return jsonify({'prediction': int(predicted.item())})


if __name__ == '__main__':
    app.run(debug=True)
