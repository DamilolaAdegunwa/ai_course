# Project Title: Advanced Multimodal AI for Cancer Disease Detection (project #1)

# File Name: advanced_multimodal_ai_cancer_detection.py

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import glob
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Dataset URL
dataset_url = "https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images"


# Data Preprocessing
def load_images_from_folder(folder, label, transform=None):
    images = []
    labels = []
    for filename in glob.glob(os.path.join(folder, '*.jpeg')):
        img = Image.open(filename)
        if transform:
            img = transform(img)
        images.append(img)
        labels.append(label)
    return images, labels


def preprocess_data(base_path, transform=None):
    classes = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']
    images = []
    labels = []
    for idx, cls in enumerate(classes):
        cls_folder = os.path.join(base_path, cls)
        cls_images, cls_labels = load_images_from_folder(cls_folder, idx, transform)
        images.extend(cls_images)
        labels.extend(cls_labels)
    return images, labels


# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load and preprocess data
base_path = 'path_to_downloaded_dataset'  # Update this path to your dataset location
images, labels = preprocess_data(base_path, transform=transform)

# Convert lists to tensors
images = torch.stack(images)
labels = torch.tensor(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


# Define PyTorch Custom Dataset
class CancerDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


# Create datasets and dataloaders
train_dataset = CancerDataset(X_train, y_train)
test_dataset = CancerDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define Model (Using ResNet)
class CancerDetectionModel(torch.nn.Module):
    def __init__(self):
        super(CancerDetectionModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(512, 5)  # 5 classes

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
    print(classification_report(y_true, y_pred, target_names=['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']))


evaluate_model(model, test_loader)
