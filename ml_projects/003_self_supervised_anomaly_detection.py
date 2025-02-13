import datasets
import numpy
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pandas import DataFrame
from torch.nn import Sequential, MSELoss
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from typing import Any

# Load dataset from Hugging Face
dataset: datasets.arrow_dataset.Dataset = load_dataset("hangyeol522/anomaly-detection-model", split="train")

# Convert dataset to DataFrame
df: DataFrame = pd.DataFrame(dataset)

# Select numerical features
numerical_columns: pandas.core.indexes.base.Index = df.select_dtypes(include=[np.number]).columns

df: DataFrame = df[numerical_columns]
print(f"before normalizing {df.head(3)}")
# Normalize data
scaler: StandardScaler = StandardScaler()
df[numerical_columns]: pandas.core.frame.DataFrame = scaler.fit_transform(df[numerical_columns])
print(f"after normalizing {df.head(3)}")
# Convert to PyTorch tensors
data_tensor: torch.Tensor = torch.tensor(df.values, dtype=torch.float32)


# Custom PyTorch Dataset
class NetworkDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Create DataLoader
batch_size: int = 256
data_loader: DataLoader = DataLoader(NetworkDataset(data_tensor), batch_size=batch_size, shuffle=True)


# Define the Masked Autoencoder
class MaskedAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(MaskedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, mask_ratio=0.3):
        # Randomly mask input features
        from torch import Tensor
        mask: Tensor = torch.rand_like(x) > mask_ratio
        masked_x = x * mask.float()

        # Encode & Decode
        encoded: Sequential = self.encoder(masked_x)
        reconstructed: Sequential = self.decoder(encoded)

        return reconstructed, mask


# Initialize model
input_dim: int = data_tensor.shape[1]
model: MaskedAutoencoder = MaskedAutoencoder(input_dim)
criterion: MSELoss = nn.MSELoss()
optimizer: torch.optim.Adam = optim.Adam(model.parameters(), lr=0.001)

# Train Model
num_epochs = 20
for epoch in range(num_epochs):
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        reconstructed, mask = model(batch)
        loss = criterion(reconstructed * mask, batch * mask)  # Compute loss only on unmasked data
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Detect Anomalies (Reconstruction Error)
with torch.no_grad():
    reconstructed_data, _ = model(data_tensor)
    anomaly_scores: numpy.ndarray = torch.mean((data_tensor - reconstructed_data) ** 2, dim=1).numpy()

# Plot anomaly scores
plt.hist(anomaly_scores, bins=50)
plt.title("Anomaly Score Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.show()

# Define anomaly threshold
threshold: numpy.float64 = np.percentile(anomaly_scores, 95)  # Top 5% are anomalies
print(f"the anomaly threshold: {threshold}")
anomalies: np.ndarray[Any, np.dtype[np.bool_]] = anomaly_scores > threshold
print(f"Detected {sum(anomalies)} anomalies out of {len(anomalies)} samples.")

comments = """
Here's your next **far more advanced** Machine Learning project. This one focuses on **self-supervised learning**, an area that bridges unsupervised and supervised learning techniques.  

---  

# **Project Title**  
**Self-Supervised Learning for Anomaly Detection in Network Traffic**  

---  

### **File Name:**  
`self_supervised_anomaly_detection.py`  

---  

### **Project Description**  
In this project, we implement **Self-Supervised Learning (SSL)** for **Anomaly Detection** in **network traffic** data using a **Masked Autoencoder (MAE)**.  
The model learns to reconstruct normal network behavior and detects anomalies when the reconstruction error is high.  

We use the **"Kitsune Network Attack Dataset" from Hugging Face**, which contains both normal and attack network traffic.  

This project is particularly useful for **cybersecurity**, **fraud detection**, and **predictive maintenance**.  

---  

## **Python Code**  
### **(Self-Supervised Anomaly Detection using a Masked Autoencoder)**  
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Load dataset from Hugging Face
dataset = load_dataset("kitsune-network-attack", split="train")

# Convert dataset to DataFrame
df = pd.DataFrame(dataset)

# Select numerical features
numerical_columns = df.select_dtypes(include=[np.number]).columns
df = df[numerical_columns]

# Normalize data
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Convert to PyTorch tensors
data_tensor = torch.tensor(df.values, dtype=torch.float32)

# Custom PyTorch Dataset
class NetworkDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Create DataLoader
batch_size = 256
data_loader = DataLoader(NetworkDataset(data_tensor), batch_size=batch_size, shuffle=True)

# Define the Masked Autoencoder
class MaskedAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(MaskedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, mask_ratio=0.3):
        # Randomly mask input features
        mask = torch.rand_like(x) > mask_ratio
        masked_x = x * mask.float()

        # Encode & Decode
        encoded = self.encoder(masked_x)
        reconstructed = self.decoder(encoded)

        return reconstructed, mask

# Initialize model
input_dim = data_tensor.shape[1]
model = MaskedAutoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train Model
num_epochs = 20
for epoch in range(num_epochs):
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        reconstructed, mask = model(batch)
        loss = criterion(reconstructed * mask, batch * mask)  # Compute loss only on unmasked data
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Detect Anomalies (Reconstruction Error)
with torch.no_grad():
    reconstructed_data, _ = model(data_tensor)
    anomaly_scores = torch.mean((data_tensor - reconstructed_data) ** 2, dim=1).numpy()

# Plot anomaly scores
plt.hist(anomaly_scores, bins=50)
plt.title("Anomaly Score Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.show()

# Define anomaly threshold
threshold = np.percentile(anomaly_scores, 95)  # Top 5% are anomalies
anomalies = anomaly_scores > threshold
print(f"Detected {sum(anomalies)} anomalies out of {len(anomalies)} samples.")
```

---

## **Example Inputs & Expected Outputs**
### **Example 1: Normal Network Traffic**
**Input:**  
```python
sample_input = torch.tensor([[0.5, -0.2, 1.3, 0.7, -0.4]])  # Normalized sample
```
**Expected Output:**  
Reconstruction Error: **Low** (Model reconstructs well)  
```python
Reconstruction Error: 0.003
Anomaly Detected: False
```

### **Example 2: Malicious Network Traffic**
**Input:**  
```python
sample_input = torch.tensor([[10.5, -5.2, 25.3, 12.7, -9.4]])  # Out-of-distribution attack sample
```
**Expected Output:**  
Reconstruction Error: **High** (Model fails to reconstruct anomalies)  
```python
Reconstruction Error: 2.75
Anomaly Detected: True
```

---

## **Key Learnings & Research Areas**
🔹 **Self-Supervised Learning (SSL)** – Learning representations without labels.  
🔹 **Masked Autoencoders (MAE)** – Similar to Transformers, they learn by reconstructing missing data.  
🔹 **Reconstruction-Based Anomaly Detection** – Uses high reconstruction error to detect outliers.  
🔹 **Hugging Face Datasets API** – Directly loads datasets into Pandas/PyTorch.  
🔹 **Neural Network Regularization** – Masking forces the model to learn meaningful features.  
🔹 **Feature Engineering for Security** – How to preprocess network traffic data.  
🔹 **Scalable Model Deployment** – Could be adapted for real-time anomaly detection in cybersecurity.  

---

## **Why This Project is Advanced**
✅ Uses **Self-Supervised Learning** (not traditional supervised/unsupervised).  
✅ Implements a **Masked Autoencoder (MAE)** instead of standard autoencoders.  
✅ **Directly integrates with Hugging Face** for dataset retrieval (no manual download).  
✅ Suitable for **high-stakes applications** like **cybersecurity and fraud detection**.  
✅ Introduces **deep learning-based anomaly detection** with PyTorch.  

---

🚀 **Next Steps**  
- **Try different masking strategies (e.g., variable mask ratios).**  
- **Extend this to real-time network monitoring using Kafka.**  
- **Train on multiple network datasets to generalize better.**  
- **Experiment with Transformer-based architectures instead of simple Autoencoders.**  

---

🔴 **Let me know if you need modifications or explanations on any part!** 🔥
"""
