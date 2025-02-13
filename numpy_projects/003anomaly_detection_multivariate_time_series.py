import numpy as np


def normalize_data(data):
    """
    Normalizes the data using z-score normalization.
    :param data: 2D Numpy array where rows are observations and columns are features.
    :return: Normalized dataset.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std


def compute_mahalanobis_distance(data, mean_vector, inv_cov_matrix):
    """
    Computes the Mahalanobis distance for each observation in the dataset.
    :param data: 2D Numpy array where rows are observations.
    :param mean_vector: 1D Numpy array of mean values for each feature.
    :param inv_cov_matrix: Inverse of the covariance matrix.
    :return: 1D Numpy array of Mahalanobis distances.
    """
    diff = data - mean_vector
    distances = np.sqrt(np.sum((diff @ inv_cov_matrix) * diff, axis=1))
    return distances


def detect_anomalies(data, threshold=3.0):
    """
    Detects anomalies in the dataset based on Mahalanobis distance.
    :param data: 2D Numpy array where rows are observations.
    :param threshold: Distance threshold to classify anomalies.
    :return: Indices of anomalies.
    """
    mean_vector = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    distances = compute_mahalanobis_distance(data, mean_vector, inv_cov_matrix)
    anomalies = np.where(distances > threshold)[0]
    return anomalies, distances


# Example Test
if __name__ == "__main__":
    # Simulated multivariate time series data (rows = observations, cols = features)
    np.random.seed(42)
    normal_data = np.random.normal(loc=0, scale=1, size=(100, 3))  # Normal observations
    anomaly_data = np.random.normal(loc=10, scale=1, size=(5, 3))  # Anomalies

    # Combine data
    data = np.vstack([normal_data, anomaly_data])

    # Normalize data
    normalized_data = normalize_data(data)

    # Detect anomalies
    anomalies, distances = detect_anomalies(normalized_data, threshold=3.0)

    print("Anomalies detected at indices:", anomalies)
    print("Mahalanobis Distances:", distances[anomalies])


comment = """
# the chatgpt link: https://chatgpt.com/c/6747eb46-b444-800c-b546-c677d0c4cd7e
### **Project Title**  
**Anomaly Detection in Multivariate Time Series using Numpy**  

**File Name**  
`anomaly_detection_multivariate_time_series.py`  

---

### **Short Description**  
This project involves creating an **anomaly detection system** for multivariate time series data using statistical analysis and matrix operations in Numpy. The system will identify unusual patterns or outliers by analyzing relationships between multiple variables over time. Advanced techniques like covariance, correlation, Mahalanobis distance, and eigenvector-based analysis will be applied. This project is practical for real-world applications like fraud detection, sensor data monitoring, and stock market anomaly tracking.

---

### **Python Code**

```python
import numpy as np

def normalize_data(data):
    
    # Normalizes the data using z-score normalization.
    # :param data: 2D Numpy array where rows are observations and columns are features.
    # :return: Normalized dataset.
    
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

def compute_mahalanobis_distance(data, mean_vector, inv_cov_matrix):
    
    # Computes the Mahalanobis distance for each observation in the dataset.
    # :param data: 2D Numpy array where rows are observations.
    # :param mean_vector: 1D Numpy array of mean values for each feature.
    # :param inv_cov_matrix: Inverse of the covariance matrix.
    # :return: 1D Numpy array of Mahalanobis distances.
    
    diff = data - mean_vector
    distances = np.sqrt(np.sum((diff @ inv_cov_matrix) * diff, axis=1))
    return distances

def detect_anomalies(data, threshold=3.0):
    
    # Detects anomalies in the dataset based on Mahalanobis distance.
    # :param data: 2D Numpy array where rows are observations.
    # :param threshold: Distance threshold to classify anomalies.
    # :return: Indices of anomalies.
    
    mean_vector = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    distances = compute_mahalanobis_distance(data, mean_vector, inv_cov_matrix)
    anomalies = np.where(distances > threshold)[0]
    return anomalies, distances

# Example Test
if __name__ == "__main__":
    # Simulated multivariate time series data (rows = observations, cols = features)
    np.random.seed(42)
    normal_data = np.random.normal(loc=0, scale=1, size=(100, 3))  # Normal observations
    anomaly_data = np.random.normal(loc=10, scale=1, size=(5, 3))  # Anomalies

    # Combine data
    data = np.vstack([normal_data, anomaly_data])

    # Normalize data
    normalized_data = normalize_data(data)

    # Detect anomalies
    anomalies, distances = detect_anomalies(normalized_data, threshold=3.0)

    print("Anomalies detected at indices:", anomalies)
    print("Mahalanobis Distances:", distances[anomalies])
```

---

### **Example Inputs and Expected Outputs**

#### **Input 1**  
```python
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=(100, 3))
anomaly_data = np.random.normal(loc=10, scale=1, size=(5, 3))
data = np.vstack([normal_data, anomaly_data])
```

**Expected Output**  
- Anomalies detected at indices: `[100, 101, 102, 103, 104]`  
- Mahalanobis Distances: Large values (e.g., >5)

---

#### **Input 2**  
```python
np.random.seed(24)
normal_data = np.random.normal(loc=5, scale=2, size=(200, 4))
anomaly_data = np.random.normal(loc=-10, scale=2, size=(10, 4))
data = np.vstack([normal_data, anomaly_data])
```

**Expected Output**  
- Anomalies detected at indices: `[200, 201, ..., 209]`  
- Mahalanobis Distances: Very high values.

---

#### **Input 3**  
```python
np.random.seed(7)
normal_data = np.random.normal(loc=3, scale=0.5, size=(50, 2))
anomaly_data = np.random.normal(loc=20, scale=0.5, size=(3, 2))
data = np.vstack([normal_data, anomaly_data])
```

**Expected Output**  
- Anomalies detected at indices: `[50, 51, 52]`  
- Mahalanobis Distances: High values due to the large deviation.

---

### **Use Cases**  
1. **Fraud Detection**: Monitor transactional data for unusual spending patterns.  
2. **Sensor Data Monitoring**: Detect faults in IoT sensor readings.  
3. **Stock Market Analysis**: Identify unusual movements in stock prices.  
4. **Network Intrusion Detection**: Spot anomalies in network traffic.  
5. **Medical Diagnostics**: Identify outliers in health metrics (e.g., unusual heart rates).

---

This project leverages **advanced matrix operations** and statistical analysis to create a highly flexible anomaly detection system. It’s a significant step forward in applying Numpy for real-world, complex problems.
"""
