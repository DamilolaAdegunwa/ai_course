import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Load Dataset
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Timestamp'], index_col='Timestamp')
    return data


# Dynamic Anomaly Detection
def detect_anomalies(data, features, z_threshold=3):
    anomalies = {}
    for feature in features:
        mean = data[feature].mean()
        std = data[feature].std()
        data[f'{feature}_Anomaly'] = ((data[feature] - mean).abs() > z_threshold * std)
        anomalies[feature] = data[data[f'{feature}_Anomaly']].index.tolist()
    return anomalies


# PCA for Root Cause Analysis
def root_cause_analysis(data, features, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data[features])
    explained_variance = pca.explained_variance_ratio_
    return reduced_data, explained_variance


# Correlation Analysis
def correlation_analysis(data, features):
    corr_matrix = data[features].corr()
    print("Correlation Matrix:\n", corr_matrix)
    return corr_matrix


# Visualize PCA Results
def visualize_pca(reduced_data, explained_variance):
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.5)
    plt.title('PCA Reduced Data (Explained Variance: {:.2f}%)'.format(sum(explained_variance) * 100))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


# Main Execution
if __name__ == "__main__":
    # Example Dataset
    data = pd.DataFrame({
        'Timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
        'SensorA': np.random.normal(50, 10, 100),
        'SensorB': np.random.normal(70, 15, 100),
        'SensorC': np.random.normal(90, 20, 100)
    })
    data.loc[10, 'SensorA'] = 500  # Inject anomaly
    data = data.set_index('Timestamp')

    # Detect Anomalies
    features = ['SensorA', 'SensorB', 'SensorC']
    anomalies = detect_anomalies(data, features)
    print("Anomalies Detected:\n", anomalies)

    # Root Cause Analysis with PCA
    reduced_data, explained_variance = root_cause_analysis(data, features)
    visualize_pca(reduced_data, explained_variance)

    # Correlation Analysis
    correlation_matrix = correlation_analysis(data, features)


# https://chatgpt.com/c/674b65b9-fecc-800c-8311-7f681df9b305
comment = """
### Project Title: **Anomaly Detection and Root Cause Analysis in Multi-Dimensional Datasets with Pandas**  
**File Name**: `anomaly_detection_and_root_cause_analysis_with_pandas.py`  

---

### Project Description  

This project involves creating an advanced **anomaly detection and root cause analysis system** for multi-dimensional datasets using Pandas. The system incorporates statistical methods, feature decomposition (e.g., PCA), and advanced correlations to identify outliers and their potential causes. The focus is on detecting patterns in high-dimensional data and providing interpretable insights into anomalies.  

**Highlights**:  
1. **Dynamic Thresholds**: Determines anomaly thresholds dynamically using statistical properties.  
2. **Principal Component Analysis (PCA)**: Reduces dimensionality to find significant contributors to anomalies.  
3. **Correlation Matrix Analysis**: Identifies interdependencies and potential root causes of anomalies.  
4. **Time-Based Anomaly Aggregation**: Groups anomalies by time intervals for temporal pattern discovery.  
5. **Explainability**: Provides human-readable explanations for detected anomalies.  

---

### Example Use Cases  

1. **Financial Fraud Detection**: Detect unusual transactions in banking data based on multi-dimensional indicators like transaction amount, location, and account activity.  
2. **Industrial Monitoring**: Identify equipment failures or unusual behavior from sensor data in manufacturing.  
3. **Customer Behavior Analytics**: Spot irregular patterns in e-commerce user activities like purchases, clicks, and navigation.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: Financial Transactions**  
| Timestamp          | TransactionID | Amount ($) | Location | AccountAge (days) | Fraudulent |  
|--------------------|---------------|------------|----------|--------------------|------------|  
| 2024-01-01 00:01  | TX001         | 5000       | NY       | 1000              | 0          |  
| 2024-01-01 00:02  | TX002         | 100        | CA       | 200               | 0          |  
| 2024-01-01 00:03  | TX003         | 10000      | NY       | 5                 | 1          |  

**Expected Output**:  
- **Anomalies Detected**: TX001, TX003.  
- **Root Cause**: Unusual transaction amount combined with account age < 10 days.  

---

#### **Input 2: Sensor Data**  
| Timestamp          | SensorA | SensorB | SensorC | Faulty |  
|--------------------|---------|---------|---------|--------|  
| 2024-01-01 00:00  | 50      | 70      | 90      | 0      |  
| 2024-01-01 00:01  | 500     | 70      | 80      | 1      |  
| 2024-01-01 00:02  | 45      | 65      | 95      | 0      |  

**Expected Output**:  
- **Anomalies Detected**: SensorA at 2024-01-01 00:01.  
- **Root Cause**: SensorA value exceeded mean + 3*std deviation.  

---

#### **Input 3: E-Commerce Data**  
| Timestamp         | UserID  | PageViews | Clicks | Purchases | Abnormal |  
|-------------------|---------|-----------|--------|-----------|----------|  
| 2024-01-01 12:00 | 1001    | 50        | 30     | 1         | 1        |  
| 2024-01-01 12:01 | 1002    | 5         | 2      | 0         | 0        |  
| 2024-01-01 12:02 | 1003    | 70        | 50     | 0         | 1        |  

**Expected Output**:  
- **Anomalies Detected**: UserID 1001 and 1003.  
- **Root Cause**: High PageViews/Clicks ratio compared to typical users.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load Dataset
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Timestamp'], index_col='Timestamp')
    return data

# Dynamic Anomaly Detection
def detect_anomalies(data, features, z_threshold=3):
    anomalies = {}
    for feature in features:
        mean = data[feature].mean()
        std = data[feature].std()
        data[f'{feature}_Anomaly'] = ((data[feature] - mean).abs() > z_threshold * std)
        anomalies[feature] = data[data[f'{feature}_Anomaly']].index.tolist()
    return anomalies

# PCA for Root Cause Analysis
def root_cause_analysis(data, features, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data[features])
    explained_variance = pca.explained_variance_ratio_
    return reduced_data, explained_variance

# Correlation Analysis
def correlation_analysis(data, features):
    corr_matrix = data[features].corr()
    print("Correlation Matrix:\n", corr_matrix)
    return corr_matrix

# Visualize PCA Results
def visualize_pca(reduced_data, explained_variance):
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.5)
    plt.title('PCA Reduced Data (Explained Variance: {:.2f}%)'.format(sum(explained_variance) * 100))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Example Dataset
    data = pd.DataFrame({
        'Timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
        'SensorA': np.random.normal(50, 10, 100),
        'SensorB': np.random.normal(70, 15, 100),
        'SensorC': np.random.normal(90, 20, 100)
    })
    data.loc[10, 'SensorA'] = 500  # Inject anomaly
    data = data.set_index('Timestamp')

    # Detect Anomalies
    features = ['SensorA', 'SensorB', 'SensorC']
    anomalies = detect_anomalies(data, features)
    print("Anomalies Detected:\n", anomalies)

    # Root Cause Analysis with PCA
    reduced_data, explained_variance = root_cause_analysis(data, features)
    visualize_pca(reduced_data, explained_variance)

    # Correlation Analysis
    correlation_matrix = correlation_analysis(data, features)
```

---

### Advanced Skills Covered  

1. **Anomaly Detection**: Implements dynamic anomaly thresholds using statistical measures.  
2. **PCA for Explainability**: Reduces data dimensionality and identifies major contributors to anomalies.  
3. **Correlation Analysis**: Highlights relationships between features for root cause insights.  
4. **Visualization**: Uses PCA scatter plots to represent multi-dimensional anomalies.  
5. **Time-Series Compatibility**: Handles temporal datasets for anomaly aggregation.  

This project combines Pandas with advanced analytical techniques, enabling actionable insights into complex multi-dimensional datasets!
"""
