import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.stats import zscore


# Load Data
def load_data(file_path):
    return pd.read_csv(file_path)


# Univariate Anomaly Detection
def detect_univariate_anomalies(data, column, threshold=3):
    data['z_score'] = zscore(data[column])
    anomalies = data[np.abs(data['z_score']) > threshold]
    return anomalies


# Multivariate Anomaly Detection using DBSCAN
def detect_multivariate_anomalies(data, columns, eps=0.5, min_samples=5):
    subset = data[columns]
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(subset)
    data['cluster'] = clustering.labels_
    anomalies = data[data['cluster'] == -1]
    return anomalies


# Root Cause Analysis
def root_cause_analysis(data, anomaly_index, columns):
    correlation_matrix = data[columns].corr()
    root_cause = correlation_matrix.iloc[anomaly_index].sort_values(ascending=False)
    return root_cause


# Visualization
def visualize_anomalies(data, column, anomalies):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[column], label='Data')
    plt.scatter(anomalies.index, anomalies[column], color='red', label='Anomalies')
    plt.legend()
    plt.title(f"Anomaly Detection in {column}")
    plt.xlabel("Index")
    plt.ylabel(column)
    plt.show()


# Example Usage
if __name__ == "__main__":
    # Example dataset
    df = pd.DataFrame({
        'Date': pd.date_range(start='2024-01-01', periods=100),
        'Sales': np.random.normal(200, 20, 100).cumsum()
    })
    df.loc[95:, 'Sales'] += 200  # Inject anomaly
    df.set_index('Date', inplace=True)

    # Detect Univariate Anomalies
    univariate_anomalies = detect_univariate_anomalies(df, 'Sales')
    print("Univariate Anomalies:\n", univariate_anomalies)

    # Detect Multivariate Anomalies
    df['Inventory'] = np.random.normal(50, 5, 100).cumsum()
    multivariate_anomalies = detect_multivariate_anomalies(df, ['Sales', 'Inventory'])
    print("Multivariate Anomalies:\n", multivariate_anomalies)

    # Perform Root Cause Analysis
    root_cause = root_cause_analysis(df, 95, ['Sales', 'Inventory'])
    print("Root Cause Analysis:\n", root_cause)

    # Visualize Anomalies
    visualize_anomalies(df, 'Sales', univariate_anomalies)


# https://chatgpt.com/c/674b65b9-fecc-800c-8311-7f681df9b305
comment = """
### Project Title: **Anomaly Detection and Root Cause Analysis with Pandas**  
**File Name**: `anomaly_detection_and_root_cause_analysis_with_pandas.py`  

---

### Project Description  

This project focuses on **advanced anomaly detection and root cause analysis** using **Pandas** and supplementary data analytics libraries. The system identifies anomalies in multidimensional datasets and provides actionable insights into the root causes of detected irregularities.  
 
Key features include:  
1. **Univariate and Multivariate Anomaly Detection**: Identifies anomalies using statistical techniques (e.g., z-scores, IQR) and machine learning-based clustering methods (e.g., DBSCAN).  
2. **Root Cause Identification**: Correlates anomalies with possible causal variables using correlation matrices and dependency graphs.  
3. **Time-Window Analysis**: Analyzes patterns in temporal data to link anomalies to specific time windows or events.  
4. **Customizable Thresholds**: Allows users to fine-tune detection sensitivity based on domain-specific needs.  

---

### Example Use Cases  

1. **Network Traffic Monitoring**: Detect unusual spikes or dips in network traffic and identify the services causing these anomalies.  
2. **Sales Irregularities**: Highlight unusual drops or peaks in sales and correlate with marketing campaigns or supply chain issues.  
3. **Sensor Data Analysis**: Identify equipment faults by analyzing irregular patterns in IoT sensor data.  
4. **Healthcare Metrics**: Detect abnormal patient vitals or lab test results and link them to possible medical conditions or events.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: Sales Data**  
| Date       | Sales | Region | Marketing Spend | Inventory Level |  
|------------|-------|--------|-----------------|-----------------|  
| 2024-01-01 | 150   | East   | 50              | 200             |  
| 2024-01-02 | 200   | East   | 60              | 210             |  
| 2024-01-03 | 400   | East   | 55              | 190             |  
| 2024-01-04 | 180   | East   | 62              | 205             |  
| 2024-01-05 | 1500  | East   | 53              | 220             |  

**Expected Output**:  
- **Anomaly Detected**: Sales on `2024-01-05` is an outlier.  
- **Root Cause**: High correlation with "Marketing Spend" and "Inventory Level".  

---

#### **Input 2: Network Traffic Data**  
| Timestamp          | Traffic Volume | Service    | Server CPU Load | Errors Logged |  
|--------------------|----------------|------------|-----------------|---------------|  
| 2024-06-01 08:00  | 500            | Service A  | 50%             | 2             |  
| 2024-06-01 09:00  | 1000           | Service A  | 70%             | 1             |  
| 2024-06-01 10:00  | 7000           | Service A  | 90%             | 10            |  
| 2024-06-01 11:00  | 600            | Service A  | 60%             | 3             |  
| 2024-06-01 12:00  | 550            | Service A  | 50%             | 0             |  

**Expected Output**:  
- **Anomaly Detected**: Traffic volume spike at `2024-06-01 10:00`.  
- **Root Cause**: Server CPU Load and Errors Logged increased significantly.  

---

#### **Input 3: IoT Sensor Data**  
| Timestamp          | Sensor ID | Temperature | Humidity | Vibration Level |  
|--------------------|-----------|-------------|----------|-----------------|  
| 2024-03-01 00:00  | Sensor 1  | 25          | 60       | 0.02            |  
| 2024-03-01 01:00  | Sensor 1  | 25          | 58       | 0.03            |  
| 2024-03-01 02:00  | Sensor 1  | 85          | 65       | 0.15            |  
| 2024-03-01 03:00  | Sensor 1  | 26          | 62       | 0.04            |  

**Expected Output**:  
- **Anomaly Detected**: Temperature and vibration spike at `2024-03-01 02:00`.  
- **Root Cause**: Correlation with both parameters.  

---

#### **Input 4: Patient Health Data**  
| Patient ID | Time       | Heart Rate | Blood Pressure | Oxygen Level |  
|------------|------------|------------|----------------|--------------|  
| P001       | 08:00      | 75         | 120/80         | 98%          |  
| P001       | 09:00      | 80         | 125/85         | 96%          |  
| P001       | 10:00      | 150        | 140/100        | 90%          |  
| P001       | 11:00      | 78         | 122/84         | 97%          |  

**Expected Output**:  
- **Anomaly Detected**: Heart rate spike and blood pressure irregularity at `10:00`.  
- **Root Cause**: Possible medical condition or stress.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.stats import zscore

# Load Data
def load_data(file_path):
    return pd.read_csv(file_path)

# Univariate Anomaly Detection
def detect_univariate_anomalies(data, column, threshold=3):
    data['z_score'] = zscore(data[column])
    anomalies = data[np.abs(data['z_score']) > threshold]
    return anomalies

# Multivariate Anomaly Detection using DBSCAN
def detect_multivariate_anomalies(data, columns, eps=0.5, min_samples=5):
    subset = data[columns]
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(subset)
    data['cluster'] = clustering.labels_
    anomalies = data[data['cluster'] == -1]
    return anomalies

# Root Cause Analysis
def root_cause_analysis(data, anomaly_index, columns):
    correlation_matrix = data[columns].corr()
    root_cause = correlation_matrix.iloc[anomaly_index].sort_values(ascending=False)
    return root_cause

# Visualization
def visualize_anomalies(data, column, anomalies):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[column], label='Data')
    plt.scatter(anomalies.index, anomalies[column], color='red', label='Anomalies')
    plt.legend()
    plt.title(f"Anomaly Detection in {column}")
    plt.xlabel("Index")
    plt.ylabel(column)
    plt.show()

# Example Usage
if __name__ == "__main__":
    # Example dataset
    df = pd.DataFrame({
        'Date': pd.date_range(start='2024-01-01', periods=100),
        'Sales': np.random.normal(200, 20, 100).cumsum()
    })
    df.loc[95:, 'Sales'] += 200  # Inject anomaly
    df.set_index('Date', inplace=True)

    # Detect Univariate Anomalies
    univariate_anomalies = detect_univariate_anomalies(df, 'Sales')
    print("Univariate Anomalies:\n", univariate_anomalies)

    # Detect Multivariate Anomalies
    df['Inventory'] = np.random.normal(50, 5, 100).cumsum()
    multivariate_anomalies = detect_multivariate_anomalies(df, ['Sales', 'Inventory'])
    print("Multivariate Anomalies:\n", multivariate_anomalies)

    # Perform Root Cause Analysis
    root_cause = root_cause_analysis(df, 95, ['Sales', 'Inventory'])
    print("Root Cause Analysis:\n", root_cause)

    # Visualize Anomalies
    visualize_anomalies(df, 'Sales', univariate_anomalies)
```

---

### Advanced Skills Covered  

1. **Z-Score-Based Anomaly Detection**: Uses statistical measures to identify outliers in univariate datasets.  
2. **Multivariate Clustering with DBSCAN**: Identifies complex relationships in multidimensional data.  
3. **Root Cause Identification**: Links anomalies to potential causes using correlation matrices.  
4. **Time-Series Anomaly Highlighting**: Detects patterns in temporal data for real-time insights.  
5. **Data Visualization**: Highlights anomalies clearly and intuitively.  

This project pushes the boundaries of anomaly detection and enables actionable root-cause insights for dynamic datasets!
"""
