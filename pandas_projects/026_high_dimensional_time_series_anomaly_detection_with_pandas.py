import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Function to preprocess time-series data
def preprocess_time_series(df, timestamp_col, resample_freq='1T'):
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col).resample(resample_freq).mean().fillna(method='ffill')
    return df


# Function to compute rolling statistics
def compute_rolling_metrics(df, window_size):
    rolling_features = pd.DataFrame()
    for col in df.select_dtypes(include=np.number).columns:
        rolling_features[f'{col}_mean'] = df[col].rolling(window=window_size).mean()
        rolling_features[f'{col}_std'] = df[col].rolling(window=window_size).std()
        rolling_features[f'{col}_zscore'] = (df[col] - rolling_features[f'{col}_mean']) / (
                    rolling_features[f'{col}_std'] + 1e-5)
    return rolling_features


# Function to calculate anomaly scores
def calculate_anomaly_scores(df, threshold=3):
    anomaly_scores = (df.abs() > threshold).sum(axis=1)
    return anomaly_scores


# Visualization function for anomalies
def visualize_anomalies(df, anomaly_scores, title='Anomaly Detection'):
    plt.figure(figsize=(12, 6))
    for col in df.columns:
        plt.plot(df[col], label=col, alpha=0.7)
    plt.scatter(anomaly_scores.index, df.max(axis=1), c='red', label='Anomalies', zorder=5)
    plt.title(title)
    plt.legend()
    plt.show()


# Main pipeline
if __name__ == "__main__":
    # Example: IoT Sensor Data
    iot_data = pd.DataFrame({
        'Timestamp': ['2024-12-01 00:00:00', '2024-12-01 00:01:00', '2024-12-01 00:02:00'],
        'Sensor_ID': ['S1', 'S1', 'S1'],
        'Temperature': [25.3, 25.7, 50.2],
        'Pressure': [101.3, 101.5, 120.0],
        'Vibration': [0.02, 0.05, 0.50]
    })

    # Preprocess data
    processed_iot_data = preprocess_time_series(iot_data, 'Timestamp')
    print("Processed IoT Data:")
    print(processed_iot_data)

    # Compute rolling metrics
    rolling_metrics = compute_rolling_metrics(processed_iot_data, window_size=2)
    print("Rolling Metrics:")
    print(rolling_metrics)

    # Calculate anomaly scores
    anomaly_scores = calculate_anomaly_scores(
        rolling_metrics[['Temperature_zscore', 'Pressure_zscore', 'Vibration_zscore']])
    print("Anomaly Scores:")
    print(anomaly_scores)

    # Visualize anomalies
    visualize_anomalies(processed_iot_data, anomaly_scores, title='IoT Sensor Anomalies')


comment = """
### Project Title: **High-Dimensional Time-Series Anomaly Detection with Pandas**  
**File Name**: `high_dimensional_time_series_anomaly_detection_with_pandas.py`  

---

### Project Description  
This project involves building a **high-dimensional time-series anomaly detection pipeline** using Pandas. It tackles:  
1. **Time-Series Aggregation and Preprocessing**: Handle datasets with millions of rows from IoT devices, financial markets, or operational logs.  
2. **Sliding Window Analysis**: Generate rolling statistics, lagged features, and derive metrics like moving averages and volatility.  
3. **Multivariate Anomaly Scoring**: Combine multiple signals into a unified anomaly score using statistical thresholds, correlation analysis, and synthetic indices.  
4. **Visualizing Anomalies**: Create detailed visualizations to pinpoint unusual patterns in time-series data.  

This project is ideal for fields such as fraud detection, industrial monitoring, and stock market irregularities.  

---

### Example Use Cases  

1. **IoT Sensor Monitoring**: Detect malfunctioning devices in a manufacturing plant.  
2. **Financial Fraud Detection**: Identify unusual transactions or stock price movements.  
3. **Website Performance Analytics**: Spot anomalies in server response times or traffic.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: IoT Sensor Data**  
**File**: `iot_data.csv`  
| Timestamp           | Sensor_ID | Temperature | Pressure | Vibration |  
|---------------------|-----------|-------------|----------|-----------|  
| 2024-12-01 00:00:00 | S1        | 25.3        | 101.3    | 0.02      |  
| 2024-12-01 00:01:00 | S1        | 25.7        | 101.5    | 0.05      |  

**Expected Output**:  
- Identify anomalies in temperature spikes or unusual vibration patterns.  
- Generate metrics like rolling mean and rolling standard deviation for each sensor.  

#### **Input 2: Financial Transactions**  
**File**: `financial_data.csv`  
| Timestamp           | Account_ID | Transaction_Amount | Balance |  
|---------------------|------------|---------------------|---------|  
| 2024-12-01 09:00:00 | 123        | 5000               | 15000   |  
| 2024-12-01 09:05:00 | 123        | 20000              | -5000   |  

**Expected Output**:  
- Flag the second transaction as anomalous due to the overdraft.  
- Compute rolling average transaction amounts for each account.  

#### **Input 3: Website Analytics**  
**File**: `website_data.csv`  
| Timestamp           | URL       | Response_Time | Errors |  
|---------------------|-----------|---------------|--------|  
| 2024-12-01 00:00:00 | /home     | 0.5           | 0      |  
| 2024-12-01 00:01:00 | /login    | 2.0           | 1      |  

**Expected Output**:  
- Highlight spikes in response times as anomalies.  
- Visualize error trends over time.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to preprocess time-series data
def preprocess_time_series(df, timestamp_col, resample_freq='1T'):
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col).resample(resample_freq).mean().fillna(method='ffill')
    return df

# Function to compute rolling statistics
def compute_rolling_metrics(df, window_size):
    rolling_features = pd.DataFrame()
    for col in df.select_dtypes(include=np.number).columns:
        rolling_features[f'{col}_mean'] = df[col].rolling(window=window_size).mean()
        rolling_features[f'{col}_std'] = df[col].rolling(window=window_size).std()
        rolling_features[f'{col}_zscore'] = (df[col] - rolling_features[f'{col}_mean']) / (rolling_features[f'{col}_std'] + 1e-5)
    return rolling_features

# Function to calculate anomaly scores
def calculate_anomaly_scores(df, threshold=3):
    anomaly_scores = (df.abs() > threshold).sum(axis=1)
    return anomaly_scores

# Visualization function for anomalies
def visualize_anomalies(df, anomaly_scores, title='Anomaly Detection'):
    plt.figure(figsize=(12, 6))
    for col in df.columns:
        plt.plot(df[col], label=col, alpha=0.7)
    plt.scatter(anomaly_scores.index, df.max(axis=1), c='red', label='Anomalies', zorder=5)
    plt.title(title)
    plt.legend()
    plt.show()

# Main pipeline
if __name__ == "__main__":
    # Example: IoT Sensor Data
    iot_data = pd.DataFrame({
        'Timestamp': ['2024-12-01 00:00:00', '2024-12-01 00:01:00', '2024-12-01 00:02:00'],
        'Sensor_ID': ['S1', 'S1', 'S1'],
        'Temperature': [25.3, 25.7, 50.2],
        'Pressure': [101.3, 101.5, 120.0],
        'Vibration': [0.02, 0.05, 0.50]
    })
    
    # Preprocess data
    processed_iot_data = preprocess_time_series(iot_data, 'Timestamp')
    print("Processed IoT Data:")
    print(processed_iot_data)
    
    # Compute rolling metrics
    rolling_metrics = compute_rolling_metrics(processed_iot_data, window_size=2)
    print("Rolling Metrics:")
    print(rolling_metrics)
    
    # Calculate anomaly scores
    anomaly_scores = calculate_anomaly_scores(rolling_metrics[['Temperature_zscore', 'Pressure_zscore', 'Vibration_zscore']])
    print("Anomaly Scores:")
    print(anomaly_scores)
    
    # Visualize anomalies
    visualize_anomalies(processed_iot_data, anomaly_scores, title='IoT Sensor Anomalies')
```

---

### How This Project Advances Your Skills  

1. **High-Dimensional Time-Series Handling**: Gain experience in managing datasets with complex temporal structures.  
2. **Statistical Anomaly Detection**: Use advanced techniques like z-scores and rolling metrics.  
3. **Cross-Domain Applications**: Explore real-world scenarios spanning IoT, finance, and web analytics.  
4. **Dynamic Visualization**: Learn to create meaningful visual representations of temporal anomalies.  
5. **Scalability**: Adapt the pipeline to integrate additional signals and scale up for large datasets.  

Elevate this project further by introducing **unsupervised learning techniques** (e.g., clustering or isolation forests) for more nuanced anomaly detection!
"""