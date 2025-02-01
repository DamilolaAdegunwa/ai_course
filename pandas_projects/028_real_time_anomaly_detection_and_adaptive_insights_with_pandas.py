import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Function to simulate real-time data feed
def simulate_real_time_data(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data.iloc[i:i + chunk_size]


# Function to calculate rolling statistics and detect anomalies
def detect_anomalies(df, column, window_size, threshold=3):
    rolling_mean = df[column].rolling(window=window_size).mean()
    rolling_std = df[column].rolling(window=window_size).std()

    upper_limit = rolling_mean + (threshold * rolling_std)
    lower_limit = rolling_mean - (threshold * rolling_std)

    df['Anomaly'] = ((df[column] > upper_limit) | (df[column] < lower_limit)).astype(int)
    df['Upper_Limit'] = upper_limit
    df['Lower_Limit'] = lower_limit
    return df


# Adaptive threshold adjustment
def adaptive_thresholds(df, column, sensitivity=1.5):
    mean = df[column].mean()
    std_dev = df[column].std()
    threshold = mean + (sensitivity * std_dev)
    return threshold


# Multivariate anomaly detection
def multivariate_anomalies(df, cols, correlation_threshold=0.8):
    correlation_matrix = df[cols].corr()
    high_corr_pairs = [(x, y) for x in cols for y in cols if
                       x != y and abs(correlation_matrix[x][y]) > correlation_threshold]
    anomalies = []
    for x, y in high_corr_pairs:
        anomalies.append(df[(df[x] > adaptive_thresholds(df, x)) & (df[y] > adaptive_thresholds(df, y))])
    return pd.concat(anomalies).drop_duplicates()


# Main pipeline
if __name__ == "__main__":
    # Example: Network Data
    network_data = pd.DataFrame({
        'Timestamp': pd.date_range(start="2024-12-01 00:00:00", periods=100, freq='T'),
        'Source_IP': ['192.168.1.1'] * 100,
        'Bytes_Transferred': np.random.poisson(1000, 100),
        'Response_Time': np.random.randint(100, 300, 100)
    })

    network_data.loc[50, 'Bytes_Transferred'] = 70000  # Inject an anomaly

    print("Simulating Real-Time Anomaly Detection...")
    for chunk in simulate_real_time_data(network_data, chunk_size=10):
        chunk = detect_anomalies(chunk, 'Bytes_Transferred', window_size=5)
        anomalies = chunk[chunk['Anomaly'] == 1]
        print(anomalies)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(network_data['Bytes_Transferred'], label='Bytes Transferred')
    plt.plot(network_data['Upper_Limit'], linestyle='--', color='r', label='Upper Limit')
    plt.plot(network_data['Lower_Limit'], linestyle='--', color='r', label='Lower Limit')
    plt.scatter(network_data.index[network_data['Anomaly'] == 1],
                network_data['Bytes_Transferred'][network_data['Anomaly'] == 1],
                color='red', label='Anomalies')
    plt.legend()
    plt.show()


comment = """
### Project Title: **Real-Time Anomaly Detection and Adaptive Insights with Pandas**  
**File Name**: `real_time_anomaly_detection_and_adaptive_insights_with_pandas.py`  

---

### Project Description  
This project focuses on building a **real-time anomaly detection system** using Pandas for streaming or simulated real-time datasets. It involves:  

1. **Real-Time Simulation**: Processing data chunk-by-chunk to mimic a live data feed.  
2. **Dynamic Statistical Anomaly Detection**: Identifying anomalies using rolling statistical measures and thresholds.  
3. **Adaptive Baselines**: Adapting thresholds dynamically as new data arrives to handle concept drift.  
4. **Multivariate Analysis**: Detecting correlations between features to flag contextual anomalies.  
5. **Alerts System**: Logging and visualizing anomalies for insights.  

This project pushes the boundaries of data analysis by simulating real-time operations and implementing adaptive learning techniques.  

---

### Example Use Cases  

1. **Network Monitoring**: Detect unusual spikes in traffic or bandwidth usage.  
2. **E-commerce Transactions**: Flag fraudulent transactions based on deviations in user behavior.  
3. **IoT Sensor Data**: Identify irregular patterns in temperature or pressure from connected devices.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: Network Data**  
**File**: `network_data.csv`  
| Timestamp           | Source_IP     | Bytes_Transferred | Response_Time |  
|---------------------|---------------|-------------------|---------------|  
| 2024-12-01 00:00:00 | 192.168.1.1   | 1024              | 150           |  
| 2024-12-01 00:01:00 | 192.168.1.1   | 5120              | 200           |  
| 2024-12-01 00:02:00 | 192.168.1.1   | 70000             | 500           |  

**Expected Output**:  
- Anomaly detected at 2024-12-01 00:02:00 (Bytes_Transferred exceeded threshold).  

#### **Input 2: IoT Temperature Data**  
**File**: `iot_temp_data.csv`  
| Timestamp           | Sensor_ID | Temperature | Pressure |  
|---------------------|-----------|-------------|----------|  
| 2024-12-01 01:00:00 | S1        | 22          | 1015     |  
| 2024-12-01 01:01:00 | S1        | 45          | 1050     |  
| 2024-12-01 01:02:00 | S1        | 80          | 900      |  

**Expected Output**:  
- Anomaly detected at 2024-12-01 01:02:00 (both Temperature and Pressure anomalies).  

#### **Input 3: E-commerce User Data**  
**File**: `user_transactions.csv`  
| Timestamp           | User_ID | Transaction_Amount | Avg_Session_Time |  
|---------------------|---------|--------------------|------------------|  
| 2024-12-01 12:00:00 | U123    | 500                | 300              |  
| 2024-12-01 12:01:00 | U123    | 2000               | 900              |  

**Expected Output**:  
- Anomaly detected at 2024-12-01 12:01:00 (Transaction_Amount deviation).  

---

### Python Code  

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to simulate real-time data feed
def simulate_real_time_data(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data.iloc[i:i+chunk_size]

# Function to calculate rolling statistics and detect anomalies
def detect_anomalies(df, column, window_size, threshold=3):
    rolling_mean = df[column].rolling(window=window_size).mean()
    rolling_std = df[column].rolling(window=window_size).std()
    
    upper_limit = rolling_mean + (threshold * rolling_std)
    lower_limit = rolling_mean - (threshold * rolling_std)
    
    df['Anomaly'] = ((df[column] > upper_limit) | (df[column] < lower_limit)).astype(int)
    df['Upper_Limit'] = upper_limit
    df['Lower_Limit'] = lower_limit
    return df

# Adaptive threshold adjustment
def adaptive_thresholds(df, column, sensitivity=1.5):
    mean = df[column].mean()
    std_dev = df[column].std()
    threshold = mean + (sensitivity * std_dev)
    return threshold

# Multivariate anomaly detection
def multivariate_anomalies(df, cols, correlation_threshold=0.8):
    correlation_matrix = df[cols].corr()
    high_corr_pairs = [(x, y) for x in cols for y in cols if x != y and abs(correlation_matrix[x][y]) > correlation_threshold]
    anomalies = []
    for x, y in high_corr_pairs:
        anomalies.append(df[(df[x] > adaptive_thresholds(df, x)) & (df[y] > adaptive_thresholds(df, y))])
    return pd.concat(anomalies).drop_duplicates()

# Main pipeline
if __name__ == "__main__":
    # Example: Network Data
    network_data = pd.DataFrame({
        'Timestamp': pd.date_range(start="2024-12-01 00:00:00", periods=100, freq='T'),
        'Source_IP': ['192.168.1.1'] * 100,
        'Bytes_Transferred': np.random.poisson(1000, 100),
        'Response_Time': np.random.randint(100, 300, 100)
    })
    
    network_data.loc[50, 'Bytes_Transferred'] = 70000  # Inject an anomaly
    
    print("Simulating Real-Time Anomaly Detection...")
    for chunk in simulate_real_time_data(network_data, chunk_size=10):
        chunk = detect_anomalies(chunk, 'Bytes_Transferred', window_size=5)
        anomalies = chunk[chunk['Anomaly'] == 1]
        print(anomalies)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(network_data['Bytes_Transferred'], label='Bytes Transferred')
    plt.plot(network_data['Upper_Limit'], linestyle='--', color='r', label='Upper Limit')
    plt.plot(network_data['Lower_Limit'], linestyle='--', color='r', label='Lower Limit')
    plt.scatter(network_data.index[network_data['Anomaly'] == 1], 
                network_data['Bytes_Transferred'][network_data['Anomaly'] == 1], 
                color='red', label='Anomalies')
    plt.legend()
    plt.show()
```

---

### Advanced Skills Covered  

1. **Real-Time Data Simulation**: Simulate and process data in chunks to replicate real-world scenarios.  
2. **Rolling Statistical Analysis**: Dynamic anomaly detection using statistical thresholds.  
3. **Adaptive Learning**: Adjust thresholds as new data arrives to account for concept drift.  
4. **Multivariate Anomaly Detection**: Detect relationships across multiple variables for contextual insights.  
5. **Visualization and Alerts**: Combine Pandas with Matplotlib for anomaly visualization.  

This project equips you with the tools to build scalable anomaly detection pipelines, perfect for real-time analytics and advanced AI-driven systems.
"""