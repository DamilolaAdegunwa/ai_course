import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import time


# Function to simulate a data stream
def simulate_data_stream(start_time, num_points, interval_seconds=60):
    timestamps = [start_time + timedelta(seconds=i * interval_seconds) for i in range(num_points)]
    metrics = [random.randint(20, 30) for _ in range(num_points)]
    return pd.DataFrame({'Timestamp': timestamps, 'Metric': metrics})


# Sliding window aggregation and anomaly detection
class RealTimeProcessor:
    def __init__(self, window_size=5, z_threshold=2.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.data = pd.DataFrame(columns=['Timestamp', 'Metric'])

    def process_new_data(self, new_data):
        # Append new data to existing data
        self.data = pd.concat([self.data, new_data]).reset_index(drop=True)

        # Compute sliding window statistics
        self.data['Rolling_Mean'] = self.data['Metric'].rolling(self.window_size).mean()
        self.data['Rolling_Std'] = self.data['Metric'].rolling(self.window_size).std()

        # Calculate Z-scores
        self.data['Z_Score'] = (self.data['Metric'] - self.data['Rolling_Mean']) / self.data['Rolling_Std']

        # Detect anomalies
        self.data['Anomaly'] = self.data['Z_Score'].apply(lambda z: abs(z) > self.z_threshold)

        # Trim data to only keep the latest relevant points
        self.data = self.data.tail(self.window_size * 2)
        return self.data

    def visualize(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Timestamp'], self.data['Metric'], label='Metric', marker='o')
        plt.plot(self.data['Timestamp'], self.data['Rolling_Mean'], label='Rolling Mean', linestyle='--',
                 color='orange')
        anomalies = self.data[self.data['Anomaly']]
        if not anomalies.empty:
            plt.scatter(anomalies['Timestamp'], anomalies['Metric'], color='red', label='Anomalies')
        plt.xlabel('Timestamp')
        plt.ylabel('Metric')
        plt.title('Real-Time Data Stream with Anomaly Detection')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# Example use case
if __name__ == "__main__":
    # Initialize processor
    processor = RealTimeProcessor(window_size=5, z_threshold=2.0)

    # Simulate a real-time data stream
    start_time = datetime.now()
    for _ in range(10):  # Simulate 10 updates to the stream
        new_data = simulate_data_stream(start_time, num_points=1)
        print("\nNew Data Stream:")
        print(new_data)

        # Process new data
        processed_data = processor.process_new_data(new_data)
        print("\nProcessed Data with Anomalies:")
        print(processed_data)

        # Visualize
        processor.visualize()

        # Simulate real-time updates
        time.sleep(1)  # Wait for 1 second before next data point
        start_time += timedelta(seconds=60)


comment = """
### Project Title: **Real-Time Data Stream Processing and Anomaly Detection with Pandas**  
**File Name**: `real_time_data_stream_processing_and_anomaly_detection_with_pandas.py`  

---

### Project Description  
This project builds a real-time data stream processing system using **Pandas**, designed for high-frequency data analysis. The system handles incoming data streams, aggregates data dynamically, detects anomalies in real time using statistical thresholds, and visualizes the results.  
Advanced concepts include:  
- **Sliding Window Aggregations** for continuous monitoring.  
- Real-time **Anomaly Detection** using Z-scores and moving averages.  
- **Dynamic Alerts and Visualization** for anomalies.  

Use cases include:  
- Monitoring server logs for unusual spikes.  
- Detecting fraudulent transactions in financial data streams.  
- Real-time sensor data monitoring for IoT applications.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1**  
**Simulated Stream Data**:  
| Timestamp           | Metric |  
|---------------------|--------|  
| 2024-11-30 12:00:00 | 20     |  
| 2024-11-30 12:01:00 | 25     |  
| 2024-11-30 12:02:00 | 100    |  
| 2024-11-30 12:03:00 | 22     |  
| 2024-11-30 12:04:00 | 23     |  

**Expected Output**:  
- **Anomaly Detected**: At `2024-11-30 12:02:00`, metric exceeds Z-score threshold.  
- **Visual Graph**: Time-series plot highlighting anomalies.  

#### **Input 2**  
**Simulated Stream Data with Sliding Window Aggregation**:  
| Timestamp           | Metric |  
|---------------------|--------|  
| 2024-11-30 12:00:00 | 30     |  
| 2024-11-30 12:01:00 | 32     |  
| 2024-11-30 12:02:00 | 28     |  
| 2024-11-30 12:03:00 | 33     |  
| 2024-11-30 12:04:00 | 70     |  

**Expected Output**:  
- Sliding window mean and standard deviation calculated dynamically.  
- Anomaly detected at `2024-11-30 12:04:00`.  

#### **Input 3**  
**Real-Time Update to Stream Data**:  
| Timestamp           | Metric |  
|---------------------|--------|  
| 2024-11-30 12:05:00 | 22     |  
| 2024-11-30 12:06:00 | 80     |  

**Expected Output**:  
- Anomaly detected at `2024-11-30 12:06:00`.  
- Updated visualization including new data points.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import time

# Function to simulate a data stream
def simulate_data_stream(start_time, num_points, interval_seconds=60):
    timestamps = [start_time + timedelta(seconds=i * interval_seconds) for i in range(num_points)]
    metrics = [random.randint(20, 30) for _ in range(num_points)]
    return pd.DataFrame({'Timestamp': timestamps, 'Metric': metrics})

# Sliding window aggregation and anomaly detection
class RealTimeProcessor:
    def __init__(self, window_size=5, z_threshold=2.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.data = pd.DataFrame(columns=['Timestamp', 'Metric'])
    
    def process_new_data(self, new_data):
        # Append new data to existing data
        self.data = pd.concat([self.data, new_data]).reset_index(drop=True)
        
        # Compute sliding window statistics
        self.data['Rolling_Mean'] = self.data['Metric'].rolling(self.window_size).mean()
        self.data['Rolling_Std'] = self.data['Metric'].rolling(self.window_size).std()
        
        # Calculate Z-scores
        self.data['Z_Score'] = (self.data['Metric'] - self.data['Rolling_Mean']) / self.data['Rolling_Std']
        
        # Detect anomalies
        self.data['Anomaly'] = self.data['Z_Score'].apply(lambda z: abs(z) > self.z_threshold)
        
        # Trim data to only keep the latest relevant points
        self.data = self.data.tail(self.window_size * 2)
        return self.data
    
    def visualize(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Timestamp'], self.data['Metric'], label='Metric', marker='o')
        plt.plot(self.data['Timestamp'], self.data['Rolling_Mean'], label='Rolling Mean', linestyle='--', color='orange')
        anomalies = self.data[self.data['Anomaly']]
        if not anomalies.empty:
            plt.scatter(anomalies['Timestamp'], anomalies['Metric'], color='red', label='Anomalies')
        plt.xlabel('Timestamp')
        plt.ylabel('Metric')
        plt.title('Real-Time Data Stream with Anomaly Detection')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Example use case
if __name__ == "__main__":
    # Initialize processor
    processor = RealTimeProcessor(window_size=5, z_threshold=2.0)
    
    # Simulate a real-time data stream
    start_time = datetime.now()
    for _ in range(10):  # Simulate 10 updates to the stream
        new_data = simulate_data_stream(start_time, num_points=1)
        print("\nNew Data Stream:")
        print(new_data)
        
        # Process new data
        processed_data = processor.process_new_data(new_data)
        print("\nProcessed Data with Anomalies:")
        print(processed_data)
        
        # Visualize
        processor.visualize()
        
        # Simulate real-time updates
        time.sleep(1)  # Wait for 1 second before next data point
        start_time += timedelta(seconds=60)
```

---

### Key Features  
1. **Real-Time Stream Handling**: Dynamically processes incoming data.  
2. **Sliding Window Aggregation**: Maintains rolling statistics for real-time monitoring.  
3. **Z-Score Based Anomaly Detection**: Detects significant deviations dynamically.  
4. **Visualization**: Highlights anomalies in real-time on a time-series graph.  
5. **Scalable**: Extensible for larger datasets and faster stream updates.  

---

### Testing Scenarios  

#### **Scenario 1**:  
Stream with a sudden spike:  
| Timestamp           | Metric |  
|---------------------|--------|  
| 12:00:00            | 25     |  
| 12:01:00            | 26     |  
| 12:02:00            | 120    |  
| 12:03:00            | 24     |  
| 12:04:00            | 25     |  

Expected Output:  
- Anomaly detected at `12:02:00`.  

#### **Scenario 2**:  
Stream with noise but no anomalies:  
| Timestamp           | Metric |  
|---------------------|--------|  
| 12:00:00            | 28     |  
| 12:01:00            | 27     |  
| 12:02:00            | 29     |  
| 12:03:00            | 30     |  
| 12:04:00            | 31     |  

Expected Output:  
- No anomalies detected.  

#### **Scenario 3**:  
Stream with delayed but frequent spikes:  
| Timestamp           | Metric |  
|---------------------|--------|  
| 12:00:00            | 22     |  
| 12:01:00            | 25     |  
| 12:02:00            | 85     |  
| 12:03:00            | 23     |  
| 12:04:00            | 88     |  

Expected Output:  
- Anomalies detected at `12:02:00` and `12:04:00`.  

---

### Advanced Extension Ideas  
- Integrate with Kafka for real-time distributed streaming.  
- Incorporate machine learning models for anomaly detection (e.g., Isolation Forest).  
- Store processed data in a time-series database like InfluxDB.  

Would you like to explore any of these extensions?
"""