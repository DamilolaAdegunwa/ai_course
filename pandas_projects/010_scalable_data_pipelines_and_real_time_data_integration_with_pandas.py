import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time


# Simulated real-time API data ingestion
def fetch_real_time_data(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        print("Failed to fetch data")
        return pd.DataFrame()


# Load batch data (CSV, JSON) and unify schema
def load_and_unify_data(csv_path, json_path, key_column):
    # Load CSV
    csv_data = pd.read_csv(csv_path)

    # Load JSON
    with open(json_path, 'r') as file:
        json_data = pd.DataFrame(json.load(file))

    # Merge Data
    unified_data = pd.merge(csv_data, json_data, on=key_column)
    return unified_data


# Transform and normalize data
def normalize_data(df, columns_to_normalize):
    for col in columns_to_normalize:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df


# Real-time anomaly detection
def detect_streaming_anomalies(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    df['Anomaly'] = (abs(df[column] - mean) > threshold * std).astype(int)
    return df


# Aggregation for hourly data
def aggregate_minute_to_hourly(df, time_column, value_column):
    df[time_column] = pd.to_datetime(df[time_column])
    df.set_index(time_column, inplace=True)
    hourly_data = df.resample('H').mean()
    return hourly_data


# Pipeline orchestration
def data_pipeline(api_url, csv_path, json_path, key_column, columns_to_normalize):
    print("Fetching real-time data...")
    real_time_data = fetch_real_time_data(api_url)

    print("Loading and unifying batch data...")
    batch_data = load_and_unify_data(csv_path, json_path, key_column)

    print("Normalizing unified data...")
    normalized_data = normalize_data(batch_data, columns_to_normalize)

    print("Detecting anomalies in real-time data...")
    real_time_data = detect_streaming_anomalies(real_time_data, 'value')

    print("Aggregating minute data to hourly...")
    hourly_data = aggregate_minute_to_hourly(real_time_data, 'timestamp', 'value')

    return normalized_data, real_time_data, hourly_data


# Visualization
def visualize_data(df, column, title):
    import matplotlib.pyplot as plt
    df[column].plot(figsize=(12, 6), title=title)
    plt.show()


# Test the pipeline
if __name__ == "__main__":
    # Simulate inputs
    api_url = "https://api.example.com/real-time-data"  # Replace with an actual API
    csv_path = "user_profiles.csv"  # Replace with actual file
    json_path = "transactions.json"  # Replace with actual file
    key_column = "user_id"
    columns_to_normalize = ['age', 'purchase_amount']

    # Run pipeline
    normalized_data, real_time_data, hourly_data = data_pipeline(api_url, csv_path, json_path, key_column,
                                                                 columns_to_normalize)

    # Display results
    print("\nNormalized Data:")
    print(normalized_data.head())

    print("\nReal-Time Data with Anomalies:")
    print(real_time_data.head())

    print("\nHourly Aggregated Data:")
    print(hourly_data.head())

    # Visualize real-time anomalies
    visualize_data(real_time_data, 'value', "Real-Time Anomaly Detection")

    # Visualize hourly trends
    visualize_data(hourly_data, 'value', "Hourly Aggregated Trends")


comment = """
### Project Title: **Scalable Data Pipelines and Real-Time Data Integration with Pandas**  
**File Name**: `scalable_data_pipelines_and_real_time_data_integration_with_pandas.py`  

---

### Project Description  
This project involves building a **scalable data pipeline** using **Pandas** to integrate **real-time data sources** like APIs, streaming data, and batch data files. The project focuses on the advanced use of **Pandas** for:  
- Real-time data ingestion.  
- Data integration across multiple sources.  
- Transformation and normalization of complex datasets.  
- Detecting and handling streaming anomalies.  
- Automating pipeline tasks with efficient scheduling.  

You’ll use APIs, JSON, and CSV files, combining them into a unified data format for advanced analytics and machine learning readiness.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1**:  
**Data Source**: API providing stock prices every minute.  
**Task**: Aggregate minute-level data into hourly data.  
**Expected Output**:  
- A DataFrame of hourly averages for each stock.  

#### **Input 2**:  
**Data Source**: JSON files of e-commerce transactions and CSV files of user profiles.  
**Task**: Merge transactions with user profiles.  
**Expected Output**:  
- A unified DataFrame linking transactions to users with normalized columns.  

#### **Input 3**:  
**Data Source**: Batch files of weather data from multiple sensors.  
**Task**: Detect data anomalies in real-time and summarize valid data.  
**Expected Output**:  
- A report of detected anomalies and a cleaned DataFrame of valid data.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time

# Simulated real-time API data ingestion
def fetch_real_time_data(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        print("Failed to fetch data")
        return pd.DataFrame()

# Load batch data (CSV, JSON) and unify schema
def load_and_unify_data(csv_path, json_path, key_column):
    # Load CSV
    csv_data = pd.read_csv(csv_path)
    
    # Load JSON
    with open(json_path, 'r') as file:
        json_data = pd.DataFrame(json.load(file))
    
    # Merge Data
    unified_data = pd.merge(csv_data, json_data, on=key_column)
    return unified_data

# Transform and normalize data
def normalize_data(df, columns_to_normalize):
    for col in columns_to_normalize:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

# Real-time anomaly detection
def detect_streaming_anomalies(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    df['Anomaly'] = (abs(df[column] - mean) > threshold * std).astype(int)
    return df

# Aggregation for hourly data
def aggregate_minute_to_hourly(df, time_column, value_column):
    df[time_column] = pd.to_datetime(df[time_column])
    df.set_index(time_column, inplace=True)
    hourly_data = df.resample('H').mean()
    return hourly_data

# Pipeline orchestration
def data_pipeline(api_url, csv_path, json_path, key_column, columns_to_normalize):
    print("Fetching real-time data...")
    real_time_data = fetch_real_time_data(api_url)
    
    print("Loading and unifying batch data...")
    batch_data = load_and_unify_data(csv_path, json_path, key_column)
    
    print("Normalizing unified data...")
    normalized_data = normalize_data(batch_data, columns_to_normalize)
    
    print("Detecting anomalies in real-time data...")
    real_time_data = detect_streaming_anomalies(real_time_data, 'value')
    
    print("Aggregating minute data to hourly...")
    hourly_data = aggregate_minute_to_hourly(real_time_data, 'timestamp', 'value')
    
    return normalized_data, real_time_data, hourly_data

# Visualization
def visualize_data(df, column, title):
    import matplotlib.pyplot as plt
    df[column].plot(figsize=(12, 6), title=title)
    plt.show()

# Test the pipeline
if __name__ == "__main__":
    # Simulate inputs
    api_url = "https://api.example.com/real-time-data"  # Replace with an actual API
    csv_path = "user_profiles.csv"  # Replace with actual file
    json_path = "transactions.json"  # Replace with actual file
    key_column = "user_id"
    columns_to_normalize = ['age', 'purchase_amount']
    
    # Run pipeline
    normalized_data, real_time_data, hourly_data = data_pipeline(api_url, csv_path, json_path, key_column, columns_to_normalize)
    
    # Display results
    print("\nNormalized Data:")
    print(normalized_data.head())
    
    print("\nReal-Time Data with Anomalies:")
    print(real_time_data.head())
    
    print("\nHourly Aggregated Data:")
    print(hourly_data.head())
    
    # Visualize real-time anomalies
    visualize_data(real_time_data, 'value', "Real-Time Anomaly Detection")
    
    # Visualize hourly trends
    visualize_data(hourly_data, 'value', "Hourly Aggregated Trends")
```

---

### Testing Scenarios  

#### **Scenario 1**:  
**API URL**: Simulated stock price data.  
**Task**: Ingest real-time data and detect anomalies.  
**Expected Output**:  
- Anomaly-labeled DataFrame with trends visualization.  

#### **Scenario 2**:  
**CSV & JSON Files**: User profiles and transactions.  
**Task**: Merge datasets and normalize numeric fields.  
**Expected Output**:  
- Unified, normalized DataFrame.  

#### **Scenario 3**:  
**Streaming Weather Data**: Batch files of temperature readings.  
**Task**: Aggregate minute-level data to hourly averages.  
**Expected Output**:  
- Hourly trends DataFrame.  

---

### Key Learnings  
- **Real-Time Data Integration**: Fetch and process streaming data dynamically.  
- **Data Normalization**: Prepare data for advanced analytics by standardizing values.  
- **Anomaly Detection**: Use statistical thresholds to highlight unusual data points.  
- **Pipeline Orchestration**: Automate complex workflows combining multiple data sources.  

Would you like to expand this project to include cloud storage or distributed computing tools like Spark?
"""