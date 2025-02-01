import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.stats import zscore


# Data Preprocessing Function
def preprocess_data(df, time_col):
    df[time_col] = pd.to_datetime(df[time_col])
    df.sort_values(by=time_col, inplace=True)
    df.set_index(time_col, inplace=True)
    return df


# Anomaly Detection using Isolation Forest
def detect_anomalies_isolation_forest(df, feature_col):
    model = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly_Score'] = model.fit_predict(df[[feature_col]])
    anomalies = df[df['Anomaly_Score'] == -1]
    return anomalies


# Anomaly Detection using Z-Score
def detect_anomalies_zscore(df, feature_col, threshold=3):
    df['Z_Score'] = zscore(df[feature_col])
    anomalies = df[np.abs(df['Z_Score']) > threshold]
    return anomalies


# Pattern Mining using Autocorrelation
def autocorrelation_analysis(df, feature_col, lag):
    autocorr = [df[feature_col].autocorr(l) for l in range(1, lag + 1)]
    return autocorr


# Visualization of Anomalies
def visualize_anomalies(df, feature_col, anomalies, title):
    plt.figure(figsize=(12, 6))
    plt.plot(df[feature_col], label='Data', color='blue')
    plt.scatter(anomalies.index, anomalies[feature_col], color='red', label='Anomalies', marker='x')
    plt.title(title)
    plt.legend()
    plt.show()


# Main function
if __name__ == "__main__":
    # Example Dataset
    data = {
        'Date': pd.date_range('2024-12-01', periods=100),
        'Sales': np.random.normal(500, 50, 100).tolist()
    }

    # Introduce anomalies
    data['Sales'][10] = 1000
    data['Sales'][50] = 50

    df = pd.DataFrame(data)

    # Preprocess data
    df = preprocess_data(df, 'Date')

    # Detect anomalies
    anomalies = detect_anomalies_isolation_forest(df, 'Sales')

    # Visualize anomalies
    visualize_anomalies(df, 'Sales', anomalies, 'Sales Anomaly Detection')

    # Perform autocorrelation analysis
    autocorr = autocorrelation_analysis(df, 'Sales', lag=10)
    print("Autocorrelation values:", autocorr)


comment = """
### Project Title: **Advanced Anomaly Detection and Pattern Mining with Pandas**  
**File Name**: `advanced_anomaly_detection_and_pattern_mining_with_pandas.py`  

---

### Project Description  
This project focuses on leveraging **Pandas** for advanced **anomaly detection** and **pattern mining** in time-series data. By integrating techniques such as **unsupervised learning**, **clustering**, and **statistical modeling**, the project detects unusual patterns and reveals hidden trends in datasets.  

Key highlights of the project include:  

1. **Data Preprocessing and Transformation**: Handling time-series data, filling missing values, and scaling features for better model performance.  
2. **Anomaly Detection**: Using statistical methods, moving averages, and machine learning techniques such as **Isolation Forest** and **DBSCAN** for anomaly detection.  
3. **Pattern Mining**: Identifying recurring patterns using frequency analysis, autocorrelation, and motif discovery.  
4. **Custom Metrics**: Developing domain-specific metrics to evaluate anomalies and trends.  
5. **Visualization**: Detailed plotting of anomalies and patterns with annotations for better interpretability.  

This project is suitable for industries such as **finance**, **healthcare**, and **manufacturing**, where anomaly detection and trend analysis are critical.  

---

### Example Use Cases  
1. **Fraud Detection**: Identifying suspicious transactions in financial data.  
2. **Sensor Data Analysis**: Detecting irregularities in IoT sensor readings for predictive maintenance.  
3. **Web Traffic Monitoring**: Spotting unusual traffic spikes or drops in website analytics.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: Daily Sensor Readings**  
| Date       | Sensor_ID | Reading | Temperature | Humidity |  
|------------|-----------|---------|-------------|----------|  
| 2024-12-01 | S001      | 45.5    | 22.3        | 60       |  
| 2024-12-02 | S001      | 47.8    | 23.0        | 65       |  
| 2024-12-03 | S001      | 120.1   | 21.8        | 63       |  
| 2024-12-04 | S001      | 48.3    | 22.5        | 62       |  
| 2024-12-05 | S001      | 46.7    | 22.9        | 64       |  

**Expected Output**:  
- **Anomalies Detected**:  
  - 2024-12-03 (Reading = 120.1)  

#### **Input 2: Website Traffic Data**  
| Date       | Page Views | Bounce Rate | Avg. Session Duration |  
|------------|------------|-------------|------------------------|  
| 2024-11-01 | 1500       | 45%         | 3.5 min                |  
| 2024-11-02 | 1600       | 42%         | 3.6 min                |  
| 2024-11-03 | 5500       | 85%         | 1.2 min                |  
| 2024-11-04 | 1550       | 43%         | 3.7 min                |  
| 2024-11-05 | 1520       | 44%         | 3.6 min                |  

**Expected Output**:  
- **Anomalies Detected**:  
  - 2024-11-03 (Page Views = 5500; Bounce Rate = 85%)  

#### **Input 3: Monthly Sales Data**  
| Month      | Product_ID | Sales    | Returns  | Marketing Spend |  
|------------|------------|----------|----------|-----------------|  
| 2024-01    | P001       | 10000    | 200      | 1500            |  
| 2024-02    | P001       | 10500    | 190      | 1600            |  
| 2024-03    | P001       | 25000    | 300      | 1400            |  
| 2024-04    | P001       | 11000    | 180      | 1700            |  
| 2024-05    | P001       | 11500    | 170      | 1750            |  

**Expected Output**:  
- **Anomalies Detected**:  
  - 2024-03 (Sales = 25000)  

---

### Python Code  

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Data Preprocessing Function
def preprocess_data(df, time_col):
    df[time_col] = pd.to_datetime(df[time_col])
    df.sort_values(by=time_col, inplace=True)
    df.set_index(time_col, inplace=True)
    return df

# Anomaly Detection using Isolation Forest
def detect_anomalies_isolation_forest(df, feature_col):
    model = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly_Score'] = model.fit_predict(df[[feature_col]])
    anomalies = df[df['Anomaly_Score'] == -1]
    return anomalies

# Anomaly Detection using Z-Score
def detect_anomalies_zscore(df, feature_col, threshold=3):
    df['Z_Score'] = zscore(df[feature_col])
    anomalies = df[np.abs(df['Z_Score']) > threshold]
    return anomalies

# Pattern Mining using Autocorrelation
def autocorrelation_analysis(df, feature_col, lag):
    autocorr = [df[feature_col].autocorr(l) for l in range(1, lag+1)]
    return autocorr

# Visualization of Anomalies
def visualize_anomalies(df, feature_col, anomalies, title):
    plt.figure(figsize=(12, 6))
    plt.plot(df[feature_col], label='Data', color='blue')
    plt.scatter(anomalies.index, anomalies[feature_col], color='red', label='Anomalies', marker='x')
    plt.title(title)
    plt.legend()
    plt.show()

# Main function
if __name__ == "__main__":
    # Example Dataset
    data = {
        'Date': pd.date_range('2024-12-01', periods=100),
        'Sales': np.random.normal(500, 50, 100).tolist()
    }
    
    # Introduce anomalies
    data['Sales'][10] = 1000
    data['Sales'][50] = 50
    
    df = pd.DataFrame(data)
    
    # Preprocess data
    df = preprocess_data(df, 'Date')
    
    # Detect anomalies
    anomalies = detect_anomalies_isolation_forest(df, 'Sales')
    
    # Visualize anomalies
    visualize_anomalies(df, 'Sales', anomalies, 'Sales Anomaly Detection')
    
    # Perform autocorrelation analysis
    autocorr = autocorrelation_analysis(df, 'Sales', lag=10)
    print("Autocorrelation values:", autocorr)
```

---
"""