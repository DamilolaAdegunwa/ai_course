import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


# Load and preprocess data
def load_multivariate_data(file_path, date_column):
    df = pd.read_csv(file_path, parse_dates=[date_column])
    df.sort_values(by=date_column, inplace=True)
    df.set_index(date_column, inplace=True)
    return df


# Detect anomalies using rolling statistics
def detect_anomalies(df, window_size, z_thresh=3):
    anomalies = {}
    for column in df.columns:
        rolling_mean = df[column].rolling(window=window_size).mean()
        rolling_std = df[column].rolling(window=window_size).std()
        z_scores = np.abs((df[column] - rolling_mean) / rolling_std)
        anomalies[column] = df[z_scores > z_thresh].index.tolist()
    return anomalies


# Multivariate trend analysis
def multivariate_trend_analysis(df):
    correlation_matrix = df.corr()
    trend_analysis = df.diff().mean().to_dict()  # Avg changes per time unit
    return correlation_matrix, trend_analysis


# Visualize anomalies and correlations
def visualize_anomalies_and_trends(df, anomalies, correlation_matrix, title):
    plt.figure(figsize=(12, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)
        plt.scatter(df.index[df.index.isin(anomalies[column])],
                    df.loc[df.index.isin(anomalies[column]), column],
                    color='red', label=f'Anomalies: {column}')
    plt.legend()
    plt.title(f"{title} - Anomaly Detection")
    plt.show()

    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title(f"{title} - Correlation Matrix")
    plt.show()


# Full pipeline
def anomaly_detection_pipeline(file_path, date_column, window_size, z_thresh):
    print("Loading data...")
    df = load_multivariate_data(file_path, date_column)

    print("Detecting anomalies...")
    anomalies = detect_anomalies(df, window_size, z_thresh)

    print("Analyzing trends...")
    correlation_matrix, trend_analysis = multivariate_trend_analysis(df)

    print("\nTrend Analysis:")
    for feature, avg_change in trend_analysis.items():
        print(f"{feature}: Avg Change per Unit: {avg_change:.2f}")

    print("\nVisualizing anomalies and correlations...")
    visualize_anomalies_and_trends(df, anomalies, correlation_matrix, "Anomaly Detection & Trends")

    return anomalies, correlation_matrix, trend_analysis


# Example usage
if __name__ == "__main__":
    # Example file_path
    file_path = "server_response_times.csv"  # Replace with your dataset
    date_column = "timestamp"
    window_size = 24  # Rolling window for daily data
    z_thresh = 3  # Z-score threshold for anomaly detection

    anomalies, correlation_matrix, trend_analysis = anomaly_detection_pipeline(
        file_path, date_column, window_size, z_thresh
    )
    print("Anomalies Detected:", anomalies)


comment = """
### Project Title: **Complex Anomaly Detection and Multivariate Trend Analysis with Pandas**  
**File Name**: `complex_anomaly_detection_and_multivariate_trend_analysis_with_pandas.py`  

---

### Project Description  
This project involves **building a scalable anomaly detection framework** and performing **multivariate trend analysis** using Pandas. Key objectives include:  
1. Identifying anomalies in time-series or transactional data using statistical and moving-window techniques.  
2. Decomposing multivariate trends to analyze relationships between features.  
3. Correlating anomalies across datasets for root cause identification.  
4. Automating anomaly reporting and visualization.  

You’ll work with rolling windows, advanced correlation methods, and multivariate data to handle scenarios like fraud detection, server performance monitoring, and trend analysis in sales or financial systems.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1**:  
**Data**: Server response times (hourly logs for a month).  
**Task**: Detect anomalies in response times.  
**Expected Output**: A DataFrame with anomalies marked and a visualization showing spikes in response times.  

#### **Input 2**:  
**Data**: Daily energy consumption across multiple regions.  
**Task**: Analyze trends and find periods with unusual deviations.  
**Expected Output**: A report with detected deviations and heatmaps for correlations between regions.  

#### **Input 3**:  
**Data**: Monthly sales and marketing spends for 5 products over 3 years.  
**Task**: Identify correlations between marketing spend and sales, with anomalies highlighted.  
**Expected Output**: Correlation matrix, anomaly data, and trend visualization per product.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Load and preprocess data
def load_multivariate_data(file_path, date_column):
    df = pd.read_csv(file_path, parse_dates=[date_column])
    df.sort_values(by=date_column, inplace=True)
    df.set_index(date_column, inplace=True)
    return df

# Detect anomalies using rolling statistics
def detect_anomalies(df, window_size, z_thresh=3):
    anomalies = {}
    for column in df.columns:
        rolling_mean = df[column].rolling(window=window_size).mean()
        rolling_std = df[column].rolling(window=window_size).std()
        z_scores = np.abs((df[column] - rolling_mean) / rolling_std)
        anomalies[column] = df[z_scores > z_thresh].index.tolist()
    return anomalies

# Multivariate trend analysis
def multivariate_trend_analysis(df):
    correlation_matrix = df.corr()
    trend_analysis = df.diff().mean().to_dict()  # Avg changes per time unit
    return correlation_matrix, trend_analysis

# Visualize anomalies and correlations
def visualize_anomalies_and_trends(df, anomalies, correlation_matrix, title):
    plt.figure(figsize=(12, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)
        plt.scatter(df.index[df.index.isin(anomalies[column])], 
                    df.loc[df.index.isin(anomalies[column]), column], 
                    color='red', label=f'Anomalies: {column}')
    plt.legend()
    plt.title(f"{title} - Anomaly Detection")
    plt.show()
    
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title(f"{title} - Correlation Matrix")
    plt.show()

# Full pipeline
def anomaly_detection_pipeline(file_path, date_column, window_size, z_thresh):
    print("Loading data...")
    df = load_multivariate_data(file_path, date_column)
    
    print("Detecting anomalies...")
    anomalies = detect_anomalies(df, window_size, z_thresh)
    
    print("Analyzing trends...")
    correlation_matrix, trend_analysis = multivariate_trend_analysis(df)
    
    print("\nTrend Analysis:")
    for feature, avg_change in trend_analysis.items():
        print(f"{feature}: Avg Change per Unit: {avg_change:.2f}")
    
    print("\nVisualizing anomalies and correlations...")
    visualize_anomalies_and_trends(df, anomalies, correlation_matrix, "Anomaly Detection & Trends")
    
    return anomalies, correlation_matrix, trend_analysis

# Example usage
if __name__ == "__main__":
    # Example file_path
    file_path = "server_response_times.csv"  # Replace with your dataset
    date_column = "timestamp"
    window_size = 24  # Rolling window for daily data
    z_thresh = 3  # Z-score threshold for anomaly detection
    
    anomalies, correlation_matrix, trend_analysis = anomaly_detection_pipeline(
        file_path, date_column, window_size, z_thresh
    )
    print("Anomalies Detected:", anomalies)
```

---

### Testing Scenarios  

#### **Scenario 1**:  
**Data**: Hourly server response times (timestamp, server1, server2, server3).  
**Task**: Detect response spikes and visualize correlation between servers.  
**Expected Output**: Red points on the graph where anomalies occur, with a heatmap of server correlations.  

#### **Scenario 2**:  
**Data**: Monthly sales and marketing spend data for 5 products.  
**Task**: Highlight anomalous spending patterns and correlate marketing and sales trends.  
**Expected Output**: Heatmap showing correlations, with anomalies in a separate list.  

#### **Scenario 3**:  
**Data**: Daily regional energy consumption (region1, region2, region3).  
**Task**: Find anomalies in energy usage and compare trends between regions.  
**Expected Output**: Anomaly-labeled graph and a heatmap of inter-regional correlations.  

---

### Key Learnings  
- **Rolling Statistics**: Use rolling windows for anomaly detection in time-series data.  
- **Multivariate Analysis**: Advanced correlation and trend decomposition techniques.  
- **Anomaly Detection**: Z-score-based thresholding for robust outlier detection.  
- **Visualization**: Seamlessly combine anomaly detection and correlation heatmaps.  

Would you like to extend this to support **real-time streaming data** or integrate with libraries like **statsmodels** or **Prophet**?
"""