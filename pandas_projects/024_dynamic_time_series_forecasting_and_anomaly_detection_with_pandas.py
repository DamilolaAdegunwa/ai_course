import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# Function to preprocess time series data
def preprocess_time_series(df: DataFrame, time_col, value_col, freq='H'):
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)
    df = df.resample(freq).mean()  # Resampling to specified frequency
    df[value_col] = df[value_col].interpolate()  # Filling missing values
    return df


# Function for anomaly detection
def detect_anomalies(df: DataFrame, value_col, threshold=3):
    mean = df[value_col].mean()
    std = df[value_col].std()
    df['Anomaly'] = ((df[value_col] - mean).abs() > threshold * std).astype(int)
    return df


# Function for forecasting
def forecast_time_series(df: DataFrame, value_col, periods, seasonal_periods=12):
    model = ExponentialSmoothing(df[value_col], seasonal='add', seasonal_periods=seasonal_periods).fit()
    forecast = model.forecast(periods)
    return forecast


# Function to visualize the time series, anomalies, and forecast
def visualize_results(df: DataFrame, value_col, forecast=None, title='Time Series Analysis'):
    plt.figure(figsize=(12, 6))
    plt.plot(df[value_col], label='Original Data', color='blue')
    if 'Anomaly' in df.columns:
        anomalies = df[df['Anomaly'] == 1]
        plt.scatter(anomalies.index, anomalies[value_col], color='red', label='Anomalies')
    if forecast is not None:
        forecast_index = pd.date_range(start=df.index[-1], periods=len(forecast)+1, freq='H')[1:]
        plt.plot(forecast_index, forecast, label='Forecast', color='green')
    plt.title(title)
    plt.legend()
    plt.show()


# Main function
if __name__ == "__main__":
    # Example: IoT Sensor Data
    sensor_data = {
        'Timestamp': ['2024-12-01 00:00:00', '2024-12-01 01:00:00', '2024-12-01 02:00:00', '2024-12-01 03:00:00'],
        'Sensor_Value': [120, 115, 200, 119]
    }
    df = pd.DataFrame(sensor_data)

    # Preprocess the data
    processed_df = preprocess_time_series(df, 'Timestamp', 'Sensor_Value')

    # Detect anomalies
    processed_df = detect_anomalies(processed_df, 'Sensor_Value')

    # Forecast future values
    forecasted_values = forecast_time_series(processed_df, 'Sensor_Value', periods=4)

    # Visualize results
    visualize_results(processed_df, 'Sensor_Value', forecast=forecasted_values, title='IoT Sensor Analysis')


comment = """
### Project Title: **Dynamic Time Series Forecasting and Anomaly Detection with Pandas**  
**File Name**: `dynamic_time_series_forecasting_and_anomaly_detection_with_pandas.py`  

---

### Project Description  
This project focuses on leveraging **Pandas** to analyze and model complex time series data for **forecasting future values** and **detecting anomalies**. The steps include:  
1. **Preprocessing Time Series**: Handling irregular time intervals, missing values, and resampling.  
2. **Feature Engineering**: Extracting time-based features like seasonality, trends, and cyclic patterns.  
3. **Anomaly Detection**: Identifying outliers using statistical methods and clustering.  
4. **Forecasting**: Predicting future values using rolling averages, exponential smoothing, and external integrations with libraries like **statsmodels** or **Prophet**.  
5. **Visualization**: Graphing trends, anomalies, and forecasted values dynamically.  

This project is suitable for domains like financial data analysis, IoT sensor data monitoring, and stock market predictions.  

---

### Example Use Cases  

1. **Energy Consumption Monitoring**: Analyze power usage patterns to predict future demands and identify unusual spikes or dips.  
2. **E-commerce Traffic**: Forecast website traffic and detect anomalies during promotional events.  
3. **Stock Price Movement**: Predict stock price trends and highlight unusual price changes.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: IoT Sensor Data**  
| Timestamp           | Sensor_Value |  
|---------------------|--------------|  
| 2024-12-01 00:00:00 | 120          |  
| 2024-12-01 01:00:00 | 115          |  
| 2024-12-01 02:00:00 | 200          |  
| 2024-12-01 03:00:00 | 119          |  

**Expected Output**:  
- Time series plot with anomalies marked.  
- Forecasted sensor values for the next 24 hours.  

#### **Input 2: Stock Prices**  
| Date       | Close_Price |  
|------------|-------------|  
| 2024-12-01 | 250         |  
| 2024-12-02 | 252         |  
| 2024-12-03 | 300         |  
| 2024-12-04 | 253         |  

**Expected Output**:  
- Detected outliers in stock prices.  
- 7-day moving average trend plot.  
- Predicted stock prices for the next week.  

#### **Input 3: E-commerce Traffic Data**  
| Timestamp           | Visits |  
|---------------------|--------|  
| 2024-12-01 08:00:00 | 1500   |  
| 2024-12-01 09:00:00 | 1800   |  
| 2024-12-01 10:00:00 | 1700   |  
| 2024-12-01 11:00:00 | 5000   |  

**Expected Output**:  
- Identification of an anomalous spike at 11:00.  
- Forecasted traffic for the next 4 hours.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Function to preprocess time series data
def preprocess_time_series(df, time_col, value_col, freq='H'):
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)
    df = df.resample(freq).mean()  # Resampling to specified frequency
    df[value_col] = df[value_col].interpolate()  # Filling missing values
    return df

# Function for anomaly detection
def detect_anomalies(df, value_col, threshold=3):
    mean = df[value_col].mean()
    std = df[value_col].std()
    df['Anomaly'] = ((df[value_col] - mean).abs() > threshold * std).astype(int)
    return df

# Function for forecasting
def forecast_time_series(df, value_col, periods, seasonal_periods=12):
    model = ExponentialSmoothing(df[value_col], seasonal='add', seasonal_periods=seasonal_periods).fit()
    forecast = model.forecast(periods)
    return forecast

# Function to visualize the time series, anomalies, and forecast
def visualize_results(df, value_col, forecast=None, title='Time Series Analysis'):
    plt.figure(figsize=(12, 6))
    plt.plot(df[value_col], label='Original Data', color='blue')
    if 'Anomaly' in df.columns:
        anomalies = df[df['Anomaly'] == 1]
        plt.scatter(anomalies.index, anomalies[value_col], color='red', label='Anomalies')
    if forecast is not None:
        forecast_index = pd.date_range(start=df.index[-1], periods=len(forecast)+1, freq='H')[1:]
        plt.plot(forecast_index, forecast, label='Forecast', color='green')
    plt.title(title)
    plt.legend()
    plt.show()

# Main function
if __name__ == "__main__":
    # Example: IoT Sensor Data
    sensor_data = {
        'Timestamp': ['2024-12-01 00:00:00', '2024-12-01 01:00:00', '2024-12-01 02:00:00', '2024-12-01 03:00:00'],
        'Sensor_Value': [120, 115, 200, 119]
    }
    df = pd.DataFrame(sensor_data)

    # Preprocess the data
    processed_df = preprocess_time_series(df, 'Timestamp', 'Sensor_Value')

    # Detect anomalies
    processed_df = detect_anomalies(processed_df, 'Sensor_Value')

    # Forecast future values
    forecasted_values = forecast_time_series(processed_df, 'Sensor_Value', periods=4)

    # Visualize results
    visualize_results(processed_df, 'Sensor_Value', forecast=forecasted_values, title='IoT Sensor Analysis')
```

---

### How This Project Advances Your Skills  
1. **Advanced Time Series Techniques**: Master the intricacies of time-based data manipulation and analysis.  
2. **Anomaly Detection Expertise**: Learn statistical methods for identifying and visualizing unusual patterns.  
3. **Predictive Analytics**: Integrate statistical models to forecast future trends dynamically.  
4. **Real-World Use Cases**: Gain hands-on experience with applicable scenarios in industries like finance, IoT, and marketing.  
5. **Scalability**: Extend this project to real-time monitoring systems by integrating APIs for data streaming and alerts.  

Push your boundaries further by incorporating **machine learning models** for anomaly detection and **probabilistic forecasting** for uncertainty estimation!"""
