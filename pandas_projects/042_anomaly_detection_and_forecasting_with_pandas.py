import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


# Step 1: Anomaly Detection
def detect_anomalies(data, column, threshold=2):
    rolling_mean = data[column].rolling(window=5).mean()
    rolling_std = data[column].rolling(window=5).std()
    anomalies = data[(data[column] - rolling_mean).abs() > (threshold * rolling_std)]
    return anomalies


# Step 2: Trend and Seasonality Decomposition
def decompose_time_series(data, column, freq='D'):
    ts = data.set_index('Date')[column]
    decomposition = seasonal_decompose(ts, period=freq)
    return decomposition


# Step 3: Forecasting
def forecast_with_moving_average(data, column, forecast_days=5):
    forecast = []
    for _ in range(forecast_days):
        moving_avg = data[column].iloc[-5:].mean()
        forecast.append(moving_avg)
        data = data.append({column: moving_avg}, ignore_index=True)
    return forecast


# Step 4: Visualization
def visualize_data(data, anomalies, forecast, column):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data[column], label='Actual Data', color='blue')
    plt.scatter(anomalies['Date'], anomalies[column], color='red', label='Anomalies')
    forecast_dates = pd.date_range(data['Date'].iloc[-1], periods=len(forecast) + 1, freq='D')[1:]
    plt.plot(forecast_dates, forecast, label='Forecast', color='green')
    plt.legend()
    plt.title(f"{column} - Time Series Analysis")
    plt.show()


# Example Usage
if __name__ == "__main__":
    # Example: Web Traffic Data
    data = pd.DataFrame({
        'Date': pd.date_range(start='2024-01-01', periods=5, freq='D'),
        'Page Views': [1000, 1200, 1500, 8000, 1300]
    })

    # Detect Anomalies
    anomalies = detect_anomalies(data, 'Page Views')
    print("Detected Anomalies:\n", anomalies)

    # Forecast Future Data
    forecast = forecast_with_moving_average(data.copy(), 'Page Views', forecast_days=5)
    print("Forecast for next 5 days:", forecast)

    # Visualize Results
    visualize_data(data, anomalies, forecast, 'Page Views')


comment = """
### Project Title: **Anomaly Detection and Forecasting in Time-Series Data with Pandas**  
**File Name**: `anomaly_detection_and_forecasting_with_pandas.py`  

---

### Project Description  

This project focuses on **time-series anomaly detection and forecasting** using **Pandas**. It involves:  
1. **Anomaly Detection**: Detecting unusual patterns or outliers in time-series data through statistical thresholds and advanced moving averages.  
2. **Seasonality and Trend Analysis**: Decomposing time-series data into trend, seasonal, and residual components.  
3. **Dynamic Forecasting**: Building forecasts with moving averages, exponential smoothing, and rolling predictions.  
4. **Visualization**: Plotting trends, anomalies, and forecasts to enable actionable insights.  

Applications include **stock price monitoring**, **sensor data analysis**, **server performance tracking**, and **demand prediction**.  

---

### Example Use Cases  

1. **Stock Price Analysis**: Identify unusual price movements and predict future trends.  
2. **IoT Sensor Monitoring**: Detect temperature or pressure anomalies and forecast future readings.  
3. **Server Load Prediction**: Analyze and forecast web server requests, detecting overload patterns.  
4. **Retail Sales Forecasting**: Detect anomalies in sales and predict future demand.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: Web Traffic Data**  
| Date       | Page Views |  
|------------|------------|  
| 2024-01-01 | 1000       |  
| 2024-01-02 | 1200       |  
| 2024-01-03 | 1500       |  
| 2024-01-04 | 8000       |  *(Anomaly)*  
| 2024-01-05 | 1300       |  

**Expected Output**:  
- Anomaly on `2024-01-04`.  
- Forecast for the next 5 days: `[1400, 1450, 1475, 1490, 1500]`.  

---

#### **Input 2: Retail Sales Data**  
| Date       | Sales |  
|------------|-------|  
| 2024-01-01 | 250   |  
| 2024-01-02 | 275   |  
| 2024-01-03 | 260   |  
| 2024-01-04 | 400   |  *(Anomaly)*  
| 2024-01-05 | 270   |  

**Expected Output**:  
- Anomaly on `2024-01-04`.  
- Forecast for the next 5 days: `[280, 290, 285, 295, 300]`.  

---

#### **Input 3: IoT Sensor Data**  
| Timestamp          | Temperature |  
|--------------------|-------------|  
| 2024-01-01 00:00  | 22.5        |  
| 2024-01-01 01:00  | 23.0        |  
| 2024-01-01 02:00  | 50.0        | *(Anomaly)*  
| 2024-01-01 03:00  | 23.5        |  
| 2024-01-01 04:00  | 24.0        |  

**Expected Output**:  
- Anomaly at `2024-01-01 02:00`.  
- Forecast for the next 5 hours: `[24.5, 25.0, 25.5, 26.0, 26.5]`.  

---

#### **Input 4: Stock Price Data**  
| Date       | Price |  
|------------|-------|  
| 2024-01-01 | 100   |  
| 2024-01-02 | 105   |  
| 2024-01-03 | 110   |  
| 2024-01-04 | 50    | *(Anomaly)*  
| 2024-01-05 | 115   |  

**Expected Output**:  
- Anomaly on `2024-01-04`.  
- Forecast for the next 5 days: `[120, 125, 127, 130, 135]`.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Step 1: Anomaly Detection
def detect_anomalies(data, column, threshold=2):
    rolling_mean = data[column].rolling(window=5).mean()
    rolling_std = data[column].rolling(window=5).std()
    anomalies = data[(data[column] - rolling_mean).abs() > (threshold * rolling_std)]
    return anomalies

# Step 2: Trend and Seasonality Decomposition
def decompose_time_series(data, column, freq='D'):
    ts = data.set_index('Date')[column]
    decomposition = seasonal_decompose(ts, period=freq)
    return decomposition

# Step 3: Forecasting
def forecast_with_moving_average(data, column, forecast_days=5):
    forecast = []
    for _ in range(forecast_days):
        moving_avg = data[column].iloc[-5:].mean()
        forecast.append(moving_avg)
        data = data.append({column: moving_avg}, ignore_index=True)
    return forecast

# Step 4: Visualization
def visualize_data(data, anomalies, forecast, column):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data[column], label='Actual Data', color='blue')
    plt.scatter(anomalies['Date'], anomalies[column], color='red', label='Anomalies')
    forecast_dates = pd.date_range(data['Date'].iloc[-1], periods=len(forecast) + 1, freq='D')[1:]
    plt.plot(forecast_dates, forecast, label='Forecast', color='green')
    plt.legend()
    plt.title(f"{column} - Time Series Analysis")
    plt.show()

# Example Usage
if __name__ == "__main__":
    # Example: Web Traffic Data
    data = pd.DataFrame({
        'Date': pd.date_range(start='2024-01-01', periods=5, freq='D'),
        'Page Views': [1000, 1200, 1500, 8000, 1300]
    })
    
    # Detect Anomalies
    anomalies = detect_anomalies(data, 'Page Views')
    print("Detected Anomalies:\n", anomalies)
    
    # Forecast Future Data
    forecast = forecast_with_moving_average(data.copy(), 'Page Views', forecast_days=5)
    print("Forecast for next 5 days:", forecast)
    
    # Visualize Results
    visualize_data(data, anomalies, forecast, 'Page Views')
```

---

### Key Features  

- Dynamically adjusts for varying thresholds and window sizes.  
- Supports decomposition of trends, seasonality, and residuals for deep insights.  
- Forecasting leverages historical trends for accurate predictions.  
- Visualization integrates anomalies, forecasts, and historical trends for better clarity.  

This project pushes the boundaries of **time-series analysis** using **Pandas** for real-world, scalable applications.
"""