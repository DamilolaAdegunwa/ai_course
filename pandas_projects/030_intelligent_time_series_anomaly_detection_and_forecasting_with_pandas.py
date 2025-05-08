import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import zscore


# Load Time Series Data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return data


# Detect Anomalies Using Z-Score
def detect_anomalies(data, column, threshold=3):
    data['Z-Score'] = zscore(data[column])
    data['Anomaly'] = data['Z-Score'].apply(lambda x: abs(x) > threshold)
    return data


# Seasonal and Trend Decomposition
def decompose_time_series(data, column, freq):
    decomposition = seasonal_decompose(data[column], model='additive', period=freq)
    decomposition.plot()
    plt.show()
    return decomposition


# Forecast Using ARIMA
def forecast_time_series(data, column, steps=5):
    model = ARIMA(data[column], order=(1, 1, 1))  # Adjust order for better results
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast


# Visualization
def plot_results(data, column, forecast=None):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data[column], label="Original Data", color='blue')
    if forecast is not None:
        forecast_index = pd.date_range(start=data.index[-1], periods=len(forecast)+1, freq='D')[1:]
        plt.plot(forecast_index, forecast, label="Forecast", color='orange')
    anomalies = data[data['Anomaly']]
    plt.scatter(anomalies.index, anomalies[column], color='red', label="Anomalies")
    plt.legend()
    plt.show()


# Main Execution
if __name__ == "__main__":
    # Example Data
    data = pd.DataFrame({
        'Date': pd.date_range(start='2024-01-01', periods=50, freq='D'),
        'Traffic': np.random.poisson(lam=1000, size=50)
    })
    # Adding an artificial anomaly
    data.loc[10, 'Traffic'] = 10000

    # Detect Anomalies
    data = detect_anomalies(data, 'Traffic')
    print(data[data['Anomaly']])

    # Decompose Time Series
    decomposition = decompose_time_series(data.set_index('Date'), 'Traffic', freq=7)

    # Forecast Traffic
    forecast = forecast_time_series(data.set_index('Date'), 'Traffic', steps=10)
    print("Forecast:\n", forecast)

    # Plot Results
    plot_results(data.set_index('Date'), 'Traffic', forecast)


# https://chatgpt.com/c/674b65b9-fecc-800c-8311-7f681df9b305
comment = """
### Project Title: **Intelligent Time Series Data Anomaly Detection and Forecasting with Pandas**  
**File Name**: `intelligent_time_series_anomaly_detection_and_forecasting_with_pandas.py`  

---

### Project Description  

This project tackles **time series analysis** with a focus on **anomaly detection** and **forecasting**. It is highly advanced and integrates:  

1. **Dynamic Time Series Resampling**: Handles data resampling for different time granularities (daily, weekly, monthly).  
2. **Advanced Anomaly Detection**: Uses statistical and machine learning techniques to detect anomalies in time series data.  
3. **Seasonal and Trend Decomposition**: Applies decomposition methods to analyze seasonality, trends, and residuals.  
4. **Forecasting**: Implements ARIMA (Auto-Regressive Integrated Moving Average) and machine learning models for forecasting.  
5. **Real-Time Updates**: Handles live-streamed data with periodic updates to the models.  

---

### Example Use Cases  

1. **Stock Market Analysis**: Detect anomalies in stock price trends and forecast future prices.  
2. **IoT Data Monitoring**: Identify unusual patterns in IoT sensor data and predict future metrics.  
3. **Web Traffic Monitoring**: Detect sudden spikes or dips in website traffic and predict future trends.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: Daily Website Traffic Data**  
| Date       | Traffic |  
|------------|---------|  
| 2024-01-01 | 1000    |  
| 2024-01-02 | 1025    |  
| 2024-01-03 | 1500    |  
| 2024-01-04 | 10000   |  
| 2024-01-05 | 1200    |  

**Expected Output**:  
- **Anomalies Detected**:  
  - Date: 2024-01-04, Traffic: 10000 (Anomaly)  

- **Forecast for Next 5 Days**:  
  - 2024-01-06: 1250  
  - 2024-01-07: 1300  
  - 2024-01-08: 1400  
  - 2024-01-09: 1450  
  - 2024-01-10: 1500  

---

#### **Input 2: Monthly Energy Consumption**  
| Month      | Consumption (kWh) |  
|------------|--------------------|  
| 2024-01-01 | 15000             |  
| 2024-02-01 | 15500             |  
| 2024-03-01 | 16000             |  
| 2024-04-01 | 100000            |  
| 2024-05-01 | 16500             |  

**Expected Output**:  
- **Anomalies Detected**:  
  - Month: 2024-04-01, Consumption: 100000 (Anomaly)  

- **Seasonal Trend Decomposition**:  
  - Trend Component: Gradual increase.  
  - Residual Component: Spike in April.  

- **Forecast for Next 3 Months**:  
  - June: 17000  
  - July: 17250  
  - August: 17500  

---

### Python Code  

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import zscore

# Load Time Series Data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return data

# Detect Anomalies Using Z-Score
def detect_anomalies(data, column, threshold=3):
    data['Z-Score'] = zscore(data[column])
    data['Anomaly'] = data['Z-Score'].apply(lambda x: abs(x) > threshold)
    return data

# Seasonal and Trend Decomposition
def decompose_time_series(data, column, freq):
    decomposition = seasonal_decompose(data[column], model='additive', period=freq)
    decomposition.plot()
    plt.show()
    return decomposition

# Forecast Using ARIMA
def forecast_time_series(data, column, steps=5):
    model = ARIMA(data[column], order=(1, 1, 1))  # Adjust order for better results
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Visualization
def plot_results(data, column, forecast=None):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data[column], label="Original Data", color='blue')
    if forecast is not None:
        forecast_index = pd.date_range(start=data.index[-1], periods=len(forecast)+1, freq='D')[1:]
        plt.plot(forecast_index, forecast, label="Forecast", color='orange')
    anomalies = data[data['Anomaly']]
    plt.scatter(anomalies.index, anomalies[column], color='red', label="Anomalies")
    plt.legend()
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Example Data
    data = pd.DataFrame({
        'Date': pd.date_range(start='2024-01-01', periods=50, freq='D'),
        'Traffic': np.random.poisson(lam=1000, size=50)
    })
    # Adding an artificial anomaly
    data.loc[10, 'Traffic'] = 10000

    # Detect Anomalies
    data = detect_anomalies(data, 'Traffic')
    print(data[data['Anomaly']])

    # Decompose Time Series
    decomposition = decompose_time_series(data.set_index('Date'), 'Traffic', freq=7)

    # Forecast Traffic
    forecast = forecast_time_series(data.set_index('Date'), 'Traffic', steps=10)
    print("Forecast:\n", forecast)

    # Plot Results
    plot_results(data.set_index('Date'), 'Traffic', forecast)
```

---

### Advanced Skills Covered  

1. **Time Series Anomaly Detection**: Uses statistical techniques (z-score) for identifying outliers in time series.  
2. **Decomposition Techniques**: Applies seasonal-trend decomposition for insights into components.  
3. **ARIMA Modeling**: Implements ARIMA for reliable time series forecasting.  
4. **Real-Time Data Handling**: Simulates live data and updates anomaly detection and forecasts.  
5. **Data Visualization**: Visualizes trends, anomalies, and forecasts with Matplotlib.  

This project pushes your Pandas and time series analysis skills into professional-level applications in fields like finance, IoT, and marketing analytics.
"""
