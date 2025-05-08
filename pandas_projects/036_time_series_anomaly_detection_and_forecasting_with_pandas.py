import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


# Load time series data
def load_time_series(file_path):
    return pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')


# Impute missing values
def impute_missing_values(data, method='linear'):
    return data.interpolate(method=method)


# Detect anomalies
def detect_anomalies(data, threshold=3):
    mean = data.mean()
    std = data.std()
    anomalies = data[np.abs(data - mean) > threshold * std]
    return anomalies


# ARIMA Forecasting
def arima_forecast(data, steps=3, order=(1, 1, 1)):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast


# Plotting time series with anomalies
def plot_anomalies(data, anomalies):
    plt.figure(figsize=(10, 6))
    plt.plot(data, label="Original Data")
    plt.scatter(anomalies.index, anomalies, color='red', label="Anomalies")
    plt.title("Time Series with Anomalies")
    plt.legend()
    plt.show()


# Example Usage
if __name__ == "__main__":
    # Example data
    data = pd.DataFrame({
        "Date": pd.date_range(start="2024-01-01", periods=5),
        "Stock Price": [100, 102, 105, 110, 98]
    }).set_index("Date")

    # Handle missing data (if any)
    data = impute_missing_values(data)

    # Detect anomalies
    anomalies = detect_anomalies(data['Stock Price'])
    print("Anomalies Detected:\n", anomalies)

    # Forecast future trends
    forecast = arima_forecast(data['Stock Price'])
    print("Forecast:\n", forecast)

    # Plot anomalies
    plot_anomalies(data['Stock Price'], anomalies)


# https://chatgpt.com/c/674b65b9-fecc-800c-8311-7f681df9b305
comment = """
### Project Title: **Time Series Anomaly Detection and Forecasting with Pandas**  
**File Name**: `time_series_anomaly_detection_and_forecasting_with_pandas.py`  

---

### Project Description  

This advanced project dives into **time series data analysis**, incorporating **anomaly detection** and **forecasting techniques** using **Pandas** and **ARIMA modeling**. The project aims to:  
1. Detect anomalies in complex time series data based on historical trends and thresholds.  
2. Build a robust forecasting system to predict future trends.  
3. Use advanced data manipulation techniques to handle missing data, outliers, and seasonality in time series datasets.  

It focuses on applying data science techniques for handling real-world challenges in industries like finance, e-commerce, and IoT.  

---

### Example Use Cases  

1. **Stock Market Trends**: Identify anomalies in stock prices and predict future trends.  
2. **E-Commerce Sales**: Detect sudden spikes or drops in sales and forecast upcoming demand.  
3. **Sensor Data from IoT Devices**: Flag unusual readings from sensors and predict future measurements.  
4. **Energy Usage Patterns**: Monitor anomalies in energy consumption and estimate future requirements.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: Stock Price Data**  
| Date       | Stock Price |  
|------------|-------------|  
| 2024-01-01 | 100         |  
| 2024-01-02 | 102         |  
| 2024-01-03 | 105         |  
| 2024-01-04 | 110         |  
| 2024-01-05 | 98          |  

**Expected Output**:  
- **Anomalies Detected**:  
  - Date: 2024-01-05, Anomaly: 98 (significant drop from trend).  
- **Forecast for Next 3 Days**:  
  - 2024-01-06: 108.5, 2024-01-07: 109.2, 2024-01-08: 110.1.  

---

#### **Input 2: E-Commerce Sales Data with Missing Values**  
| Date       | Sales |  
|------------|-------|  
| 2024-01-01 | 200   |  
| 2024-01-02 | NaN   |  
| 2024-01-03 | 220   |  
| 2024-01-04 | 230   |  
| 2024-01-05 | NaN   |  

**Expected Output**:  
- **Imputed Sales Data**:  
  - 2024-01-02: 210, 2024-01-05: 225.  
- **Anomalies Detected**: None.  
- **Forecast**:  
  - 2024-01-06: 235, 2024-01-07: 240.  

---

#### **Input 3: IoT Sensor Data**  
| Date       | Temperature |  
|------------|-------------|  
| 2024-01-01 | 22          |  
| 2024-01-02 | 23          |  
| 2024-01-03 | 50          |  
| 2024-01-04 | 24          |  
| 2024-01-05 | 25          |  

**Expected Output**:  
- **Anomalies Detected**:  
  - Date: 2024-01-03, Anomaly: 50 (unrealistic spike).  
- **Forecast**:  
  - 2024-01-06: 26, 2024-01-07: 27.  

---

#### **Input 4: Energy Usage Data with Seasonality**  
| Date       | Energy Usage |  
|------------|--------------|  
| 2024-01-01 | 1000         |  
| 2024-01-02 | 1010         |  
| 2024-01-03 | 1005         |  
| 2024-01-04 | 980          |  
| 2024-01-05 | 990          |  

**Expected Output**:  
- **Anomalies Detected**: None.  
- **Forecast**:  
  - 2024-01-06: 995, 2024-01-07: 1002.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load time series data
def load_time_series(file_path):
    return pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Impute missing values
def impute_missing_values(data, method='linear'):
    return data.interpolate(method=method)

# Detect anomalies
def detect_anomalies(data, threshold=3):
    mean = data.mean()
    std = data.std()
    anomalies = data[np.abs(data - mean) > threshold * std]
    return anomalies

# ARIMA Forecasting
def arima_forecast(data, steps=3, order=(1, 1, 1)):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Plotting time series with anomalies
def plot_anomalies(data, anomalies):
    plt.figure(figsize=(10, 6))
    plt.plot(data, label="Original Data")
    plt.scatter(anomalies.index, anomalies, color='red', label="Anomalies")
    plt.title("Time Series with Anomalies")
    plt.legend()
    plt.show()

# Example Usage
if __name__ == "__main__":
    # Example data
    data = pd.DataFrame({
        "Date": pd.date_range(start="2024-01-01", periods=5),
        "Stock Price": [100, 102, 105, 110, 98]
    }).set_index("Date")

    # Handle missing data (if any)
    data = impute_missing_values(data)

    # Detect anomalies
    anomalies = detect_anomalies(data['Stock Price'])
    print("Anomalies Detected:\n", anomalies)

    # Forecast future trends
    forecast = arima_forecast(data['Stock Price'])
    print("Forecast:\n", forecast)

    # Plot anomalies
    plot_anomalies(data['Stock Price'], anomalies)
```

---

### Advanced Skills Covered  

1. **Advanced Time Series Handling**: Implements advanced techniques for handling missing data, outliers, and seasonality.  
2. **Anomaly Detection**: Identifies statistical anomalies in time series data.  
3. **ARIMA Modeling**: Uses ARIMA for robust time series forecasting.  
4. **Real-World Application**: Handles diverse inputs such as stock prices, sales data, IoT sensor readings, and energy usage.  
5. **Data Visualization**: Provides clear and intuitive visualizations of anomalies and trends.  

This project introduces advanced techniques for time series analysis, preparing you for real-world data challenges across multiple domains!
"""
