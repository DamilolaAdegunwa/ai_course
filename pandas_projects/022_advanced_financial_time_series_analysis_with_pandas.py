import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest


# 1. Function to preprocess time series data
def preprocess_time_series(df: DataFrame, date_col, value_col):
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df = df[[value_col]].sort_index()
    return df


# 2. Function for time series decomposition
def decompose_time_series(df: DataFrame, column):
    decomposition = seasonal_decompose(df[column], model='additive', period=7)
    return decomposition


# 3. Function for anomaly detection
def detect_anomalies(df: DataFrame, column):
    model = IsolationForest(contamination=0.01, random_state=42)
    df['Anomaly'] = model.fit_predict(df[[column]])
    anomalies = df[df['Anomaly'] == -1]
    return anomalies


# 4. Function for multi-step forecasting
def forecast_time_series(df: DataFrame, column, steps=5):
    model = ARIMA(df[column], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast


# 5. Visualization function
def plot_decomposition(decomposition):
    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
    decomposition.observed.plot(ax=axes[0], title='Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    plt.tight_layout()
    plt.show()


# Main function
if __name__ == "__main__":
    # Example 1: Stock Price Analysis
    stock_data = {
        'Date': ['2024-11-01', '2024-11-02', '2024-11-03'],
        'Close': [102.00, 107.00, 104.00]
    }
    stock_df = pd.DataFrame(stock_data)
    stock_df = preprocess_time_series(stock_df, 'Date', 'Close')

    # Decompose time series
    decomposition = decompose_time_series(stock_df, 'Close')
    plot_decomposition(decomposition)

    # Anomaly detection
    anomalies = detect_anomalies(stock_df, 'Close')
    print("Anomalies in Stock Prices:")
    print(anomalies)

    # Forecast
    forecast = forecast_time_series(stock_df, 'Close', steps=3)
    print("Forecasted Stock Prices:", forecast)

    # Example 2: Sales Data
    sales_data = {
        'Date': ['2024-10-01', '2024-10-02', '2024-10-03'],
        'Sales': [5000, 5200, 4900]
    }
    sales_df = pd.DataFrame(sales_data)
    sales_df = preprocess_time_series(sales_df, 'Date', 'Sales')

    # Forecast sales
    sales_forecast = forecast_time_series(sales_df, 'Sales', steps=7)
    print("Forecasted Sales:", sales_forecast)


#1/2
comment = """
### Project Title: **Advanced Financial Time Series Analysis and Forecasting with Pandas**  
**File Name**: `advanced_financial_time_series_analysis_with_pandas.py`  

---

### Project Description  
This project involves analyzing, modeling, and forecasting financial time series data using **Pandas** and advanced statistical and machine learning techniques. The project integrates the following concepts:

1. **Time Series Decomposition**: Break down a time series into trend, seasonal, and residual components.  
2. **Advanced Feature Engineering**: Generate lag features, rolling statistics, and Fourier transformations for predictive modeling.  
3. **Anomaly Detection**: Identify irregularities in financial data using statistical thresholds and machine learning models.  
4. **Multi-Step Forecasting**: Predict future stock prices, sales, or other financial metrics using ARIMA, SARIMAX, or machine learning regressors.  
5. **Backtesting Framework**: Evaluate model performance using historical data.  

---

### Example Use Cases  
1. **Stock Price Analysis**: Predict future stock prices and identify outliers in historical trading data.  
2. **Sales Forecasting**: Analyze past sales data to make multi-step predictions for revenue.  
3. **Economic Indicator Modeling**: Forecast unemployment rates or GDP growth using lagged and derived indicators.  

---

### Example Input(s) and Expected Output(s)

#### **Input 1: Stock Price Data**  
| Date       | Open   | High   | Low    | Close  | Volume   |  
|------------|--------|--------|--------|--------|----------|  
| 2024-11-01 | 100.00 | 105.00 | 99.00  | 102.00 | 2000000  |  
| 2024-11-02 | 102.00 | 108.00 | 101.00 | 107.00 | 2500000  |  
| 2024-11-03 | 107.00 | 109.00 | 103.00 | 104.00 | 1800000  |  

**Expected Output**:  
- **Trend Component**: The overall price increase over the date range.  
- **Seasonal Component**: Daily variations in trading volume.  
- **Prediction**: Stock price for the next 3 days.

#### **Input 2: Sales Data**  
| Date       | Sales    |  
|------------|----------|  
| 2024-10-01 | 5000     |  
| 2024-10-02 | 5200     |  
| 2024-10-03 | 4900     |  

**Expected Output**:  
- **Anomalies**: No anomaly detected.  
- **Forecast**: Sales for the next 7 days with confidence intervals.

#### **Input 3: Economic Indicators**  
| Date       | Inflation Rate (%) | Unemployment Rate (%) | GDP Growth (%) |  
|------------|--------------------|-----------------------|----------------|  
| 2024-10-01 | 2.5                | 5.1                   | 1.8            |  
| 2024-10-02 | 2.6                | 5.2                   | 1.7            |  
| 2024-10-03 | 2.5                | 5.0                   | 1.9            |  

**Expected Output**:  
- **Prediction**: GDP Growth forecast for the next 5 periods.  
- **Feature Importance**: Unemployment rate contributes most to GDP growth prediction.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest

# Function to preprocess time series data
def preprocess_time_series(df, date_col, value_col):
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df = df[[value_col]].sort_index()
    return df

# Function for time series decomposition
def decompose_time_series(df, column):
    decomposition = seasonal_decompose(df[column], model='additive', period=7)
    return decomposition

# Function for anomaly detection
def detect_anomalies(df, column):
    model = IsolationForest(contamination=0.01, random_state=42)
    df['Anomaly'] = model.fit_predict(df[[column]])
    anomalies = df[df['Anomaly'] == -1]
    return anomalies

# Function for multi-step forecasting
def forecast_time_series(df, column, steps=5):
    model = ARIMA(df[column], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Visualization function
def plot_decomposition(decomposition):
    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
    decomposition.observed.plot(ax=axes[0], title='Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    plt.tight_layout()
    plt.show()

# Main function
if __name__ == "__main__":
    # Example 1: Stock Price Analysis
    stock_data = {
        'Date': ['2024-11-01', '2024-11-02', '2024-11-03'],
        'Close': [102.00, 107.00, 104.00]
    }
    stock_df = pd.DataFrame(stock_data)
    stock_df = preprocess_time_series(stock_df, 'Date', 'Close')

    # Decompose time series
    decomposition = decompose_time_series(stock_df, 'Close')
    plot_decomposition(decomposition)

    # Anomaly detection
    anomalies = detect_anomalies(stock_df, 'Close')
    print("Anomalies in Stock Prices:")
    print(anomalies)

    # Forecast
    forecast = forecast_time_series(stock_df, 'Close', steps=3)
    print("Forecasted Stock Prices:", forecast)

    # Example 2: Sales Data
    sales_data = {
        'Date': ['2024-10-01', '2024-10-02', '2024-10-03'],
        'Sales': [5000, 5200, 4900]
    }
    sales_df = pd.DataFrame(sales_data)
    sales_df = preprocess_time_series(sales_df, 'Date', 'Sales')

    # Forecast sales
    sales_forecast = forecast_time_series(sales_df, 'Sales', steps=7)
    print("Forecasted Sales:", sales_forecast)
```

---

This project focuses on **financial modeling**, **forecasting**, and **advanced anomaly detection**. It’s extensible to other time-series datasets, incorporates **machine learning**, and provides deep insights into temporal trends and patterns.
"""
