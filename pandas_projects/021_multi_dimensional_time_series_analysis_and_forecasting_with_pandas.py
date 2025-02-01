import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# Function to preprocess time-series data
def preprocess_time_series(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    return df


# Function for rolling averages and lagged features
def create_features(df, target_col, window=3):
    df[f'{target_col}_rolling_avg'] = df[target_col].rolling(window=window).mean()
    df[f'{target_col}_lag_1'] = df[target_col].shift(1)
    return df


# Function for seasonal decomposition
def seasonal_analysis(df, target_col, frequency):
    decomposition = seasonal_decompose(df[target_col], model='additive', period=frequency)
    decomposition.plot()
    plt.show()
    return decomposition


# Function for forecasting with Exponential Smoothing
def forecast_exponential_smoothing(df, target_col, forecast_steps):
    model = ExponentialSmoothing(df[target_col], seasonal='additive', seasonal_periods=12)
    fit = model.fit()
    forecast = fit.forecast(forecast_steps)
    return forecast


# Function for visualization
def plot_time_series(df, columns, title):
    plt.figure(figsize=(12, 6))
    for col in columns:
        plt.plot(df.index, df[col], label=col)
    plt.legend()
    plt.title(title)
    plt.show()


# Main function
if __name__ == "__main__":
    # Example: Weather data
    weather_data = {
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
        'Temperature': [15, 16, 15, 14],
        'Rainfall': [2.5, 3.0, 0.0, 0.5],
        'Humidity': [80, 78, 75, 70]
    }
    df = pd.DataFrame(weather_data)

    # Preprocess data
    df = preprocess_time_series(df, date_col='Date')

    # Create rolling averages and lagged features
    df = create_features(df, target_col='Temperature', window=3)
    print(df)

    # Perform seasonal decomposition
    decomposition = seasonal_analysis(df, target_col='Temperature', frequency=12)

    # Forecast future trends
    forecast = forecast_exponential_smoothing(df, target_col='Temperature', forecast_steps=7)
    print("Forecasted Values:", forecast)

    # Visualize results
    plot_time_series(df, columns=['Temperature', 'Temperature_rolling_avg'], title="Temperature Trends")


#2/2
comment = """
### Project Title: **Multi-Dimensional Time Series Analysis and Forecasting with Pandas**  
**File Name**: `multi_dimensional_time_series_analysis_and_forecasting_with_pandas.py`  

---

### Project Description  
This project uses **Pandas** to handle multi-dimensional time-series datasets and applies advanced data transformations, analysis, and forecasting techniques. The goal is to prepare and analyze datasets with multiple time-varying variables, detect patterns, and forecast future trends. This includes:

1. **Complex Aggregations**: Analyzing temporal and multi-dimensional trends such as seasonal, weekly, or daily variations.  
2. **Feature Engineering**: Creating lagged features, rolling averages, and cyclical encodings for time components (e.g., day of week, month).  
3. **Advanced Forecasting**: Using advanced models like **ARIMA**, **SARIMA**, or **Prophet** to forecast trends and visualize outcomes.  
4. **Data Visualization**: Comprehensive plotting to illustrate patterns, correlations, and forecast accuracy.  
5. **Predictive Analytics**: Identifying anomalies or future outliers based on trends.  

---

### Example Use Cases  
1. **Weather Forecasting**: Analyzing multi-dimensional time-series data to forecast temperature, rainfall, and humidity trends.  
2. **Sales Prediction**: Forecasting future sales based on historical data with seasonal trends.  
3. **Traffic Analysis**: Analyzing and predicting vehicular or web traffic trends for optimization.

---

### Example Input(s) and Expected Output(s)

#### **Input 1: Weather Data (Multi-Feature Time Series)**  
| Date       | Temperature | Rainfall | Humidity | Wind Speed |  
|------------|-------------|----------|----------|------------|  
| 2024-01-01 | 15          | 2.5      | 80       | 10         |  
| 2024-01-02 | 16          | 3.0      | 78       | 12         |  
| 2024-01-03 | 15          | 0.0      | 75       | 8          |  
| 2024-01-04 | 14          | 0.5      | 70       | 6          |  

**Expected Output**:  
- **Rolling Averages**: Daily 3-day rolling average of temperature and rainfall.  
- **Forecasting**: Predict the temperature for the next 7 days based on patterns.

#### **Input 2: Sales Data**  
| Date       | Product A Sales | Product B Sales | Total Sales |  
|------------|-----------------|-----------------|-------------|  
| 2024-06-01 | 200             | 150             | 350         |  
| 2024-06-02 | 220             | 160             | 380         |  
| 2024-06-03 | 210             | 170             | 380         |  
| 2024-06-04 | 230             | 180             | 410         |  

**Expected Output**:  
- **Seasonal Patterns**: Identification of weekly trends in sales data.  
- **Forecasting**: Predict sales for the next 5 days.  

#### **Input 3: Web Traffic Data**  
| Timestamp            | Page Views | Clicks | Sessions |  
|----------------------|------------|--------|----------|  
| 2024-10-01 00:00:00 | 500        | 50     | 40       |  
| 2024-10-01 01:00:00 | 520        | 60     | 45       |  
| 2024-10-01 02:00:00 | 480        | 55     | 38       |  
| 2024-10-01 03:00:00 | 460        | 40     | 35       |  

**Expected Output**:  
- **Cyclical Encoding**: Highlight traffic peaks and valleys over the day.  
- **Anomaly Detection**: Identify hours with significant deviations.

---

### Python Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Function to preprocess time-series data
def preprocess_time_series(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    return df

# Function for rolling averages and lagged features
def create_features(df, target_col, window=3):
    df[f'{target_col}_rolling_avg'] = df[target_col].rolling(window=window).mean()
    df[f'{target_col}_lag_1'] = df[target_col].shift(1)
    return df

# Function for seasonal decomposition
def seasonal_analysis(df, target_col, frequency):
    decomposition = seasonal_decompose(df[target_col], model='additive', period=frequency)
    decomposition.plot()
    plt.show()
    return decomposition

# Function for forecasting with Exponential Smoothing
def forecast_exponential_smoothing(df, target_col, forecast_steps):
    model = ExponentialSmoothing(df[target_col], seasonal='additive', seasonal_periods=12)
    fit = model.fit()
    forecast = fit.forecast(forecast_steps)
    return forecast

# Function for visualization
def plot_time_series(df, columns, title):
    plt.figure(figsize=(12, 6))
    for col in columns:
        plt.plot(df.index, df[col], label=col)
    plt.legend()
    plt.title(title)
    plt.show()

# Main function
if __name__ == "__main__":
    # Example: Weather data
    weather_data = {
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
        'Temperature': [15, 16, 15, 14],
        'Rainfall': [2.5, 3.0, 0.0, 0.5],
        'Humidity': [80, 78, 75, 70]
    }
    df = pd.DataFrame(weather_data)
    
    # Preprocess data
    df = preprocess_time_series(df, date_col='Date')
    
    # Create rolling averages and lagged features
    df = create_features(df, target_col='Temperature', window=3)
    print(df)
    
    # Perform seasonal decomposition
    decomposition = seasonal_analysis(df, target_col='Temperature', frequency=12)
    
    # Forecast future trends
    forecast = forecast_exponential_smoothing(df, target_col='Temperature', forecast_steps=7)
    print("Forecasted Values:", forecast)
    
    # Visualize results
    plot_time_series(df, columns=['Temperature', 'Temperature_rolling_avg'], title="Temperature Trends")
```

---

### How This Project Advances Your Skills  
1. **Multi-Dimensional Analysis**: You’ll work with complex datasets involving multiple time-dependent variables.  
2. **Temporal Patterns**: Learn how to detect trends, seasonality, and anomalies.  
3. **Forecasting Techniques**: Implement advanced models like exponential smoothing for predictions.  
4. **Cyclical Features**: Encode time-based features to capture periodicity effectively.  
5. **Data Visualization**: Advanced plotting techniques to illustrate trends and forecasts.  

You can further enhance this project by incorporating **machine learning models** (e.g., LSTM or XGBoost) for time-series predictions.
"""