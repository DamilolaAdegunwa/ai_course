import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# 1. Load and preprocess time-series data
def load_time_series_data(file_path, date_column, target_column):
    df = pd.read_csv(file_path, parse_dates=[date_column])
    df.sort_values(by=date_column, inplace=True)
    df.set_index(date_column, inplace=True)
    return df[[target_column]]


# 2. Create lagged features for time-series data
def create_lag_features(df, target_column, lags):
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df[target_column].shift(lag)
    df.dropna(inplace=True)
    return df


# 3. Train a predictive model
def train_forecasting_model(df, target_column, test_size=0.2):
    features = [col for col in df.columns if col != target_column]
    X = df[features]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        'MAE': mean_absolute_error(y_test, predictions),
        'RMSE': np.sqrt(mean_squared_error(y_test, predictions))
    }
    return model, predictions, y_test, metrics


# 4. Forecast future values
def forecast_future_values(model, df, steps, target_column):
    features = [col for col in df.columns if col != target_column]
    last_row = df.iloc[-1][features].values.reshape(1, -1)
    forecast = []

    for step in range(steps):
        next_value = model.predict(last_row)[0]
        forecast.append(next_value)
        # Update the feature set with the forecasted value
        last_row = np.append(last_row[:, 1:], next_value).reshape(1, -1)

    return forecast


# 5. Visualization function
def plot_forecast(actual, predictions, future_forecast, title):
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual.values, label='Actual')
    plt.plot(predictions.index, predictions.values, label='Predicted')
    future_index = pd.date_range(predictions.index[-1], periods=len(future_forecast) + 1, freq='D')[1:]
    plt.plot(future_index, future_forecast, label='Forecast', linestyle='--')
    plt.legend()
    plt.title(title)
    plt.show()


# 6. Example pipeline
def forecasting_pipeline(file_path, date_column, target_column, lags, steps):
    print("Loading and preprocessing data...")
    data = load_time_series_data(file_path, date_column, target_column)

    print("Creating lagged features...")
    data = create_lag_features(data, target_column, lags)

    print("Training forecasting model...")
    model, predictions, actual, metrics = train_forecasting_model(data, target_column)

    print("Forecasting future values...")
    future_forecast = forecast_future_values(model, data, steps, target_column)

    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

    print("\nPlotting forecast...")
    plot_forecast(data[target_column], pd.Series(predictions, index=actual.index), future_forecast,
                  "Time-Series Forecast")

    return metrics, future_forecast


# Test the pipeline
if __name__ == "__main__":
    # Example data
    file_path = "daily_stock_prices.csv"  # Replace with your time-series dataset
    date_column = "date"
    target_column = "price"
    lags = 7
    steps = 7

    metrics, forecast = forecasting_pipeline(file_path, date_column, target_column, lags, steps)
    print("Future Forecast:", forecast)


comment = """
### Project Title: **Dynamic Forecasting and Predictive Analysis with Pandas**  
**File Name**: `dynamic_forecasting_and_predictive_analysis_with_pandas.py`  

---

### Project Description  
This project explores **dynamic forecasting** and **predictive analytics** using **Pandas**. You will build a forecasting model with Pandas, seamlessly integrating time-series analysis, feature engineering, and predictive modeling. The project involves:  
- Multi-step time-series forecasting.  
- Feature extraction and lag analysis.  
- Correlation and regression modeling for future value prediction.  
- Automated evaluation metrics to validate predictions.  
- Visualization for trend and forecast evaluation.  

This project will enable handling complex predictive scenarios like sales forecasting, weather prediction, or energy usage estimation.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1**:  
**Data**: Time-series dataset of daily stock prices.  
**Task**: Predict stock prices for the next 7 days.  
**Expected Output**: A DataFrame with predicted prices and evaluation metrics like RMSE and MAE.  

#### **Input 2**:  
**Data**: Hourly weather data (temperature, humidity).  
**Task**: Forecast the next 24 hours of temperature.  
**Expected Output**: A DataFrame showing the forecast and a visualization of actual vs. predicted values.  

#### **Input 3**:  
**Data**: Monthly sales data across regions.  
**Task**: Identify the top 5 correlated features and predict sales for the next 3 months.  
**Expected Output**: Correlation table, regression model output, and predictions for each region.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load and preprocess time-series data
def load_time_series_data(file_path, date_column, target_column):
    df = pd.read_csv(file_path, parse_dates=[date_column])
    df.sort_values(by=date_column, inplace=True)
    df.set_index(date_column, inplace=True)
    return df[[target_column]]

# Create lagged features for time-series data
def create_lag_features(df, target_column, lags):
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df[target_column].shift(lag)
    df.dropna(inplace=True)
    return df

# Train a predictive model
def train_forecasting_model(df, target_column, test_size=0.2):
    features = [col for col in df.columns if col != target_column]
    X = df[features]
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    metrics = {
        'MAE': mean_absolute_error(y_test, predictions),
        'RMSE': np.sqrt(mean_squared_error(y_test, predictions))
    }
    return model, predictions, y_test, metrics

# Forecast future values
def forecast_future_values(model, df, steps, target_column):
    features = [col for col in df.columns if col != target_column]
    last_row = df.iloc[-1][features].values.reshape(1, -1)
    forecast = []
    
    for step in range(steps):
        next_value = model.predict(last_row)[0]
        forecast.append(next_value)
        # Update the feature set with the forecasted value
        last_row = np.append(last_row[:, 1:], next_value).reshape(1, -1)
    
    return forecast

# Visualization function
def plot_forecast(actual, predictions, future_forecast, title):
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual.values, label='Actual')
    plt.plot(predictions.index, predictions.values, label='Predicted')
    future_index = pd.date_range(predictions.index[-1], periods=len(future_forecast) + 1, freq='D')[1:]
    plt.plot(future_index, future_forecast, label='Forecast', linestyle='--')
    plt.legend()
    plt.title(title)
    plt.show()

# Example pipeline
def forecasting_pipeline(file_path, date_column, target_column, lags, steps):
    print("Loading and preprocessing data...")
    data = load_time_series_data(file_path, date_column, target_column)
    
    print("Creating lagged features...")
    data = create_lag_features(data, target_column, lags)
    
    print("Training forecasting model...")
    model, predictions, actual, metrics = train_forecasting_model(data, target_column)
    
    print("Forecasting future values...")
    future_forecast = forecast_future_values(model, data, steps, target_column)
    
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    
    print("\nPlotting forecast...")
    plot_forecast(data[target_column], pd.Series(predictions, index=actual.index), future_forecast, "Time-Series Forecast")
    
    return metrics, future_forecast

# Test the pipeline
if __name__ == "__main__":
    # Example data
    file_path = "daily_stock_prices.csv"  # Replace with your time-series dataset
    date_column = "date"
    target_column = "price"
    lags = 7
    steps = 7
    
    metrics, forecast = forecasting_pipeline(file_path, date_column, target_column, lags, steps)
    print("Future Forecast:", forecast)
```

---

### Testing Scenarios  

#### **Scenario 1**:  
**Data**: Daily stock prices for 2 years.  
**Task**: Predict next 7 days of stock prices.  
**Expected Output**: Forecasted prices with RMSE and MAE.  

#### **Scenario 2**:  
**Data**: Hourly weather data for a month.  
**Task**: Forecast next 24 hours of temperature.  
**Expected Output**: Predicted hourly temperatures and trend visualization.  

#### **Scenario 3**:  
**Data**: Monthly sales data for 5 years.  
**Task**: Identify the top correlated features and forecast sales for the next 3 months.  
**Expected Output**: Correlation matrix, regression results, and future sales estimates.  

---

### Key Learnings  
- **Time-Series Analysis**: Advanced lag creation and multi-step forecasting.  
- **Predictive Modeling**: Building regression models with evaluation metrics.  
- **Feature Engineering**: Enhancing datasets for better model performance.  
- **Visualization**: Compare actual vs. forecasted data effectively.  

Would you like to expand this project to incorporate external libraries like **Prophet** or **ARIMA models**?
"""
