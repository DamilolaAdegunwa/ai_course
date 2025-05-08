import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


# Step 1: Data Preprocessing
def preprocess_data(data, target_col, datetime_col):
    # Convert datetime column to datetime type
    data[datetime_col] = pd.to_datetime(data[datetime_col])

    # Set datetime column as index
    data.set_index(datetime_col, inplace=True)

    # Handle missing values
    data = data.fillna(method='ffill')  # Forward fill for missing data

    # Feature scaling (if needed for ML models)
    scaler = StandardScaler()
    data[target_col] = scaler.fit_transform(data[target_col].values.reshape(-1, 1))

    return data


# Step 2: Time Series Decomposition
def decompose_timeseries(data, target_col):
    decomposition = seasonal_decompose(data[target_col], model='additive', period=12)
    decomposition.plot()
    plt.show()
    return decomposition


# Step 3: ARIMA Model
def arima_forecasting(data, target_col, p=1, d=1, q=1):
    model = ARIMA(data[target_col], order=(p, d, q))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=5)
    return forecast


# Step 4: XGBoost Model for Forecasting
def xgboost_forecasting(data, features, target_col):
    X = data[features]
    y = data[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6)
    model.fit(X_train, y_train)

    # Forecasting
    forecast = model.predict(X_test)

    # Evaluate model
    mae = mean_absolute_error(y_test, forecast)
    rmse = np.sqrt(mean_squared_error(y_test, forecast))

    print(f"MAE: {mae}, RMSE: {rmse}")

    return forecast


# Step 5: Forecast Evaluation
def evaluate_forecast(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {mape}%")


# Example Usage
if __name__ == "__main__":
    # Example: Stock Market Data
    stock_data = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'Open': [100, 102, 104, 106],
        'High': [105, 106, 108, 110],
        'Low': [98, 100, 102, 104],
        'Close': [102, 104, 106, 108],
        'Volume': [500000, 520000, 510000, 530000]
    })

    stock_data = preprocess_data(stock_data, 'Close', 'Date')
    decompose_timeseries(stock_data, 'Close')

    # ARIMA Model Forecast
    arima_forecast = arima_forecasting(stock_data, 'Close')
    print(f"ARIMA Forecast: {arima_forecast}")

    # XGBoost Forecast
    features = ['Open', 'High', 'Low', 'Volume']
    xgb_forecast = xgboost_forecasting(stock_data, features, 'Close')

    # Evaluate Forecast
    evaluate_forecast(stock_data['Close'], xgb_forecast)

    # Plotting Forecast Results
    plt.plot(stock_data.index, stock_data['Close'], label='Actual')
    plt.plot(stock_data.index[-len(xgb_forecast):], xgb_forecast, label='Forecast')
    plt.legend()
    plt.show()


comment = """
### Project Title: **Advanced Predictive Analytics and Time Series Forecasting with Pandas and Machine Learning**  
**File Name**: `advanced_predictive_analytics_and_time_series_forecasting_with_pandas_and_ml.py`

---

### Project Description  

This project demonstrates **advanced predictive analytics** and **time series forecasting** using **Pandas** and **machine learning** techniques. The goal is to forecast future trends based on historical data and provide predictive insights into time-dependent variables. The project includes:

1. **Data Preprocessing**: Handle missing data, perform feature engineering, and scale the data appropriately for time series forecasting.
2. **Feature Engineering**: Create lag features, rolling statistics, and time-based features to enhance predictive models.
3. **Time Series Decomposition**: Use statistical techniques to decompose the time series into trend, seasonality, and residuals.
4. **Modeling**: Apply advanced time series forecasting models such as ARIMA (AutoRegressive Integrated Moving Average), SARIMA (Seasonal ARIMA), and machine learning algorithms like XGBoost.
5. **Model Evaluation**: Use cross-validation and performance metrics like MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and MAPE (Mean Absolute Percentage Error) for model evaluation.
6. **Forecasting**: Generate future predictions based on the selected model and visualize the forecasted vs actual data.

This project is useful for applications in stock market prediction, sales forecasting, energy consumption prediction, weather forecasting, and any domain that involves time series data.

---

### Example Use Cases  

1. **Stock Market Prediction**: Forecast future stock prices using historical data to make informed investment decisions.  
2. **Sales Forecasting**: Predict future product sales to optimize inventory and supply chain management.  
3. **Weather Prediction**: Use historical weather data to forecast future weather patterns.  
4. **Energy Consumption Forecasting**: Predict future energy consumption based on past usage data to manage energy resources effectively.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: Stock Market Data (Price Prediction)**  
| Date       | Open   | High   | Low    | Close  | Volume  |  
|------------|--------|--------|--------|--------|---------|  
| 2023-01-01 | 100    | 105    | 98     | 102    | 500000  |  
| 2023-01-02 | 102    | 106    | 100    | 104    | 520000  |  
| 2023-01-03 | 104    | 108    | 102    | 106    | 510000  |  
| 2023-01-04 | 106    | 110    | 104    | 108    | 530000  |  

**Expected Output**:  
- **Forecast**: Predict the future closing prices for the next 5 days.  
- **Result**: A predicted closing price for 2023-01-05 to 2023-01-09.  

---

#### **Input 2: Retail Sales Data (Sales Forecasting)**  
| Date       | Store_ID | Sales |  
|------------|----------|-------|  
| 2023-01-01 | 1        | 500   |  
| 2023-01-02 | 1        | 450   |  
| 2023-01-03 | 1        | 600   |  
| 2023-01-04 | 1        | 550   |  

**Expected Output**:  
- **Forecast**: Predict the future sales for the next 7 days.  
- **Result**: Sales forecasts for the next week, such as 2023-01-05 to 2023-01-11.  

---

#### **Input 3: Energy Consumption Data (Energy Forecasting)**  
| Date       | Hour | Consumption |  
|------------|------|-------------|  
| 2023-01-01 | 0    | 50          |  
| 2023-01-01 | 1    | 55          |  
| 2023-01-01 | 2    | 60          |  
| 2023-01-01 | 3    | 58          |  

**Expected Output**:  
- **Forecast**: Predict energy consumption for the next 12 hours.  
- **Result**: Forecasted consumption for hours 4 to 15.  

---

#### **Input 4: Weather Data (Temperature Prediction)**  
| Date       | Temperature |  
|------------|-------------|  
| 2023-01-01 | 30          |  
| 2023-01-02 | 32          |  
| 2023-01-03 | 31          |  
| 2023-01-04 | 33          |  

**Expected Output**:  
- **Forecast**: Predict the next week's temperature.  
- **Result**: Temperature forecasts for 2023-01-05 to 2023-01-11.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Step 1: Data Preprocessing
def preprocess_data(data, target_col, datetime_col):
    # Convert datetime column to datetime type
    data[datetime_col] = pd.to_datetime(data[datetime_col])
    
    # Set datetime column as index
    data.set_index(datetime_col, inplace=True)
    
    # Handle missing values
    data = data.fillna(method='ffill')  # Forward fill for missing data
    
    # Feature scaling (if needed for ML models)
    scaler = StandardScaler()
    data[target_col] = scaler.fit_transform(data[target_col].values.reshape(-1, 1))
    
    return data

# Step 2: Time Series Decomposition
def decompose_timeseries(data, target_col):
    decomposition = seasonal_decompose(data[target_col], model='additive', period=12)
    decomposition.plot()
    plt.show()
    return decomposition

# Step 3: ARIMA Model
def arima_forecasting(data, target_col, p=1, d=1, q=1):
    model = ARIMA(data[target_col], order=(p, d, q))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=5)
    return forecast

# Step 4: XGBoost Model for Forecasting
def xgboost_forecasting(data, features, target_col):
    X = data[features]
    y = data[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6)
    model.fit(X_train, y_train)
    
    # Forecasting
    forecast = model.predict(X_test)
    
    # Evaluate model
    mae = mean_absolute_error(y_test, forecast)
    rmse = np.sqrt(mean_squared_error(y_test, forecast))
    
    print(f"MAE: {mae}, RMSE: {rmse}")
    
    return forecast

# Step 5: Forecast Evaluation
def evaluate_forecast(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {mape}%")

# Example Usage
if __name__ == "__main__":
    # Example: Stock Market Data
    stock_data = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'Open': [100, 102, 104, 106],
        'High': [105, 106, 108, 110],
        'Low': [98, 100, 102, 104],
        'Close': [102, 104, 106, 108],
        'Volume': [500000, 520000, 510000, 530000]
    })
    
    stock_data = preprocess_data(stock_data, 'Close', 'Date')
    decompose_timeseries(stock_data, 'Close')
    
    # ARIMA Model Forecast
    arima_forecast = arima_forecasting(stock_data, 'Close')
    print(f"ARIMA Forecast: {arima_forecast}")
    
    # XGBoost Forecast
    features = ['Open', 'High', 'Low', 'Volume']
    xgb_forecast = xgboost_forecasting(stock_data, features, 'Close')
    
    # Evaluate Forecast
    evaluate_forecast(stock_data['Close'], xgb_forecast)
    
    # Plotting Forecast Results
    plt.plot(stock_data.index, stock_data['Close'], label='Actual')
    plt.plot(stock_data.index[-len(xgb_forecast):], xgb_forecast, label='Forecast')
    plt.legend()
    plt.show()
```

### Explanation of Code:
1. **Data Preprocessing**: The datetime column is converted to a `datetime` type

, set as an index, and missing values are handled by forward filling.
2. **Time Series Decomposition**: The time series is decomposed into trend, seasonal, and residual components using `seasonal_decompose`.
3. **ARIMA Model**: An ARIMA model is built to forecast future values based on past data.
4. **XGBoost Model**: XGBoost regression is used for forecasting based on multiple features like Open, High, Low, and Volume.
5. **Model Evaluation**: The performance is evaluated using MAE, RMSE, and MAPE metrics.

This project enables you to forecast future trends and evaluate your models based on advanced machine learning algorithms, making it highly applicable for business and research purposes in predictive analytics and time series analysis.
"""
