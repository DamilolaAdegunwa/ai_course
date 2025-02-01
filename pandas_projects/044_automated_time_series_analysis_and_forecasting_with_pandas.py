import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


# Step 1: Data Preprocessing
def preprocess_time_series(data, date_column, value_column, freq='D'):
    data[date_column] = pd.to_datetime(data[date_column])
    data = data.set_index(date_column)
    data = data.asfreq(freq)
    data[value_column] = data[value_column].fillna(method='ffill')  # Forward fill missing values
    return data


# Step 2: Time-Series Feature Engineering
def create_lagged_features(data, value_column, lags=3):
    for lag in range(1, lags + 1):
        data[f'lag_{lag}'] = data[value_column].shift(lag)
    data.dropna(inplace=True)
    return data


# Step 3: Trend and Seasonality Analysis
def analyze_seasonality(data, value_column, freq='D'):
    result = seasonal_decompose(data[value_column], model='additive', period=freq)
    result.plot()
    plt.show()


# Step 4: Forecasting with ARIMA
def forecast_with_arima(data, value_column, steps=5):
    model = ARIMA(data[value_column], order=(5, 1, 0))
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=steps)
    return forecast


# Example Usage
if __name__ == "__main__":
    # Example: Stock Price Dataset
    stock_data = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
        'Close Price': [150.10, 152.20, 151.00, 153.50]
    })

    # Preprocess data
    stock_data = preprocess_time_series(stock_data, 'Date', 'Close Price')

    # Feature engineering
    stock_data = create_lagged_features(stock_data, 'Close Price', lags=3)
    print("Lagged Features:\n", stock_data)

    # Analyze seasonality
    analyze_seasonality(stock_data, 'Close Price', freq=4)

    # Forecasting
    forecast = forecast_with_arima(stock_data, 'Close Price', steps=5)
    print("Forecasted Prices:\n", forecast)


comment = """
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Step 1: Data Preprocessing
def preprocess_time_series(data, date_column, value_column, freq='D'):
    data[date_column] = pd.to_datetime(data[date_column])
    data = data.set_index(date_column)
    data = data.asfreq(freq)
    data[value_column] = data[value_column].fillna(method='ffill')  # Forward fill missing values
    return data

# Step 2: Time-Series Feature Engineering
def create_lagged_features(data, value_column, lags=3):
    for lag in range(1, lags + 1):
        data[f'lag_{lag}'] = data[value_column].shift(lag)
    data.dropna(inplace=True)
    return data

# Step 3: Trend and Seasonality Analysis
def analyze_seasonality(data, value_column, freq='D'):
    result = seasonal_decompose(data[value_column], model='additive', period=freq)
    result.plot()
    plt.show()

# Step 4: Forecasting with ARIMA
def forecast_with_arima(data, value_column, steps=5):
    model = ARIMA(data[value_column], order=(5, 1, 0))
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=steps)
    return forecast

# Example Usage
if __name__ == "__main__":
    # Example: Stock Price Dataset
    stock_data = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
        'Close Price': [150.10, 152.20, 151.00, 153.50]
    })
    
    # Preprocess data
    stock_data = preprocess_time_series(stock_data, 'Date', 'Close Price')
    
    # Feature engineering
    stock_data = create_lagged_features(stock_data, 'Close Price', lags=3)
    print("Lagged Features:\n", stock_data)
    
    # Analyze seasonality
    analyze_seasonality(stock_data, 'Close Price', freq=4)
    
    # Forecasting
    forecast = forecast_with_arima(stock_data, 'Close Price', steps=5)
    print("Forecasted Prices:\n", forecast)

"""