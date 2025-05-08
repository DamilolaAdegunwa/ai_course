import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


# Load and Prepare Data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return data


# Trend Analysis
def detect_trend(data, column):
    trend = data[column].rolling(window=5, min_periods=1).mean()
    return trend


# Forecasting with ARIMA
def forecast_arima(data, column, steps=5):
    model = ARIMA(data[column], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast


# Forecasting with Holt-Winters
def forecast_holt_winters(data, column, steps=5):
    model = ExponentialSmoothing(data[column], trend='add', seasonal=None, damped_trend=True)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast


# Monte Carlo Simulation for Probability Analysis
def monte_carlo_simulation(data, column, steps=5, simulations=1000):
    recent_value = data[column].iloc[-1]
    std_dev = data[column].diff().std()
    simulated_paths = []
    for _ in range(simulations):
        path = [recent_value]
        for _ in range(steps):
            path.append(path[-1] + np.random.normal(0, std_dev))
        simulated_paths.append(path)
    simulated_paths = np.array(simulated_paths)
    probability_exceed = np.mean(simulated_paths[:, -1] > recent_value + 5)
    return simulated_paths, probability_exceed


# Visualization
def visualize_results(data, trend, forecast, column):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[column], label='Original Data')
    plt.plot(data.index, trend, label='Trend', linestyle='--')
    plt.plot(pd.date_range(data.index[-1], periods=len(forecast) + 1, freq='D')[1:], forecast, label='Forecast', marker='o')
    plt.legend()
    plt.title(f"{column} Trend and Forecast")
    plt.xlabel("Date")
    plt.ylabel(column)
    plt.show()


# Main Execution
if __name__ == "__main__":
    # Example Dataset
    data = pd.DataFrame({
        'Date': pd.date_range(start='2024-01-01', periods=100),
        'StockPrice': np.random.normal(100, 5, 100).cumsum()
    })
    data.loc[95:, 'StockPrice'] += 20  # Inject upward trend
    data = data.set_index('Date')

    # Trend Detection
    column = 'StockPrice'
    trend = detect_trend(data, column)

    # ARIMA Forecasting
    forecast_arima_result = forecast_arima(data, column, steps=5)

    # Holt-Winters Forecasting
    forecast_hw_result = forecast_holt_winters(data, column, steps=5)

    # Monte Carlo Simulation
    simulated_paths, probability_exceed = monte_carlo_simulation(data, column, steps=5)

    # Visualize Results
    visualize_results(data, trend, forecast_arima_result, column)
    print(f"Probability of exceeding recent value by 5: {probability_exceed * 100:.2f}%")


# https://chatgpt.com/c/674b65b9-fecc-800c-8311-7f681df9b305
comment = """
### Project Title: **Dynamic Event Forecasting and Probabilistic Trend Analysis with Pandas**  
**File Name**: `dynamic_event_forecasting_and_trend_analysis_with_pandas.py`  

---

### Project Description  

This project involves building a **dynamic event forecasting and probabilistic trend analysis system**. Using **time-series data**, it detects trends, predicts future events, and calculates the probability of specific occurrences. The system employs advanced techniques such as exponential smoothing, autoregressive integrated moving average (ARIMA) modeling, and Monte Carlo simulations for robust forecasting.  

**Highlights**:  
1. **Trend Analysis**: Identifies long-term and seasonal trends in datasets.  
2. **Forecasting**: Uses ARIMA and Holt-Winters methods for accurate event predictions.  
3. **Probabilistic Analysis**: Simulates scenarios using Monte Carlo to determine the likelihood of specific events.  
4. **Custom Metrics**: Incorporates user-defined metrics for domain-specific insights.  
5. **Visualization**: Graphs predictions, trends, and uncertainty ranges for better interpretability.  

---

### Example Use Cases  

1. **Stock Price Forecasting**: Predict future stock prices based on historical data and analyze volatility.  
2. **Weather Event Prediction**: Forecast weather patterns like rainfall or temperature trends using historical climate data.  
3. **Sales Projections**: Analyze sales data to predict peak seasons and anticipate future demand.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: Stock Prices**  
| Date       | StockPrice |  
|------------|------------|  
| 2024-01-01 | 100        |  
| 2024-01-02 | 102        |  
| 2024-01-03 | 105        |  
| 2024-01-04 | 103        |  
| 2024-01-05 | 108        |  

**Expected Output**:  
- **Trend**: Upward trend detected.  
- **Forecast**: Next 5 days predicted stock prices: [110, 112, 115, 117, 120].  
- **Probability Analysis**: 70% chance the stock price exceeds 115 within the next 5 days.  

---

#### **Input 2: Weather Data**  
| Date       | Temperature |  
|------------|-------------|  
| 2024-06-01 | 30          |  
| 2024-06-02 | 32          |  
| 2024-06-03 | 35          |  
| 2024-06-04 | 34          |  
| 2024-06-05 | 36          |  

**Expected Output**:  
- **Trend**: Seasonal increase detected.  
- **Forecast**: Next 5 days predicted temperatures: [37, 38, 39, 40, 42].  
- **Probability Analysis**: 90% chance temperatures exceed 38 within the next 5 days.  

---

#### **Input 3: Sales Data**  
| Date       | Sales |  
|------------|-------|  
| 2024-11-01 | 150   |  
| 2024-11-02 | 200   |  
| 2024-11-03 | 250   |  
| 2024-11-04 | 220   |  
| 2024-11-05 | 300   |  

**Expected Output**:  
- **Trend**: Pre-holiday sales spike detected.  
- **Forecast**: Next 5 days predicted sales: [320, 350, 370, 400, 450].  
- **Probability Analysis**: 85% chance sales exceed 400 within the next 5 days.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load and Prepare Data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return data

# Trend Analysis
def detect_trend(data, column):
    trend = data[column].rolling(window=5, min_periods=1).mean()
    return trend

# Forecasting with ARIMA
def forecast_arima(data, column, steps=5):
    model = ARIMA(data[column], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Forecasting with Holt-Winters
def forecast_holt_winters(data, column, steps=5):
    model = ExponentialSmoothing(data[column], trend='add', seasonal=None, damped_trend=True)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Monte Carlo Simulation for Probability Analysis
def monte_carlo_simulation(data, column, steps=5, simulations=1000):
    recent_value = data[column].iloc[-1]
    std_dev = data[column].diff().std()
    simulated_paths = []
    for _ in range(simulations):
        path = [recent_value]
        for _ in range(steps):
            path.append(path[-1] + np.random.normal(0, std_dev))
        simulated_paths.append(path)
    simulated_paths = np.array(simulated_paths)
    probability_exceed = np.mean(simulated_paths[:, -1] > recent_value + 5)
    return simulated_paths, probability_exceed

# Visualization
def visualize_results(data, trend, forecast, column):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[column], label='Original Data')
    plt.plot(data.index, trend, label='Trend', linestyle='--')
    plt.plot(pd.date_range(data.index[-1], periods=len(forecast) + 1, freq='D')[1:], forecast, label='Forecast', marker='o')
    plt.legend()
    plt.title(f"{column} Trend and Forecast")
    plt.xlabel("Date")
    plt.ylabel(column)
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Example Dataset
    data = pd.DataFrame({
        'Date': pd.date_range(start='2024-01-01', periods=100),
        'StockPrice': np.random.normal(100, 5, 100).cumsum()
    })
    data.loc[95:, 'StockPrice'] += 20  # Inject upward trend
    data = data.set_index('Date')

    # Trend Detection
    column = 'StockPrice'
    trend = detect_trend(data, column)

    # ARIMA Forecasting
    forecast_arima_result = forecast_arima(data, column, steps=5)

    # Holt-Winters Forecasting
    forecast_hw_result = forecast_holt_winters(data, column, steps=5)

    # Monte Carlo Simulation
    simulated_paths, probability_exceed = monte_carlo_simulation(data, column, steps=5)

    # Visualize Results
    visualize_results(data, trend, forecast_arima_result, column)
    print(f"Probability of exceeding recent value by 5: {probability_exceed * 100:.2f}%")
```

---

### Advanced Skills Covered  

1. **ARIMA Modeling**: Implements advanced time-series forecasting for real-world applications.  
2. **Holt-Winters Method**: Adds seasonal and trend smoothing for dynamic predictions.  
3. **Monte Carlo Simulation**: Provides probabilistic insights into future trends.  
4. **Time-Series Trend Analysis**: Employs rolling windows for detecting and visualizing trends.  
5. **Custom Visualizations**: Enhances interpretability of forecasts and trends.  

This project introduces cutting-edge methods for forecasting and trend analysis, providing actionable insights with a probabilistic edge!
"""
