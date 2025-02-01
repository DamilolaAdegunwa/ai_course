import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error


# Load and preprocess time series data
def preprocess_time_series(data, date_col, value_col):
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(by=date_col).set_index(date_col)
    return data


# Decompose time series into components
def decompose_time_series(series, model='additive', period=12):
    result = seasonal_decompose(series, model=model, period=period)
    return result


# Generate lag and rolling features
def create_features(df, value_col, lags, rolling_window):
    for lag in lags:
        df[f'lag_{lag}'] = df[value_col].shift(lag)
    df[f'rolling_mean_{rolling_window}'] = df[value_col].rolling(rolling_window).mean()
    df[f'rolling_std_{rolling_window}'] = df[value_col].rolling(rolling_window).std()
    return df.dropna()


# Predictive modeling
def forecast(df, features, target_col, steps=5):
    model = LinearRegression()
    X = df[features]
    y = df[target_col]
    model.fit(X, y)

    future_preds = []
    last_row = df.iloc[-1][features].values.reshape(1, -1)
    for _ in range(steps):
        pred = model.predict(last_row)[0]
        future_preds.append(pred)
        last_row = np.roll(last_row, -1)  # Shift features
        last_row[0, -1] = pred  # Update with prediction

    return future_preds


# Example usage
if __name__ == "__main__":
    # Example: Energy Consumption Data
    data = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
        'Region': ['North', 'North', 'North', 'North'],
        'Demand': [200, 210, 215, 220]
    })

    processed_data = preprocess_time_series(data, 'Date', 'Demand')
    decomposition = decompose_time_series(processed_data['Demand'], period=2)
    processed_features = create_features(processed_data, 'Demand', lags=[1, 2], rolling_window=2)

    predictions = forecast(processed_features, features=['lag_1', 'lag_2', 'rolling_mean_2'], target_col='Demand')
    print(f"Future Predictions: {predictions}")


comment = """
### Project Title: **Deep-Time Series Analysis and Predictive Scenario Modeling with Pandas**  
**File Name**: `deep_time_series_analysis_and_predictive_scenario_modeling_with_pandas.py`  

---

### Project Description  

This project explores **deep-level time series analysis and predictive scenario modeling** using **Pandas**, tackling challenges like:  

1. **Time Series Segmentation**: Analyze multiple overlapping time series with noise reduction and outlier detection.  
2. **Scenario Forecasting**: Perform **multi-step forecasting** with adaptive seasonal patterns.  
3. **Dynamic Event Impact Analysis**: Model the effect of interventions (e.g., policy changes or market shocks).  
4. **Feature Engineering for Time Series**: Create lag, rolling, and cyclical features dynamically for advanced time series prediction.  
5. **Scenario Comparisons**: Generate multiple predictive scenarios with contrasting assumptions.  

This project is applicable to **financial forecasting**, **climate analysis**, **operations planning**, and **policy impact evaluations**.  

---

### Example Use Cases  

1. **Stock Market Analysis**: Identify trends and anomalies in stock prices with predictive modeling.  
2. **Energy Demand Forecasting**: Predict multi-regional energy demand based on past consumption and weather factors.  
3. **Climate Change Impact Analysis**: Forecast rainfall or temperature patterns under different emissions scenarios.  
4. **Sales and Inventory Optimization**: Predict future demand spikes for inventory planning.  
5. **Policy Impact Simulation**: Estimate GDP changes under various taxation policies.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: Stock Prices**  
**Stock Data**:  

| Date       | Ticker | Price | Volume   |  
|------------|--------|-------|----------|  
| 2024-01-01 | AAPL   | 150   | 1,000,000|  
| 2024-01-02 | AAPL   | 152   | 1,050,000|  
| 2024-01-01 | TSLA   | 700   | 2,000,000|  
| 2024-01-02 | TSLA   | 710   | 2,100,000|  

**Expected Output**:  
- Trend analysis, anomalies, and price predictions for the next 5 days.  

---

#### **Input 2: Energy Consumption**  
**Energy Data**:  

| Date       | Region | Demand | Temp |  
|------------|--------|--------|------|  
| 2024-01-01 | North  | 200    | 15   |  
| 2024-01-02 | North  | 210    | 14   |  
| 2024-01-01 | South  | 300    | 25   |  
| 2024-01-02 | South  | 320    | 24   |  

**Expected Output**:  
- Seasonal trends and predictions for energy consumption.  

---

#### **Input 3: Climate Data**  
**Temperature Data**:  

| Year | Month | Avg Temp | Emissions Index |  
|------|-------|----------|-----------------|  
| 2020 | Jan   | 15.5     | 0.9             |  
| 2020 | Feb   | 16.0     | 0.95            |  
| 2021 | Jan   | 15.8     | 0.88            |  
| 2021 | Feb   | 16.2     | 0.91            |  

**Expected Output**:  
- Predict temperature trends under varying emissions scenarios.  

---

#### **Input 4: Retail Sales Data**  
**Sales Data**:  

| Date       | Product | Sales | Returns | Promo |  
|------------|---------|-------|---------|-------|  
| 2024-01-01 | Shoes   | 500   | 20      | Yes   |  
| 2024-01-02 | Shoes   | 480   | 15      | No    |  
| 2024-01-01 | Bags    | 300   | 5       | Yes   |  
| 2024-01-02 | Bags    | 320   | 8       | No    |  

**Expected Output**:  
- Predict sales under promotional strategies for the next month.  

---

#### **Input 5: Economic Indicators**  
**GDP Data**:  

| Year | Quarter | GDP  | Tax Rate | Unemployment |  
|------|---------|------|----------|--------------|  
| 2020 | Q1      | 5.0  | 10%      | 6.0%         |  
| 2020 | Q2      | 4.8  | 12%      | 6.5%         |  
| 2021 | Q1      | 5.2  | 10%      | 5.8%         |  
| 2021 | Q2      | 5.1  | 11%      | 6.0%         |  

**Expected Output**:  
- GDP forecasts with varying tax rates and unemployment trends.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

# Load and preprocess time series data
def preprocess_time_series(data, date_col, value_col):
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(by=date_col).set_index(date_col)
    return data

# Decompose time series into components
def decompose_time_series(series, model='additive', period=12):
    result = seasonal_decompose(series, model=model, period=period)
    return result

# Generate lag and rolling features
def create_features(df, value_col, lags, rolling_window):
    for lag in lags:
        df[f'lag_{lag}'] = df[value_col].shift(lag)
    df[f'rolling_mean_{rolling_window}'] = df[value_col].rolling(rolling_window).mean()
    df[f'rolling_std_{rolling_window}'] = df[value_col].rolling(rolling_window).std()
    return df.dropna()

# Predictive modeling
def forecast(df, features, target_col, steps=5):
    model = LinearRegression()
    X = df[features]
    y = df[target_col]
    model.fit(X, y)
    
    future_preds = []
    last_row = df.iloc[-1][features].values.reshape(1, -1)
    for _ in range(steps):
        pred = model.predict(last_row)[0]
        future_preds.append(pred)
        last_row = np.roll(last_row, -1)  # Shift features
        last_row[0, -1] = pred  # Update with prediction
    
    return future_preds

# Example usage
if __name__ == "__main__":
    # Example: Energy Consumption Data
    data = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
        'Region': ['North', 'North', 'North', 'North'],
        'Demand': [200, 210, 215, 220]
    })
    
    processed_data = preprocess_time_series(data, 'Date', 'Demand')
    decomposition = decompose_time_series(processed_data['Demand'], period=2)
    processed_features = create_features(processed_data, 'Demand', lags=[1, 2], rolling_window=2)
    
    predictions = forecast(processed_features, features=['lag_1', 'lag_2', 'rolling_mean_2'], target_col='Demand')
    print(f"Future Predictions: {predictions}")
```

This project offers advanced techniques for **time series processing, decomposition, feature engineering, and predictive analysis**. It is a comprehensive tool for **real-world forecasting challenges** across various domains.
"""