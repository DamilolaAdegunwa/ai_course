import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Load Dataset
def load_time_series_data(file_path, date_col, value_col):
    df = pd.read_csv(file_path, parse_dates=[date_col])
    df.set_index(date_col, inplace=True)
    return df[[value_col]].dropna()


# Preprocess Data: Fill Missing Values and Resample
def preprocess_time_series(df, freq='D'):
    df = df.resample(freq).mean()
    df.fillna(method='ffill', inplace=True)
    return df


# Decompose Time Series
def decompose_time_series(df, column, model='additive'):
    decomposition = seasonal_decompose(df[column], model=model)
    decomposition.plot()
    plt.show()
    return decomposition


# Detect Anomalies
def detect_anomalies(df, column, contamination=0.01):
    model = IsolationForest(contamination=contamination, random_state=42)
    df['Anomaly'] = model.fit_predict(df[[column]])
    return df


# Forecast Future Values using Linear Regression
def forecast_time_series(df, column, forecast_days=7):
    df['Time'] = np.arange(len(df))
    X = df[['Time']]
    y = df[column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    future_time = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
    predictions = model.predict(future_time)
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

    forecast_df = pd.DataFrame({column: predictions}, index=future_dates)
    return forecast_df


# Visualization: Plot Trends and Anomalies
def visualize_trends_and_anomalies(df, column):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=df.index, y=column, label='Observed')
    if 'Anomaly' in df:
        anomalies = df[df['Anomaly'] == -1]
        plt.scatter(anomalies.index, anomalies[column], color='red', label='Anomalies')
    plt.title(f"Trends and Anomalies in {column}")
    plt.legend()
    plt.show()


# Testing the Pipeline
if __name__ == "__main__":
    # File Path and Columns
    file_path = "titanic.csv"  # Replace with actual time-series file
    date_col = "Date"  # Replace with actual date column
    value_col = "Value"  # Replace with actual value column

    # Step 1: Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_time_series_data(file_path, date_col, value_col)
    df = preprocess_time_series(df)

    # Step 2: Decompose time series
    print("Decomposing time series...")
    decomposition = decompose_time_series(df, value_col)

    # Step 3: Detect anomalies
    print("Detecting anomalies...")
    df = detect_anomalies(df, value_col)
    visualize_trends_and_anomalies(df, value_col)

    # Step 4: Forecast future values
    print("Forecasting future values...")
    forecast = forecast_time_series(df, value_col)
    print("Future Forecast:")
    print(forecast)

    # Step 5: Visualize forecast
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=df.index, y=value_col, label='Observed')
    sns.lineplot(data=forecast, x=forecast.index, y=value_col, label='Forecast')
    plt.title("Time Series Forecasting")
    plt.legend()
    plt.show()


comment = """
### Project Title: **AI-Driven Time Series Forecasting and Anomaly Detection with Pandas**  
**File Name**: `ai_driven_time_series_forecasting_and_anomaly_detection_with_pandas.py`  

---

### Project Description  
This project combines **Pandas**, **machine learning**, and advanced statistical methods to analyze time-series data, detect anomalies, and forecast future trends. You’ll:  

1. Preprocess raw data into time-series format using **Pandas**.  
2. Apply rolling windows, seasonal decomposition, and trend analysis.  
3. Use machine learning models for time-series forecasting.  
4. Perform anomaly detection by identifying deviations from expected behavior.  
5. Visualize trends, anomalies, and forecasts using **Matplotlib** and **Seaborn**.  

The focus is on integrating **Pandas** with predictive modeling and advanced data transformations for real-world applications such as stock price prediction, weather forecasting, or monitoring system failures.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1**:  
**Dataset**: Time-series data of daily website traffic.  
**Task**: Forecast traffic for the next 7 days.  
**Expected Output**:  
- A line chart showing observed data and predicted traffic for 7 days.  

#### **Input 2**:  
**Dataset**: Time-series data of machine sensor readings.  
**Task**: Identify anomalies in the readings.  
**Expected Output**:  
- A scatter plot with anomalies marked.  

#### **Input 3**:  
**Dataset**: Time-series data of sales trends.  
**Task**: Decompose the data into seasonal, trend, and residual components.  
**Expected Output**:  
- Visualizations of each decomposed component.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Dataset
def load_time_series_data(file_path, date_col, value_col):
    df = pd.read_csv(file_path, parse_dates=[date_col])
    df.set_index(date_col, inplace=True)
    return df[[value_col]].dropna()

# Preprocess Data: Fill Missing Values and Resample
def preprocess_time_series(df, freq='D'):
    df = df.resample(freq).mean()
    df.fillna(method='ffill', inplace=True)
    return df

# Decompose Time Series
def decompose_time_series(df, column, model='additive'):
    decomposition = seasonal_decompose(df[column], model=model)
    decomposition.plot()
    plt.show()
    return decomposition

# Detect Anomalies
def detect_anomalies(df, column, contamination=0.01):
    model = IsolationForest(contamination=contamination, random_state=42)
    df['Anomaly'] = model.fit_predict(df[[column]])
    return df

# Forecast Future Values using Linear Regression
def forecast_time_series(df, column, forecast_days=7):
    df['Time'] = np.arange(len(df))
    X = df[['Time']]
    y = df[column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    future_time = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
    predictions = model.predict(future_time)
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    
    forecast_df = pd.DataFrame({column: predictions}, index=future_dates)
    return forecast_df

# Visualization: Plot Trends and Anomalies
def visualize_trends_and_anomalies(df, column):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=df.index, y=column, label='Observed')
    if 'Anomaly' in df:
        anomalies = df[df['Anomaly'] == -1]
        plt.scatter(anomalies.index, anomalies[column], color='red', label='Anomalies')
    plt.title(f"Trends and Anomalies in {column}")
    plt.legend()
    plt.show()

# Testing the Pipeline
if __name__ == "__main__":
    # File Path and Columns
    file_path = "titanic.csv"  # Replace with actual time-series file
    date_col = "Date"  # Replace with actual date column
    value_col = "Value"  # Replace with actual value column
    
    # Step 1: Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_time_series_data(file_path, date_col, value_col)
    df = preprocess_time_series(df)

    # Step 2: Decompose time series
    print("Decomposing time series...")
    decomposition = decompose_time_series(df, value_col)

    # Step 3: Detect anomalies
    print("Detecting anomalies...")
    df = detect_anomalies(df, value_col)
    visualize_trends_and_anomalies(df, value_col)

    # Step 4: Forecast future values
    print("Forecasting future values...")
    forecast = forecast_time_series(df, value_col)
    print("Future Forecast:")
    print(forecast)

    # Step 5: Visualize forecast
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=df.index, y=value_col, label='Observed')
    sns.lineplot(data=forecast, x=forecast.index, y=value_col, label='Forecast')
    plt.title("Time Series Forecasting")
    plt.legend()
    plt.show()
```

---

### Testing Scenarios  

#### **Scenario 1**:  
**Dataset**: Monthly stock prices.  
**Task**: Forecast stock prices for the next 6 months.  
**Expected Output**:  
- A table of predicted prices and a chart of observed vs. forecasted prices.  

#### **Scenario 2**:  
**Dataset**: Daily power usage.  
**Task**: Identify days with anomalous usage.  
**Expected Output**:  
- A chart showing anomalies as red points on the time-series plot.  

#### **Scenario 3**:  
**Dataset**: Weekly sales trends.  
**Task**: Decompose the trends into seasonal and residual components.  
**Expected Output**:  
- Subplots of seasonal, trend, and residual components.  

---

### Key Learnings  
- **Time Series Manipulation**: Resampling, missing value handling, and advanced analysis with Pandas.  
- **Anomaly Detection**: Use machine learning models like Isolation Forest for anomaly detection.  
- **Forecasting**: Predict future trends using regression techniques.  
- **Visualization**: Communicate insights through effective visualizations.  

Would you like to integrate more advanced forecasting models like ARIMA or LSTM?
"""