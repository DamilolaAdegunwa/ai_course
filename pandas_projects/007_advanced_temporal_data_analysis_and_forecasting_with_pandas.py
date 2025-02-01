import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Load the Air Quality dataset
file_path = "air_quality_no2.csv"
df = pd.read_csv(file_path, parse_dates=["datetime"], index_col="datetime")


# Example Function 1: Temporal Feature Extraction
def extract_temporal_features(df):
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day_of_week"] = df.index.day_name()
    df["hour"] = df.index.hour
    return df


# Example Function 2: Aggregating Data
def aggregate_by_day_of_week(df, column):
    return df.groupby("day_of_week")[column].mean().sort_index()


# Example Function 3: Rolling Statistics
def apply_rolling_mean(df, column, window):
    df[f"{column}_RollingMean_{window}"] = df[column].rolling(window=window).mean()
    return df


# Example Function 4: Seasonal Decomposition
def decompose_time_series(df, column, freq):
    decomposition = seasonal_decompose(df[column].dropna(), model='additive', period=freq)
    decomposition.plot()
    plt.show()
    return decomposition.trend, decomposition.seasonal, decomposition.resid


# Example Function 5: Naive Forecasting
def naive_forecast(df, column, steps):
    forecast = [df[column].iloc[-1]] * steps
    return forecast


# Test the pipeline
if __name__ == "__main__":
    # Step 1: Temporal Feature Extraction
    df = extract_temporal_features(df)
    print("Temporal features added:")
    print(df.head())

    # Step 2: Aggregate NO2 by Day of the Week
    weekly_avg = aggregate_by_day_of_week(df, "NO2")
    print("Weekly Average NO2 Concentrations:")
    print(weekly_avg)

    # Step 3: Apply Rolling Mean
    df = apply_rolling_mean(df, "NO2", window=7)
    print("Rolling Mean applied. Sample data:")
    print(df.head(10))

    # Step 4: Decompose Time Series
    print("Decomposing the time series...")
    trend, seasonal, resid = decompose_time_series(df, "NO2", freq=365)
    print("Trend Sample:")
    print(trend.head())

    # Step 5: Naive Forecast
    print("Naive Forecast for 7 steps ahead:")
    forecast = naive_forecast(df, "NO2", steps=7)
    print(forecast)


# https://chatgpt.com/c/674b65b9-fecc-800c-8311-7f681df9b305 (all pandas projects)
# https://chatgpt.com/c/67681655-c04c-800c-a8e9-bea0cacfa327 (pandas project 007)
comment = """
### Project Title: Advanced Temporal Data Analysis and Forecasting with Pandas  
**File Name**: `advanced_temporal_data_analysis_and_forecasting_with_pandas.py`  

---

### Project Description  
This project takes you deeper into working with time-series datasets using **Pandas**. The goal is to explore advanced techniques such as:  

1. Temporal feature extraction (e.g., month, day of the week, is_holiday).  
2. Seasonal decomposition to identify trends, seasonality, and residuals.  
3. Rolling window calculations and exponential weighted statistics for trend smoothing.  
4. Generating predictions using ARIMA-like naive models.  

The dataset can be any time-series data (e.g., stock prices, air quality data). We'll demonstrate this using the **Air Quality dataset** you uploaded.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1**:  
**Dataset**: Air Quality dataset.  
**Task**: Extract temporal features and aggregate NO2 concentrations by day of the week.  
**Expected Output**:  
- A summary table showing average NO2 concentrations for each day of the week.  

#### **Input 2**:  
**Dataset**: Air Quality dataset.  
**Task**: Perform rolling mean with a window of 7 days to smooth NO2 levels.  
**Expected Output**:  
- A new column `NO2_RollingMean_7` with smoothed values.  

#### **Input 3**:  
**Dataset**: Air Quality dataset.  
**Task**: Decompose the time series into trend, seasonality, and residuals.  
**Expected Output**:  
- Decomposed components plotted and returned as separate DataFrames.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Load the Air Quality dataset
file_path = "air_quality_no2.csv"
df = pd.read_csv(file_path, parse_dates=["datetime"], index_col="datetime")

# Example Function 1: Temporal Feature Extraction
def extract_temporal_features(df):
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day_of_week"] = df.index.day_name()
    df["hour"] = df.index.hour
    return df

# Example Function 2: Aggregating Data
def aggregate_by_day_of_week(df, column):
    return df.groupby("day_of_week")[column].mean().sort_index()

# Example Function 3: Rolling Statistics
def apply_rolling_mean(df, column, window):
    df[f"{column}_RollingMean_{window}"] = df[column].rolling(window=window).mean()
    return df

# Example Function 4: Seasonal Decomposition
def decompose_time_series(df, column, freq):
    decomposition = seasonal_decompose(df[column].dropna(), model='additive', period=freq)
    decomposition.plot()
    plt.show()
    return decomposition.trend, decomposition.seasonal, decomposition.resid

# Example Function 5: Naive Forecasting
def naive_forecast(df, column, steps):
    forecast = [df[column].iloc[-1]] * steps
    return forecast

# Test the pipeline
if __name__ == "__main__":
    # Step 1: Temporal Feature Extraction
    df = extract_temporal_features(df)
    print("Temporal features added:")
    print(df.head())

    # Step 2: Aggregate NO2 by Day of the Week
    weekly_avg = aggregate_by_day_of_week(df, "NO2")
    print("Weekly Average NO2 Concentrations:")
    print(weekly_avg)

    # Step 3: Apply Rolling Mean
    df = apply_rolling_mean(df, "NO2", window=7)
    print("Rolling Mean applied. Sample data:")
    print(df.head(10))

    # Step 4: Decompose Time Series
    print("Decomposing the time series...")
    trend, seasonal, resid = decompose_time_series(df, "NO2", freq=365)
    print("Trend Sample:")
    print(trend.head())
    
    # Step 5: Naive Forecast
    print("Naive Forecast for 7 steps ahead:")
    forecast = naive_forecast(df, "NO2", steps=7)
    print(forecast)
```

---

### Testing Scenarios  

#### **Scenario 1**:  
**Dataset**: Air Quality dataset.  
**Task**: Extract temporal features and group by month.  
**Expected Output**:  
- Columns for year, month, day of the week, and hour, with aggregated data grouped by month.  

#### **Scenario 2**:  
**Dataset**: Air Quality dataset.  
**Task**: Apply a 14-day rolling mean to NO2 concentrations.  
**Expected Output**:  
- A column `NO2_RollingMean_14` containing smoothed values for better trend visibility.  

#### **Scenario 3**:  
**Dataset**: Air Quality dataset.  
**Task**: Decompose NO2 levels into trend, seasonality, and residuals.  
**Expected Output**:  
- Plots and dataframes for each component of the time series.  

---

### Key Learnings  
- **Time-Series Analysis**: Learn advanced techniques for analyzing temporal datasets.  
- **Rolling Statistics**: Smoothing data to identify trends.  
- **Seasonal Decomposition**: Breaking time series into meaningful components.  
- **Naive Forecasting**: Building a simple yet effective forecasting model.  

Let me know if you'd like additional challenges or insights into specific steps!
"""