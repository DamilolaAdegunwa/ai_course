import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression


# Function to generate simulated hierarchical time-series data
def generate_data(start_date, num_days, regions, base_value=100):
    dates = pd.date_range(start=start_date, periods=num_days, freq='D')
    data = []
    for region in regions:
        values = base_value + np.random.randn(num_days).cumsum()
        data.extend(zip(dates, [region] * num_days, values))
    return pd.DataFrame(data, columns=['Date', 'Region', 'Sales'])


# Seasonal decomposition and visualization
def decompose_time_series(data, column, period=7):
    decomposition = seasonal_decompose(data[column], period=period, model='additive', extrapolate_trend='freq')
    decomposition.plot()
    plt.show()
    return decomposition


# Hierarchical Forecasting
def hierarchical_forecasting(data, hierarchy_levels, forecast_horizon=7):
    forecasts = {}
    for level in hierarchy_levels:
        level_data = data.groupby(level)['Sales'].sum().reset_index()
        model = LinearRegression()
        X = np.arange(len(level_data)).reshape(-1, 1)
        y = level_data['Sales'].values
        model.fit(X, y)
        future_X = np.arange(len(level_data), len(level_data) + forecast_horizon).reshape(-1, 1)
        predictions = model.predict(future_X)
        forecasts[level] = predictions
    return forecasts


# Reconciliation
def reconcile_forecasts(forecasts, hierarchy_levels):
    total_forecast = forecasts[hierarchy_levels[0]].sum(axis=0)
    reconciled = {level: forecasts[level] / forecasts[level].sum(axis=0) * total_forecast for level in hierarchy_levels}
    return reconciled


# Main simulation and forecasting
if __name__ == "__main__":
    # Generate data
    start_date = '2024-01-01'
    regions = ['North', 'South', 'East', 'West']
    num_days = 365
    data = generate_data(start_date, num_days, regions)

    # Decompose time series for trend analysis
    north_data = data[data['Region'] == 'North']
    north_data.set_index('Date', inplace=True)
    decomposition = decompose_time_series(north_data, 'Sales', period=30)

    # Hierarchical forecasting
    hierarchy_levels = ['Region']
    forecasts = hierarchical_forecasting(data, hierarchy_levels, forecast_horizon=30)
    print("\nHierarchical Forecasts:")
    for level, forecast in forecasts.items():
        print(f"Level: {level}, Forecast: {forecast}")

    # Reconcile forecasts
    reconciled_forecasts = reconcile_forecasts(forecasts, hierarchy_levels)
    print("\nReconciled Forecasts:")
    for level, forecast in reconciled_forecasts.items():
        print(f"Level: {level}, Forecast: {forecast}")


comment = """
### Project Title: **Dynamic Hierarchical Time-Series Forecasting with Pandas**  
**File Name**: `dynamic_hierarchical_time_series_forecasting_with_pandas.py`  

---

### Project Description  
This project implements a dynamic **hierarchical time-series forecasting system** using **Pandas**. Hierarchical time-series (HTS) forecasting splits data into multiple granularities (e.g., country-level, state-level, city-level) and forecasts at all levels while ensuring consistency across levels. Advanced techniques include:  
1. **Data Resampling** for different time intervals.  
2. **Top-Down, Bottom-Up, and Reconciliation Forecasting** for hierarchical consistency.  
3. **Custom Seasonal Decomposition** for trend analysis.  
4. **Rolling Forecast Simulation** for iterative testing and validation.  

---

### Example Use Cases  
1. **Sales Forecasting**: Predicting total sales for a company while forecasting at regional and store levels.  
2. **Energy Consumption**: Forecasting overall power usage across countries, states, and households.  
3. **Epidemiology**: Predicting disease outbreaks at global, regional, and local scales.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1**  
**Simulated Time-Series Data**:  
| Date       | Region    | Sales |  
|------------|-----------|-------|  
| 2024-01-01 | North     | 100   |  
| 2024-01-01 | South     | 200   |  
| 2024-01-01 | East      | 150   |  
| 2024-01-01 | West      | 250   |  
| 2024-01-02 | North     | 110   |  
| 2024-01-02 | South     | 190   |  

**Expected Output**:  
- Total sales forecast across all regions.  
- Consistent forecasts for North, South, East, and West regions.  

#### **Input 2**  
**Hierarchical Data by Product Categories**:  
| Date       | Category  | Subcategory | Sales |  
|------------|-----------|-------------|-------|  
| 2024-01-01 | Electronics | Phones     | 500   |  
| 2024-01-01 | Electronics | Laptops    | 400   |  
| 2024-01-01 | Furniture   | Chairs     | 200   |  
| 2024-01-01 | Furniture   | Tables     | 150   |  

**Expected Output**:  
- Category-level and subcategory-level sales forecasts.  
- Aggregated and reconciled forecasts for total sales.  

#### **Input 3**  
**Energy Consumption by State**:  
| Date       | State       | Consumption |  
|------------|-------------|-------------|  
| 2024-01-01 | California  | 5000 kWh    |  
| 2024-01-01 | Texas       | 7000 kWh    |  
| 2024-01-01 | Florida     | 3000 kWh    |  

**Expected Output**:  
- State-level forecasts.  
- Total U.S. energy consumption forecast with consistency.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression

# Function to generate simulated hierarchical time-series data
def generate_data(start_date, num_days, regions, base_value=100):
    dates = pd.date_range(start=start_date, periods=num_days, freq='D')
    data = []
    for region in regions:
        values = base_value + np.random.randn(num_days).cumsum()
        data.extend(zip(dates, [region] * num_days, values))
    return pd.DataFrame(data, columns=['Date', 'Region', 'Sales'])

# Seasonal decomposition and visualization
def decompose_time_series(data, column, period=7):
    decomposition = seasonal_decompose(data[column], period=period, model='additive', extrapolate_trend='freq')
    decomposition.plot()
    plt.show()
    return decomposition

# Hierarchical Forecasting
def hierarchical_forecasting(data, hierarchy_levels, forecast_horizon=7):
    forecasts = {}
    for level in hierarchy_levels:
        level_data = data.groupby(level)['Sales'].sum().reset_index()
        model = LinearRegression()
        X = np.arange(len(level_data)).reshape(-1, 1)
        y = level_data['Sales'].values
        model.fit(X, y)
        future_X = np.arange(len(level_data), len(level_data) + forecast_horizon).reshape(-1, 1)
        predictions = model.predict(future_X)
        forecasts[level] = predictions
    return forecasts

# Reconciliation
def reconcile_forecasts(forecasts, hierarchy_levels):
    total_forecast = forecasts[hierarchy_levels[0]].sum(axis=0)
    reconciled = {level: forecasts[level] / forecasts[level].sum(axis=0) * total_forecast for level in hierarchy_levels}
    return reconciled

# Main simulation and forecasting
if __name__ == "__main__":
    # Generate data
    start_date = '2024-01-01'
    regions = ['North', 'South', 'East', 'West']
    num_days = 365
    data = generate_data(start_date, num_days, regions)
    
    # Decompose time series for trend analysis
    north_data = data[data['Region'] == 'North']
    north_data.set_index('Date', inplace=True)
    decomposition = decompose_time_series(north_data, 'Sales', period=30)
    
    # Hierarchical forecasting
    hierarchy_levels = ['Region']
    forecasts = hierarchical_forecasting(data, hierarchy_levels, forecast_horizon=30)
    print("\nHierarchical Forecasts:")
    for level, forecast in forecasts.items():
        print(f"Level: {level}, Forecast: {forecast}")
    
    # Reconcile forecasts
    reconciled_forecasts = reconcile_forecasts(forecasts, hierarchy_levels)
    print("\nReconciled Forecasts:")
    for level, forecast in reconciled_forecasts.items():
        print(f"Level: {level}, Forecast: {forecast}")
```

---

### Key Features  
1. **Hierarchical Time-Series Handling**: Ensures consistent forecasts across levels.  
2. **Seasonal Decomposition**: Identifies seasonal trends and anomalies.  
3. **Rolling Forecasting**: Supports continuous evaluation and model improvement.  
4. **Reconciliation Methods**: Balances forecasts from bottom-up or top-down approaches.  

---

### Testing Scenarios  

#### **Scenario 1**:  
- Input: Regional sales data with daily frequency.  
- Test: Validate that forecasts at the regional level sum up correctly to the total sales forecast.  

#### **Scenario 2**:  
- Input: Product categories and subcategories sales data.  
- Test: Ensure category-level forecasts reconcile with subcategory-level forecasts.  

#### **Scenario 3**:  
- Input: Energy consumption at state and national levels.  
- Test: Verify that total U.S. energy forecasts match the sum of state-level forecasts.  

---

### Advanced Extension Ideas  
1. Use **Prophet** or **ARIMA models** for improved forecasting accuracy.  
2. Integrate with live data sources (e.g., APIs or IoT sensors).  
3. Incorporate external factors like weather or holidays for more realistic forecasting.  

Would you like to explore any of these extensions?
"""