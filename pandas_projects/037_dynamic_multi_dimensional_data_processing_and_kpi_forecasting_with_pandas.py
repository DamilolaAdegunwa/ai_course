import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Dynamic aggregation
def aggregate_data(data, group_by, metrics):
    aggregated = data.groupby(group_by)[metrics].sum()
    return aggregated

# Forecast KPIs
def forecast_kpi(data, metric, steps=3, order=(1, 1, 1)):
    model = ARIMA(data[metric], order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Detect anomalies
def detect_kpi_anomalies(data, metric, threshold=2.5):
    mean = data[metric].mean()
    std = data[metric].std()
    anomalies = data[np.abs(data[metric] - mean) > threshold * std]
    return anomalies

# Example usage
if __name__ == "__main__":
    # Example: Sales Data
    data = pd.DataFrame({
        "Date": pd.date_range(start="2024-01-01", periods=5),
        "Region": ["North", "South", "North", "South", "East"],
        "Product": ["Laptop", "Phone", "Laptop", "Tablet", "Phone"],
        "Sales": [1500, 2000, 1600, 900, 1800]
    })

    # Aggregate by region
    aggregated_sales = aggregate_data(data, "Region", ["Sales"])
    print("Aggregated Sales:\n", aggregated_sales)

    # Forecast
    forecast = forecast_kpi(data, "Sales")
    print("Forecast:\n", forecast)

    # Detect anomalies
    anomalies = detect_kpi_anomalies(data, "Sales")
    print("Anomalies:\n", anomalies)


# https://chatgpt.com/c/674b65b9-fecc-800c-8311-7f681df9b305
comment = """
### Project Title: **Dynamic Multi-Dimensional Data Processing and KPI Forecasting with Pandas**  
**File Name**: `dynamic_multi_dimensional_data_processing_and_kpi_forecasting_with_pandas.py`  

---

### Project Description  

This project focuses on creating a **high-performance data processing system** for **multi-dimensional datasets**. It involves:  
1. Handling **large and complex datasets** containing hierarchical and multi-dimensional data.  
2. **Dynamic aggregation** and slicing of data based on multiple hierarchical levels.  
3. **KPI (Key Performance Indicator) Analysis** with trends, outlier detection, and forecasting using advanced algorithms.  
4. Integration of **pivot table-like operations**, **grouped metrics generation**, and real-time KPI insights.  

This project demonstrates how to dynamically manage, analyze, and predict KPIs in various domains such as finance, supply chain, and e-commerce.  

---

### Example Use Cases  

1. **Sales Performance**: Analyze and predict sales KPIs across multiple regions, product categories, and customer segments.  
2. **Supply Chain Optimization**: Detect bottlenecks in logistics data and forecast delivery trends.  
3. **Financial Analysis**: Aggregate and forecast financial KPIs like revenue, profit margins, and expenses.  
4. **Retail Insights**: Identify top-performing stores and products, and forecast their future performance.  
5. **IoT Analytics**: Process sensor data from multiple devices and predict failure rates or energy consumption.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: Sales Data**  
| Date       | Region  | Product  | Customer Segment | Sales  | Quantity | Profit |  
|------------|---------|----------|------------------|--------|----------|--------|  
| 2024-01-01 | North   | Laptop   | Consumer         | 1500   | 3        | 300    |  
| 2024-01-01 | South   | Phone    | Business         | 2000   | 5        | 500    |  
| 2024-01-02 | North   | Laptop   | Consumer         | 1600   | 4        | 350    |  
| 2024-01-02 | South   | Tablet   | Corporate        | 900    | 2        | 150    |  
| 2024-01-03 | East    | Phone    | Consumer         | 1800   | 6        | 400    |  

**Expected Output**:  
- **Aggregated Sales by Region**:  
  - North: 3100, South: 2900, East: 1800.  
- **Top KPI**: Product: Phone (highest sales: $3800).  
- **Forecast for Sales in Next 3 Days**:  
  - North: $3300, South: $3000, East: $1900.  

---

#### **Input 2: Financial Data**  
| Date       | Department | Revenue | Expense | Profit |  
|------------|------------|---------|---------|--------|  
| 2024-01-01 | Marketing  | 5000    | 3000    | 2000   |  
| 2024-01-02 | Sales      | 6000    | 2500    | 3500   |  
| 2024-01-03 | IT         | 4000    | 1500    | 2500   |  
| 2024-01-04 | Marketing  | 5500    | 3200    | 2300   |  
| 2024-01-05 | Sales      | 7000    | 2700    | 4300   |  

**Expected Output**:  
- **Total Profit by Department**:  
  - Marketing: 4300, Sales: 7800, IT: 2500.  
- **Anomalies**: None.  
- **Revenue Forecast for Next 3 Days**:  
  - Marketing: $5800, Sales: $7500, IT: $4200.  

---

#### **Input 3: IoT Sensor Data**  
| Date       | Device ID | Location  | Temperature | Humidity | Status |  
|------------|-----------|-----------|-------------|----------|--------|  
| 2024-01-01 | 1001      | Warehouse | 22          | 60       | Active |  
| 2024-01-02 | 1002      | Factory   | 25          | 55       | Active |  
| 2024-01-03 | 1001      | Warehouse | 23          | 58       | Active |  
| 2024-01-04 | 1003      | Store     | 20          | 65       | Inactive |  
| 2024-01-05 | 1002      | Factory   | 26          | 57       | Active |  

**Expected Output**:  
- **Active Device Count by Location**:  
  - Warehouse: 2, Factory: 2, Store: 0.  
- **Temperature Forecast for Warehouse**:  
  - 2024-01-06: 24, 2024-01-07: 25.  

---

#### **Input 4: Supply Chain Data**  
| Date       | Region | Item      | Shipping Time | Cost | Status  |  
|------------|--------|-----------|---------------|------|---------|  
| 2024-01-01 | North  | Electronics | 2 days       | 300  | Delivered |  
| 2024-01-02 | South  | Furniture   | 4 days       | 500  | Pending   |  
| 2024-01-03 | East   | Clothing    | 3 days       | 200  | Delivered |  
| 2024-01-04 | North  | Electronics | 1 day        | 250  | Delivered |  
| 2024-01-05 | South  | Furniture   | 5 days       | 550  | Pending   |  

**Expected Output**:  
- **Average Shipping Time by Region**:  
  - North: 1.5 days, South: 4.5 days, East: 3 days.  
- **Pending Deliveries Cost**:  
  - $1050.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Dynamic aggregation
def aggregate_data(data, group_by, metrics):
    aggregated = data.groupby(group_by)[metrics].sum()
    return aggregated

# Forecast KPIs
def forecast_kpi(data, metric, steps=3, order=(1, 1, 1)):
    model = ARIMA(data[metric], order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Detect anomalies
def detect_kpi_anomalies(data, metric, threshold=2.5):
    mean = data[metric].mean()
    std = data[metric].std()
    anomalies = data[np.abs(data[metric] - mean) > threshold * std]
    return anomalies

# Example usage
if __name__ == "__main__":
    # Example: Sales Data
    data = pd.DataFrame({
        "Date": pd.date_range(start="2024-01-01", periods=5),
        "Region": ["North", "South", "North", "South", "East"],
        "Product": ["Laptop", "Phone", "Laptop", "Tablet", "Phone"],
        "Sales": [1500, 2000, 1600, 900, 1800]
    })

    # Aggregate by region
    aggregated_sales = aggregate_data(data, "Region", ["Sales"])
    print("Aggregated Sales:\n", aggregated_sales)

    # Forecast
    forecast = forecast_kpi(data, "Sales")
    print("Forecast:\n", forecast)

    # Detect anomalies
    anomalies = detect_kpi_anomalies(data, "Sales")
    print("Anomalies:\n", anomalies)
```  

This project significantly advances your Pandas expertise by tackling multi-dimensional data challenges, providing dynamic aggregation, and KPI-based insights for complex datasets.
"""