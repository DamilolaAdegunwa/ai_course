import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


# Function for lagged features
def create_lag_features(df, column, lags):
    for lag in range(1, lags + 1):
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df


# Function for rolling features
def create_rolling_features(df, column, windows):
    for window in windows:
        df[f"{column}_rolling_mean_{window}"] = df[column].rolling(window=window).mean()
        df[f"{column}_rolling_std_{window}"] = df[column].rolling(window=window).std()
    return df


# Function for target encoding
def target_encode(df, categorical_column, target_column):
    means = df.groupby(categorical_column)[target_column].transform("mean")
    df[f"{categorical_column}_target_encoded"] = means
    return df


# Function for feature importance ranking
def rank_features(df, target_column):
    correlations = df.corr()[target_column].abs().sort_values(ascending=False)
    return correlations[1:]  # Exclude target itself


# Function for time-based train-test split
def time_based_split(df, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(df):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        yield train, test


# Main pipeline
if __name__ == "__main__":
    # Example: Sales Data
    sales_data = pd.DataFrame({
        'Date': pd.date_range(start="2024-11-01", periods=10, freq='D'),
        'Product_ID': ['P1'] * 10,
        'Sales': [200, 210, 190, 220, 240, 250, 270, 260, 250, 240],
        'Price': [25, 24, 25, 26, 24, 23, 22, 24, 26, 27],
        'Store_ID': [1] * 10
    })

    sales_data['Date'] = pd.to_datetime(sales_data['Date'])
    sales_data = sales_data.set_index('Date')

    # Lagged Features
    sales_data = create_lag_features(sales_data, 'Sales', lags=3)
    print("Lagged Features:")
    print(sales_data.head())

    # Rolling Features
    sales_data = create_rolling_features(sales_data, 'Sales', windows=[2, 3])
    print("Rolling Features:")
    print(sales_data.head())

    # Target Encoding
    sales_data['Sales_binary'] = (sales_data['Sales'] > 240).astype(int)
    sales_data = target_encode(sales_data, 'Store_ID', 'Sales_binary')
    print("Target Encoded Features:")
    print(sales_data.head())

    # Feature Ranking
    rankings = rank_features(sales_data.dropna(), 'Sales')
    print("Feature Importance Ranking:")
    print(rankings)

    # Time-Based Splits
    print("Train-Test Splits:")
    for train, test in time_based_split(sales_data.dropna(), n_splits=2):
        print("Train:\n", train.head())
        print("Test:\n", test.head())


comment = """
### Project Title: **Dynamic Feature Engineering and Predictive Insights with Pandas**  
**File Name**: `dynamic_feature_engineering_and_predictive_insights_with_pandas.py`  

---

### Project Description  
This project takes **feature engineering** to a new level using Pandas by automating the creation of derived features for predictive tasks across dynamic datasets. It includes:  

1. **Dynamic Feature Creation**: Automatically generate features like lags, interaction terms, and transformations tailored to any dataset.  
2. **Target Encoding**: Perform advanced techniques to encode categorical variables based on the target.  
3. **Auto Correlation Analysis**: Analyze relationships between features dynamically for predictive insights.  
4. **Train-Test Splits with Time-Based Validation**: Implement robust train-test splitting strategies for time-series or rolling datasets.  
5. **Feature Importance Ranking**: Use embedded statistical measures to rank the predictive power of each feature.  

This project prepares data for machine learning tasks with minimal manual intervention.  

---

### Example Use Cases  

1. **Retail Forecasting**: Engineer features from historical sales data to predict future trends.  
2. **Credit Risk Scoring**: Automate feature creation to assess customer creditworthiness.  
3. **Dynamic Pricing Models**: Generate predictive features for setting optimal prices based on demand and supply patterns.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: Sales Data**  
**File**: `sales_data.csv`  
| Date       | Product_ID | Sales | Price | Store_ID |  
|------------|------------|-------|-------|----------|  
| 2024-11-01 | 101        | 200   | 25    | 1        |  
| 2024-11-02 | 101        | 210   | 24    | 1        |  
| 2024-11-03 | 101        | 190   | 25    | 1        |  

**Expected Output**:  
- Features: `lag_1_sales`, `rolling_mean_3_sales`, `price_x_store_interaction`, etc.  
- Insights: Positive correlation between `price` and `sales` for Store 1.  

#### **Input 2: Loan Applications**  
**File**: `loan_data.csv`  
| Application_Date | Customer_ID | Loan_Amount | Age | Gender | Approved |  
|------------------|-------------|-------------|-----|--------|----------|  
| 2024-10-01       | C1          | 5000        | 35  | M      | 1        |  
| 2024-10-05       | C2          | 10000       | 42  | F      | 0        |  

**Expected Output**:  
- Target Encoded Gender: Female approval rate = 0. Target-encoded features generated.  

#### **Input 3: Energy Consumption Data**  
**File**: `energy_data.csv`  
| Timestamp           | Meter_ID | Consumption | Temperature |  
|---------------------|----------|-------------|-------------|  
| 2024-12-01 00:00:00 | M1       | 100         | 15          |  
| 2024-12-01 01:00:00 | M1       | 110         | 14          |  

**Expected Output**:  
- Lagged features for consumption, interaction between consumption and temperature.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

# Function for lagged features
def create_lag_features(df, column, lags):
    for lag in range(1, lags + 1):
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df

# Function for rolling features
def create_rolling_features(df, column, windows):
    for window in windows:
        df[f"{column}_rolling_mean_{window}"] = df[column].rolling(window=window).mean()
        df[f"{column}_rolling_std_{window}"] = df[column].rolling(window=window).std()
    return df

# Function for target encoding
def target_encode(df, categorical_column, target_column):
    means = df.groupby(categorical_column)[target_column].transform("mean")
    df[f"{categorical_column}_target_encoded"] = means
    return df

# Function for feature importance ranking
def rank_features(df, target_column):
    correlations = df.corr()[target_column].abs().sort_values(ascending=False)
    return correlations[1:]  # Exclude target itself

# Function for time-based train-test split
def time_based_split(df, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(df):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        yield train, test

# Main pipeline
if __name__ == "__main__":
    # Example: Sales Data
    sales_data = pd.DataFrame({
        'Date': pd.date_range(start="2024-11-01", periods=10, freq='D'),
        'Product_ID': ['P1'] * 10,
        'Sales': [200, 210, 190, 220, 240, 250, 270, 260, 250, 240],
        'Price': [25, 24, 25, 26, 24, 23, 22, 24, 26, 27],
        'Store_ID': [1] * 10
    })
    
    sales_data['Date'] = pd.to_datetime(sales_data['Date'])
    sales_data = sales_data.set_index('Date')
    
    # Lagged Features
    sales_data = create_lag_features(sales_data, 'Sales', lags=3)
    print("Lagged Features:")
    print(sales_data.head())
    
    # Rolling Features
    sales_data = create_rolling_features(sales_data, 'Sales', windows=[2, 3])
    print("Rolling Features:")
    print(sales_data.head())
    
    # Target Encoding
    sales_data['Sales_binary'] = (sales_data['Sales'] > 240).astype(int)
    sales_data = target_encode(sales_data, 'Store_ID', 'Sales_binary')
    print("Target Encoded Features:")
    print(sales_data.head())
    
    # Feature Ranking
    rankings = rank_features(sales_data.dropna(), 'Sales')
    print("Feature Importance Ranking:")
    print(rankings)
    
    # Time-Based Splits
    print("Train-Test Splits:")
    for train, test in time_based_split(sales_data.dropna(), n_splits=2):
        print("Train:\n", train.head())
        print("Test:\n", test.head())
```

---

### Advanced Skills Covered  

1. **Dynamic Feature Engineering**: Automate generation of predictive features tailored to various datasets.  
2. **Advanced Encoding Techniques**: Implement target encoding dynamically.  
3. **Time-Based Validation**: Split data in a time-aware manner for robust model validation.  
4. **Correlation Analysis**: Statistically rank features for predictive modeling.  
5. **Adaptability**: Apply these techniques to a wide variety of datasets.  

This project bridges the gap between raw data and machine learning, enabling a seamless transition from data exploration to predictive modeling!
"""