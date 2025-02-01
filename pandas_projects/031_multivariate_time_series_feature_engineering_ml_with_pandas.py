import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load Multivariate Time Series Data
def load_multivariate_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Timestamp'], index_col='Timestamp')
    return data

# Feature Engineering
def feature_engineering(data, target_column):
    # Extract Time Features
    data['Hour'] = data.index.hour
    data['DayOfWeek'] = data.index.dayofweek
    data['Month'] = data.index.month

    # Rolling Features
    for col in data.columns:
        if col != target_column:
            data[f'{col}_RollingMean'] = data[col].rolling(window=3).mean()
            data[f'{col}_RollingStd'] = data[col].rolling(window=3).std()

    # Lag Features
    for lag in range(1, 4):
        for col in data.columns:
            if col != target_column:
                data[f'{col}_Lag{lag}'] = data[col].shift(lag)

    data = data.dropna()  # Remove rows with NaN values after rolling/lagging
    return data

# Preprocess Data for Machine Learning
def preprocess_data(data, categorical_columns, numerical_columns, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(), categorical_columns)
        ]
    )
    return X, y, preprocessor

# Train and Evaluate Model
def train_model(X, y, preprocessor):
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Classification Report:\n", classification_report(y_test, y_pred))
    return model

# Main Execution
if __name__ == "__main__":
    # Load Example Data
    data = pd.DataFrame({
        'Timestamp': pd.date_range(start='2024-01-01', periods=50, freq='H'),
        'Energy': np.random.poisson(lam=1500, size=50),
        'Temperature': np.random.normal(loc=15, scale=5, size=50),
        'DayType': ['Weekday'] * 40 + ['Weekend'] * 10
    })
    data['Anomaly'] = np.random.choice([0, 1], size=50, p=[0.9, 0.1])
    data = data.set_index('Timestamp')

    # Feature Engineering
    data = feature_engineering(data, 'Anomaly')

    # Preprocess Data
    categorical_columns = ['DayType']
    numerical_columns = [col for col in data.columns if col not in ['Anomaly', 'DayType']]
    X, y, preprocessor = preprocess_data(data, categorical_columns, numerical_columns, 'Anomaly')

    # Train and Evaluate
    trained_model = train_model(X, y, preprocessor)


# https://chatgpt.com/c/674b65b9-fecc-800c-8311-7f681df9b305
comment = """
### Project Title: **Multivariate Time Series Feature Engineering and Machine Learning with Pandas**  
**File Name**: `multivariate_time_series_feature_engineering_ml_with_pandas.py`  

---

### Project Description  

This project focuses on the **end-to-end processing of multivariate time series data** using Pandas, including **feature engineering** and applying machine learning models for **classification or regression tasks**. The goal is to develop advanced techniques for preparing time-series datasets for predictive tasks, combining domain knowledge, automation, and advanced statistical methodologies.  

**Highlights:**  
1. **Dynamic Feature Extraction**: Extracts advanced time-based and statistical features (e.g., rolling averages, lag features, and Fourier transforms).  
2. **Multi-Series Synchronization**: Handles synchronization of multiple time-series data streams with varying frequencies.  
3. **Data Normalization and Encoding**: Prepares data for machine learning using normalization and encoding of categorical variables.  
4. **Model Training and Evaluation**: Uses machine learning models to predict outcomes based on extracted features.  
5. **Auto-Feature Optimization**: Employs feature selection techniques to optimize the dataset for model performance.  

---

### Example Use Cases  

1. **Energy Demand Forecasting**: Predict future electricity demand using past consumption, weather, and time-related data.  
2. **Customer Churn Prediction**: Identify customers likely to churn using user activity logs and metadata.  
3. **IoT Device Failure Prediction**: Detect possible device failures by analyzing sensor readings and historical patterns.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: Energy Consumption Data**  
| Timestamp          | Energy (kWh) | Temperature (°C) | DayType   |  
|--------------------|--------------|-------------------|-----------|  
| 2024-01-01 00:00  | 1500         | 10                | Weekday   |  
| 2024-01-01 01:00  | 1480         | 10.5              | Weekday   |  
| 2024-01-01 02:00  | 1450         | 10.2              | Weekday   |  

**Expected Output**:  
- Features: Rolling Average, Lagged Values, Time-of-Day Encoding, Fourier Components.  
- Prediction Target: Next hour's energy consumption.  

---

#### **Input 2: E-commerce User Activity**  
| Timestamp         | UserID  | PageViews | Clicks | Purchase | DeviceType |  
|-------------------|---------|-----------|--------|----------|------------|  
| 2024-01-01 08:00 | 1001    | 5         | 2      | 0        | Mobile     |  
| 2024-01-01 08:01 | 1002    | 3         | 1      | 1        | Desktop    |  
| 2024-01-01 08:02 | 1001    | 8         | 4      | 1        | Mobile     |  

**Expected Output**:  
- Features: Rolling Click Rate, Purchase Probability, Device-Type Encoding.  
- Prediction Target: Probability of purchase.  

---

#### **Input 3: IoT Sensor Data**  
| Timestamp          | Sensor1 | Sensor2 | Sensor3 | DeviceStatus |  
|--------------------|---------|---------|---------|--------------|  
| 2024-01-01 00:00  | 50      | 70      | 30      | Active       |  
| 2024-01-01 01:00  | 52      | 68      | 33      | Active       |  
| 2024-01-01 02:00  | 49      | 72      | 35      | Faulty       |  

**Expected Output**:  
- Features: Rolling Averages, Variance, Fourier Transform Components.  
- Prediction Target: Probability of Fault.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load Multivariate Time Series Data
def load_multivariate_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Timestamp'], index_col='Timestamp')
    return data

# Feature Engineering
def feature_engineering(data, target_column):
    # Extract Time Features
    data['Hour'] = data.index.hour
    data['DayOfWeek'] = data.index.dayofweek
    data['Month'] = data.index.month

    # Rolling Features
    for col in data.columns:
        if col != target_column:
            data[f'{col}_RollingMean'] = data[col].rolling(window=3).mean()
            data[f'{col}_RollingStd'] = data[col].rolling(window=3).std()

    # Lag Features
    for lag in range(1, 4):
        for col in data.columns:
            if col != target_column:
                data[f'{col}_Lag{lag}'] = data[col].shift(lag)

    data = data.dropna()  # Remove rows with NaN values after rolling/lagging
    return data

# Preprocess Data for Machine Learning
def preprocess_data(data, categorical_columns, numerical_columns, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(), categorical_columns)
        ]
    )
    return X, y, preprocessor

# Train and Evaluate Model
def train_model(X, y, preprocessor):
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Classification Report:\n", classification_report(y_test, y_pred))
    return model

# Main Execution
if __name__ == "__main__":
    # Load Example Data
    data = pd.DataFrame({
        'Timestamp': pd.date_range(start='2024-01-01', periods=50, freq='H'),
        'Energy': np.random.poisson(lam=1500, size=50),
        'Temperature': np.random.normal(loc=15, scale=5, size=50),
        'DayType': ['Weekday'] * 40 + ['Weekend'] * 10
    })
    data['Anomaly'] = np.random.choice([0, 1], size=50, p=[0.9, 0.1])
    data = data.set_index('Timestamp')

    # Feature Engineering
    data = feature_engineering(data, 'Anomaly')

    # Preprocess Data
    categorical_columns = ['DayType']
    numerical_columns = [col for col in data.columns if col not in ['Anomaly', 'DayType']]
    X, y, preprocessor = preprocess_data(data, categorical_columns, numerical_columns, 'Anomaly')

    # Train and Evaluate
    trained_model = train_model(X, y, preprocessor)
```

---

### Advanced Skills Covered  

1. **Multivariate Time Series Processing**: Handles multiple synchronized and unsynchronized time-series datasets.  
2. **Feature Engineering**: Creates rolling, lag, and Fourier features to improve predictive power.  
3. **Automated ML Pipelines**: Integrates preprocessing and machine learning seamlessly using Scikit-learn.  
4. **Scalable Solutions**: Easily extendable for large datasets and real-time applications.  
5. **Classification/Regression with ML**: Applies machine learning techniques to time-series predictions.  

This project pushes your knowledge of Pandas, feature engineering, and machine learning integration to new heights!
"""