import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np

# Load the Titanic dataset
# file_path = "titanic.csv"
# df = pd.read_csv(file_path)

file_path = "titanic.xlsx"
df = pd.read_excel(file_path)


# Data Preprocessing
def preprocess_titanic_data(df):
    # Fill missing values with median for numerical columns and mode for categorical
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            # df[column].fillna(df[column].median(), inplace=True)  # old!
            df.fillna({column: df[column].median()}, inplace=True)
            # or df[column] = df[column].fillna(df[column].median())
        else:
            # df[column].fillna(df[column].mode()[0], inplace=True) # old
            df.fillna({column: df[column].mode()[0]}, inplace=True)
            # or df[column] = df[column].fillna(df[column].mode()[0])
    return df


df = preprocess_titanic_data(df)


# Example Function 1: Feature Scaling and Encoding
def scale_and_encode_features(df):
    scaler = MinMaxScaler()
    label_encoders = {}

    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            df[column] = scaler.fit_transform(df[[column]])
        elif df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le  # Store encoder for inverse transformation if needed

    return df, label_encoders


# Example Function 2: Generate Lag Features
def generate_lag_features(df, columns, lags):
    for column in columns:
        if column in df.columns:
            for lag in range(1, lags + 1):
                df[f"{column}_lag{lag}"] = df[column].shift(lag)
    df = df.fillna(0)  # Replace NaN introduced by shifting with 0
    return df


# Example Function 3: Feature Selection Based on Correlation
def select_features_by_correlation(df, target_column, threshold=0.1):
    correlations = df.corr()[target_column].abs()
    selected_features = correlations[correlations > threshold].index.tolist()
    return df[selected_features]


# Test the pipeline
if __name__ == "__main__":
    print("Scaling and Encoding Features...")
    df_scaled, encoders = scale_and_encode_features(df)
    print(f"df_scaled:\n {df_scaled}")

    print("Generating Lag Features...")
    lagged_df = generate_lag_features(df_scaled, columns=['Age', 'Fare'], lags=3)
    print(f"lagged_df:\n {lagged_df}")

    print("Selecting Features by Correlation...")
    selected_features = select_features_by_correlation(lagged_df, target_column='Survived')
    print("Selected Features:\n ", selected_features)


comment = """
### Project Title: Machine Learning Data Preparation and Feature Engineering with Pandas  
**File Name**: `machine_learning_data_preparation_with_pandas.py`  

---

### Project Description  
In this project, you'll focus on preparing raw datasets for machine learning models using **Pandas**. The project covers:  

1. Advanced feature engineering techniques, such as creating lag features, rolling window statistics, and polynomial features.  
2. Cleaning and encoding data, including handling missing values and outliers.  
3. Automating one-hot encoding, label encoding, and feature scaling.  
4. Advanced feature selection based on correlation and variance thresholds.  

We will leverage the **Titanic dataset** to predict survival probabilities, ensuring the features generated are meaningful for a machine-learning pipeline.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1**:  
**Dataset**: Titanic dataset.  
**Task**: One-hot encode categorical columns and scale numerical columns.  
**Expected Output**:  
- Transformed dataset with numerical features scaled between 0-1 and categorical features one-hot encoded.  

#### **Input 2**:  
**Dataset**: Titanic dataset.  
**Task**: Generate lag features for numerical columns like "Age" and "Fare".  
**Expected Output**:  
- New columns like `Age_lag1`, `Fare_lag2` added to the dataset.  

#### **Input 3**:  
**Dataset**: Titanic dataset.  
**Task**: Select features with high correlation to the target column (`Survived`).  
**Expected Output**:  
- A reduced dataset with only the most relevant features retained.  

---

### Python Code  

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np

# Load the Titanic dataset
file_path = "titanic.csv"
df = pd.read_csv(file_path)

# Data Preprocessing
def preprocess_titanic_data(df):
    # Fill missing values with median for numerical columns and mode for categorical
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            df[column].fillna(df[column].median(), inplace=True)
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)
    return df

df = preprocess_titanic_data(df)

# Example Function 1: Feature Scaling and Encoding
def scale_and_encode_features(df):
    scaler = MinMaxScaler()
    label_encoders = {}
    
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            df[column] = scaler.fit_transform(df[[column]])
        elif df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le  # Store encoder for inverse transformation if needed
    
    return df, label_encoders

# Example Function 2: Generate Lag Features
def generate_lag_features(df, columns, lags):
    for column in columns:
        if column in df.columns:
            for lag in range(1, lags + 1):
                df[f"{column}_lag{lag}"] = df[column].shift(lag)
    df = df.fillna(0)  # Replace NaN introduced by shifting with 0
    return df

# Example Function 3: Feature Selection Based on Correlation
def select_features_by_correlation(df, target_column, threshold=0.1):
    correlations = df.corr()[target_column].abs()
    selected_features = correlations[correlations > threshold].index.tolist()
    return df[selected_features]

# Test the pipeline
if __name__ == "__main__":
    print("Scaling and Encoding Features...")
    df_scaled, encoders = scale_and_encode_features(df)
    
    print("Generating Lag Features...")
    lagged_df = generate_lag_features(df_scaled, columns=['Age', 'Fare'], lags=3)
    
    print("Selecting Features by Correlation...")
    selected_features = select_features_by_correlation(lagged_df, target_column='Survived')
    print("Selected Features:", selected_features)
```

---

### Testing Scenarios  

#### **Scenario 1**:  
**Dataset**: Titanic dataset.  
**Task**: Scale and encode all features.  
**Expected Output**:  
- Numerical features scaled between 0-1 and categorical features converted to numeric labels.  

#### **Scenario 2**:  
**Dataset**: Titanic dataset.  
**Task**: Generate lag features for `Age` and `Fare` columns.  
**Expected Output**:  
- New columns like `Age_lag1`, `Age_lag2`, `Fare_lag1`, `Fare_lag2`, `Fare_lag3` added to the dataset.  

#### **Scenario 3**:  
**Dataset**: Titanic dataset.  
**Task**: Select features with correlation above 0.1 to `Survived`.  
**Expected Output**:  
- Reduced dataset with only features strongly correlated with the target column.  

---

### Key Learnings  
- **Data Preparation**: Cleaning, imputing missing values, scaling, and encoding features.  
- **Feature Engineering**: Generating meaningful features (lags, rolling statistics, polynomial terms).  
- **Feature Selection**: Identifying and retaining the most relevant features for ML models.  

Let me know if you'd like additional tasks or enhancements for this project!
"""
