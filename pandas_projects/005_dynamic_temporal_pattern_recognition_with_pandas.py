import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Air Quality dataset as an example
# file_path = "air_quality_no2.csv"
file_path = "air_quality_long_edited.csv"
df = pd.read_csv(file_path, parse_dates=['datetime'])

# Ensure datetime is correctly parsed and set as index
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

# remove reduntant lines
df = df.drop(['city', 'country', 'location', 'unit'], axis=1)

# Print unique values in the 'parameter' column
unique_parameters = df['parameter'].unique()
print(unique_parameters)  # ['pm25' 'no2']

df['parameter'] = df['parameter'].replace({'pm25': 1, 'no2': 2})


# Data Cleaning and Preprocessing
def preprocess_air_quality_data(df):
    # Handle missing values by interpolation
    df = df.interpolate(method='time')

    # Add useful time-based features
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['hour'] = df.index.hour
    return df


df = preprocess_air_quality_data(df)


# Example Function 1: Seasonal Trend Analysis
def analyze_monthly_trends(df, pollutant=2):
    monthly_avg = df.resample('ME').mean()  # Resample to monthly averages
    print("monthly_avg")
    print(monthly_avg)

    plt.figure(figsize=(10, 6))
    plt.plot(monthly_avg.index, monthly_avg["parameter"], marker='o', linestyle='-', label=f'{pollutant} Levels')
    plt.title('Monthly NO2 Trends')
    plt.xlabel('Month')
    plt.ylabel(f'{pollutant} Levels')
    plt.grid(True)
    plt.legend()
    plt.show()
    return monthly_avg


# Example Function 2: Anomaly Detection
def detect_anomalies(df, pollutant=2, threshold=1.5):
    # Use IQR to detect anomalies
    q1 = df["parameter"].quantile(0.25)
    q3 = df["parameter"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    anomalies = df[(df["parameter"] < lower_bound) | (df["parameter"] > upper_bound)]

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["parameter"], label=f'{pollutant} Levels', alpha=0.7)
    plt.scatter(anomalies.index, anomalies["parameter"], color='red', label='Anomalies')
    plt.title('Anomaly Detection in NO2 Levels')
    plt.xlabel('Time')
    plt.ylabel(f'{pollutant} Levels')
    plt.legend()
    plt.show()
    return anomalies


# Example Function 3: Correlation Between Pollutants
def pollutant_correlation(df):
    # print(df)
    # correlation_matrix = df["parameter"].corr(df['month'])
    grouped = df.groupby('parameter').mean()  # Compute mean for each parameter
    correlation = grouped['value'].corr(grouped['month'])

    print("correlation")
    print(correlation)

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Pollutant Correlation Heatmap')
    plt.show()
    return correlation_matrix


# Test the pipeline
if __name__ == "__main__":
    print("Analyzing Monthly Trends...")
    monthly_trends = analyze_monthly_trends(df)

    print("Detecting Anomalies...")
    anomalies = detect_anomalies(df)

    print("Pollutant Correlation Analysis...")
    correlation_matrix = pollutant_correlation(df)


# https://chatgpt.com/c/674b65b9-fecc-800c-8311-7f681df9b305 (all pandas projects)
# https://chatgpt.com/c/676810bf-2c48-800c-96ec-18a9647de1bb (pandas project 5)
comment = """
### Project Title: Dynamic Temporal Pattern Recognition with Pandas and Visualization  
**File Name**: `dynamic_temporal_pattern_recognition_with_pandas.py`  

---

### Project Description  
This project dives into advanced temporal data analysis and visualization. Using **Pandas** and **Matplotlib/Seaborn**, we will:  

1. Process temporal data for insights using advanced time-series techniques.  
2. Identify dynamic trends and patterns, such as seasonalities, anomalies, and correlations.  
3. Automate the extraction of statistical patterns (e.g., moving averages, autocorrelation, trend analysis).  
4. Visualize insights dynamically to capture changes over time.  

We’ll leverage the **Air Quality dataset** (or any time-series dataset) for practical use cases. The goal is to not just clean and prepare the data but also derive actionable insights from temporal patterns.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1**:  
**Dataset**: Air Quality dataset.  
**Task**: Identify seasonal NO2 patterns over months.  
**Expected Output**:  
- Monthly NO2 levels with a line graph showing trends.  
- Highlight anomalies (e.g., unusual peaks).  

#### **Input 2**:  
**Dataset**: Titanic dataset (departure time hypothetical).  
**Task**: Analyze ticket purchases over time for peak sales.  
**Expected Output**:  
- Hourly sales trend with timestamps.  
- Heatmap of passenger class sales vs. time.  

#### **Input 3**:  
**Dataset**: Custom sales dataset.  
**Task**: Detect correlations between product sales over time.  
**Expected Output**:  
- Cross-correlation matrix between product sales.  
- Bar chart of best-selling products with their time-to-sale peak.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Air Quality dataset as an example
file_path = "air_quality_no2.csv"
df = pd.read_csv(file_path, parse_dates=['datetime'])

# Ensure datetime is correctly parsed and set as index
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

# Data Cleaning and Preprocessing
def preprocess_air_quality_data(df):
    # Handle missing values by interpolation
    df = df.interpolate(method='time')
    
    # Add useful time-based features
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['hour'] = df.index.hour
    return df

df = preprocess_air_quality_data(df)

# Example Function 1: Seasonal Trend Analysis
def analyze_monthly_trends(df, pollutant='NO2'):
    monthly_avg = df.resample('M').mean()  # Resample to monthly averages
    
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_avg.index, monthly_avg[pollutant], marker='o', linestyle='-', label=f'{pollutant} Levels')
    plt.title('Monthly NO2 Trends')
    plt.xlabel('Month')
    plt.ylabel(f'{pollutant} Levels')
    plt.grid(True)
    plt.legend()
    plt.show()
    return monthly_avg

# Example Function 2: Anomaly Detection
def detect_anomalies(df, pollutant='NO2', threshold=1.5):
    # Use IQR to detect anomalies
    q1 = df[pollutant].quantile(0.25)
    q3 = df[pollutant].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    anomalies = df[(df[pollutant] < lower_bound) | (df[pollutant] > upper_bound)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[pollutant], label=f'{pollutant} Levels', alpha=0.7)
    plt.scatter(anomalies.index, anomalies[pollutant], color='red', label='Anomalies')
    plt.title('Anomaly Detection in NO2 Levels')
    plt.xlabel('Time')
    plt.ylabel(f'{pollutant} Levels')
    plt.legend()
    plt.show()
    return anomalies

# Example Function 3: Correlation Between Pollutants
def pollutant_correlation(df, pollutants=['NO2', 'O3', 'PM10']):
    correlation_matrix = df[pollutants].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Pollutant Correlation Heatmap')
    plt.show()
    return correlation_matrix

# Test the pipeline
if __name__ == "__main__":
    print("Analyzing Monthly Trends...")
    monthly_trends = analyze_monthly_trends(df)

    print("Detecting Anomalies...")
    anomalies = detect_anomalies(df)

    print("Pollutant Correlation Analysis...")
    correlation_matrix = pollutant_correlation(df)
```

---

### Testing Scenarios  

#### **Scenario 1**:  
**Dataset**: Air Quality dataset.  
**Task**: Analyze monthly NO2 trends.  
**Expected Output**: A line graph showing trends of NO2 levels monthly with clear markers.  

#### **Scenario 2**:  
**Dataset**: Air Quality dataset.  
**Task**: Detect anomalies in pollutant levels based on IQR thresholding.  
**Expected Output**: A scatter plot marking anomalies in pollutant levels over time.  

#### **Scenario 3**:  
**Dataset**: Air Quality dataset.  
**Task**: Identify correlations between NO2, PM10, and O3.  
**Expected Output**: Heatmap showing correlations between the pollutants, e.g., NO2 vs. O3.  

---

This project focuses on **time-series analysis**, anomaly detection, and multi-variable relationships, providing a more advanced challenge than previous projects. Let me know if you’d like further enhancements!
"""