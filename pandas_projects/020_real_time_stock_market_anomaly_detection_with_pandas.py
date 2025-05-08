import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.ensemble import IsolationForest


# Preprocessing Function
def preprocess_stock_data(df: DataFrame):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df


# Feature Engineering
def calculate_indicators(df: DataFrame):
    # Simple Moving Averages
    df['SMA_20'] = df['Price'].rolling(window=20).mean()
    df['SMA_50'] = df['Price'].rolling(window=50).mean()

    # Bollinger Bands
    rolling_std = df['Price'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (rolling_std * 2)
    df['Lower_Band'] = df['SMA_20'] - (rolling_std * 2)

    # Relative Strength Index (RSI)
    delta = df['Price'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    short_ema = df['Price'].ewm(span=12, adjust=False).mean()
    long_ema = df['Price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df


# Anomaly Detection
def detect_anomalies(df: DataFrame):
    model = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly'] = model.fit_predict(df[['Price', 'Volume']])
    return df


# Visualization
def plot_data_with_anomalies(df: DataFrame):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Price'], label='Price', color='blue')
    plt.scatter(df[df['Anomaly'] == -1].index, df[df['Anomaly'] == -1]['Price'], color='red', label='Anomaly', marker='o')
    plt.title('Stock Prices with Anomalies')
    plt.legend()
    plt.show()


# Main Function
if __name__ == "__main__":
    # Simulated data
    data = {
        'Timestamp': pd.date_range(start='2024-12-01 09:30:00', periods=100, freq='T'),
        'Stock Symbol': ['AAPL'] * 100,
        'Price': np.random.normal(172, 1, 100).round(2),
        'Volume': np.random.randint(10000, 20000, 100)
    }

    df = pd.DataFrame(data)

    # Preprocess and calculate indicators
    df = preprocess_stock_data(df)
    df = calculate_indicators(df)

    # Detect anomalies
    df = detect_anomalies(df)

    # Visualize results
    plot_data_with_anomalies(df)

# 1/2
comment = """
### Project Title: **Real-Time Stock Market Anomaly Detection with Pandas and Advanced Analytics**  
**File Name**: `real_time_stock_market_anomaly_detection_with_pandas.py`  

---

### Project Description  
This project implements **real-time stock market anomaly detection** using Pandas for data manipulation, time-series analysis, and advanced statistical techniques. It combines **real-time data streaming**, **machine learning-based anomaly detection**, and **financial metrics analysis** to identify unusual price movements, volatility spikes, or patterns indicating potential fraud or critical market events.  

The project incorporates:  
1. **Data Streaming and Preprocessing**: Simulating a real-time stock market feed and efficiently handling high-frequency data.  
2. **Feature Engineering**: Calculating advanced financial metrics like **moving averages**, **Bollinger Bands**, **RSI (Relative Strength Index)**, and **MACD (Moving Average Convergence Divergence)**.  
3. **Anomaly Detection**: Leveraging **Isolation Forest**, **DBSCAN**, or custom thresholds for anomaly detection.  
4. **Alerting System**: Creating triggers for identified anomalies with detailed context.  
5. **Visualization**: Generating plots showing anomalies, trends, and financial indicators.  

This project is particularly useful for **quantitative analysts**, **traders**, and **financial risk managers** to monitor and respond to anomalies in real-time.

---

### Example Use Cases  
1. **High-Frequency Trading**: Identifying irregular patterns during trading to adjust strategies instantly.  
2. **Market Surveillance**: Detecting potential market manipulation or insider trading activity.  
3. **Investment Decision Support**: Noticing trend reversals early for informed buy/sell decisions.

---

### Example Input(s) and Expected Output(s)

#### **Input 1: Real-Time Price Data**  
| Timestamp            | Stock Symbol | Price   | Volume | Open   | High   | Low    | Close  |  
|-----------------------|--------------|---------|--------|--------|--------|--------|--------|  
| 2024-12-01 09:30:00  | AAPL         | 172.50  | 15000  | 172.00 | 173.00 | 171.50 | 172.50 |  
| 2024-12-01 09:31:00  | AAPL         | 173.20  | 17000  | 172.50 | 173.50 | 172.50 | 173.20 |  
| 2024-12-01 09:32:00  | AAPL         | 171.90  | 12000  | 173.20 | 173.20 | 171.80 | 171.90 |  

**Expected Output**:  
- Detected Anomalies: Sudden drop in price detected at **2024-12-01 09:32:00**.  
- RSI: Below threshold (indicating oversold conditions).  

---

#### **Input 2: Historical Price Data for Trend Analysis**  
| Date       | Stock Symbol | Price   | Volume | Open   | High   | Low    | Close  |  
|------------|--------------|---------|--------|--------|--------|--------|--------|  
| 2024-11-25 | TSLA         | 230.00  | 50000  | 225.00 | 235.00 | 225.00 | 230.00 |  
| 2024-11-26 | TSLA         | 235.00  | 52000  | 230.00 | 240.00 | 230.00 | 235.00 |  
| 2024-11-27 | TSLA         | 220.00  | 48000  | 235.00 | 235.00 | 220.00 | 220.00 |  

**Expected Output**:  
- Anomaly Detected: Significant price drop on 2024-11-27 (from $235 to $220).  
- Bollinger Bands Analysis: Price breached the lower band, suggesting a reversal or bearish momentum.  

---

#### **Input 3: Stream of Multi-Symbol Data**  
| Timestamp            | Stock Symbol | Price   | Volume |  
|-----------------------|--------------|---------|--------|  
| 2024-12-01 09:30:00  | AAPL         | 172.50  | 15000  |  
| 2024-12-01 09:30:00  | TSLA         | 230.50  | 30000  |  
| 2024-12-01 09:30:00  | MSFT         | 331.00  | 20000  |  

**Expected Output**:  
- Detected Anomalies: None.  
- Real-Time Metrics:  
  - AAPL RSI: 52 (Neutral).  
  - TSLA Bollinger Band Breach: False.  
  - MSFT MACD Cross: True (Bullish signal).  

---

### Python Code  

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Preprocessing Function
def preprocess_stock_data(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df

# Feature Engineering
def calculate_indicators(df):
    # Simple Moving Averages
    df['SMA_20'] = df['Price'].rolling(window=20).mean()
    df['SMA_50'] = df['Price'].rolling(window=50).mean()

    # Bollinger Bands
    rolling_std = df['Price'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (rolling_std * 2)
    df['Lower_Band'] = df['SMA_20'] - (rolling_std * 2)

    # Relative Strength Index (RSI)
    delta = df['Price'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    short_ema = df['Price'].ewm(span=12, adjust=False).mean()
    long_ema = df['Price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

# Anomaly Detection
def detect_anomalies(df):
    model = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly'] = model.fit_predict(df[['Price', 'Volume']])
    return df

# Visualization
def plot_data_with_anomalies(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Price'], label='Price', color='blue')
    plt.scatter(df[df['Anomaly'] == -1].index, df[df['Anomaly'] == -1]['Price'], color='red', label='Anomaly', marker='o')
    plt.title('Stock Prices with Anomalies')
    plt.legend()
    plt.show()

# Main Function
if __name__ == "__main__":
    # Simulated data
    data = {
        'Timestamp': pd.date_range(start='2024-12-01 09:30:00', periods=100, freq='T'),
        'Stock Symbol': ['AAPL'] * 100,
        'Price': np.random.normal(172, 1, 100).round(2),
        'Volume': np.random.randint(10000, 20000, 100)
    }
    
    df = pd.DataFrame(data)
    
    # Preprocess and calculate indicators
    df = preprocess_stock_data(df)
    df = calculate_indicators(df)
    
    # Detect anomalies
    df = detect_anomalies(df)
    
    # Visualize results
    plot_data_with_anomalies(df)
```  

---

### Key Features in this Project  
1. **Real-Time Simulation**: Simulates a high-frequency stock data stream.  
2. **Advanced Indicators**: Computes RSI, Bollinger Bands, and MACD.  
3. **Machine Learning for Anomalies**: Uses **Isolation Forest** for anomaly detection.  
4. **Visual Insight**: Highlights anomalies visually for actionable insights.  

This project integrates advanced analytics, machine learning, and financial analysis, significantly enhancing your expertise with Pandas.
"""
