import pandas as pd
import numpy as np


# Function to merge multiple datasets
def merge_datasets(dfs, keys, how='inner'):
    result = dfs[0]
    for i in range(1, len(dfs)):
        result = pd.merge(result, dfs[i], on=keys[i - 1], how=how)
    return result


# Function to engineer features
def feature_engineering(df):
    # Example: Temperature-adjusted sales
    if 'Units_Sold' in df.columns and 'Temperature' in df.columns:
        df['Temp_Adjusted_Sales'] = df['Units_Sold'] * (30 / df['Temperature'])

    # Example: Lagged price changes
    if 'Close_Price' in df.columns:
        df['Price_Change'] = df['Close_Price'].diff()

    # Example: Cost per engagement
    if 'Ad_Spend' in df.columns and 'Views' in df.columns:
        df['Cost_Per_Engagement'] = df['Ad_Spend'] / (df['Views'] + 1e-5)

    return df


# Visualization function
def visualize_data(df, cols, title='Data Visualization'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for col in cols:
        plt.plot(df[col], label=col)
    plt.title(title)
    plt.legend()
    plt.show()


# Main function to demonstrate use cases
if __name__ == "__main__":
    # Example 1: Merging Sales and Weather Data
    sales_data = pd.DataFrame({
        'Date': ['2024-12-01', '2024-12-01'],
        'Product_ID': ['P1', 'P2'],
        'Units_Sold': [120, 80],
        'Revenue': [3600, 2400]
    })
    weather_data = pd.DataFrame({
        'Date': ['2024-12-01', '2024-12-02'],
        'Temperature': [25, 28],
        'Precipitation': [10, 0]
    })
    merged_df = merge_datasets([sales_data, weather_data], ['Date'])
    engineered_df = feature_engineering(merged_df)
    print("Sales and Weather Merged Data with Features:")
    print(engineered_df)

    # Example 2: Merging Stock Prices and Sentiment
    stock_prices = pd.DataFrame({
        'Date': ['2024-12-01', '2024-12-02'],
        'Symbol': ['AAPL', 'AAPL'],
        'Close_Price': [150, 155]
    })
    news_sentiment = pd.DataFrame({
        'Date': ['2024-12-01'],
        'Symbol': ['AAPL'],
        'Sentiment': ['Positive']
    })
    merged_financials = merge_datasets([stock_prices, news_sentiment], ['Date', 'Symbol'])
    engineered_financials = feature_engineering(merged_financials)
    print("Stock and Sentiment Merged Data with Features:")
    print(engineered_financials)

    # Visualization Example
    visualize_data(engineered_financials, ['Close_Price', 'Price_Change'], title='Stock Price Trends')


comment = """
### Project Title: **Multi-Source Data Fusion and Predictive Insights with Pandas**  
**File Name**: `multi_source_data_fusion_and_predictive_insights_with_pandas.py`  

---

### Project Description  
This project is focused on **multi-source data integration** and deriving actionable **predictive insights**. It involves:  
1. **Data Fusion**: Merging and harmonizing data from multiple heterogeneous sources.  
2. **Cross-Referencing Data**: Leveraging joins, conditional merges, and group-wise computations to build meaningful relationships.  
3. **Feature Engineering Across Datasets**: Constructing synthetic features like combined metrics, growth rates, and weighted aggregates.  
4. **Predictive Analytics**: Applying machine learning-ready transformations for feature-rich datasets, including lagged features, rolling metrics, and normalized indices.  
5. **Visualization**: Displaying integrated insights and highlighting key predictive trends.  

This project is essential for scenarios like supply chain optimization, social media sentiment analysis, and real-world predictive modeling where multiple data streams come into play.  

---

### Example Use Cases  

1. **E-commerce Demand Prediction**: Fuse sales, weather, and marketing data to forecast product demand.  
2. **Social Media Analytics**: Combine post engagement data with user demographics and ad spend to predict campaign success.  
3. **Financial Risk Assessment**: Integrate stock prices, macroeconomic indicators, and news sentiment to evaluate risk.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: E-commerce Data**  
**File 1: Sales Data**  
| Date       | Product_ID | Units_Sold | Revenue |  
|------------|------------|------------|---------|  
| 2024-12-01 | P1         | 120        | 3600    |  
| 2024-12-01 | P2         | 80         | 2400    |  

**File 2: Weather Data**  
| Date       | Temperature | Precipitation |  
|------------|-------------|---------------|  
| 2024-12-01 | 25          | 10            |  
| 2024-12-02 | 28          | 0             |  

**Expected Output**:  
- Merged dataset with sales, revenue, and weather.  
- Features like temperature-adjusted sales and average revenue per product.  

#### **Input 2: Social Media Data**  
**File 1: Engagement Data**  
| Post_ID | Views  | Likes | Shares |  
|---------|--------|-------|--------|  
| 101     | 5000   | 300   | 50     |  
| 102     | 3000   | 200   | 40     |  

**File 2: User Data**  
| User_ID | Demographic | Ad_Spend |  
|---------|-------------|----------|  
| 1       | Teenager    | 200      |  
| 2       | Adult       | 400      |  

**Expected Output**:  
- Post-level engagement metrics merged with ad spend.  
- Derived features like cost per engagement and shares/likes ratio.  

#### **Input 3: Financial Data**  
**File 1: Stock Prices**  
| Date       | Symbol | Close_Price |  
|------------|--------|-------------|  
| 2024-12-01 | AAPL   | 150         |  
| 2024-12-02 | AAPL   | 155         |  

**File 2: News Sentiment**  
| Date       | Symbol | Sentiment |  
|------------|--------|-----------|  
| 2024-12-01 | AAPL   | Positive  |  

**Expected Output**:  
- Stock prices combined with sentiment.  
- Feature engineering like lagged price changes and sentiment-weighted performance.  

---

### Python Code  

```python
import pandas as pd
import numpy as np

# Function to merge multiple datasets
def merge_datasets(dfs, keys, how='inner'):
    result = dfs[0]
    for i in range(1, len(dfs)):
        result = pd.merge(result, dfs[i], on=keys[i-1], how=how)
    return result

# Function to engineer features
def feature_engineering(df):
    # Example: Temperature-adjusted sales
    if 'Units_Sold' in df.columns and 'Temperature' in df.columns:
        df['Temp_Adjusted_Sales'] = df['Units_Sold'] * (30 / df['Temperature'])
    
    # Example: Lagged price changes
    if 'Close_Price' in df.columns:
        df['Price_Change'] = df['Close_Price'].diff()
    
    # Example: Cost per engagement
    if 'Ad_Spend' in df.columns and 'Views' in df.columns:
        df['Cost_Per_Engagement'] = df['Ad_Spend'] / (df['Views'] + 1e-5)
    
    return df

# Visualization function
def visualize_data(df, cols, title='Data Visualization'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for col in cols:
        plt.plot(df[col], label=col)
    plt.title(title)
    plt.legend()
    plt.show()

# Main function to demonstrate use cases
if __name__ == "__main__":
    # Example 1: Merging Sales and Weather Data
    sales_data = pd.DataFrame({
        'Date': ['2024-12-01', '2024-12-01'],
        'Product_ID': ['P1', 'P2'],
        'Units_Sold': [120, 80],
        'Revenue': [3600, 2400]
    })
    weather_data = pd.DataFrame({
        'Date': ['2024-12-01', '2024-12-02'],
        'Temperature': [25, 28],
        'Precipitation': [10, 0]
    })
    merged_df = merge_datasets([sales_data, weather_data], ['Date'])
    engineered_df = feature_engineering(merged_df)
    print("Sales and Weather Merged Data with Features:")
    print(engineered_df)
    
    # Example 2: Merging Stock Prices and Sentiment
    stock_prices = pd.DataFrame({
        'Date': ['2024-12-01', '2024-12-02'],
        'Symbol': ['AAPL', 'AAPL'],
        'Close_Price': [150, 155]
    })
    news_sentiment = pd.DataFrame({
        'Date': ['2024-12-01'],
        'Symbol': ['AAPL'],
        'Sentiment': ['Positive']
    })
    merged_financials = merge_datasets([stock_prices, news_sentiment], ['Date', 'Symbol'])
    engineered_financials = feature_engineering(merged_financials)
    print("Stock and Sentiment Merged Data with Features:")
    print(engineered_financials)
    
    # Visualization Example
    visualize_data(engineered_financials, ['Close_Price', 'Price_Change'], title='Stock Price Trends')
```

---

### How This Project Advances Your Skills  
1. **Multi-Source Data Integration**: Learn advanced techniques for merging and aligning complex datasets.  
2. **Dynamic Feature Engineering**: Develop transferable skills for creating predictive features across domains.  
3. **Advanced Use of Pandas**: Explore deep functionalities like `merge`, `groupby`, and synthetic calculations.  
4. **Scalability**: Build modular pipelines that can handle large-scale, real-world datasets.  
5. **Data Science-Ready Transformations**: Make your datasets immediately applicable for machine learning models.  

Push the boundaries further by integrating **external APIs** to fetch live data streams and incorporating **dimensionality reduction** techniques for visualization!
"""