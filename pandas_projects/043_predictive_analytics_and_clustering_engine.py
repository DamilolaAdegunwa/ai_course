import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


# Step 1: Dynamic Feature Engineering
def scale_data(data, columns):
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data


# Step 2: Cluster Analysis
def cluster_data(data, columns, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[columns])
    return data, kmeans


# Step 3: Regression Modeling
def linear_regression_model(data, features, target):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return model, mse


# Step 4: Classification Modeling
def logistic_regression_model(data, features, target):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy


# Example Usage
if __name__ == "__main__":
    # Example 1: E-Commerce Data Clustering
    data = pd.DataFrame({
        'CustomerID': [1, 2, 3, 4],
        'Age': [19, 35, 45, 25],
        'Annual Income (k$)': [15, 25, 45, 90],
        'Spending Score': [39, 81, 61, 3]
    })
    data = scale_data(data, ['Age', 'Annual Income (k$)', 'Spending Score'])
    clustered_data, kmeans = cluster_data(data, ['Age', 'Annual Income (k$)', 'Spending Score'], n_clusters=2)
    print("Clustered Data:\n", clustered_data)

    # Example 2: Predict Sales
    sales_data = pd.DataFrame({
        'Sales': [200, 300, 250, 400],
        'Advertising Spend': [50, 70, 60, 100],
        'Social Media Spend': [30, 40, 35, 50]
    })
    features = ['Advertising Spend', 'Social Media Spend']
    target = 'Sales'
    model, mse = linear_regression_model(sales_data, features, target)
    print("Mean Squared Error for Sales Prediction:", mse)


comment = """
### Project Title: **Pandas-Based Predictive Analytics and Clustering Engine**  
**File Name**: `predictive_analytics_and_clustering_engine.py`  

---

### Project Description  

This project introduces an **advanced predictive analytics engine** using **Pandas** for real-world decision-making. The project integrates:  

1. **Dynamic Feature Engineering**: Extract meaningful features automatically from raw data.  
2. **Cluster Analysis**: Perform clustering using techniques like k-means or hierarchical clustering to group data.  
3. **Regression and Classification Modeling**: Use linear regression for continuous predictions and logistic regression for binary classifications.  
4. **Custom Pipelines**: Create a flexible data pipeline for transformations, model fitting, and evaluation.  
5. **Scalability**: Easily adapt for large datasets with optimizations.  

This project is suitable for applications like customer segmentation, sales predictions, and targeted marketing analytics.  

---

### Example Use Cases  

1. **Customer Segmentation**: Group customers based on spending patterns, age, or geographical locations.  
2. **Sales Forecasting**: Predict future sales trends using historical data.  
3. **Loan Default Prediction**: Identify the likelihood of a customer defaulting on a loan.  
4. **Anomaly Group Detection**: Cluster suspicious activities or outliers in transactions.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: E-Commerce Data**  
| CustomerID | Age | Annual Income (k$) | Spending Score |  
|------------|-----|---------------------|----------------|  
| 1          | 19  | 15                  | 39             |  
| 2          | 35  | 25                  | 81             |  
| 3          | 45  | 45                  | 61             |  
| 4          | 25  | 90                  | 3              |  

**Expected Output**:  
- **Clusters**: 2 groups based on spending behavior and income levels.  
- **Prediction**: Predict spending score for a new user with an income of $50k.  

---

#### **Input 2: Loan Dataset**  
| LoanID | Age | Income (k$) | LoanAmount (k$) | Defaulted |  
|--------|-----|-------------|-----------------|-----------|  
| 101    | 30  | 60          | 20              | 0         |  
| 102    | 40  | 80          | 50              | 1         |  
| 103    | 25  | 30          | 10              | 0         |  
| 104    | 50  | 100         | 70              | 1         |  

**Expected Output**:  
- **Clusters**: Group borrowers by risk levels.  
- **Prediction**: Predict if a borrower with $40k income and $25k loan will default.  

---

#### **Input 3: Sales Data**  
| Date       | Sales | Advertising Spend | Social Media Spend |  
|------------|-------|-------------------|--------------------|  
| 2024-01-01 | 200   | 50                | 30                 |  
| 2024-01-02 | 300   | 70                | 40                 |  
| 2024-01-03 | 250   | 60                | 35                 |  
| 2024-01-04 | 400   | 100               | 50                 |  

**Expected Output**:  
- **Regression Model**: Predict sales based on advertising spend.  

---

#### **Input 4: Server Performance Data**  
| ServerID | CPU Usage (%) | Memory Usage (%) | Downtime (minutes) |  
|----------|---------------|------------------|--------------------|  
| 1        | 80            | 70               | 5                  |  
| 2        | 90            | 85               | 15                 |  
| 3        | 75            | 60               | 2                  |  
| 4        | 95            | 90               | 20                 |  

**Expected Output**:  
- **Clusters**: Identify high-risk servers.  
- **Prediction**: Predict downtime based on CPU and memory usage.  

---

### Python Code  

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Step 1: Dynamic Feature Engineering
def scale_data(data, columns):
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

# Step 2: Cluster Analysis
def cluster_data(data, columns, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[columns])
    return data, kmeans

# Step 3: Regression Modeling
def linear_regression_model(data, features, target):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return model, mse

# Step 4: Classification Modeling
def logistic_regression_model(data, features, target):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy

# Example Usage
if __name__ == "__main__":
    # Example 1: E-Commerce Data Clustering
    data = pd.DataFrame({
        'CustomerID': [1, 2, 3, 4],
        'Age': [19, 35, 45, 25],
        'Annual Income (k$)': [15, 25, 45, 90],
        'Spending Score': [39, 81, 61, 3]
    })
    data = scale_data(data, ['Age', 'Annual Income (k$)', 'Spending Score'])
    clustered_data, kmeans = cluster_data(data, ['Age', 'Annual Income (k$)', 'Spending Score'], n_clusters=2)
    print("Clustered Data:\n", clustered_data)
    
    # Example 2: Predict Sales
    sales_data = pd.DataFrame({
        'Sales': [200, 300, 250, 400],
        'Advertising Spend': [50, 70, 60, 100],
        'Social Media Spend': [30, 40, 35, 50]
    })
    features = ['Advertising Spend', 'Social Media Spend']
    target = 'Sales'
    model, mse = linear_regression_model(sales_data, features, target)
    print("Mean Squared Error for Sales Prediction:", mse)
```

---

### Key Features  

- **Dynamic Feature Engineering**: Automated scaling and preparation for analysis.  
- **Scalability**: Easily handle datasets across different domains.  
- **Flexibility**: Clustering, regression, and classification in a single pipeline.  
- **Insights-Driven**: Identify groups, predict outcomes, and evaluate accuracy.  

This project builds on the concepts of clustering and predictive analytics for advanced real-world applications.
"""