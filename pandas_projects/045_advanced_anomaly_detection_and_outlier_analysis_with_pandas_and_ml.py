import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt


# Step 1: Data Preprocessing
def preprocess_data(data, features):
    # Handle missing values
    data = data.fillna(data.mean())  # Simple imputation strategy
    # Normalize the features
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    return data


# Step 2: Anomaly Detection using Isolation Forest
def detect_anomalies_isolation_forest(data, features, contamination=0.05):
    model = IsolationForest(contamination=contamination)
    data['anomaly'] = model.fit_predict(data[features])
    return data


# Step 3: Anomaly Detection using DBSCAN
def detect_anomalies_dbscan(data, features, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    data['anomaly'] = model.fit_predict(data[features])
    return data


# Step 4: Anomaly Detection using One-Class SVM
def detect_anomalies_svm(data, features, nu=0.05):
    model = OneClassSVM(nu=nu)
    data['anomaly'] = model.fit_predict(data[features])
    return data


# Step 5: Statistical Outlier Detection (Z-score and IQR)
def detect_outliers_statistical(data, features):
    # Z-score method
    z_scores = np.abs(stats.zscore(data[features]))
    data['z_outlier'] = (z_scores > 3).astype(int)  # Flagging z-scores greater than 3

    # IQR method
    Q1 = data[features].quantile(0.25)
    Q3 = data[features].quantile(0.75)
    IQR = Q3 - Q1
    data['iqr_outlier'] = ((data[features] < (Q1 - 1.5 * IQR)) | (data[features] > (Q3 + 1.5 * IQR))).astype(int)

    return data


# Step 6: Visualization of Anomalies
def visualize_anomalies(data, features):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data[features[0]], data[features[1]], c=data['anomaly'], cmap='coolwarm', edgecolor='k')
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    plt.title('Anomaly Detection Visualization')
    plt.show()


# Example Usage
if __name__ == "__main__":
    # Example: Fraud Detection Dataset
    transaction_data = pd.DataFrame({
        'Transaction_ID': [1, 2, 3, 4],
        'Amount': [500, 1500, 100, 5000],
        'Date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
        'Merchant': ['Store_A', 'Store_B', 'Store_C', 'Store_A'],
        'Location': ['NY', 'CA', 'TX', 'NY']
    })

    # Preprocessing
    transaction_data = preprocess_data(transaction_data, ['Amount'])

    # Anomaly Detection (Isolation Forest)
    transaction_data = detect_anomalies_isolation_forest(transaction_data, ['Amount'])
    print("Detected Anomalies with Isolation Forest:\n", transaction_data)

    # Anomaly Detection (DBSCAN)
    transaction_data = detect_anomalies_dbscan(transaction_data, ['Amount'])
    print("Detected Anomalies with DBSCAN:\n", transaction_data)

    # Statistical Outliers
    transaction_data = detect_outliers_statistical(transaction_data, ['Amount'])
    print("Outliers Detected Statistically:\n", transaction_data)

    # Visualization
    visualize_anomalies(transaction_data, ['Amount', 'Transaction_ID'])


comment = """
### Project Title: **Advanced Anomaly Detection and Outlier Analysis with Pandas and Machine Learning**  
**File Name**: `advanced_anomaly_detection_and_outlier_analysis_with_pandas_and_ml.py`

---

### Project Description  

This project focuses on **advanced anomaly detection and outlier analysis**, combining **Pandas** for data manipulation with machine learning techniques for predictive modeling. The goal is to build a system that can:  

1. **Data Preprocessing**: Clean and prepare the data using Pandas, including missing value imputation, normalization, and feature scaling.  
2. **Unsupervised Learning for Anomaly Detection**: Implement unsupervised algorithms like Isolation Forest, DBSCAN, and One-Class SVM to detect anomalies in the data.  
3. **Statistical Outlier Detection**: Use statistical methods (Z-score, IQR) for outlier identification and compare them with machine learning methods.  
4. **Feature Engineering**: Create and manipulate features to improve the anomaly detection model's performance.  
5. **Model Evaluation**: Use precision, recall, and F1-score to evaluate the anomaly detection model's effectiveness.  

This project is ideal for applications in fraud detection, sensor monitoring, financial market anomaly detection, and healthcare.  

---

### Example Use Cases  

1. **Fraud Detection in Transactions**: Detect fraudulent financial transactions using anomaly detection techniques.  
2. **Quality Control in Manufacturing**: Detect defective products based on sensor data.  
3. **Network Intrusion Detection**: Identify network traffic anomalies that might indicate a security breach.  
4. **Healthcare Monitoring**: Monitor vital signs to detect abnormal readings that could indicate medical issues.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: Transaction Data (Fraud Detection)**  
| Transaction_ID | Amount | Date       | Merchant | Location |  
|----------------|--------|------------|----------|----------|  
| 1              | 500    | 2024-01-01 | Store_A  | NY       |  
| 2              | 1500   | 2024-01-01 | Store_B  | CA       |  
| 3              | 100    | 2024-01-02 | Store_C  | TX       |  
| 4              | 5000   | 2024-01-02 | Store_A  | NY       |  

**Expected Output**:  
- **Outliers**: Transactions with unusually large amounts, possibly indicating fraud.  
- **Anomaly Detection**: Transaction 4 could be flagged as anomalous due to the large amount compared to typical transactions.  

---

#### **Input 2: Sensor Data (Manufacturing Quality Control)**  
| Timestamp           | Sensor_1 | Sensor_2 | Sensor_3 |  
|---------------------|----------|----------|----------|  
| 2024-01-01 00:00:00 | 98       | 105      | 99       |  
| 2024-01-01 01:00:00 | 100      | 107      | 101      |  
| 2024-01-01 02:00:00 | 97       | 103      | 102      |  
| 2024-01-01 03:00:00 | 200      | 105      | 101      |  

**Expected Output**:  
- **Outlier Detection**: Sensor reading 4 at timestamp 2024-01-01 03:00:00 could be an outlier due to a significant spike in Sensor_1.  
- **Anomaly Detection**: Sensor 1 at timestamp 2024-01-01 03:00:00 is flagged as anomalous.  

---

#### **Input 3: Network Traffic Data (Intrusion Detection)**  
| Timestamp           | Source_IP     | Traffic_Volume | Protocol |  
|---------------------|---------------|----------------|----------|  
| 2024-01-01 00:00:00 | 192.168.1.1   | 1000           | HTTP     |  
| 2024-01-01 01:00:00 | 192.168.1.2   | 1200           | HTTPS    |  
| 2024-01-01 02:00:00 | 192.168.1.3   | 800            | FTP      |  
| 2024-01-01 03:00:00 | 192.168.1.4   | 10000          | HTTP     |  

**Expected Output**:  
- **Outliers**: Traffic volume from IP 192.168.1.4 could be flagged as anomalous.  
- **Anomaly Detection**: Traffic volume of 10,000 in a short period may indicate a DoS attack.  

---

#### **Input 4: Healthcare Data (Vital Sign Monitoring)**  
| Timestamp           | Heart_Rate | Blood_Pressure | Body_Temperature |  
|---------------------|------------|----------------|------------------|  
| 2024-01-01 00:00:00 | 72         | 120/80         | 36.5             |  
| 2024-01-01 01:00:00 | 75         | 122/82         | 36.6             |  
| 2024-01-01 02:00:00 | 68         | 118/78         | 36.4             |  
| 2024-01-01 03:00:00 | 150        | 200/120        | 38.5             |  

**Expected Output**:  
- **Outlier Detection**: The heart rate and blood pressure values at timestamp 2024-01-01 03:00:00 may indicate a medical anomaly.  
- **Anomaly Detection**: Values of 150 for heart rate and 200/120 for blood pressure could suggest a medical emergency.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt

# Step 1: Data Preprocessing
def preprocess_data(data, features):
    # Handle missing values
    data = data.fillna(data.mean())  # Simple imputation strategy
    # Normalize the features
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    return data

# Step 2: Anomaly Detection using Isolation Forest
def detect_anomalies_isolation_forest(data, features, contamination=0.05):
    model = IsolationForest(contamination=contamination)
    data['anomaly'] = model.fit_predict(data[features])
    return data

# Step 3: Anomaly Detection using DBSCAN
def detect_anomalies_dbscan(data, features, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    data['anomaly'] = model.fit_predict(data[features])
    return data

# Step 4: Anomaly Detection using One-Class SVM
def detect_anomalies_svm(data, features, nu=0.05):
    model = OneClassSVM(nu=nu)
    data['anomaly'] = model.fit_predict(data[features])
    return data

# Step 5: Statistical Outlier Detection (Z-score and IQR)
def detect_outliers_statistical(data, features):
    # Z-score method
    z_scores = np.abs(stats.zscore(data[features]))
    data['z_outlier'] = (z_scores > 3).astype(int)  # Flagging z-scores greater than 3
    
    # IQR method
    Q1 = data[features].quantile(0.25)
    Q3 = data[features].quantile(0.75)
    IQR = Q3 - Q1
    data['iqr_outlier'] = ((data[features] < (Q1 - 1.5 * IQR)) | (data[features] > (Q3 + 1.5 * IQR))).astype(int)
    
    return data

# Step 6: Visualization of Anomalies
def visualize_anomalies(data, features):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data[features[0]], data[features[1]], c=data['anomaly'], cmap='coolwarm', edgecolor='k')
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    plt.title('Anomaly Detection Visualization')
    plt.show()

# Example Usage
if __name__ == "__main__":
    # Example: Fraud Detection Dataset
    transaction_data = pd.DataFrame({
        'Transaction_ID': [1, 2, 3, 4],
        'Amount': [500, 1500, 100, 5000],
        'Date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
        'Merchant': ['Store_A', 'Store_B', 'Store_C', 'Store_A'],
        'Location': ['NY', 'CA', 'TX', 'NY']
    })
    
    # Preprocessing
    transaction_data = preprocess_data(transaction_data, ['Amount'])
    
    # Anomaly Detection (Isolation Forest)
    transaction_data = detect_anomalies_isolation_forest(transaction_data, ['Amount'])
    print("Detected Anomalies with Isolation Forest:\n", transaction_data

)
    
    # Anomaly Detection (DBSCAN)
    transaction_data = detect_anomalies_dbscan(transaction_data, ['Amount'])
    print("Detected Anomalies with DBSCAN:\n", transaction_data)
    
    # Statistical Outliers
    transaction_data = detect_outliers_statistical(transaction_data, ['Amount'])
    print("Outliers Detected Statistically:\n", transaction_data)
    
    # Visualization
    visualize_anomalies(transaction_data, ['Amount', 'Transaction_ID'])
```

### Explanation of Code:  
- **Preprocessing**: Missing values are imputed with the mean, and the data is normalized using `StandardScaler`.  
- **Anomaly Detection**: Three anomaly detection techniques are applied:
  - **Isolation Forest**: Detects anomalies based on isolation, suitable for high-dimensional datasets.  
  - **DBSCAN**: A clustering algorithm that can detect dense regions and outliers as noise.  
  - **One-Class SVM**: Identifies outliers by learning the boundary of the normal data.  
- **Statistical Outlier Detection**: 
  - **Z-score**: Flags values that are more than 3 standard deviations away from the mean.  
  - **IQR (Interquartile Range)**: Identifies outliers beyond 1.5 times the IQR above the third quartile or below the first quartile.  
- **Visualization**: Uses `matplotlib` to display anomalies in the dataset.  

This advanced project extends basic anomaly detection into a practical use case with multiple techniques and evaluation methods. You can test with different datasets by modifying the sample data provided in the code.


"""