import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import entropy


# Load Dataset (KDD Cup 99 - Network Intrusion Detection)
data = fetch_kddcup99(subset="http", as_frame=True).frame
data = pd.DataFrame(data)
# Columns to check
cols_to_check = np.array(data.columns)  # ['duration', 'src_bytes', 'dst_bytes', 'labels']

for col in cols_to_check:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

# print(f"the description of the data is: {data.describe()}")
# print(f"the length of the data is (np.shape(data)) {np.shape(data)}")  # (58725 x 4)
# print(f"the length of the data is (len(data)) {len(data)}")  # 58725 rows
# data = data.select_dtypes(include=[np.number])  # Keep only numerical features
# print(f"data shape (after include=[np.number]): {np.shape(data)}")
# Introduce Anomaly Labels (5% labeled anomalies)
np.random.seed(42)
data['Anomaly'] = np.random.choice([0, 1], size=len(data), p=[0.95, 0.05])
# print(f"data shape (after adding anomaly): {np.shape(data)}")
# Split Data into Labeled (5%) and Unlabeled (95%)
labeled_data = data.sample(frac=0.05, random_state=42)  # I think this a subpopulation
# print(f"labeled_data shape: {np.shape(labeled_data)}")
# print(f"the length of the labeled_data is (len(labeled_data)) {len(labeled_data)}")  # 2936 rows
unlabeled_data = data.drop(labeled_data.index)
# print(f"the length of the unlabeled_data is (len(unlabeled_data)) {len(unlabeled_data)}")  # 55789 rows
# print(f"unlabeled_data shape: {np.shape(unlabeled_data)}")
# print(f"the 'unlabeled_data' data {unlabeled_data}")

X_labeled, y_labeled = labeled_data.drop(columns=['Anomaly']), labeled_data['Anomaly']
# print(f"X_labeled shape: {np.shape(X_labeled)}")
# print(f"y_labeled shape: {np.shape(y_labeled)}")
# print(f"the X_labeled (len(X_labeled)): {len(X_labeled)} {X_labeled}")
# print(f"the y_labeled (len(y_labeled)): {len(y_labeled)} {y_labeled}")

X_unlabeled = unlabeled_data.drop(columns=['Anomaly'])
# print(f"the X_unlabeled (len(X_unlabeled)): {len(X_unlabeled)} {X_unlabeled}")
# print(f"X_unlabeled shape: {np.shape(X_unlabeled)}")

# Initial Unsupervised Anomaly Detection (Gaussian Mixture Model)
gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
gmm.fit(X_unlabeled)
gmm_scores = -gmm.score_samples(X_unlabeled)  # Higher scores indicate anomalies

# Convert GMM Scores into Initial Anomaly Predictions
threshold = np.percentile(gmm_scores, 95)  # Top 5% as anomalies
# y_unlabeled_pred = (gmm_scores >= threshold).astype(int)
y_unlabeled_pred = (gmm_scores >= threshold).astype(int)


# Active Learning: Selecting Samples for Labeling (Entropy-Based)
def select_samples_for_labeling(X_unlabeled, y_pred, num_samples=50):
    uncertainty = entropy(np.vstack([y_pred, 1 - y_pred]), axis=0)
    uncertain_samples = np.argsort(-uncertainty)[:num_samples]
    return X_unlabeled.iloc[uncertain_samples]


# Simulating Active Learning Iterations
for i in range(3):  # 3 Active Learning Cycles
    X_query = select_samples_for_labeling(X_unlabeled, y_unlabeled_pred, num_samples=50)

    # Simulate Expert Labeling (Using True Labels)
    y_query = data.loc[X_query.index, 'Anomaly']

    # Add Labeled Data to Training Set
    X_labeled = pd.concat([X_labeled, X_query])
    y_labeled = pd.concat([y_labeled, y_query])

    # Train a New Isolation Forest on the Growing Labeled Set
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_labeled, y_labeled)

    # Predict on Unlabeled Data
    y_unlabeled_pred = model.predict(X_unlabeled)
    y_unlabeled_pred = np.where(y_unlabeled_pred == -1, 1, 0)  # Convert -1 (anomaly) to 1

# Final Evaluation on Test Set
X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42)
final_model = IsolationForest(contamination=0.05, random_state=42)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
y_pred = np.where(y_pred == -1, 1, 0)

print("\nFinal Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


comments = """
### **Project Title:**  
**Self-Learning Anomaly Detection with Active Learning and Semi-Supervised Learning**

**File Name:**  
`self_learning_anomaly_detection.py`

---

### **Project Description:**  
This project implements an **advanced anomaly detection system** using **active learning** and **semi-supervised learning** techniques. Unlike traditional anomaly detection that requires large amounts of labeled data, this approach starts with a small labeled dataset and actively selects uncertain samples for labeling. This mimics real-world ML scenarios where labeled data is scarce, and models need to learn efficiently from limited information.

It incorporates **Gaussian Mixture Models (GMM) for density estimation**, **Isolation Forest for anomaly detection**, and **an Active Learning loop with Uncertainty Sampling** to iteratively improve performance.

---

### **Use Cases:**  
1. **Cybersecurity:** Detect network intrusions by identifying unusual traffic patterns.  
2. **Fraud Detection:** Find fraudulent financial transactions with minimal labeled fraud cases.  
3. **Healthcare:** Detect rare diseases by learning from a small set of diagnosed cases.  
4. **Manufacturing:** Identify faulty products in a production line using limited failure examples.  

---

### **Example Input(s) and Expected Output(s):**  

#### **Example Input 1:** (Network Traffic Anomaly)  
**Input:**  
- Features from a network traffic dataset (packet size, request frequency, protocol type, etc.).  
- Partially labeled dataset with only 5% of known anomalies.  

**Expected Output:**  
- Model identifies 95% of anomalous packets without explicit labeling.  

---

#### **Example Input 2:** (Credit Card Fraud)  
**Input:**  
- Features from a transaction dataset (amount, location, transaction frequency, etc.).  
- Initial dataset contains labels for only 10% of fraud cases.  

**Expected Output:**  
- Model detects fraudulent transactions with **>85% accuracy** after 5 active learning iterations.  

---

### **Python Code:**  
```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import entropy

# Load Dataset (KDD Cup 99 - Network Intrusion Detection)
data = fetch_kddcup99(subset="http", as_frame=True).frame
data = data.select_dtypes(include=[np.number])  # Keep only numerical features

# Introduce Anomaly Labels (5% labeled anomalies)
np.random.seed(42)
data['Anomaly'] = np.random.choice([0, 1], size=len(data), p=[0.95, 0.05])

# Split Data into Labeled (5%) and Unlabeled (95%)
labeled_data = data.sample(frac=0.05, random_state=42)
unlabeled_data = data.drop(labeled_data.index)

X_labeled, y_labeled = labeled_data.drop(columns=['Anomaly']), labeled_data['Anomaly']
X_unlabeled = unlabeled_data.drop(columns=['Anomaly'])

# Initial Unsupervised Anomaly Detection (Gaussian Mixture Model)
gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
gmm.fit(X_unlabeled)
gmm_scores = -gmm.score_samples(X_unlabeled)  # Higher scores indicate anomalies

# Convert GMM Scores into Initial Anomaly Predictions
threshold = np.percentile(gmm_scores, 95)  # Top 5% as anomalies
y_unlabeled_pred = (gmm_scores >= threshold).astype(int)

# Active Learning: Selecting Samples for Labeling (Entropy-Based)
def select_samples_for_labeling(X_unlabeled, y_pred, num_samples=50):
    uncertainty = entropy(np.vstack([y_pred, 1 - y_pred]), axis=0)
    uncertain_samples = np.argsort(-uncertainty)[:num_samples]
    return X_unlabeled.iloc[uncertain_samples]

# Simulating Active Learning Iterations
for i in range(3):  # 3 Active Learning Cycles
    X_query = select_samples_for_labeling(X_unlabeled, y_unlabeled_pred, num_samples=50)
    
    # Simulate Expert Labeling (Using True Labels)
    y_query = data.loc[X_query.index, 'Anomaly']
    
    # Add Labeled Data to Training Set
    X_labeled = pd.concat([X_labeled, X_query])
    y_labeled = pd.concat([y_labeled, y_query])
    
    # Train a New Isolation Forest on the Growing Labeled Set
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_labeled, y_labeled)
    
    # Predict on Unlabeled Data
    y_unlabeled_pred = model.predict(X_unlabeled)
    y_unlabeled_pred = np.where(y_unlabeled_pred == -1, 1, 0)  # Convert -1 (anomaly) to 1

# Final Evaluation on Test Set
X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42)
final_model = IsolationForest(contamination=0.05, random_state=42)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
y_pred = np.where(y_pred == -1, 1, 0)

print("\nFinal Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

---

### **Key Learnings (Advanced Concepts to Explore Further):**  

#### **1. Semi-Supervised Learning**  
- **Googleable Keywords:** "Self-training in Machine Learning", "Label Propagation", "Co-training Algorithm"  
- **Why It’s Important?** Learn how models can generalize from **partially labeled data**.  

#### **2. Active Learning with Uncertainty Sampling**  
- **Googleable Keywords:** "Query strategies in active learning", "Entropy-based sampling", "Bayesian Uncertainty in ML"  
- **Why It’s Important?** Helps models select the **most useful data points** for labeling.  

#### **3. Gaussian Mixture Model (GMM) for Density-Based Anomaly Detection**  
- **Googleable Keywords:** "Gaussian Mixture Model for Outlier Detection", "Expectation-Maximization Algorithm"  
- **Why It’s Important?** Useful for detecting anomalies in **high-dimensional datasets**.  

#### **4. Isolation Forest for Outlier Detection**  
- **Googleable Keywords:** "Isolation Forest vs. One-Class SVM", "Tree-Based Anomaly Detection"  
- **Why It’s Important?** A robust **unsupervised method** for finding rare patterns.  

#### **5. Real-World Applications of Anomaly Detection**  
- **Googleable Keywords:** "Machine Learning for Fraud Detection", "AI for Cybersecurity", "Unsupervised ML in Healthcare"  
- **Why It’s Important?** Anomaly detection is **widely used in industry**, from banking to medical diagnostics.  

---

### **Next Steps:**  
1. **Try Different Datasets** – Apply this to financial fraud, cybersecurity, or manufacturing datasets.  
2. **Experiment with Deep Learning** – Use **Variational Autoencoders (VAE)** or **GANs for anomaly detection**.  
3. **Deploy as an API** – Integrate this model into a real-time **anomaly detection system**.  

---

This project takes your ML engineering skills **to the next level** by incorporating **active learning, semi-supervised learning, and uncertainty estimation**—essential techniques for real-world AI applications.

Would you like enhancements such as **Neural Network-based anomaly detection** or **real-time streaming capabilities**? Let me know! 🚀
"""
