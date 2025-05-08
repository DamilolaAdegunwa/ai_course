# advanced_customer_retention_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from ucimlrepo import fetch_ucirepo

# fetch dataset
iranian_churn = fetch_ucirepo(id=563)  # https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset

# data (as pandas dataframes)
X = iranian_churn.data.features
y = iranian_churn.data.targets

print(f"X (before get_dummies): ", X.head(3))

# Display basic information about the dataset
print(f"Feature shape: {X.shape}")
print(f"Feature columns: {X.columns}")

# Handle missing values if any
X.fillna(method='ffill', inplace=True)

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)
print(f"X (after get_dummies): ", X.head(3))

# metadata
print(f"iranian_churn.metadata: ", iranian_churn.metadata)

# variable information
print(f"iranian_churn.variables: ", iranian_churn.variables)

# Train-test split
X_train: pd.DataFrame
X_test: pd.DataFrame
y_train: pd.Series
y_test: pd.Series
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Address class imbalance using SMOTE
smote: SMOTE = SMOTE(random_state=42)
X_train_resampled: pd.DataFrame
y_train_resampled: pd.Series
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Feature scaling
scaler: StandardScaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Initialize and train models
# Gradient Boosting Machine
gbm: GradientBoostingClassifier = GradientBoostingClassifier(random_state=42)
gbm.fit(X_train_resampled, y_train_resampled)

# Extreme Gradient Boosting
xgb: XGBClassifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train_resampled, y_train_resampled)

# Predictions
y_pred_gbm: np.ndarray = gbm.predict(X_test)
y_pred_xgb: np.ndarray = xgb.predict(X_test)

# Evaluation
print("Gradient Boosting Machine Results:")
print(confusion_matrix(y_test, y_pred_gbm))
print(classification_report(y_test, y_pred_gbm))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_gbm):.4f}\n")

print("Extreme Gradient Boosting Results:")
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_xgb):.4f}")
