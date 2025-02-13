import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
data_path = "ML_sample_dataset.xlsx"
df = pd.read_excel(data_path)

# Data Preparation
numerical_cols = [col for col in df.columns if "Numerical" in col]
categorical_cols = [col for col in df.columns if "Categorical" in col]
binary_cols = [col for col in df.columns if "Binary" in col]

# Generate target column (binary classification)
df['Target'] = np.random.choice([0, 1], len(df))

# Splitting Features and Target
X = df[numerical_cols + categorical_cols + binary_cols]
y = df['Target']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocessing for numerical and categorical data
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough'  # Leave binary columns as is
)

# Define the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Create pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predict on test data
y_pred = pipeline.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
feature_names = (
    numerical_cols +
    list(pipeline.named_steps['preprocessor']
         .transformers_[1][1]
         .get_feature_names_out(categorical_cols)) +
    binary_cols
)
importances = model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importance)

# Test Input Example
test_input = pd.DataFrame([{
    "Numerical_1": 67.2, "Numerical_2": 45.1, "Numerical_3": 89.5,
    "Numerical_4": 23.7, "Numerical_5": 12.9,
    "Categorical_1": 'B', "Categorical_2": 'D', "Categorical_3": 'A',
    "Binary_1": 0, "Binary_2": 1
}])

test_prediction = pipeline.predict(test_input)
print("\nTest Prediction for Example Input:")
print(test_prediction)

# other metadata
comments = """
https://chatgpt.com/c/678cb759-4994-800c-adaa-9d6bb947dfc6 (projects: machine learning)
https://chatgpt.com/c/679e7ed8-d36c-800c-aeff-91270d88c51b (001_advanced_multimodal_data_fusion_for_predictive_modeling.py)

"""
