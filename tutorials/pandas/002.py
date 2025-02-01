import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "expanded_employee_dataset.xlsx"
df = pd.read_excel(file_path)
df['Join_Year'] = df['Join_Date'].dt.year


# 0. Feature Engineering
def preprocess_data(df):
    # df['Join_Year'] = df['Join_Date'].dt.year
    feature_columns = ['Age', 'Salary', 'Department']
    target_column = 'Performance_Score'

    # One-hot encode categorical data
    X = df[feature_columns]
    y = df[target_column]
    return X, y


# 1. Predicting Employee Performance
def predict_performance(df):
    X,y= preprocess_data(df)

    #Define the preprocessed steps
    categorical_features = ["Department"]
    numeric_features = ["Age", "Salary"]

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ]
    )

    # Define the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # create pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train the model
    pipeline.fit(X_train, y_train)
    print("Model Training Completed.")

    # Predict on the test set
    predictions = pipeline.predict(X_test)
    results = pd.DataFrame({'Actual': y_test, 'predicted': predictions})
    print("Predictions on Test data")
    print(results.head())

    return pipeline


# Main
if __name__ == "__main__":
    output = predict_performance(df)
























