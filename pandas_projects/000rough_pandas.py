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
from pandas import DataFrame
# Load the dataset
# file_path = "750000_employee_dataset.xlsx"  # 750,000 rows
file_path = "expanded_employee_dataset.xlsx"  # 100,000 rows
df = pd.read_excel(file_path)


# 1. Feature Engineering
def preprocess_data(df):
    df['Join_Year'] = df['Join_Date'].dt.year
    feature_columns = ['Age', 'Salary', 'Department']
    target_column = 'Performance_Score'

    # One-hot encode categorical data
    X = df[feature_columns]
    y = df[target_column]
    return X, y


# 2. Predicting Employee Performance
def predict_performance(df):
    X, y = preprocess_data(df)

    # Define preprocessing steps
    categorical_features = ['Department']
    numeric_features = ['Age', 'Salary']

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Define the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Create pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    pipeline.fit(X_train, y_train)
    print("Model Training Completed.")

    # Predict on the test set
    predictions = pipeline.predict(X_test)
    results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    print("Predictions on Test Data:")
    print(results.head())

    return pipeline


# 3. Time-series analysis of hiring trends
def hiring_trend_analysis(df: DataFrame):
    """
    Time-series analysis of hiring trends
    :param df:
    """
    df['Join_Year'] = df['Join_Date'].dt.year
    hiring_trend = df.groupby('Join_Year').size()
    print("Hiring Trend Over the Years:")
    print(hiring_trend)

    # Plotting the trend
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=hiring_trend.index, y=hiring_trend.values, marker='o')
    plt.title("Hiring Trends (2010-2023)")
    plt.xlabel("Year")
    plt.ylabel("Number of Hires")
    plt.grid(True)
    plt.show()


# Example Use Cases
if __name__ == "__main__":
    # Predict Employee Performance
    # pipeline = predict_performance(df)
    hiring_trend_analysis(df)