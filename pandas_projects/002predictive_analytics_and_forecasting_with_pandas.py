import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
# Load the dataset
file_path = "expanded_employee_dataset.xlsx"
df = pd.read_excel(file_path)
df['Join_Year'] = df['Join_Date'].dt.year


# 0. Feature Engineering
def preprocess_data(df: DataFrame) -> (Series | None | DataFrame, Series | None | DataFrame):
    # df['Join_Year'] = df['Join_Date'].dt.year
    feature_columns = ['Age', 'Salary', 'Department']
    target_column = 'Performance_Score'

    # One-hot encode categorical data
    X = df[feature_columns]
    y = df[target_column]
    # a :Series | None | DataFrame = None  # (Series | None | DataFrame, Series | None | DataFrame)
    return X, y


# 1. Predicting Employee Performance
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

    # return pipeline
    return results, X_test, y_test, pipeline


# 2. Forecasting Hiring Trends
def forecast_hiring_trends(df: DataFrame):
    hiring_data = df['Join_Year'].value_counts().sort_index()
    X = np.array(hiring_data.index).reshape(-1, 1)
    y = hiring_data.values

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Forecast next 5 years
    future_years = np.array(range(X.max() + 1, X.max() + 6)).reshape(-1, 1)
    forecast = model.predict(future_years)

    forecast_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted_Hires': forecast.astype(int)})
    print("Hiring Forecast for Next 5 Years:")
    print(forecast_df)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(X, y, label="Historical Data")
    plt.plot(future_years, forecast, label="Forecast", linestyle='--', color='orange')
    plt.xlabel("Year")
    plt.ylabel("Number of Hires")
    plt.title("Hiring Trends and Forecast")
    plt.legend()
    plt.show()

    return forecast_df


# 3. Identify High-Performing Departments
def high_performance_departments(df):
    avg_performance = df.groupby('Department')['Performance_Score'].mean()
    print(avg_performance)
    return avg_performance.sort_values(ascending=False)


# 4a. Analyze Salary Distribution
def analyze_salary_distribution_old(df):
    salary_stats = df['Salary'].describe()
    print(salary_stats)
    return salary_stats


# 4b. Analyze Salary Distribution
def analyze_salary_distribution(df):
    # by department
    avg_performance_department: DataFrame = df.groupby('Department')['Salary'].mean().sort_values(ascending=False)
    print(avg_performance_department)

    # by Join_Year
    avg_performance_join_year: DataFrame = df.groupby('Join_Year')['Salary'].mean().sort_values(ascending=False)
    print(avg_performance_join_year)

    # by Age
    avg_performance_age: DataFrame = df.groupby('Age')['Salary'].mean().sort_values(ascending=False)
    print(avg_performance_age)
    return avg_performance_department, avg_performance_join_year, avg_performance_age


# 5. Predict Salary Based on Features
def predict_salary(df):
    feature_columns = ['Age', 'Department', 'Performance_Score']
    target_column = 'Salary'

    X = df[feature_columns]
    y = df[target_column]

    categorical_features = ['Department']
    numeric_features = ['Age', 'Performance_Score']

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numeric_transformer = StandardScaler()

    commented_transformer = """
    # you could opt for a more compound option
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    """

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    print(results)
    return results, X_test, y_test, pipeline


# 6. Forecast Average Performance Score per Year
def forecast_performance_trends(df):
    performance_data = df.groupby(df['Join_Year'])['Performance_Score'].mean()
    X = np.array(performance_data.index).reshape(-1, 1)
    y = performance_data.values

    model = LinearRegression()
    model.fit(X, y)

    future_years = np.array(range(X.max() + 1, X.max() + 6)).reshape(-1, 1)
    forecast = model.predict(future_years)

    forecast_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted_Performance': forecast})
    print(forecast_df)
    return forecast_df


# 7. Correlation Analysis
def correlation_analysis(df: DataFrame):
    data = df[['Age', 'Performance_Score', 'Salary', 'Join_Year']]
    mean_vals = np.mean(data, axis=0); std_vals = np.std(data, axis=0)
    standardized_data = (data - mean_vals) / std_vals
    t_corrcoef = np.corrcoef(standardized_data.T)
    print(t_corrcoef)
    return t_corrcoef


# 8. Evaluate Model Performance
def evaluate_model_performance(X_test, y_test, pipeline):
    predictions = pipeline.predict(X_test)
    results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

    return results


# 9. Evaluate Total Salary Paid by the Company
def total_salary_paid_by_the_company(df):
    salary_sum = df['Salary'].sum()
    return salary_sum


# 10. Performance Clusters
def performance_clusters(df):
    from sklearn.cluster import KMeans

    X = df[['Performance_Score']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Performance_Cluster'] = kmeans.fit_predict(X)
    return df[['Name', 'Performance_Cluster']]


# Generate Report
def generate_report(df):
    print("Predict Employee Performance")
    performance_predictions, X_test, y_test, pipeline = predict_performance(df)

    print("Forecast Hiring Trends")
    hiring_trends = forecast_hiring_trends(df)

    print("Identify High-Performance Departments")
    high_performance_depts = high_performance_departments(df)

    print("Analyze Salary Distribution")
    salary_distribution = analyze_salary_distribution(df)

    print("Predict Salary")
    salary_predictions = predict_salary(df)

    print("Forecast Performance Trends")
    performance_trends = forecast_performance_trends(df)

    print("Perform Correlation Analysis")
    correlation_analysis_result = correlation_analysis(df)

    print("Evaluate Model Performance")
    model_performance = evaluate_model_performance(X_test, y_test, pipeline)

    print("total_salary_paid_by_the_company")
    total_salary_paid_by_the_company_ = total_salary_paid_by_the_company(df)

    print("Identify Performance Clusters")
    performance_clusters_result = performance_clusters(df)

    # Create the report object
    report = {
        'Performance Predictions': performance_predictions,
        'Hiring Trends': hiring_trends,
        'High-Performance Departments': high_performance_depts,
        'Salary Distribution': salary_distribution,
        'Salary Predictions': salary_predictions,
        'Performance Trends': performance_trends,
        'Correlation Analysis': correlation_analysis_result,
        'Model Performance': model_performance,
        'Total Salary Paid By the Company': total_salary_paid_by_the_company_,
        'Performance Clusters': performance_clusters_result
    }

    #  report = {}
    return report


# test this (rough work)
def test1(df):
    hiring_data = df['Join_Year'].value_counts().sort_index()
    X = np.array(hiring_data.index).reshape(-1, 1)
    print(f"hiring_data.index {hiring_data.index}")
    y = hiring_data.values
    print(f"y {y}")


# Main
if __name__ == "__main__":
    # test1(df)
    # predict_performance(df)  # 1
    # forecast_hiring_trends(df)  # 2
    # high_performance_departments(df)  # 3
    # analyze_salary_distribution_old(df)  # 4a
    # analyze_salary_distribution(df)  # 4b
    # predict_salary(df)  # 5
    # forecast_performance_trends(df)  # 6
    correlation_analysis(df)  # 7
    # output = generate_report(df)
'''    for key, value in report.items():
        print(f"\n{key}:\n{value}\n")'''


comment = """
Project Title: Predictive Analytics and Forecasting with Pandas and Machine Learning
File Name: predictive_analytics_and_forecasting_with_pandas.py

Project Description
This project incorporates advanced pandas functionality combined with machine learning to forecast and analyze employee trends and future performance. Key components include:

Feature engineering to prepare data for machine learning.
Prediction of employee performance scores based on historical trends.
Forecasting future hiring trends using regression models.
Data scaling, train-test splitting, and cross-validation.
Integration of scikit-learn for predictive modeling.
Example Input(s) and Expected Output(s)
Input 1:
Task: Predict the performance score of employees based on age, department, and salary.
Expected Output:
Predicted performance scores for employees, e.g.:

ID	Age	Salary	Department	Predicted_Performance
100	45	75000	HR	3.87
101	30	55000	Engineering	4.12
Input 2:
Task: Forecast the number of employees hired per year for the next 5 years.
Expected Output:
Forecast of hiring numbers:

Year	Predicted_Hires
2024	120
2025	135
2026	140
Input 3:
Task: Generate insights about the most influential features on performance scores.
Expected Output:
Feature importance metrics, e.g.:

Feature	Importance
Salary	0.45
Age	0.30
Department	0.25
Python Code
"""

"""
Testing Scenarios
Scenario 1:
Input the dataset and execute predict_performance().
Expected: A DataFrame showing actual vs predicted performance scores for a subset of employees.

Scenario 2:
Input the dataset and run forecast_hiring_trends().
Expected: A DataFrame of predicted hires for the next 5 years and a line chart visualization.

Scenario 3:
Modify the dataset to add a new department and test if the predict_performance() method handles unseen categorical data.
Expected: Predictions generated successfully without errors.

Let me know how you'd like to refine or extend this project further!
"""
