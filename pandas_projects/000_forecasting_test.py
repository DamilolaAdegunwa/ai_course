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
                                                         

# 2. Forecasting Hiring Trends
def forecast_hiring_trends(df):
    hiring_data = df['Join_Year'].value_counts().sort_index()
    print(hiring_data)
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
    plt.plot(X, y, label="Historical Data", linestyle='solid', color='#000')
    # the linestyle choices are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    # the colors choices includes  '#rrggbb', '#rrggbbaa', '#rgb', '#rgba' valid names
    plt.plot(future_years, forecast, label="Forecast", linestyle='--', color='orange')
    plt.xlabel("Year")
    plt.ylabel("Number of Hires")
    plt.title("Hiring Trends and Forecast")
    plt.legend()
    plt.show()

    return forecast_df


def test(df):
    hiring_data = df['Join_Year'].value_counts().sort_index()
    X = np.array(hiring_data.index).reshape(-1, 1)
    y = hiring_data.values

    print("----------------X---------------")
    print(X)
    print("-----------y-----------")
    print(y)

    return X, y


# Main
if __name__ == "__main__":
    forecast_hiring_trends(df)
    # output = test(df)
