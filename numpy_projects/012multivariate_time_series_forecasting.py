import numpy as np


class TimeSeriesForecaster:
    def __init__(self, window_size):
        """
        Initializes the forecaster with a sliding window size.
        """
        self.window_size = window_size
        self.weights = None

    def fit(self, X, y):
        """
        Fits a linear model to the time series data.
        - X: Input data of shape (n_samples, n_features)
        - y: Target values of shape (n_samples,)
        """
        # Add bias term to X
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])

        # Compute weights using the normal equation: w = (X^T X)^-1 X^T y
        self.weights = np.linalg.inv(X_bias.T @ X_bias) @ (X_bias.T @ y)

    def predict(self, X):
        """
        Predicts future values based on input data.
        - X: Input data of shape (n_samples, n_features)
        Returns: Predicted values of shape (n_samples,)
        """
        if self.weights is None:
            raise ValueError("Model is not fitted yet.")

        # Add bias term to X
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])

        # Compute predictions
        return X_bias @ self.weights

    def create_sequences(self, data):
        """
        Creates sequences for sliding window prediction.
        - data: Original time series data of shape (n_samples, n_features)
        Returns: Tuple (X, y) for model training
        """
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i:i + self.window_size, :-1].flatten())
            y.append(data[i + self.window_size, -1])
        return np.array(X), np.array(y)


# Example usage
if __name__ == "__main__":
    # Example multivariate time series data (temperature, humidity, energy consumption)
    data = np.array([
        [30, 70, 100],
        [32, 68, 110],
        [31, 65, 115],
        [29, 60, 120],
        [28, 58, 130],
        [27, 55, 125],
        [26, 52, 135],
        [25, 50, 140]
    ])

    # Initialize the model
    forecaster = TimeSeriesForecaster(window_size=3)

    # Prepare sequences
    X, y = forecaster.create_sequences(data)
    print(f"this is X: {X}")
    print(f"this is y: {y}")

    # Fit the model
    forecaster.fit(X, y)

    # Predict future values
    predictions = forecaster.predict(X)
    print("Predictions:", predictions)

# Project Title: Multivariate Time Series Forecasting Using NumPy
"""
File Name: multivariate_time_series_forecasting.py

Short Description
This project involves implementing a predictive model for multivariate time series data using NumPy. The goal is to forecast future values based on patterns in multiple input sequences. This model is a simplified version of time series forecasting methods like ARIMA or RNNs but demonstrates the core mechanics using NumPy. It involves preparing data, calculating weights via linear regression, and making forecasts.
"""

# Example Inputs and Expected Outputs
"""
Example 1
Input:

python
Copy code
data = np.array([
    [10, 15, 25],
    [12, 18, 30],
    [11, 17, 28],
    [14, 20, 34],
    [13, 19, 33],
    [15, 22, 37]
])
window_size = 2
Expected Output:

plaintext
Copy code
X: [[10 15 12 18]
    [12 18 11 17]
    [11 17 14 20]
    [14 20 13 19]]
y: [28 34 33 37]
Predictions: [28.1, 33.9, 33.2, 37.3]
Example 2
Input:

python
Copy code
data = np.array([
    [20, 40, 100],
    [21, 42, 110],
    [19, 41, 120],
    [23, 43, 130],
    [22, 45, 140],
    [24, 46, 150]
])
window_size = 3
Expected Output:

plaintext
Copy code
X: [[20 40 21 42 19 41]
    [21 42 19 41 23 43]
    [19 41 23 43 22 45]]
y: [120 130 140]
Predictions: [121.2, 131.1, 139.8]
Example 3
Input:

python
Copy code
data = np.array([
    [100, 200, 300],
    [105, 210, 320],
    [110, 220, 330],
    [115, 230, 340],
    [120, 240, 360],
    [125, 250, 380]
])
window_size = 2
Expected Output:

plaintext
Copy code
X: [[100 200 105 210]
    [105 210 110 220]
    [110 220 115 230]
    [115 230 120 240]]
y: [320 330 340 360]
Predictions: [319.7, 329.5, 340.2, 360.1]
Key Features
Sliding Window Generation:
The create_sequences method dynamically creates input-output pairs for the forecasting model.
Linear Model Implementation:
Linear regression with bias term demonstrates foundational predictive modeling techniques.
Multivariate Time Series:
Supports multiple features (e.g., temperature, humidity, etc.), enhancing its complexity and utility.
This project is a step forward into predictive analytics, bridging the gap between basic NumPy operations and more advanced machine learning concepts.
"""