import numpy as np

# Example dataset: Age, Height (cm), Weight (kg), Income (USD)
sample_data = np.array([
    [25, 180, 75, 50000],
    [32, 165, 68, 48000],
    [40, 170, 72, 55000],
    [23, 175, 78, 52000]
])

# Separate target variable (Income)
Features = sample_data[:, :3]  # Features: Age, Height, Weight
Target = sample_data[:, 3]  # Target: Income


# Function to calculate correlation matrix.
# (addition) the data was first z-score normalized (meaning (data - mean)/ std), then the transpose of the data was then ran thru np.corrcoef(data)
def compute_correlation_matrix(data):
    """
    Computes the correlation matrix for a given dataset.
    :param data: 2D Numpy array with numerical data
    :return: Correlation matrix as a 2D Numpy array
    """
    mean_vals = np.mean(data, axis=0)
    # print(f"here is the mean_vals given by: {mean_vals}")
    std_vals = np.std(data, axis=0)
    # print(f"here is the std_vals given by: {std_vals}")
    standardized_data = (data - mean_vals) / std_vals
    # print(f"here is the standardized_data given by: {standardized_data}")
    t_corrcoef = np.corrcoef(standardized_data.T)
    # print(f"here is the standardized_data.T given by: {standardized_data.T}")
    # print(f"here is the t_corrcoef given by: {t_corrcoef}")
    return t_corrcoef


# Function to normalize data (min-max scaling)
def normalize_data(data):
    """
    Normalizes the data using min-max scaling.
    :param data: 2D Numpy array
    :return: Normalized dataset
    """
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    min_max = (data - min_vals) / (max_vals - min_vals)
    return min_max


# Function to compute covariance matrix
def compute_covariance_matrix(data):
    """
    Computes the covariance matrix of the dataset.
    :param data: 2D Numpy array with numerical data
    :return: Covariance matrix
    """
    mean_vals = np.mean(data, axis=0)
    centered_data = data - mean_vals  # this is called centering, it shifts the centered data mean to 0
    return np.cov(centered_data, rowvar=False)


# Linear regression using Numpy
def linear_regression(X, y):
    """
    Performs simple linear regression using the normal equation.
    :param X: 2D Numpy array for features (independent variables)
    :param y: 1D Numpy array for the target variable (dependent variable)
    :return: Coefficients for the linear regression model
    """
    X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
    coeffs = np.linalg.inv(X.T @ X) @ X.T @ y
    return coeffs


def output():
    data = sample_data
    # Separate target variable (Income)
    X = Features
    y = Target

    # The raw data
    print("\ndata (Features: Age, Height, Weight - Target: Income):")
    print(data)

    # Compute correlation matrix
    correlation_matrix = compute_correlation_matrix(data)
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # Normalize data
    normalized_data = normalize_data(data)
    print("\nNormalized Data:")
    print(normalized_data)

    # Compute covariance matrix
    covariance_matrix = compute_covariance_matrix(data)
    print("\nCovariance Matrix:")
    print(covariance_matrix)

    # Perform linear regression
    regression_coeffs = linear_regression(X, y)
    print("\nLinear Regression Coefficients:")
    print(regression_coeffs)


# Test the functions with example inputs
if __name__ == "__main__":
    # output()
    # print("output of the method: compute_correlation_matrix(sample_data)")
    # print(compute_correlation_matrix(sample_data))
    #
    # print("out of the method: compute_covariance_matrix(sample_data)")
    # print(compute_covariance_matrix(sample_data))

    print("output of the method: linear_regression(features, target)")
    print(linear_regression(Features, Target))
