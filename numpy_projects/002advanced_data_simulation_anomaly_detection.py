import numpy as np


# Normalizing Function:  Z-Score Normalization (Standardization)
def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std


# Data Simulation Function
def simulate_data(samples=1000, features=5, anomaly_ratio=0.05):
    """
    Simulates synthetic data with a mix of normal and anomalous values.
    """
    normal_data = np.random.normal(loc=50, scale=10, size=(int(samples * (1 - anomaly_ratio)), features))  # mean=50, sd=10, size=95% of the sample, and 5 columns
    anomalies = np.random.uniform(low=0, high=100, size=(int(samples * anomaly_ratio), features))  # min=0, max=100, size=5% of the sample, and 5 columns
    data = np.vstack([normal_data, anomalies])  # stack the "normal_data" up top of the "anomalies" data
    return data


# Anomaly Detection
def detect_anomalies(data, threshold=3.0):
    """
    Detects anomalies based on Z-scores.
    """
    z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
    anomalies = np.any(z_scores > threshold, axis=1)
    return anomalies, z_scores


# Covariance and Correlation Matrix
def compute_covariance_and_correlation(data):
    """
    Computes covariance and correlation matrices.
    """
    covariance_matrix = np.cov(data, rowvar=False)
    correlation_matrix = np.corrcoef(data, rowvar=False)
    return covariance_matrix, correlation_matrix


# Main Workflow
def main():
    # Step 1: Simulate Data
    np.random.seed(42)
    data = simulate_data(samples=1000, features=5, anomaly_ratio=0.1)

    # Step 2: Normalize Data
    normalized_data = normalize_data(data)

    # Step 3: Detect Anomalies
    anomalies, z_scores = detect_anomalies(normalized_data, threshold=3.0)

    # Step 4: Compute Covariance and Correlation
    covariance_matrix, correlation_matrix = compute_covariance_and_correlation(normalized_data)

    # Display Results
    print("Simulated Data Shape:", data.shape)
    print("Anomalies Detected:", anomalies)
    print("Number of Anomalies Detected:", np.sum(anomalies))
    print("\nCovariance Matrix:\n", covariance_matrix)
    print("\nCorrelation Matrix:\n", correlation_matrix)


if __name__ == "__main__":
    main()
    # print(simulate_data())
