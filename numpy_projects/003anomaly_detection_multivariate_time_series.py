import numpy as np

def normalize_data(data):
    """
    Normalizes the data using z-score normalization.
    :param data: 2D Numpy array where rows are observations and columns are features.
    :return: Normalized dataset.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

def compute_mahalanobis_distance(data, mean_vector, inv_cov_matrix):
    """
    Computes the Mahalanobis distance for each observation in the dataset.
    :param data: 2D Numpy array where rows are observations.
    :param mean_vector: 1D Numpy array of mean values for each feature.
    :param inv_cov_matrix: Inverse of the covariance matrix.
    :return: 1D Numpy array of Mahalanobis distances.
    """
    diff = data - mean_vector
    distances = np.sqrt(np.sum((diff @ inv_cov_matrix) * diff, axis=1))
    return distances

def detect_anomalies(data, threshold=3.0):
    """
    Detects anomalies in the dataset based on Mahalanobis distance.
    :param data: 2D Numpy array where rows are observations.
    :param threshold: Distance threshold to classify anomalies.
    :return: Indices of anomalies.
    """
    mean_vector = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    distances = compute_mahalanobis_distance(data, mean_vector, inv_cov_matrix)
    anomalies = np.where(distances > threshold)[0]
    return anomalies, distances

# Example Test
if __name__ == "__main__":
    # Simulated multivariate time series data (rows = observations, cols = features)
    np.random.seed(42)
    normal_data = np.random.normal(loc=0, scale=1, size=(100, 3))  # Normal observations
    anomaly_data = np.random.normal(loc=10, scale=1, size=(5, 3))  # Anomalies

    # Combine data
    data = np.vstack([normal_data, anomaly_data])

    # Normalize data
    normalized_data = normalize_data(data)

    # Detect anomalies
    anomalies, distances = detect_anomalies(normalized_data, threshold=3.0)

    print("Anomalies detected at indices:", anomalies)
    print("Mahalanobis Distances:", distances[anomalies])
