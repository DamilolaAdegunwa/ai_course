import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        """
        Initialize the KMeans class.
        - n_clusters: Number of clusters (k).
        - max_iter: Maximum number of iterations.
        - tol: Tolerance for centroid movement.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None

    def fit(self, X):
        """
        Fit the K-Means model to the data.
        - X: Data matrix of shape (n_samples, n_features).
        """
        # Step 1: Initialize centroids randomly from data points
        n_samples, _ = X.shape
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iter):
            # Step 2: Assign each point to the nearest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            # Step 3: Calculate new centroids as the mean of assigned points
            new_centroids = np.array([X[self.labels == j].mean(axis=0) for j in range(self.n_clusters)])

            # Step 4: Check for convergence (if centroids do not change significantly)
            if np.all(np.linalg.norm(new_centroids - self.centroids, axis=1) < self.tol):
                break

            self.centroids = new_centroids

    def predict(self, X):
        """
        Predict the nearest cluster for new data points.
        - X: Data matrix of shape (n_samples, n_features).
        Returns: Array of shape (n_samples,) with cluster labels.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def inertia(self, X):
        """
        Calculate the sum of squared distances (inertia) of points to their centroids.
        - X: Data matrix of shape (n_samples, n_features).
        Returns: Inertia (float).
        """
        distances = np.linalg.norm(X - self.centroids[self.labels], axis=1)
        return np.sum(distances**2)


if __name__ == "__main__":
    # Example 1: Simple 2D dataset
    X_2D = np.array([
        [1, 2], [1.5, 1.8], [5, 8],
        [8, 8], [1, 0.6], [9, 11]
    ])
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X_2D)
    print("Example 1 - Cluster Labels:", kmeans.labels)
    print("Example 1 - Centroids:\n", kmeans.centroids)

    # Example 2: Random 3D dataset
    X_3D = np.random.rand(100, 3) * 10
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_3D)
    print("\nExample 2 - Cluster Labels:", kmeans.labels[:10])  # Display first 10 labels
    print("Example 2 - Inertia:", kmeans.inertia(X_3D))

    # Example 3: Larger synthetic dataset
    np.random.seed(42)
    cluster_1 = np.random.normal(0, 1, (50, 2))
    cluster_2 = np.random.normal(5, 1, (50, 2))
    cluster_3 = np.random.normal(10, 1, (50, 2))
    X_large = np.vstack([cluster_1, cluster_2, cluster_3])
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_large)
    print("\nExample 3 - Centroids:\n", kmeans.centroids)

# Implementing K-Means Clustering Using NumPy
"""
Project Title: Implementing K-Means Clustering Using NumPy
File Name: kmeans_clustering_with_numpy.py

Short Description
This project involves implementing the K-Means Clustering algorithm using only NumPy. K-Means is a popular unsupervised learning algorithm used for clustering data into k groups. The algorithm iteratively assigns points to the nearest cluster centroid and updates the centroids until convergence.

This implementation will handle multi-dimensional data and include features such as random initialization of centroids and a method to calculate the sum of squared distances (inertia) for performance evaluation.
"""

# Example Inputs and Expected Outputs
"""
Example Inputs and Expected Outputs
Example 1: Simple 2D Dataset
Input:

python
Copy code
X = np.array([
    [1, 2], [1.5, 1.8], [5, 8],
    [8, 8], [1, 0.6], [9, 11]
])
n_clusters = 2
Expected Output:

plaintext
Copy code
Cluster Labels: [0, 0, 1, 1, 0, 1]
Centroids:
[[1.16666667 1.46666667]
 [7.33333333 9.     ]]
Example 2: Random 3D Dataset
Input:

python
Copy code
X = np.random.rand(100, 3) * 10  # Random dataset with 100 samples, 3 features
n_clusters = 3
Expected Output:

plaintext
Copy code
Cluster Labels: [Array of integers indicating cluster assignments, e.g., [2, 0, 1, ...]]
Inertia: A positive float value representing the sum of squared distances of samples to their centroids.
Example 3: Synthetic 2D Dataset with Three Clusters
Input:

python
Copy code
cluster_1 = np.random.normal(0, 1, (50, 2))
cluster_2 = np.random.normal(5, 1, (50, 2))
cluster_3 = np.random.normal(10, 1, (50, 2))
X = np.vstack([cluster_1, cluster_2, cluster_3])
n_clusters = 3
Expected Output:

plaintext
Copy code
Centroids:
[[ 0.1  0.1]  # Example approximate centroid values for cluster 1
 [ 5.0  5.0]  # Example approximate centroid values for cluster 2
 [10.0 10.0]] # Example approximate centroid values for cluster 3
Key Features
Unsupervised Learning:
Capable of discovering patterns in unlabeled data.
Customizable Parameters:
User-defined number of clusters (n_clusters) and maximum iterations (max_iter).
Inertia Calculation:
Provides a performance metric to evaluate clustering quality.
Scalable:
Handles multi-dimensional and large datasets efficiently using NumPy operations.
This project takes you deeper into the realm of unsupervised learning and clustering techniques, focusing exclusively on NumPy for mathematical operations and data manipulations.
"""