import numpy as np

class PCA:
    def __init__(self, n_components):
        """
        Initialize the PCA class.
        - n_components: Number of principal components to retain.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """
        Fit the PCA model to the data.
        - X: Data matrix of shape (n_samples, n_features).
        """
        # Step 1: Compute the mean of each feature
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Step 2: Compute the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Step 3: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Step 4: Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Step 5: Select the top n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        """
        Project the data onto the principal components.
        - X: Data matrix of shape (n_samples, n_features).
        Returns: Transformed data matrix of shape (n_samples, n_components).
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """
        Fit the PCA model to the data and return the transformed data.
        - X: Data matrix of shape (n_samples, n_features).
        Returns: Transformed data matrix of shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X)


if __name__ == "__main__":
    # Example 1: Basic 2D dataset
    X_2D = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ])
    pca = PCA(n_components=1)
    X_transformed = pca.fit_transform(X_2D)
    print("Transformed Data (Example 1):")
    print(X_transformed)

    # Example 2: 3D dataset
    X_3D = np.random.rand(10, 3) * 10  # Random 3D dataset
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X_3D)
    print("\nTransformed Data (Example 2):")
    print(X_transformed)

    # Example 3: High-dimensional dataset
    X_high_dim = np.random.rand(100, 50)  # Random high-dimensional dataset
    pca = PCA(n_components=5)
    X_transformed = pca.fit_transform(X_high_dim)
    print("\nTransformed Data (Example 3):")
    print(X_transformed)

# Project Title: Implementing Principal Component Analysis (PCA) Using NumPy
"""
Project Title: Implementing Principal Component Analysis (PCA) Using NumPy
File Name: pca_with_numpy.py

Short Description
This project involves implementing Principal Component Analysis (PCA), a popular dimensionality reduction algorithm used in machine learning and data analysis. PCA transforms high-dimensional data into a lower-dimensional form while retaining as much variance as possible. It computes the eigenvalues and eigenvectors of the covariance matrix to determine the directions (principal components) in which the data varies the most.

This implementation uses NumPy only and provides the ability to project data onto a user-specified number of principal components.
"""

# Example Inputs and Expected Outputs
"""
Example 1: Simple 2D Dataset
Input:

python
Copy code
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
])
n_components = 1
Expected Output:

plaintext
Copy code
Transformed Data: A 4x1 matrix showing data projected onto the first principal component.
Example 2: Random 3D Dataset
Input:

python
Copy code
X = np.random.rand(10, 3) * 10  # 10 samples, 3 features
n_components = 2
Expected Output:

plaintext
Copy code
Transformed Data: A 10x2 matrix showing data projected onto the top 2 principal components.
Example 3: High-Dimensional Dataset
Input:

python
Copy code
X = np.random.rand(100, 50)  # 100 samples, 50 features
n_components = 5
Expected Output:

plaintext
Copy code
Transformed Data: A 100x5 matrix showing data projected onto the top 5 principal components.
Key Features
Dimensionality Reduction:
Enables reduction of high-dimensional datasets to fewer dimensions while retaining significant variance.
Custom Number of Components:
User can specify how many principal components to retain.
Efficient Computation:
Utilizes NumPy's matrix operations and eigen decomposition for fast computation.
Scalability:
Works on datasets with large numbers of features and samples.
This project provides a foundation for understanding PCA and its practical application in data preprocessing and visualization. It’s a step up in complexity, focusing entirely on NumPy while tackling a core concept in machine learning.
"""