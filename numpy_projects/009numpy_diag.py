import numpy as np


# 1a. Create a diagonal matrix from a given array
def example_1a():
    array = np.array([1, 2, 3, 4])
    diag_matrix = np.diag(array)
    print("1a. Diagonal Matrix:\n", diag_matrix)
    # Output:
    # [[1 0 0 0]
    #  [0 2 0 0]
    #  [0 0 3 0]
    #  [0 0 0 4]]


# 1b. Extract the diagonal elements from a 2D matrix
def example_1b():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    diagonal = np.diag(matrix)
    print("1b. Diagonal Elements:", diagonal)
    # Output: [1 5 9]


# 2a. Create a 2D matrix with a diagonal from a flattened array
def example_2a():
    array = [5, 10, 15]
    diag_flat = np.diagflat(array)
    print("2a. Diagonal from Flat Array:\n", diag_flat)
    # Output:
    # [[ 5  0  0]
    #  [ 0 10  0]
    #  [ 0  0 15]]


# 2b. Embed diagonal elements within a larger matrix
def example_2b():
    array = [2, 4, 6]
    diag_flat = np.diagflat(array, k=1)
    print("2b. Diagonal at Offset +1:\n", diag_flat)
    # Output:
    # [[0 2 0 0]
    #  [0 0 4 0]
    #  [0 0 0 6]
    #  [0 0 0 0]]


# 3a. Create a triangular matrix with ones below the main diagonal
def example_3a():
    tri_matrix = np.tri(4, 4, k=0, dtype=int)
    print("3a. Lower Triangular Matrix:\n", tri_matrix)
    # Output:
    # [[1 0 0 0]
    #  [1 1 0 0]
    #  [1 1 1 0]
    #  [1 1 1 1]]


# 3b. Triangular matrix for masking operations
def example_3b():
    tri_mask = np.tri(3, 5, k=-1, dtype=int)
    print("3b. Triangular Mask:\n", tri_mask)
    # Output:
    # [[0 0 0 0 0]
    #  [1 0 0 0 0]
    #  [1 1 0 0 0]]


# 4a. Extract the lower triangle of a matrix
def example_4a():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    lower_tri = np.tril(matrix)
    print("4a. Lower Triangle:\n", lower_tri)
    # Output:
    # [[1 0 0]
    #  [4 5 0]
    #  [7 8 9]]


# 4b. Lower triangular matrix with offset
def example_4b():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    lower_tri_offset = np.tril(matrix, k=-1)
    print("4b. Lower Triangle (Offset -1):\n", lower_tri_offset)
    # Output:
    # [[0 0 0]
    #  [4 0 0]
    #  [7 8 0]]


# 5a. Extract the upper triangle of a matrix
def example_5a():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    upper_tri = np.triu(matrix)
    print("5a. Upper Triangle:\n", upper_tri)
    # Output:
    # [[1 2 3]
    #  [0 5 6]
    #  [0 0 9]]


# 5b. Upper triangular matrix with offset
def example_5b():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    upper_tri_offset = np.triu(matrix, k=1)
    print("5b. Upper Triangle (Offset +1):\n", upper_tri_offset)
    # Output:
    # [[0 2 3]
    #  [0 0 6]
    #  [0 0 0]]


# 6a. Create a Vandermonde matrix
def example_6a():
    x = np.array([1, 2, 3, 4])
    vander_matrix = np.vander(x, increasing=True)
    print("6a. Vandermonde Matrix (Increasing Order):\n", vander_matrix)
    # Output:
    # [[ 1  1  1  1]
    #  [ 1  2  4  8]
    #  [ 1  3  9 27]
    #  [ 1  4 16 64]]


# 6b. Generate a Vandermonde matrix for polynomial fitting
def example_6b():
    x = np.array([1, 2, 3])
    vander_poly = np.vander(x, N=3, increasing=False)
    print("6b. Vandermonde Matrix (Polynomial Fit):\n", vander_poly)
    # Output:
    # [[ 1  1  1]
    #  [ 4  2  1]
    #  [ 9  3  1]]


# 7a. Block matrix construction
def example_7a():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    # block_matrix = np.bmat([[A, B.T], [B, None]])
    block_matrix = np.bmat([[A, B.T], [B, A.T]])
    print("7a. Block Matrix:\n", block_matrix)
    # Output:
    #  [[1 2 5 7]
    #  [3 4 6 8]
    #  [5 6 1 3]
    #  [7 8 2 4]]


# 7b. Combine identity and random matrices
def example_7b():
    I = np.eye(2)
    R = np.random.randint(1, 10, (2, 2))
    block_combo = np.bmat([[I, R], [R, I]])
    print("7b. Combined Block Matrix:\n", block_combo)
    # Output:
    # Block matrix combining identity and random matrices


# Run all examples
example_1a()
example_1b()
example_2a()
example_2b()
example_3a()
example_3b()
example_4a()
example_4b()
example_5a()
example_5b()
example_6a()
example_6b()
example_7a()
example_7b()
