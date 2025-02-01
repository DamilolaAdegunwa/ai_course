import numpy as np

class DynamicTimeWarping:
    def __init__(self):
        """
        Initialize the DTW class.
        """
        self.cost_matrix = None
        self.path = None

    def compute_distance(self, seq1, seq2):
        """
        Compute the DTW distance and alignment path between two sequences.
        - seq1: First sequence (1D NumPy array).
        - seq2: Second sequence (1D NumPy array).
        Returns: DTW distance and alignment path.
        """
        n, m = len(seq1), len(seq2)
        self.cost_matrix = np.zeros((n, m))

        # Initialize the cost matrix
        self.cost_matrix[0, 0] = abs(seq1[0] - seq2[0])
        for i in range(1, n):
            self.cost_matrix[i, 0] = self.cost_matrix[i - 1, 0] + abs(seq1[i] - seq2[0])
        for j in range(1, m):
            self.cost_matrix[0, j] = self.cost_matrix[0, j - 1] + abs(seq1[0] - seq2[j])

        # Fill in the rest of the cost matrix
        for i in range(1, n):
            for j in range(1, m):
                cost = abs(seq1[i] - seq2[j])
                self.cost_matrix[i, j] = cost + min(
                    self.cost_matrix[i - 1, j],    # insertion
                    self.cost_matrix[i, j - 1],    # deletion
                    self.cost_matrix[i - 1, j - 1] # match
                )

        # Backtrack to find the optimal alignment path
        i, j = n - 1, m - 1
        path = [(i, j)]
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                step = np.argmin([
                    self.cost_matrix[i - 1, j],    # insertion
                    self.cost_matrix[i, j - 1],    # deletion
                    self.cost_matrix[i - 1, j - 1] # match
                ])
                if step == 0:
                    i -= 1
                elif step == 1:
                    j -= 1
                else:
                    i -= 1
                    j -= 1
            path.append((i, j))

        self.path = path[::-1]
        return self.cost_matrix[-1, -1], self.path

if __name__ == "__main__":
    # Example 1: Basic similarity
    seq1 = np.array([1, 2, 3, 4, 5])
    seq2 = np.array([2, 3, 4])
    dtw = DynamicTimeWarping()
    distance, path = dtw.compute_distance(seq1, seq2)
    print("Example 1:")
    print(f"DTW Distance: {distance}")
    print(f"Optimal Path: {path}")

    # Example 2: Different sequence lengths
    seq1 = np.array([1, 3, 4, 9])
    seq2 = np.array([1, 2, 3, 4, 7])
    distance, path = dtw.compute_distance(seq1, seq2)
    print("\nExample 2:")
    print(f"DTW Distance: {distance}")
    print(f"Optimal Path: {path}")

    # Example 3: Sinusoidal sequences
    seq1 = np.sin(np.linspace(0, np.pi, 50))
    seq2 = np.sin(np.linspace(0, np.pi, 30))
    distance, path = dtw.compute_distance(seq1, seq2)
    print("\nExample 3:")
    print(f"DTW Distance: {distance}")
    print(f"Optimal Path: {path}")

# Dynamic Time Warping (DTW) Implementation Using NumPy
"""
Project Title: Dynamic Time Warping (DTW) Implementation Using NumPy
File Name: dtw_with_numpy.py

Short Description
Dynamic Time Warping (DTW) is an advanced algorithm for measuring similarity between two temporal sequences that may vary in speed or time. Unlike traditional distance metrics, DTW allows for non-linear alignments of sequences, making it especially useful for time series analysis in fields such as speech recognition, bioinformatics, and finance.

This project implements DTW using NumPy only, allowing for both distance computation and path alignment between two sequences. This is a significant step up in complexity, involving dynamic programming, optimization, and multi-dimensional matrix manipulations.
"""

# Example Inputs and Expected Outputs
"""
Example Inputs and Expected Outputs
Example 1: Basic Similarity
Input:

python
Copy code
seq1 = np.array([1, 2, 3, 4, 5])
seq2 = np.array([2, 3, 4])
Expected Output:

plaintext
Copy code
DTW Distance: A scalar value representing the total alignment cost.
Optimal Path: A list of index pairs showing the alignment between the sequences.
Example 2: Different Sequence Lengths
Input:

python
Copy code
seq1 = np.array([1, 3, 4, 9])
seq2 = np.array([1, 2, 3, 4, 7])
Expected Output:

plaintext
Copy code
DTW Distance: A scalar value representing the alignment cost for the sequences.
Optimal Path: Index pairs indicating how the sequences align optimally.
Example 3: Sinusoidal Sequences
Input:

python
Copy code
seq1 = np.sin(np.linspace(0, np.pi, 50))
seq2 = np.sin(np.linspace(0, np.pi, 30))
Expected Output:

plaintext
Copy code
DTW Distance: A scalar value reflecting the alignment cost between the sinusoidal sequences.
Optimal Path: The alignment path for the sequences.
Key Features
Time Series Analysis:
Ideal for comparing temporal sequences with differing lengths and speeds.
Dynamic Programming:
Uses a cost matrix to compute the optimal alignment efficiently.
Customization:
Works on numeric sequences of varying complexity (e.g., simple integers, continuous functions).
Visualization Potential:
Can be extended to plot alignment paths for better interpretability.
This project significantly advances your NumPy expertise while introducing you to practical algorithms for time series and sequence analysis.
"""