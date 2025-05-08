# panel 1
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
X, y = make_classification(
    n_samples=100_000, n_features=20, n_informative=2, n_redundant=10, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.99, random_state=42
)

# --- features and target to numpy
X: np.ndarray = X
y: np.ndarray = y

# --- train, test to numpy
X_train: np.ndarray = np.array(X_train)
X_test: np.ndarray = np.array(X_test)
y_train: np.ndarray = np.array(y_train)
y_test: np.ndarray = np.array(y_test)

# --- features std_out
# print(f"X: {pd.DataFrame(X).describe()}")  # count,mean,std,min,25%,50%,75%,max
# print(f"X[:3, :3]: {X[:3, :3]}")
# print(f"np.array(X.columns): {np.array(X.columns)}")  # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
print(f"X[:3]: {X[:3].__dir__()}")
# print(f"X.shape: ", X.shape)  # X.shape:  (100000, 20)

# target std_out
# print(f"y.shape", y.shape)  # y.shape (100000, 1)
