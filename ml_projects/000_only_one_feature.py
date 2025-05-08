import pandas as pd
import numpy as np
X:np.ndarray = np.array([
    [1, 2, 3],
    [2, 4, 6],
    [3, 6, 9],
    [4, 8, 12],
    [5, 10, 15],
])

X_1f = X[:, [2]]
print(f"X_1f: ", X_1f);

X_1fb = X[:, 2]
print(f"X_1fb: ", X_1fb);
