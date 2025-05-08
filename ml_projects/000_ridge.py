# 1) Non-negative least squares
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# 2) Generate some random data
np.random.seed(42)
n_samples, n_features = 200, 50
X = np.random.randn(n_samples, n_features)
true_coef = 3 * np.random.randn(n_features)
print(f"true_coef: {true_coef}, true_coeff shape {true_coef.shape} \n\n\n\n")
# Threshold coefficients to render them non-negative
true_coef[true_coef < 0] = 0
y = np.dot(X, true_coef)
print(f"y (np.dot(X, true_coef)): {y}, y-dot-shape {y.shape}  \n\n\n\n")
# 3) Add some noise
y += 5 * np.random.normal(size=(n_samples,))
print(f"y (y += 5 * np.random.normal(size=(n_samples,))) {y}, , y-random-normal-shape {y.shape}  \n\n\n\n")

# Split the data in train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0, solver='saga')
ridge.fit(X_train, y_train)
