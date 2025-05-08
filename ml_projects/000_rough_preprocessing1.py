import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing

# Data
x = np.array([[100_000, 150_000, 350_000, 200_000],
              [1, 2, 3, 2]])
x = x.T

# preprocess data
x_standardized = preprocessing.scale(x)  # i guess it uses 'StandardScaler' by default
print("\n\n\n --- x_standardized: \n", x_standardized)

# manual
x_std_manual = (x - np.mean(x, axis=0))/(np.std(x, axis=0))
# axis=0 => columns & axis=1 => rows
# print("x_std_manual: \n", x_std_manual)

# fit & transform
x_train = x
# print(x_train)

model1 = preprocessing.StandardScaler().fit(x_train)
# print("scaler.mean_: \n", scaler.mean_)
# print("scaler.var_: \n", scaler.var_)

x_train_pred = model1.transform(x_train)  # 'transform' is better called 'predict'
print("\n\n\n --- x_train_pred: \n", x_train_pred)

# test
x_test = np.array([[110_000, 130_000, 250_000, 290_000],
                   [2, 1, 3, 3]])
x_test = x_test.T
# print("x_test: \n", x_test)

x_test_pred = model1.transform(x_test)  # 'transform' is better called 'predict'
print("\n\n\n --- x_test_pred: \n", x_test_pred)

x_test_standardized = preprocessing.scale(x_test)
print("\n\n\n --- x_test_standardized: ", x_test_standardized)

print("randint: ", np.random.randint(0, 10, 50))

fraction: float = 12.34567
print("round: = ", round(fraction, ndigits=2))
