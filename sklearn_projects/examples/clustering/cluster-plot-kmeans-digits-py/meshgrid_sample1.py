import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = [2, 3, 3]; print(f"x: {x}")
y = [5, 6, 7, 8, 9, 10]; print(f"y: {y}")
X, Y = np.meshgrid(x, y); print(f"X: {X}"); print(f"Y: {Y}")

print(X.shape); # 6 by 3 (yrows, xrows)
"""
[2, 3, 3]
[2, 3, 3]
[2, 3, 3]
[2, 3, 3]
[2, 3, 3]
[2, 3, 3]
"""
print(Y.shape); # 6 by 3
"""
[5,5,5]
[6,6,6]
[7,7,7]
[8,8,8]
[9,9,9]
[10,10,10]
"""
what = np.c_[X.ravel(), Y.ravel()]
print(f"what: {what}")
