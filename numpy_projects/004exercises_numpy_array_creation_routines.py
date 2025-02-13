# exercises_numpy_array_creation_routines.py

import numpy as np

# ===========================
# 1. empty(shape[, dtype, order, device, like])
# ===========================
# Exercise 1: Create an uninitialized array of shape (3, 4).
array1 = np.empty((3, 4))
print("Exercise 1: empty((3, 4))\n", array1, "\n")

# Exercise 2: Create an uninitialized 1D array with 5 elements and dtype float32.
array2 = np.empty(5, dtype=np.float32)
print("Exercise 2: empty(5, dtype=np.float32)\n", array2, "\n")

# Exercise 2b:
array2b = np.empty(5, dtype=np.int64)
print("Exercise 2b: empty(5, dtype=np.int64)\n", array2b, "\n")

# ===========================
# 2. empty_like(prototype[, dtype, order, subok, ...])
# ===========================
# Exercise 3: Create an uninitialized array with the same shape as a given array.
prototype = np.array([[1, 2], [3, 4], [5, 6]])
array3 = np.empty_like(prototype)
print("Exercise 3: empty_like(prototype)\n", array3, "\n")

# Exercise 4: Create an uninitialized array with the same shape and dtype as the prototype, overriding the dtype to int.
array4 = np.empty_like(prototype, dtype=int)
print("Exercise 4: empty_like(prototype, dtype=int)\n", array4, "\n")

# ===========================
# 3. eye(N[, M, k, dtype, order, device, like])
# ===========================
# Exercise 5: Create a 2D array with ones on the main diagonal, size 4x4.
array5 = np.eye(4)
print("Exercise 5: eye(4)\n", array5, "\n")

# Exercise 6: Create a 3x5 array with ones on the diagonal starting from the second column.
array6 = np.eye(3, 5, k=1)
print("Exercise 6: eye(3, 5, k=1)\n", array6, "\n")

array6b = np.eye(3, 5)
print("Exercise 6b: eye(3, 5)\n", array6b, "\n")

# ===========================
# 4. identity(n[, dtype, like])
# ===========================
# Exercise 7: Create a 5x5 identity matrix.
array7 = np.identity(5)
print("Exercise 7: identity(5)\n", array7, "\n")

# Exercise 8: Create a 3x3 identity matrix with dtype float64.
array8 = np.identity(3, dtype=np.float64)
print("Exercise 8: identity(3, dtype=np.float64)\n", array8, "\n")

# ===========================
# 5. ones(shape[, dtype, order, device, like])
# ===========================
# Exercise 9: Create a 2x3 array filled with ones.
array9 = np.ones((2, 3))
print("Exercise 9: ones((2, 3))\n", array9, "\n")

# Exercise 10: Create a 3D array of shape (4, 2, 2) filled with ones, dtype int.
array10 = np.ones((4, 2, 2), dtype=int)
print("Exercise 10: ones((4, 2, 2), dtype=int)\n", array10, "\n")

# ===========================
# 6. ones_like(a[, dtype, order, subok, shape, ...])
# ===========================
# Exercise 11: Create an array of ones with the same shape as the prototype array.
array11 = np.ones_like(prototype)
print("Exercise 11: ones_like(prototype)\n", array11, "\n")

# Exercise 12: Create an array of ones with the same shape as the prototype array but dtype float.
array12 = np.ones_like(prototype, dtype=float)
print("Exercise 12: ones_like(prototype, dtype=float)\n", array12, "\n")

# ===========================
# 7. zeros(shape[, dtype, order, like])
# ===========================
# Exercise 13: Create a 4x4 array filled with zeros.
array13 = np.zeros((4, 4))
print("Exercise 13: zeros((4, 4))\n", array13, "\n")

# Exercise 14: Create a 1D array with 6 elements filled with zeros, dtype int.
array14 = np.zeros(6, dtype=int)
print("Exercise 14: zeros(6, dtype=int)\n", array14, "\n")

# ===========================
# 8. zeros_like(a[, dtype, order, subok, shape, ...])
# ===========================
# Exercise 15: Create an array of zeros with the same shape as the prototype array.
array15 = np.zeros_like(prototype)
print("Exercise 15: zeros_like(prototype)\n", array15, "\n")

# Exercise 16: Create an array of zeros with the same shape and dtype as the prototype, overriding the dtype to float64.
array16 = np.zeros_like(prototype, dtype=np.float64)
print("Exercise 16: zeros_like(prototype, dtype=np.float64)\n", array16, "\n")

# ===========================
# 9. full(shape, fill_value[, dtype, order, ...])
# ===========================
# Exercise 17: Create a 2x3x3 array filled with the value 7.
array17 = np.full((2, 3, 3), 7)
print("Exercise 17: full((2, 3, 3), 7)\n", array17, "\n")

# Exercise 18: Create a 2x4 array filled with the value -5, dtype float.
array18 = np.full((2, 4), -5, dtype=float)
print("Exercise 18: full((2, 4), -5, dtype=float)\n", array18, "\n")

# ===========================
# 10. full_like(a, fill_value[, dtype, order, ...])
# ===========================
# Exercise 19: Create a full array with the same shape as the prototype and filled with the value 9.
array19 = np.full_like(prototype, 9)
print("Exercise 19: full_like(prototype, 9)\n", array19, "\n")

# Exercise 20: Create a full array with the same shape as the prototype, filled with the value 3.14 and dtype float.
array20 = np.full_like(prototype, 3.14, dtype=float)
print("Exercise 20: full_like(prototype, 3.14, dtype=float)\n", array20, "\n")
