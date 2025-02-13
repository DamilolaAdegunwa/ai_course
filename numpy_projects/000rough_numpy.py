import numpy as np

## test 1
# data = [('Laptop', 999.99, 25), ('Smartphone', 699.99, 50), ('Tablet', 499.99, 30)]
# dtype = [('product', 'U15'), ('price', 'f4'), ('stock', 'i4')]
# catalog = np.rec.array(data, dtype=dtype)
# catalog_valuation = (catalog.price * catalog.stock).sum()
# print(catalog_valuation)

## test 2
# numbers = [1, 20, 300]
# serial = [str(x).zfill(4) for x in numbers]
# print(serial)

## test 3
x = np.arange(0, 4, 1)  # column
y = np.arange(10, 24, 1)  # row
grid = np.meshgrid(x, y)
print('**************************')
print(grid[0])
print('**************************')
print(grid[1])
print(np.size(grid))
print(np.shape(grid))
