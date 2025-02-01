import numpy as np

data = [('Laptop', 999.99, 25), ('Smartphone', 699.99, 50), ('Tablet', 499.99, 30)]
dtype = [('product', 'U15'), ('price', 'f4'), ('stock', 'i4')]
catalog = np.rec.array(data, dtype=dtype)
catalog_valuation = (catalog.price * catalog.stock).sum()
print(catalog_valuation)