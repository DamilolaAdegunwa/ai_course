import numpy as np
import pandas as pd
a = np.array(np.meshgrid([2, 3], [5, 7]))
print(a.flatten().__len__())
print(a.ravel().__len__())
print(a)
print(a.ndim)
