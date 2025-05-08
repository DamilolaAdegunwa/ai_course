import pandas as pd
import numpy as np
from scipy import stats

new = np.array(
    [
        [0, 1, 1, 2],
        [3, 0, 2, 1],
        [1, 0, 5, 4]
    ]
)

_, p, _, _ = stats.chi2_contingency(new)

print(p)

m: pd.DataFrame = pd.DataFrame([
    [1,2,3,4],
    [10,20,30,40],
    [11,21,31,41],
])

# print(m.info())
# print(f"count(): ", m.count())
