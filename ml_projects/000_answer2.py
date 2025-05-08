import pandas as pd

arr = [[1, 2], [4, 5], [7, 8]]
df = pd.DataFrame(arr, index=['a', 'b', 'c'], columns=['A', 'B'])
output = df.loc[:, 'A']
print(output)
