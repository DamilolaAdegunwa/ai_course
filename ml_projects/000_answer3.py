import pandas as pd

one = ['one', '1', 'one1']
a = ['A', 'a', 'Aa']

df1 = pd.DataFrame({'one': one, 'a': a})
df1 = df1[['one', 'a']]

a = ['A', 'a', 'Aa']
b = ['B', 'b', 'Bb']

df2 = pd.DataFrame({'a': a, 'b': b})
df2 = df2[['a', 'b']]

output = df1.merge(df2, on='a', how='inner')

print(df1)
print(df2)
print(output)
