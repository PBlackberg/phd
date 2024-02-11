import numpy as np

a = np.array(['a', 'b', 'c'])
b = np.array(['a', 'b', 'd'])

result = np.intersect1d(a, b)
print(result)