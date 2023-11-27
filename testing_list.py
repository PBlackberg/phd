import matplotlib.pyplot as plt
import numpy as np

a = np.array([1, 2, 3])
b = np.array([np.nan, 2, 3])


# plt.plot(a, b)
# plt.show()


print(np.nanmin(b))

