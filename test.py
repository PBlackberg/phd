

import numpy as np
import xarray as xr

a = xr.DataArray(data = [1, 2, 3], dims = 'time', coords = {'time': [1, 2, 3]})
b = a.mean(dim='time').data*np.ones(shape = len(a.time))
print(b)


c = xr.DataArray(data = a.mean(dim='time').data*np.ones(shape = len(a.time)), dims = 'time', coords = {'time': a.time.data}) 
print(c)