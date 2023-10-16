


import xarray as xr
import numpy as np

a = xr.DataArray([1, 2, np.nan])

print(a.sum())

