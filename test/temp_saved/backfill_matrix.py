import numpy as np
import xarray as xr

# Creating a 3-dimensional numpy array (shape is (layers, rows, columns))
my_3d_array = np.array([
                        [
                        [1, 2, 3], 
                        [4, np.nan, 6],
                        [1, 2, 3],
                        [1, np.nan, 3],
                        [1, 2, 3]
                        ],

                        [
                        [7, 8, 9], 
                        [10, np.nan, 12],
                        [1, 2, 3],
                        [1, 2, 3],
                        [1, 2, 3]
                        ],

                        [
                        [13, 14, 15], 
                        [16, 17, 18],
                        [1, 2, 3],
                        [1, 2, 3],
                        [1, 2, 3]
                        ],

                        [
                        [13, 14, 15], 
                        [16, 17, 18],
                        [1, 2, 3],
                        [1, 2, 3],
                        [1, 2, 3]
                        ]
                        ])
print(my_3d_array)


w = xr.DataArray(my_3d_array, dims=['plev', 'lat', 'lon'], coords = {'plev':np.arange(1, 5), 'lat':np.arange(1, 6), 'lon':np.arange(1, 4)})
print(w)


backfilled_w = w.bfill(dim='plev')
first_non_nan_at_plev = backfilled_w.isel(plev=0)
print(first_non_nan_at_plev)



