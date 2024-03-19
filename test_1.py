
import xarray as xr
import numpy as np


# ds = xr.open_dataset('/scratch/w40/cb4968/sample_data/conv_obj/cmip6/NorESM2-MM_conv_obj_daily_ssp585_regridded_144x72.nc')
# print(ds)
# ds = ds.sel(time = '2070')
# print(ds)
# ds = ds.sel(time = '2070-01')
# print(ds)
# print(ds.time)

# for year in np.unique(ds['time'].dt.year):
#     print(year)
#     print(ds['time'].sel(time = f'{year}'))
#     exit()

# print(ds.sel(time=ds['time'].dt.year == year))


# import itertools

# a = [[1, 2], [3, 4]]
# b = list(itertools.chain.from_iterable(a))

# print(b)
