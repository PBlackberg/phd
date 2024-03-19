
# import xarray as xr
# import numpy as np


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


# import pandas as pd
# import dask
# import distributed

# print(f"Pandas version: {pd.__version__}")
# print(f"Dask version: {dask.__version__}")
# print(f"Distributed version: {distributed.__version__}")



# import pandas as pd
# import pickle

# # Example object: Creating a simple pandas DataFrame
# obj = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# # Serializing the object
# obj_serialized = pickle.dumps(obj)

# # Deserializing the object
# obj_deserialized = pickle.loads(obj_serialized)

# # Verifying the deserialization
# print(obj_deserialized)


# import numpy as np
# import xarray as xr
# import pickle

# # Step 1: Create a NumPy array
# array = np.array([[1, 2, 3], [4, 5, 6]])

# # Step 2: Convert to xarray DataArray
# data_array = xr.DataArray(array, dims=["x", "y"], coords={"x": [0, 1], "y": [0, 1, 2]})

# # Step 3: Serialize and Deserialize
# data_array_serialized = pickle.dumps(data_array)
# data_array_deserialized = pickle.loads(data_array_serialized)

# # Print the deserialized DataArray
# print(data_array_deserialized)


# import cftime
# import netCDF4

# print("cftime version:", cftime.__version__)
# print("netCDF4 version:", netCDF4.__version__)




# import shapely
# print(shapely.__version__)



objects = [1, 2, 3]

a, b, c = objects

print(a)
print(b)
print(c)
