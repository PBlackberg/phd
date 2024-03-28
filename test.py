
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



# objects = [1, 2, 3]

# a, b, c = objects

# print(a)
# print(b)
# print(c)

# import xarray as xr
# import matplotlib.pyplot as plt
# ds = xr.open_dataset('/scratch/w40/cb4968/temp_calc/rome/TaiESM1_historical/rome_1984.nc')
# print(ds['rome'])

# # ds['rome'].plot()
# # plt.show()

# import os
# plot_object = ds['rome'].plot()
# fig = plot_object[0].figure
# fig.savefig(f'{os.getcwd()}/test.png')



# sub_dirs = ['a', 'c']
# sub_folders =  ['d', 'e']

# for sub_dir, sub_folder in zip(sub_dirs, sub_folders):
#     print(sub_dir)
#     print(sub_folder)
        

# import xarray as xr
# da = xr.open_dataset('/scratch/w40/cb4968/metrics/conv_obj/o_prop_lat_95thprctile/cmip6/CESM2-WACCM_o_prop_lat_95thprctile_daily_ssp585_2070-2100regridded_144x72.nc')['o_prop_lat_95thprctile']
# print(da.isel(time = slice(0,2)).data)




# import os
# import sys
# home = os.path.expanduser("~")
# sys.path.insert(0, f'{os.getcwd()}/util-core')
# import choose_datasets as cD                          # chosen datasets
# a = 1970
# print(type(a))

# experiment = cD.experiments[0]
# print(experiment)

# if experiment == 'historical':
#     year1 = cD.cmip_years[0][0].split('-')[0]
#     year2 = cD.cmip_years[0][0].split('-')[1]
# else:
#     year1 = cD.cmip_years[0][1].split('-')[0]
#     year2 = cD.cmip_years[0][1].split('-')[1]

# print(year1)
# print(year2)



# year1, year2 = (1970, 1999)                       else (2070, 2099)                # range of years to concatenate files f


