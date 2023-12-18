''' 
This script shows an example of rearranging the data dimensions while preserving the coordaintes. 
Sometimes datasets will for example have latitudes in ascending or descending order
'''



import xarray as xr
import numpy as np



lat_values = np.array([30, 20, 10, 0, -10, -20, -30])
lon_values = np.array([0, 10, 20, 30])
data = np.random.rand(len(lat_values), len(lon_values))
dataset = xr.DataArray(data, coords=[("lat", lat_values), ("lon", lon_values)])     # create matrix
original_dataset = dataset.copy()


sorted_dataset = dataset.sortby("lat")                                              # sort latitudes by ascending order, while preserving coordiantes


print('orginal dataset: \n', original_dataset)
print('orginal dataset: \n', sorted_dataset)

































