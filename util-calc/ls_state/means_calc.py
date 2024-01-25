'''
Mean calculation of different dimensions
'''


import numpy as np

def get_tMean(da):
    return da.mean(dim='time', keep_attrs=True)

def get_sMean(da):
    return da.weighted(np.cos(np.deg2rad(da.lat))).mean(dim=('lat','lon'), keep_attrs=True).compute() # dask objects require the compute part






