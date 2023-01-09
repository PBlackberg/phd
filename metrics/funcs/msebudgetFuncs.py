
import xarray as xr

import numpy as np
import skimage.measure as skm
import scipy
import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat

import os
#from cmip6_metrics.funcs import myPlots

from funcs.vars import myFuncs
from funcs.vars import myPlots


model='MPI-ESM1-2-HR'
experiment_id='historical'
folder = '/Users/cbla0002/Documents/data/cmip6/' + model


fileName = model + '_mse_vInt_' + experiment_id + '.nc'
path = folder + '/' + fileName
ds = xr.open_dataset(path)
mse = ds.mse_vInt



myPlots.plot_snapshot(mse.isel(time=0), 'viridis', 'mse_vInt', model)
plt.show()



# mse_mean = mse.mean(dim=('lat','lon'))
# print(np.shape(mse_mean))

# mse_anom = mse - mse_mean
# print(np.shape(mse_anom))


# mse_anom.plot()


#myPlots.plot_snapshot(mse_anom.isel(time=0), 'PRGn', 'mse anomaly', model)







