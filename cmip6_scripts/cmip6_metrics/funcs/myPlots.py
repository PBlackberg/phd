import intake
import xarray as xr

import numpy as np
import skimage.measure as skm

import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeat

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 






def plot_snapshot(var, cmap, variable_name, model):
    projection = ccrs.PlateCarree(central_longitude=180)
    lat = var.lat
    lon = var.lon

    f, ax = plt.subplots(subplot_kw=dict(projection=projection), figsize=(15, 5))

    var.plot(transform=ccrs.PlateCarree(), cbar_kwargs={'orientation': 'horizontal','pad':0.125, 'aspect':50,'fraction':0.055}, cmap=cmap)
    ax.add_feature(cfeat.COASTLINE)
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
    ax.set_title(variable_name + ' snapshot, model:' + model)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels([0, 90, 180, 270, 360])
    ax.set_yticks([-20, 0, 20])
    plt.tight_layout()



    


# f, ax = plt.subplots(figsize=(15, 5))

# da = xr.DataArray(rx1day.mean(dim=('lat','lon'),keep_attrs=True).data)
# da.plot(ax=ax, label='rx1day_sMean')
# ax.set_title('rx1day_sMean, model:' + model + ' exp:' + experiment)
# ax.set_ylabel('Rx1day (mm/day)')
# ax.set_xlabel('year')
# ax.legend(loc = 'upper left')

# plt.tight_layout()




