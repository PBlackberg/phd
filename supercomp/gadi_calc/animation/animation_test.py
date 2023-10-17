import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import matplotlib.pyplot as plt
from matplotlib import animation

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util')
import myFuncs as mF            # imports common operators
import myVars as mV             # imports common variables
import constructed_fields as cF # imports fields for testing
import get_data as gD           # imports functions to get data from gadi


def plot_ax_scene(frame, fig, switch, da_0, da_1, timesteps, variable_0, variable_1, title):
    lat, lon = da_0.lat, da_0.lon
    lonm,latm = np.meshgrid(lon,lat)
    ax = fig.add_subplot(projection=ccrs.PlateCarree(central_longitude=180))
    ax.add_feature(cfeat.COASTLINE)
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
    timestep = timesteps[frame]
    pcm_0 = ax.pcolormesh(lonm,latm, da_0.isel(time = timestep), transform=ccrs.PlateCarree(),zorder=0, cmap = 'Purples', vmin= 150, vmax = 350)
    pcm_1 = ax.pcolormesh(lonm,latm, da_1.isel(time = timestep), transform=ccrs.PlateCarree(),zorder=0, cmap = variable_1.cmap, vmin=0, vmax=80) if switch['field ontop'] else None
    mF.scale_ax(ax, scaleby = 1.15)
    mF.move_col(ax, moveby = -0.055)
    mF.move_row(ax, moveby = 0.075)

    if frame == 0:
        mF.plot_axtitle(fig, ax, title, xpad = 0.005, ypad = 0.035, fontsize=15)
        mF.plot_xlabel(fig, ax, xlabel='Lon', pad = 0.1, fontsize = 12)
        mF.plot_ylabel(fig, ax, ylabel='Lat', pad = 0.055, fontsize = 12)
        mF.format_ticks(ax, labelsize = 11)
        mF.cbar_below_axis(fig, ax, pcm_1, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = variable_1.label, text_pad = 0.125) if switch['field ontop'] \
            else mF.cbar_below_axis(fig, ax, pcm_0, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = '', text_pad = 0.125) 
    plt.close()

def animate(switch, da_0, da_1, timesteps, variable_0, variable_1, title):
    fig= plt.figure(figsize=(12, 4))
    ani = animation.FuncAnimation(
        fig,                          
        plot_ax_scene,                                                               # name of the function
        frames = len(timesteps),                                                     # can also be iterable or list
        interval = 500,                                                              # ms between frames
        fargs=(fig, switch, da_0, da_1, timesteps, variable_0, variable_1, title)    # additional function arguments
        )
    return ani



# da_0 = xr.open_dataset('/Users/cbla0002/Documents/data/lw/sample_data/cmip6/TaiESM1_rlut_monthly_historical_regridded.nc')['rlut'] # outgoing longwave radiation


switch = {'field ontop': False}
da_1 = '' 
timesteps = np.arange(1, 50) 
variable_0 = '' 
variable_1 = '' 
title = ''

ani = animate(switch, da_0, da_1, timesteps, variable_0, variable_1, title)
ani.save(f'{home}/Desktop/test.mp4')



