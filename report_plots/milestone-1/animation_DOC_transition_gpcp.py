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
home = os.path.expanduser("~")
import myFuncs_m as mF                                
import myVars_m as mV                                 

# -------------------------------------------------------------------------------------------- Get data ----------------------------------------------------------------------------------------------------- #
# subset on days between certain areafraction
areafraction = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/ni/obs/GPCP_ni_95thPrctile_daily__orig_short.nc')['areafraction']
condition = (areafraction >= 3.5) & (areafraction <= 6.5)
subset_times = areafraction['time'].where(condition, drop=True)

# pr
da = xr.open_dataset('/Users/cbla0002/Documents/data/pr/sample_data/obs/GPCP_pr_daily__orig.nc')['pr'].sel(time= slice('2010', '2022'))
conv_threshold = da.quantile(0.95, dim=('lat', 'lon'), keep_attrs=True)
conv_threshold = xr.DataArray(data = conv_threshold.mean(dim='time').data * np.ones(shape = len(da.time)), dims = 'time', coords = {'time': da.time.data}) 
da_0 = da.where(da >= conv_threshold)
da_0 = da_0.sel(time=subset_times)

# pr99
conv_threshold = da.quantile(0.99, dim=('lat', 'lon'), keep_attrs=True)
da_1 = da.where(da >= conv_threshold)
da_1 = da_1.sel(time=subset_times)

# timestep based on rome
rome = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/rome/obs/GPCP_rome_95thPrctile_daily__orig_short.nc')['rome']
rome = rome.sel(time=subset_times)
low, mid_1, mid_2, high = 3, 48.5, 51.5, 97
# low, mid_1, mid_2, high = 0.25, 49.75, 50.25, 99.9
timesteps_low  = np.squeeze(np.argwhere(rome.data  <= np.percentile(rome, low)))
timesteps_mid  = np.squeeze(np.argwhere((rome.data >= np.percentile(rome, mid_1)) & (rome.data <= np.percentile(rome, mid_2))))
timesteps_high = np.squeeze(np.argwhere(rome.data  >= np.percentile(rome, high)))
timesteps = np.concatenate((timesteps_low, timesteps_mid, timesteps_high))
# timesteps = timesteps_low
# timesteps = np.concatenate((timesteps_low, timesteps_high))

class variable():
    def __init__(self, cmap, label):
        self.cmap = cmap
        self.label = label
variable_0 = variable('Greys', r'pr [mm day$^{-1}$]')
variable_1 = variable('Reds', r'pr [mm day$^{-1}$]')
title = 'transition of DOC'

# -------------------------------------------------------------------------------------- animate / format plot ----------------------------------------------------------------------------------------------------- #
def plot_ax_scene(frame, fig, da_0, da_1, timesteps, variable_0, variable_1, title):
    lat, lon = da_0.lat, da_0.lon
    lonm,latm = np.meshgrid(lon,lat)
    ax = fig.add_subplot(projection=ccrs.PlateCarree(central_longitude=180))
    ax.add_feature(cfeat.COASTLINE)
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
    timestep = timesteps_high[frame]
    pcm_0 = ax.pcolormesh(lonm,latm, da_0.isel(time = timestep), transform=ccrs.PlateCarree(),zorder=0, cmap = variable_0.cmap, vmin=0, vmax=80)
    pcm_1 = ax.pcolormesh(lonm,latm, da_1.isel(time = timestep), transform=ccrs.PlateCarree(),zorder=0, cmap = variable_1.cmap, vmin=0, vmax=80)
    mF.scale_ax(ax, scaleby = 1.15)
    mF.move_col(ax, moveby = -0.055)
    mF.move_row(ax, moveby = 0.075)

    if frame == 0:
        # mF.plot_axtitle(fig, ax, title, xpad = 0.005, ypad = 0.035, fontsize=15)
        mF.plot_xlabel(fig, ax, xlabel='Lon', pad = 0.1, fontsize = 12)
        mF.plot_ylabel(fig, ax, ylabel='Lat', pad = 0.055, fontsize = 12)
        mF.format_ticks(ax, labelsize = 11)
        mF.cbar_below_axis(fig, ax, pcm_1, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = variable_1.label, text_pad = 0.125)
    plt.close()

def animate(da_0, da_1, timesteps, variable_0, variable_1, title):
    fig= plt.figure(figsize=(12, 4))
    ani = animation.FuncAnimation(
        fig,                          
        plot_ax_scene,                                                               # name of the function
        frames = len(timesteps_high),                                                     # can also be iterable or list
        interval = 500,                                                              # ms between frames
        fargs=(fig, da_0, da_1, timesteps, variable_0, variable_1, title)            # additional function arguments
        )
    return ani

# ----------------------------------------------------------------------------------- Run animation and save ----------------------------------------------------------------------------------------------------- #
print(f'{len(timesteps)} frames included')
ani = animate(da_0, da_1, timesteps, variable_0, variable_1, title)
ani.save(f'{home}/Desktop/DOC_high_gpcp.mp4')
print('finished')

