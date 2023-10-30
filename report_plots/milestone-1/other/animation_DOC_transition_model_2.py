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
# Find subset of days conditioned on areafraction
areafraction = xr.open_dataset(f'/Users/cbla0002/Documents/data/org/metrics/ni/cmip6/{mV.datasets[0]}_ni_95thPrctile_daily_historical_regridded.nc')['areafraction']
condition = (areafraction >= 3.5) & (areafraction <= 6.5)
subset_times = areafraction['time'].where(condition, drop=True)

# Get original grid variable
# pr
da = xr.open_dataset(f'/Users/cbla0002/Documents/data/pr/sample_data/cmip6/{mV.datasets[0]}_pr_daily_historical_orig.nc')['pr']
conv_threshold = da.quantile(0.95, dim=('lat', 'lon'), keep_attrs=True)
conv_threshold = xr.DataArray(data = conv_threshold.mean(dim='time').data * np.ones(shape = len(da.time)), dims = 'time', coords = {'time': da.time.data}) 
da_0 = da.where(da >= conv_threshold)
da_0 = da_0.sel(time=subset_times)
# pr99
conv_threshold = da.quantile(0.99, dim=('lat', 'lon'), keep_attrs=True)
da_1 = da.where(da >= conv_threshold)
da_1 = da_1.sel(time=subset_times)

# Get regridded field
# pr_regrid
da = xr.open_dataset(f'/Users/cbla0002/Documents/data/pr/sample_data/cmip6/{mV.datasets[0]}_pr_daily_historical_regridded.nc')['pr']
conv_threshold = da.quantile(0.95, dim=('lat', 'lon'), keep_attrs=True)
conv_threshold = xr.DataArray(data = conv_threshold.mean(dim='time').data * np.ones(shape = len(da.time)), dims = 'time', coords = {'time': da.time.data}) 
da_0_regrid = da.where(da >= conv_threshold)
da_0_regrid = da_0_regrid.sel(time=subset_times)
# pr99_regrid
conv_threshold = da.quantile(0.99, dim=('lat', 'lon'), keep_attrs=True)
da_1_regrid = da.where(da >= conv_threshold)
da_1_regrid = da_1_regrid.sel(time=subset_times)


# timestep based on rome
rome = xr.open_dataset(f'/Users/cbla0002/Documents/data/org/metrics/rome/cmip6/{mV.datasets[0]}_rome_95thPrctile_daily_historical_regridded.nc')['rome']
rome = rome.sel(time=subset_times)
# low, mid_1, mid_2, high = 1, 49.5, 51.5, 99
low, mid_1, mid_2, high = 0.2, 49.9, 50.1, 99.8
timesteps_low  = np.squeeze(np.argwhere(rome.data  <= np.percentile(rome, low)))
timesteps_mid  = np.squeeze(np.argwhere((rome.data >= np.percentile(rome, mid_1)) & (rome.data <= np.percentile(rome, mid_2))))
timesteps_high = np.squeeze(np.argwhere(rome.data  >= np.percentile(rome, high)))
timesteps = np.concatenate((timesteps_low, timesteps_mid, timesteps_high))
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
    ax = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    ax.add_feature(cfeat.COASTLINE)
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
    timestep = timesteps_high[frame]
    pcm_0 = ax.pcolormesh(lonm,latm, da_0.isel(time = timestep), transform=ccrs.PlateCarree(),zorder=0, cmap = variable_0.cmap, vmin=0, vmax=80)
    pcm_1 = ax.pcolormesh(lonm,latm, da_1.isel(time = timestep), transform=ccrs.PlateCarree(),zorder=0, cmap = variable_1.cmap, vmin=0, vmax=80)
    mF.scale_ax(ax, scaleby = 1.15)
    mF.move_col(ax, moveby = -0.025)
    # mF.move_row(ax, moveby = 0.075)
    if frame == 0:
        mF.plot_axtitle(fig, ax, title, xpad = 0.005, ypad = 0.02, fontsize=10)
        mF.plot_ylabel(fig, ax, ylabel='Lat', pad = 0.1, fontsize = 7)
        mF.format_ticks(ax, labelsize = 7)
        ax.set_xticklabels('')


    lat, lon = da_0_regrid.lat, da_0_regrid.lon
    lonm,latm = np.meshgrid(lon,lat)
    ax = fig.add_subplot(3, 1, 2, projection=ccrs.PlateCarree(central_longitude=180))
    ax.add_feature(cfeat.COASTLINE)
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
    timestep = timesteps[frame]
    pcm_0 = ax.pcolormesh(lonm,latm, da_0_regrid.isel(time = timestep), transform=ccrs.PlateCarree(),zorder=0, cmap = variable_0.cmap, vmin=0, vmax=80)
    pcm_1 = ax.pcolormesh(lonm,latm, da_1_regrid.isel(time = timestep), transform=ccrs.PlateCarree(),zorder=0, cmap = variable_1.cmap, vmin=0, vmax=80)
    mF.scale_ax(ax, scaleby = 1.15)
    mF.move_col(ax, moveby = -0.025)
    mF.move_row(ax, moveby = -0.05)
    if frame == 0:
        mF.plot_xlabel(fig, ax, xlabel='Lon', pad = 0.1, fontsize = 7)
        mF.plot_ylabel(fig, ax, ylabel='Lat', pad = 0.1, fontsize = 7)
        mF.format_ticks(ax, labelsize = 7)
        mF.cbar_below_axis(fig, ax, pcm_1, cbar_height = 0.05, pad = 0.135, numbersize = 7, cbar_label = variable_1.label, text_pad = 0.1, text_fontsize = 7)
    plt.close()

def animate(da_0, da_1, timesteps, variable_0, variable_1, title):
    fig= plt.figure(figsize=(5.5, 3.5))
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
ani.save(f'{home}/Desktop/DOC_model_regrid.mp4')
print('finished')





