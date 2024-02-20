import numpy as np
import xarray as xr
import skimage.measure as skm
import matplotlib.pyplot as plt

import os
home = os.path.expanduser("~")
import myVars_m as mV         # imports common variables
import myFuncs_m as mF        # imports common operators and plotting

switch = {'show': True, 'save_to_desktop':True}

# -------------------------------------------------------------------------------------------- Get data ----------------------------------------------------------------------------------------------------- #
areafraction = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/ni/cmip6/TaiESM1_ni_95thPrctile_daily_historical_regridded.nc')['areafraction']
condition = (areafraction >= 4.5) & (areafraction <= 5.5)
subset_times = areafraction['time'].where(condition, drop=True)

da =           xr.open_dataset('/Users/cbla0002/Documents/data/pr/sample_data/cmip6/TaiESM1_pr_daily_historical_regridded.nc')['pr'].sel(time=subset_times)
rome =        xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/rome/cmip6/TaiESM1_rome_95thPrctile_daily_historical_regridded.nc')['rome'].sel(time=subset_times)
number_index = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/ni/cmip6/TaiESM1_ni_95thPrctile_daily_historical_regridded.nc')['ni'].sel(time=subset_times)
areafraction = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/ni/cmip6/TaiESM1_ni_95thPrctile_daily_historical_regridded.nc')['areafraction'].sel(time=subset_times)

conv_percentile = 0.95
conv_threshold = da.quantile(conv_percentile, dim=('lat', 'lon'), keep_attrs=True).mean(dim='time')

def find_timesteps(alist):
    min_index, median_index, max_index = np.argmin(alist), np.argsort(alist)[len(alist) // 2], np.argmax(alist)
    timesteps = [min_index, median_index, max_index]

    min_value, median_value, max_value = alist[min_index], alist[median_index], alist[max_index]
    doc_values = [min_value, median_value, max_value]
    return timesteps, doc_values
timesteps, doc_values = find_timesteps(rome.values)

# -------------------------------------------------------------------------------------------- plot ----------------------------------------------------------------------------------------------------- #
title = 'Transition of Degree of Organization of Convection (DOC)'
nrows = 3
ncols = 1
num_subplots = 3

fig, axes = mF.create_map_figure(width = 9, height = 5.5, nrows=nrows, ncols=ncols)
for i, timestep in enumerate(timesteps):
    row = i // ncols  # determine row index
    col = i % ncols   # determine col index
    ax = axes.flatten()[i]
    scene = da.isel(time=timestep)
    scene = skm.label(scene.where(scene>=conv_threshold,0)>0, background=np.nan,connectivity=2) # label matrix first gives a unique number to all groups of connected components.
    scene = (scene>0)*1  # This line sets all unique labels to 1 
    scene = xr.DataArray(data=scene, dims=['lat', 'lon'], coords={'lat': da.lat.data, 'lon': da.lon.data})
    pcm = mF.plot_axScene(ax, scene, cmap='Greys', vmin = None, vmax = None)

    ax_title = f'Low DOC: ROME = {int(rome[timestep])} km$^2$, NI = {int(number_index.data[timestep])}, areafraction = {round(areafraction.data[timestep], 2)}%'      if row == 0 else None
    ax_title = f'Moderate DOC: ROME = {int(rome[timestep])} km$^2$, NI = {int(number_index.data[timestep])}, areafraction = {round(areafraction.data[timestep], 2)}%' if row == 1 else ax_title
    ax_title = f'High DOC: ROME = {int(rome[timestep])} km$^2$, NI = {int(number_index.data[timestep])}, areafraction = {round(areafraction.data[timestep], 2)}%'     if row == 2 else ax_title
    mF.move_col(ax, -0.04)       if col == 0 else None
    mF.move_row(ax, 0.04+0.005)  if row == 0 else None
    mF.move_row(ax, 0.0+0.005)   if row == 1 else None
    mF.move_row(ax, -0.04+0.005) if row == 2 else None
    mF.scale_ax(ax, 1.15)
    if row == 2:
        mF.plot_xlabel(fig, ax, 'lon', pad=0.075, fontsize = 12)

    if col == 0:
        mF.plot_ylabel(fig, ax, 'lat', pad = 0.075, fontsize = 12)
    mF.plot_axtitle(fig, ax, ax_title, xpad = 0, ypad = 0.01, fontsize = 12)

    mF.format_ticks(ax, i, num_subplots, nrows, col, labelsize = 10)
    ax.set_xticklabels('') if row == 0 else None
    ax.set_xticklabels('') if row == 1 else None


# ax.text(0.5, 0.95, title, ha = 'center', fontsize = 15, transform=fig.transFigure)

mF.save_figure(fig, f'{home}/Desktop', 'test.pdf') if switch['save_to_desktop'] else None
plt.show() if switch['show'] else None
























