import numpy as np
import xarray as xr
import skimage.measure as skm
import matplotlib.pyplot as plt

import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/functions')
import myVars as mV         # imports common variables
import myFuncs as mF        # imports common operators and plotting
sys.path.insert(0, f'{folder_code}/plotting')
import map_plot_scene as mp # imports functions to plot maps


da = xr.open_dataset('/Users/cbla0002/Documents/data/pr/sample_data/cmip6/TaiESM1_pr_daily_historical_regridded.nc')['pr']
array = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/rome/cmip6/TaiESM1_rome_daily_historical_regridded.nc')['rome']
number_index = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/ni/cmip6/TaiESM1_ni_daily_historical_regridded.nc')['ni']
areafraction = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/ni/cmip6/TaiESM1_ni_daily_historical_regridded.nc')['areafraction']

conv_percentile = 0.97
conv_threshold = da.quantile(conv_percentile, dim=('lat', 'lon'), keep_attrs=True).mean(dim='time')


def find_timesteps(array):
    min_index = np.argmin(array)
    median_index = np.argsort(array)[len(array) // 2]
    max_index = np.argmax(array)
    timesteps = [min_index, median_index, max_index]

    min_value = array[min_index]
    median_value = array[median_index]
    max_value = array[max_index]
    doc_values = [min_value, median_value, max_value]
    return timesteps, doc_values


title = 'Transition of Degree of Organization of Convection (DOC)'
nrows = 3
ncols = 1
fig, axes = mp.create_map_figure(width = 9, height = 5.5, nrows=nrows, ncols=ncols)
num_subplots = 3

timesteps, doc_values = find_timesteps(array.values)

for i, timestep in enumerate(timesteps):
    row = i // ncols  # determine row index
    col = i % ncols   # determine col index
    ax = axes.flatten()[i]
    scene = da.isel(time=timestep)
    scene = skm.label(scene.where(scene>=conv_threshold,0)>0, background=np.nan,connectivity=2) # label matrix first gives a unique number to all groups of connected components.
    scene = (scene>0)*1  # This line sets all unique labels to 1 
    scene = xr.DataArray(
        data=scene,
        dims=['lat', 'lon'],
        coords={'lat': da.lat.data, 'lon': da.lon.data}
        )

    pcm = mp.plot_axScene(ax, scene, cmap='Greys', vmin = None, vmax = None)

    ax_title = f'Low DOC: ROME = {int(array[timestep])}, NI = {int(number_index.data[timestep])}, areafraction = {round(areafraction.data[timestep], 2)}' if row == 0 else None
    ax_title = f'Moderate DOC: ROME = {int(array[timestep])}, NI = {int(number_index.data[timestep])}, areafraction = {round(areafraction.data[timestep], 2)}'  if row == 1 else ax_title
    ax_title = f'High DOC: ROME = {int(array[timestep])}, NI = {int(number_index.data[timestep])}, areafraction = {round(areafraction.data[timestep], 2)}'      if row == 2 else ax_title

    mF.move_col(ax, -0.04) if col == 0 else None

    mF.move_row(ax, 0.04+0.005) if row == 0 else None
    mF.move_row(ax, 0.0+0.005) if row == 1 else None
    mF.move_row(ax, -0.04+0.005) if row == 2 else None

    mF.scale_ax(ax, 1.15)

    if row == 2:
        mF.plot_xlabel(fig, ax, 'lon', pad=0.075, fontsize = 12)

    if col == 0:
        mF.plot_ylabel(fig, ax, 'lat', pad = 0.075, fontsize = 12)
    mF.plot_axtitle(fig, ax, ax_title, xpad = 0, ypad = 0.01, fontsize = 12)

    mp.format_ticks(ax, i, num_subplots, nrows, col, labelsize = 10)
    ax.set_xticklabels('') if row == 0 else None
    ax.set_xticklabels('') if row == 1 else None


# ax.text(0.5, 0.95, title, ha = 'center', fontsize = 15, transform=fig.transFigure)


folder = f'{home}/Desktop/GASS-CFMIP_poster'
filename = 'doc_transition'    
mV.save_figure(fig, folder, f'{filename}.pdf')
plt.show()













