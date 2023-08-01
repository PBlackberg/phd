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
conv_percentile = 0.97
conv_threshold = da.quantile(conv_percentile, dim=('lat', 'lon'), keep_attrs=True).mean(dim='time')


def find_timesteps(array):
    min_index = np.argmin(array)
    median_index = np.argsort(array)[len(array) // 2]
    max_index = np.argmax(array)
    timesteps = 0 #[min_index, median_index, max_index]

    min_value = array[min_index]
    median_value = array[median_index]
    max_value = array[max_index]
    doc_values = [min_value, median_value, max_value]
    return timesteps, doc_values

nrows = 1
ncols = 1
fig, ax = mp.create_map_figure(width = 9, height = 2.75, nrows=nrows, ncols=ncols)
num_subplots = 2

timestep = 4 # 4

scene = da.isel(time=timestep)
pcm = mp.plot_axScene(ax, scene, cmap='Blues', zorder = 0, vmin = None, vmax = None)

threshold_pr99 = scene.quantile(0.99, dim=('lat', 'lon'), keep_attrs=True)
pcm = mp.plot_axScene(ax, scene.where(scene>=threshold_pr99), cmap='Reds', zorder = 0, vmin = 0, vmax = None)

# scene = da.isel(time=timestep)
# scene = skm.label(scene.where(scene>=conv_threshold,0)>0, background=np.nan,connectivity=2) # label matrix first gives a unique number to all groups of connected components.
# scene = (scene>0)*1  # This line sets all unique labels to 1 
# scene = xr.DataArray(
#     data=scene,
#     dims=['lat', 'lon'],
#     coords={'lat': da.lat.data, 'lon': da.lon.data}
#     )
# mp.plot_axScene(ax, scene, cmap='Greys', vmin = None, vmax = None)

mF.move_col(ax, -0.04)

mF.move_row(ax, 0.175)

mF.scale_ax(ax, 1.15)

mF.plot_xlabel(fig, ax, 'lon', pad = 0.15, fontsize = 12)
mF.plot_ylabel(fig, ax, 'lat', pad = 0.075, fontsize = 12)

# mF.plot_axtitle(fig, ax, ax_title, xpad = 0, ypad = 0.01, fontsize = 12)

mp.format_ticks(ax, labelsize = 10)
# ax.set_xticklabels('') if row == 0 else None
# ax.set_xticklabels('') if row == 1 else None


cbar_position = [0.1565, 0.195, 0.75, 0.05]
cbar_label = 'pr [mm day{}]'.format(mF.get_super('-1'))
cbar_text_x = cbar_position[0] + cbar_position[2] / 2   # In th middle of colorbar
cbar_text_y = cbar_position[1]-0.15                      # essentially pad
cbar_text_fontsize = 11
ticklabel_size = 11
cbar_ax = fig.add_axes(cbar_position)
cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
ax.text(cbar_text_x, cbar_text_y, cbar_label, ha = 'center', fontsize = cbar_text_fontsize , transform=fig.transFigure)




folder = f'{home}/Desktop/GASS-CFMIP_poster'
filename = 'pr_extreme_snapshot'    
mV.save_figure(fig, folder, f'{filename}.pdf')
plt.show()













