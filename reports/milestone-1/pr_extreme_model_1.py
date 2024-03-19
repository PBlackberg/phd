import numpy as np
import xarray as xr
import skimage.measure as skm
import matplotlib.pyplot as plt

import os
home = os.path.expanduser('~')
import myVars_m as mV         # imports common variables
import myFuncs_m as mF        # imports common operators and plotting



switch = {'show': True, 'save_to_desktop':True}

# -------------------------------------------------------------------------------------------- Get data ----------------------------------------------------------------------------------------------------- #
da = xr.open_dataset('/Users/cbla0002/Documents/data/pr/sample_data/cmip6/TaiESM1_pr_daily_historical_regridded.nc')['pr']
alist = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/rome/cmip6/TaiESM1_rome_95thPrctile_daily_historical_regridded.nc')['rome']
timestep = 4

conv_percentile = 0.95
conv_threshold = da.quantile(conv_percentile, dim=('lat', 'lon'), keep_attrs=True).mean(dim='time')
scene = da.where(da>=conv_threshold).isel(time=timestep)
threshold_pr99 = da.quantile(0.99, dim=('lat', 'lon'), keep_attrs=True).isel(time=timestep)


# -------------------------------------------------------------------------------------------- plot ----------------------------------------------------------------------------------------------------- #
nrows = 1
ncols = 1
fig, ax = mF.create_map_figure(width = 9, height = 2.75, nrows=nrows, ncols=ncols)
pcm = mF.plot_axScene(ax, scene, cmap='Greys', zorder = 0, vmin = 0, vmax = 60)
pcm = mF.plot_axScene(ax, scene.where(scene>=threshold_pr99), cmap='Reds', zorder = 0, vmin = 0, vmax = None)
mF.move_col(ax, -0.04)
mF.move_row(ax, 0.175)
mF.scale_ax(ax, 1.15)
mF.plot_xlabel(fig, ax, 'lon', pad = 0.15, fontsize = 12)
mF.plot_ylabel(fig, ax, 'lat', pad = 0.075, fontsize = 12)
mF.format_ticks(ax, labelsize = 10)

cbar_position = [0.1565, 0.195, 0.75, 0.05]
cbar_label = r'pr [mm day$^{-1}$]'
cbar_text_x = cbar_position[0] + cbar_position[2] / 2   # In th middle of colorbar
cbar_text_y = cbar_position[1]-0.15                      # essentially pad
cbar_text_fontsize = 11
ticklabel_size = 11
cbar_ax = fig.add_axes(cbar_position)
cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
ax.text(cbar_text_x, cbar_text_y, cbar_label, ha = 'center', fontsize = cbar_text_fontsize , transform=fig.transFigure)

mF.save_figure(fig, f'{home}/Desktop/', 'test.pdf') if switch['save_to_desktop'] else None
plt.show() if switch['show'] else None






