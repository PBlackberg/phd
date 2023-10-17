import numpy as np
import xarray as xr
import skimage.measure as skm
import matplotlib.pyplot as plt

import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/util')
import myFuncs as mF        # imports common operators and plotting

scene = xr.open_dataset('/Users/cbla0002/Documents/data/hur/metrics/hur_snapshot/cmip6/TaiESM1_hur_snapshot_monthly_historical_regridded.nc')['hur_snapshot']

nrows = 1
ncols = 1
fig, ax = mF.create_map_figure(width = 9, height = 2.75, nrows=nrows, ncols=ncols)


pcm = mF.plot_axScene(ax, scene, cmap='Greens', zorder = 0, vmin = None, vmax = None)

mF.move_col(ax, -0.04)

mF.move_row(ax, 0.175)

mF.scale_ax(ax, 1.15)

mF.plot_xlabel(fig, ax, 'lon', pad = 0.15, fontsize = 12)
mF.plot_ylabel(fig, ax, 'lat', pad = 0.075, fontsize = 12)
mF.format_ticks(ax, labelsize = 10)

cbar_position = [0.1565, 0.195, 0.75, 0.05]
cbar_label = 'Relative humidity [%]'
cbar_text_x = cbar_position[0] + cbar_position[2] / 2   # In th middle of colorbar
cbar_text_y = cbar_position[1]-0.15                      # essentially pad
cbar_text_fontsize = 11
ticklabel_size = 11
cbar_ax = fig.add_axes(cbar_position)
cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
ax.text(cbar_text_x, cbar_text_y, cbar_label, ha = 'center', fontsize = cbar_text_fontsize , transform=fig.transFigure)


folder = f'{home}/Desktop/GASS-CFMIP_poster'
filename = 'hur_snapshot'    
mF.save_figure(fig, folder, f'{filename}.pdf')
plt.show()













