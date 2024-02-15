import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib
matplotlib.use('Agg')

import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-core')
import myFuncs_plot as mFp
import myFuncs_dask as mFd


# ds = xr.open_dataset('/scratch/b/b382628/pr_144x72.nc')
# ds = xr.open_dataset('/scratch/b/b382628/pr/ICON-ESM_ngc2013/ICON-ESM_ngc2013_pr_daily_2020_regridded_144x72.nc')



ds = xr.open_dataset()
# print(ds)
da = ds['pr'].isel(time=0).sel(lat = slice(-30, 30))*60*60*24
print(da)

def plot_scene(scene, cmap = 'Blues', label = '[units]', figure_title = 'test', ax_title= 'test', vmin = None, vmax = None):
    fig, ax = mFp.create_map_figure(width = 12, height = 4)
    pcm = mFp.plot_axMapScene(ax, scene, cmap, vmin = vmin, vmax = vmax)
    mFp.move_col(ax, moveby = -0.055)
    mFp.move_row(ax, moveby = 0.075)
    mFp.scale_ax(ax, scaleby = 1.15)
    mFp.cbar_below_axis(fig, ax, pcm, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = label, text_pad = 0.125)
    mFp.plot_xlabel(fig, ax, 'Lon', pad = 0.1, fontsize = 12)
    mFp.plot_ylabel(fig, ax, 'Lat', pad = 0.055, fontsize = 12)
    ax.text(0.5, 0.925, figure_title, ha = 'center', fontsize = 15, transform=fig.transFigure)
    mFp.plot_axtitle(fig, ax, ax_title, xpad = 0.005, ypad = 0.025, fontsize = 15)
    mFp.format_ticks(ax, labelsize = 11)
    return fig


fig = plot_scene(da, vmin = 0, vmax = 20)
fig.savefig("my_plot.png", dpi=300) # deafult is 100 dpi


# plt.figure(figsize=(10, 5))
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.coastlines()  # Add coastlines
# da.plot(ax=ax, transform=ccrs.PlateCarree(), vmin=0, vmax=20)
# ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
# plt.savefig("my_plot.png", bbox_inches='tight', dpi=300)
# plt.close()

