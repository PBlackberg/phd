

import xarray as xr
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV
import myFuncs as mF    
import myClasses as mC
import matplotlib.pyplot as plt



a = xr.open_dataset('/Users/cbla0002/Documents/data/sample_data/cl/cmip6/BCC-CSM2-MR_cl_monthly_historical_regridded.nc')
print(a)
scene = a['cl'].isel(time=0).sel(plev = slice(1000e2, 600e2)) #.max(dim='plev')


# print(scene)

# fig, ax = mF.create_map_figure(width = 12, height = 4)
# pcm = mF.plot_axMapScene(ax, scene, 'Blues')
# mF.move_col(ax, moveby = -0.055)
# mF.move_row(ax, moveby = 0.075)
# mF.scale_ax(ax, scaleby = 1.15)
# mF.cbar_below_axis(fig, ax, pcm, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = '[%]', text_pad = 0.125)
# mF.plot_xlabel(fig, ax, 'Lon', pad = 0.1, fontsize = 12)
# mF.plot_ylabel(fig, ax, 'Lat', pad = 0.055, fontsize = 12)
# mF.plot_axtitle(fig, ax, 'cl', xpad = 0.005, ypad = 0.025, fontsize = 15)
# mF.format_ticks(ax, labelsize = 11)
# plt.show()












