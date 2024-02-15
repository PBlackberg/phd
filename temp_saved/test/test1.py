
import xarray as xr
import matplotlib.pyplot as plt
  


path = '/scratch/w40/cb4968/metrics/wap/wap_500hpa_itcz_width/cmip6/TaiESM1_wap_500hpa_itcz_width_monthly_historical_regridded_144x72.nc'
ds = xr.open_dataset(path)
print(ds)
da = ds['wap_500hpa_itcz_width']
print(da.data)


import os
import sys
home = os.path.expanduser("~")                                        
sys.path.insert(0, f'{os.getcwd()}/util-core')
import myFuncs_plots as mFd   
fig = mFd.plot_scene(da, ax_title = 'pr_mean', vmin = 0, vmax = 20)    #, vmin = 0, vmax = 60) #, cmap = 'RdBu')
mFd.show_plot(fig, show_type = 'show', cycle_time = 0.5)        # 3.25 # show_type = [show, save_cwd, cycle] (cycle wont break the loop)


# da.plot.pcolormesh('object', 'time')
# plt.show()



