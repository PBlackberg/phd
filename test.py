import xarray as xr

'/g/data/oi10/replicas/CMIP6/CMIP/NOAA-GFDL/GFDL-ESM4/historical/r1i1p1f1/day/pr/gn'



a = xr.open_dataset('/g/data/fs38/publications/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/historical/r1i1p1f1/day/pr/gn/latest/pr_day_ACCESS-ESM1-5_historical_r1i1p1f1_gn_19500101-19991231.nc')

da = a['pr']
fig_obj = da.isel(time=0).plot()


import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/switch')
import myFuncs as mF
fig = fig_obj.figure
mF.save_plot(switch = {'save_folder_cwd':True}, fig=fig, home=home, filename='test')





