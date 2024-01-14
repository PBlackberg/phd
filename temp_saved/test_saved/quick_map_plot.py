
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV
import myFuncs as mF    
import myClasses as mC


dataset = 'BCC-CSM2-MR'
da = xr.open_dataset(f'/Users/cbla0002/Documents/data/sample_data/pr/cmip6/{dataset}_pr_daily_historical_regridded.nc')['pr']
conv_threshold = da.quantile(int(mV.conv_percentiles[0])*0.01, dim=('lat', 'lon'), keep_attrs=True).mean(dim='time').data
metric_class = mC.get_metric_class('pr')


for day in np.arange(1, 1000):
    scene = da.isel(time = day)
    scene = xr.where(scene>= conv_threshold, 1, 0)
    mF.plot_one_scene(scene, metric_class, figure_title = '', ax_title= '', vmin = None, vmax = None)
    plt.show()
















