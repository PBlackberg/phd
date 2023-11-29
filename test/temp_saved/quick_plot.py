
import xarray as xr
import matplotlib.pyplot as plt

import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV

for dataset in mV.datasets:
    # a = xr.open_dataset(f'/Users/cbla0002/Documents/data/metrics/rsdt/rsdt_sMean/cmip6/{dataset}_rsdt_sMean_monthly_historical_regridded.nc')
    a = xr.open_dataset(f'/Users/cbla0002/Documents/data/metrics/rsdt/rsdt_sMean/cmip6/{dataset}_rsdt_sMean_monthly_historical_regridded.nc')
    plt.figure()
    plt.title(dataset)
    plt.plot(a['rsdt_sMean'])
    plt.show()















