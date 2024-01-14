



import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


da = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/ni/cmip6/TaiESM1_ni_95thPrctile_daily_historical_regridded.nc')['areafraction']

def plot_timeseries(y, ylabel, xlabel):
    plt.figure(figsize=(25,5))
    plt.plot(y)
    plt.axhline(y=y.mean(dim='time'), color='k')
    plt.title('areafraction picked out by precipitation threshold')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

plot_timeseries(da, 'areafraction [%]', 'timee (days)')
plt.show()
























