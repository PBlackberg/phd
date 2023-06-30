import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

def plot_ax_bins(ax, x, y, color='k'):    
    bin_width = (x.max() - x.min())/100 # Each bin is one percent of the range of x values
    bin_end = x.max()
    bins = np.arange(x.min(), bin_end+bin_width, bin_width)
    y_bins = []
    for i in np.arange(0,len(bins)-1):
        y_bins = np.append(y_bins, y.where((x>=bins[i]) & (x<bins[i+1])).mean())
    ax.plot(bins[:-1], y_bins, color)





# Generate some example data
x = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/rome/cmip6/TaiESM1_rome_daily_historical_regridded.nc')['rome']
y0 = xr.open_dataset('/Users/cbla0002/Documents/data/hur/metrics/hur_sMean/cmip6/TaiESM1_hur_sMean_monthly_historical_regridded.nc')['hur_sMean']
y1 = xr.open_dataset('/Users/cbla0002/Documents/data/lw/metrics/rlut_sMean/cmip6/TaiESM1_rlut_sMean_monthly_historical_regridded.nc')['rlut_sMean']

x = x.resample(time='M').mean(dim='time')
x = x.assign_coords(time=y0.time)


fig, ax = plt.subplots()
plot_ax_bins(ax, x, y0, color='b')
ax.hist2d(x,y0,[20,20], cmap ='Blues')
ax.tick_params('y', colors='b')

ax1 = ax.twinx()
plot_ax_bins(ax1, x, y1, color='g')
ax1.hist2d(x,y1,[20,20], cmap ='Greens', alpha=0.5)
ax1.tick_params('y', colors='g')
plt.show()









