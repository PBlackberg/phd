import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
a = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/rome/obs/GPCP_rome_daily_regridded.nc')['rome'] # GPCP
print(a.time[0:5])

# b = xr.open_dataset('/Users/cbla0002/Documents/data/lw/metrics/rlut_sMean/obs/CERES_rlut_sMean_monthly_regridded.nc')['rlut_sMean'] # CERES

x = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/rome/cmip6/TaiESM1_rome_daily_historical_regridded.nc')['rome'] 
y = xr.open_dataset('/Users/cbla0002/Documents/data/pr/metrics/meanInPercentiles_pr/cmip6/TaiESM1_meanInPercentiles_pr_daily_historical_regridded.nc')['pr99'] 

def plot_bins(x,y, ax):    
    bin_width = (x.max() - x.min())/100
    bin_end = x.max()
    bins = np.arange(0, bin_end+bin_width, bin_width)
    y_bins = []
    for i in np.arange(0,len(bins)-1):
        y_bins = np.append(y_bins, y.where((x>=bins[i]) & (x<=bins[i+1])).mean())
    ax.plot(bins[:-1], y_bins, 'k')

# fig= plt.figure(figsize=(22.5,17.5))
# ax= fig.add_subplot(2,2,1)
# plt.hist2d(x,y,[20,20], cmap = 'Greys') #, vmin=0, vmax=300)
# plot_bins(x,y, ax)
# ax.set_title('model')
# plt.colorbar()
# plt.show()



fig, ax = plt.subplots(1, 1, figsize=(5,5))
h2d = plt.hist2d(x,y,[20,20], cmap ='Purples')
color = 'purple'
sp = plot_bins(x, y, ax)
plt.colorbar()
plt.show()



print('finished')







