
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats


def ax_plot(switch, ax, x, y, metric_y):
    if switch['bins']:
        pcm = ax.hist2d(x,y,[20,20], cmap = metric_y.cmap)
        bin_width = (x.max() - x.min())/100 # Each bin is one percent of the range of x values
        bins = np.arange(x.min(), x.max() + bin_width, bin_width)
        y_bins = []
        for i in np.arange(0,len(bins)-1):
            y_bins = np.append(y_bins, y.where((x>=bins[i]) & (x<bins[i+1])).mean())
        ax.plot(bins[:-1], y_bins, metric_y.color)
    return pcm

def calc_meanInPercentile(da, percentile):
    ''' Mean precipitation rate of the gridboxes included in the percentile of each scene (precipiration rate threshold) '''
    aWeights = np.cos(np.deg2rad(da.lat))
    percentile_value = da.quantile(percentile, dim=('lat', 'lon'), keep_attrs=True)
    return da.where(da >= percentile_value).weighted(aWeights).mean(dim=('lat', 'lon'), keep_attrs=True)

# -------------------------------------------------------------------- mean precipitation in high percentile ---------------------------------------------------------------- 

# looping over days
show = False
if show:
    da = xr.open_dataset('/Users/cbla0002/Documents/data/pr/sample_data/cmip6/MPI-ESM1-2-LR_pr_daily_historical_regridded.nc')['pr']
    a = da.isel(time= 0).where(da.isel(time= 0) >= da.isel(time= 0).quantile(0.99))
    pr99_mean = []
    for day in np.arange(0,len(da.time.data)):
        pr_day = da.isel(time= day)

        pr99_mean = np.append(pr99_mean, pr_day.where(pr_day >= pr_day.quantile(0.99)).mean(dim = ('lat', 'lon')))
    
    pr99_mean = xr.DataArray(pr99_mean, dims = 'time', coords = {'time': da.time})


# matrix operations
calc = True
if calc:
    da = xr.open_dataset('/Users/cbla0002/Documents/data/pr/sample_data/cmip6/MPI-ESM1-2-LR_pr_daily_historical_regridded.nc')['pr']
    percentile = 0.99
    pr99_mean = calc_meanInPercentile(da, percentile)

    rolling_mean = pr99_mean.rolling(time=12, center=True).mean()
    y = pr99_mean - rolling_mean
    y = y.dropna(dim='time')

    rome = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/rome/cmip6/MPI-ESM1-2-LR_rome_97thPrctile_daily_historical_regridded.nc')['rome']

    rolling_mean = rome.rolling(time=12, center=True).mean()
    x = rome - rolling_mean
    x = x.dropna(dim='time')





# ---------------------------------------------------------------------------------- plot ------------------------------------------------------------------------------ 


switch = {'bins':True}
class metric():
    cmap = 'Blues'
    color = 'blue'
metric_y = metric()
    
show = True
if show:
    fig, ax = plt.subplots()
    ax_plot(switch, ax, x, y, metric_y)
    res= stats.pearsonr(x, y)
    ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', 
            xytext=(0.8, 0.85), textcoords='axes fraction', fontsize = 12, color = 'r') if res[1]<=0.05 else None
    plt.xlim(-0.5e6, 1e6)
    plt.show()


















