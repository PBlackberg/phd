import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

pr = xr.open_dataset('/Users/cbla0002/Documents/other/saved/GPCP/GPCP_precip_raw.nc')['precip']
valid_range = [0, 10000] # There are some e+33 values in the dataset

pr = pr.where((pr >= valid_range[0]) & (pr <= valid_range[1]), np.nan)
nb_nan = pr.isnull().sum(dim=('lat', 'lon')) # That leaves one day missing (2017-08-01)
index =np.where(nb_nan>0)
show_missing = False
if show_missing:
    print(np.count_nonzero(nb_nan), 'days with NaN')
    print(nb_nan.max().data, 'NaN values that day')

plot = False
if plot:
    print(index[0][0])
    pr.isel(time=index[0][0]).plot()
    plt.show()
pr = pr.dropna('time', how='all') # drop that day

# # The dataset has some extreme values (are they valid? - there are only a few so might not matter)
show_them = False
if show_them:
    print((pr > 250).sum().values)
    print(np.sort(pr.values.flatten())[-50:])

# checking the dates of the file
show_it = False
if show_it:
    print(pr.sel(time = '1997').time[0:3].data)
    print(pr.time[-3:].data)
    print(pr.sel(time='2021').time[-3:].data)

# There is a double trend in high percentile precipitation rate
prPercentile = xr.open_dataset('/Volumes/Philip_hd/data/saved/obs/metrics_obs_orig/GPCP/GPCP_prPercentiles_orig.nc')['pr99']
a_month = prPercentile.resample(time='M').mean(dim='time')
section = a_month.sel(time=slice('1997', '2021'))
show_range = False
if show_range:
    print(section.time[0].values)
    print(section.time[-1].values)

plot = False # double trend in high percentile precipitation rate
if plot:
    fig, ax = plt.subplots()
    section.plot()
    ax.axvline('1998', color='green', linestyle='--')
    ax.axvline('2009', color='green', linestyle='--')
    ax.axvline('2010', color='red', linestyle='--')
    ax.axvline('2022', color='red', linestyle='--')
    plt.title('GPCP double trend in high percentile precipitation rate')
    plt.show()

print('finsihed')

















































# ----------------------------------------------------------------------------------- saved for the moment ----------------------------------------------------------------------------------------------------- #



# pr = pr.sel(time=slice('1998-01, 2022-01'))


# a = xr.open_dataset('/Users/cbla0002/Documents/data/pr/sample_data/obs/GPCP_precip_orig.nc')['precip']
# b = a.time
# print(b[0:2].data)
# print(b[-2:].data)


# a = xr.open_dataset('/Users/cbla0002/Documents/data/pr/sample_data/obs/GPCP_precip.nc')['precip']
# b = a.time
# print(b[0:2].data)
# print(b[-2:].data)





# Linearly interpolate where there is missing data or outliers (I have set valied range to [0, 250] mm/day)
# from scipy.interpolate import griddata
# valid_range = [0, 250] 
# da = da.where((da >= valid_range[0]) & (da <= valid_range[1]), np.nan)
# da = da.where(da.sum(dim =('lat','lon')) != 0, np.nan)
# threshold = 0.5
# da = da.where(da.isnull().sum(dim=('lat','lon'))/(da.shape[1]*da.shape[2]) < threshold, other=np.nan)
# da = da.dropna('time', how='all')
# nb_nan = da.isnull().sum(dim=('lat', 'lon'))
# nan_days =np.nonzero(nb_nan.data)[0]
# for day in nan_days:
#     time_slice = da.isel(time=day)
#     nan_indices = np.argwhere(np.isnan(time_slice.values))
#     nonnan_indices = np.argwhere(~np.isnan(time_slice.values))
#     interpolated_values = griddata(nonnan_indices, time_slice.values[~np.isnan(time_slice.values)], nan_indices, method='linear')
#     time_slice.values[nan_indices[:, 0], nan_indices[:, 1]] = interpolated_values













