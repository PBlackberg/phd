import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats


def plot_correlation(ax, x,y, position, fontsize):
    res= stats.pearsonr(x,y)
    if res[1]<=0.05:
        ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', xytext=position, textcoords='axes fraction', fontsize = fontsize, color = 'r')


# model: rome and rome fixed area
plot = False
if plot:
    x = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/rome/cmip6/TaiESM1_rome_daily_historical_regridded.nc')['rome']
    y = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/rome_fixed_area/cmip6/TaiESM1_rome_fixed_area_daily_historical_regridded.nc')['rome']
    fig, ax = plt.subplots(1, 1)
    plt.scatter(x, y)
    plot_correlation(ax, x,y, position = (0.8, 0.9), fontsize = 12)
    
    plt.title('for TaiESM1')
    plt.xlabel('rome')
    plt.ylabel('rome fixed area threshold')
    plt.show()


# model: rome and areafraction
plot = False
if plot:
    x = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/rome/cmip6/TaiESM1_rome_daily_historical_regridded.nc')['rome']
    y = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/ni/cmip6/TaiESM1_ni_daily_historical_regridded.nc')['areafraction']
    fig, ax = plt.subplots(1, 1)
    plt.scatter(x, y)
    plot_correlation(ax, x,y, position = (0.8, 0.9), fontsize = 12)
    
    plt.title('for TaiESM1')
    plt.xlabel('rome')
    plt.ylabel('areafraction')
    plt.show()


# random field: rome and rome fixed area
plot = False
if plot:
    x = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/rome/cmip6/random_fields_rome_daily_historical_regridded.nc')['rome']
    y = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/rome_fixed_area/cmip6/random_fields_rome_fixed_area_daily_historical_regridded.nc')['rome']
    fig, ax = plt.subplots(1, 1)
    plt.scatter(x, y)
    plot_correlation(ax, x,y, position = (0.8, 0.9), fontsize = 12)
    
    plt.title('for random field')
    plt.xlabel('rome')
    plt.ylabel('rome fixed area')
    plt.show()


# random field: rome and areafraction
plot = True
if plot:
    x = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/rome/cmip6/random_fields_rome_daily_historical_regridded.nc')['rome']
    y = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/ni/cmip6/random_fields_ni_daily_historical_regridded.nc')['areafraction']
    fig, ax = plt.subplots(1, 1)
    plt.scatter(x, y)
    plot_correlation(ax, x,y, position = (0.8, 0.9), fontsize = 12)
    
    plt.title('for random field')
    plt.xlabel('rome')
    plt.ylabel('areafraction')
    plt.show()









