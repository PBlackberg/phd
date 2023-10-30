
''' This script plots the resolution of a model with the organization index '''

import xarray as xr

def get_orig_res(dataset, source):
    da = xr.open_dataset(f'/Users/cbla0002/Documents/data/sample_data/pr/{source}/{dataset}_pr_daily_historical_orig.nc')['pr']
    dlat, dlon = da['lat'].diff(dim='lat').data[0], da['lon'].diff(dim='lon').data[0]
    resolution = dlat * dlon
    return dlat, dlon, resolution











