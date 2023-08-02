
import xarray as xr
import os
def save_file(data, folder, filename):
    ''' Saves file to specified folder and filename '''
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    os.remove(path) if os.path.exists(path) else None
    data.to_netcdf(path)
    return


ds = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/rome/cmip6/TaiESM1_rome_daily_ssp585_regridded.nc')
print(ds)
ds = ds.rename({'__xarray_dataarray_variable__': 'rome'})
print(ds)

folder = '/Users/cbla0002/Documents/data/org/metrics/rome/cmip6'
filename = 'TaiESM1_rome_daily_ssp585_regridded.nc'
save_file(ds, folder, filename)


