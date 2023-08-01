''' ERA5 data '''
import xarray as xr
import numpy as np

import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/functions')
import myVars as mV # imports common variables

resolution = 'regridded'
def get_era5_monthly(resolution, variable):
    ''' Reanalysis data from ERA5 '''
    path_gen = f'/g/data/rt52/era5/pressure-levels/monthly-averaged/{variable}'
    years = range(1998,2022) # there is a constant shift in high percentile precipitation rate trend from around (2009-01-2009-06) forward
    folders = [f for f in os.listdir(path_gen) if (f.isdigit() and int(f) in years)]
    folders = sorted(folders, key=int)
    
    path_fileList = []
    for folder in folders:
        path_folder = os.path.join(path_gen, folder)
        files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
        files = sorted(files, key=lambda x: x[x.index("l_")+1:x.index("-")])
        for file in files:
            path_fileList = np.append(path_fileList, os.path.join(path_folder, file))

    ds = xr.open_mfdataset(path_fileList, combine='by_coords')
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds = ds.sortby('lat').sel(lat = slice(-35,35))
    da = ds[variable]
    print(da.units)

    if resolution == 'regridded':
        import xesmf_regrid as rD
        regridder = rD.regrid_conserv_xesmf(ds)
        da = regridder(da)
    
    ds = xr.Dataset(data_vars = {f'{variable}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)
    return ds


ds = get_era5_monthly(resolution, 't')
print(ds['t'])
# mV.save_sample_data(data=ds, folder_save=mV.folder_save[0], source='obs', dataset='ERA5', name='q', timescale='monthly', experiment='', resolution='orig')





























































