''' 
Here is an example of getting CMIP6 data (monthly relative humidity and daily precipitation from model: TaiESM1)
and regridding the variable to the grid of another model (FGOALS-g2 from cmip5)
'''



import numpy as np
import xarray as xr
import xesmf as xe
import os



def concat_files(path_folder, experiment):
    ''' Concatenates files of monthly or daily data between specified years '''
    files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
    year1, year2 = (1970, 1999)                      if experiment == 'historical' else (2070, 2099)                # range of years to concatenate files for
    fileYear1_charStart, fileYear1_charEnd = (13, 9) if 'Amon' in path_folder      else (17, 13)                    # character index range for starting year of file (counting from the end)
    fileYear2_charStart, fileYear2_charEnd = (6, 2)  if 'Amon' in path_folder      else (8, 4)                      #                             end
    files = sorted(files, key=lambda x: x[x.index(".nc")-fileYear1_charStart:x.index(".nc")-fileYear1_charEnd])
    files = [f for f in files if int(f[f.index(".nc")-fileYear1_charStart : f.index(".nc")-fileYear1_charEnd]) <= int(year2) and int(f[f.index(".nc")-fileYear2_charStart : f.index(".nc")-fileYear2_charEnd]) >= int(year1)]
    paths = []
    for file in files:
        paths = np.append(paths, os.path.join(path_folder, file))
    # print(paths[0])                                                                                               # for debugging
    ds = xr.open_mfdataset(paths, combine='by_coords').sel(time=slice(str(year1), str(year2)),lat=slice(-35,35))    # take out a little bit wider range to not exclude data when interpolating grid
    return ds

def regrid_conserv_xesmf(ds_in):
    ''' Creates regridder, interpolating to the grid of FGOALS-g2 from cmip5 here '''
    folder = '/g/data/al33/replicas/CMIP5/combined/LASG-CESS/FGOALS-g2/historical/day/atmos/day/r1i1p1/v20161204/pr'
    fileName = 'pr_day_FGOALS-g2_historical_r1i1p1_19970101-19971231.nc'
    ds_out = xr.open_dataset('{}/{}'.format(folder, fileName)).sel(lat=slice(-30,30))
    regridder = xe.Regridder(ds_in.isel(time=0), ds_out, 'conservative', periodic=True)
    return regridder


monthly_humidity, daily_pr = True, True
experiment = 'historical'
if monthly_humidity:
    path_folder = '/g/data/oi10/replicas/CMIP6/CMIP/AS-RCEC/TaiESM1/historical/r1i1p1f1/Amon/hur/gn/v20200623'   # 'hur_Amon_TaiESM1_historical_r1i1p1f1_gn_185001-201412.nc' # one of the files
    ds = concat_files(path_folder, experiment)
    da = ds['hur']
    regridder = regrid_conserv_xesmf(ds) # define regridder based of grid from other model (FGOALS-g2 from cmip5 currently)
    da = regridder(da)
    ds = xr.Dataset(data_vars = {f'hur': da.sel(lat=slice(-30,30))}, attrs = ds.attrs) # if regridded it should already be lat: [-30,30]
    print('CMIP6 hur dataset \n', ds)

if daily_pr:
    path_folder = '/g/data/oi10/replicas/CMIP6/CMIP/AS-RCEC/TaiESM1/historical/r1i1p1f1/day/pr/gn/v20200626'    # 'pr_day_TaiESM1_historical_r1i1p1f1_gn_19700101-19791231.nc' # one of the files
    ds = concat_files(path_folder, experiment)
    da = ds['pr']
    regridder = regrid_conserv_xesmf(ds) # define regridder based of grid from other model (FGOALS-g2 from cmip5 currently)
    da = regridder(da)
    ds = xr.Dataset(data_vars = {f'pr': da.sel(lat=slice(-30,30))}, attrs = ds.attrs) # if regridded it should already be lat: [-30,30]
    print('CMIP6 pr dataset \n', ds)






