import xarray as xr
import xesmf as xe
import numpy as np
import os
import scipy
from scipy.interpolate import griddata




# ---------------------------------------------------------------- functions to process the data ----------------------------------------------------------------

def concat_files(path_folder, experiment):
    if experiment == 'historical':
        yearEnd_first = 1970
        yearStart_last = 1999

    if experiment == 'rcp85':
        yearEnd_first = 2070
        yearStart_last = 2099

    files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
    if 'Amon' in path_folder:
        files = sorted(files, key=lambda x: x[x.index(".nc")-13:x.index(".nc")-9])
        files = [f for f in files if int(f[f.index(".nc")-13:f.index(".nc")-9]) <= yearStart_last and int(f[f.index(".nc")-6:f.index(".nc")-2]) >= yearEnd_first]
    else:
        files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
        files = sorted(files, key=lambda x: x[x.index(".nc")-17:x.index(".nc")-13])
        files = [f for f in files if int(f[f.index(".nc")-17:f.index(".nc")-13]) <= yearStart_last and int(f[f.index(".nc")-8:f.index(".nc")-4]) >= yearEnd_first]


    path_fileList = []
    for file in files:
        path_fileList = np.append(path_fileList, os.path.join(path_folder, file))

    ds = xr.open_mfdataset(path_fileList, combine='by_coords').sel(time=slice(str(yearEnd_first), str(yearStart_last)),lat=slice(-35,35))

    return ds

def regrid_conserv_xesmf(ds_in):
    folder = '/g/data/al33/replicas/CMIP5/combined/LASG-CESS/FGOALS-g2/historical/day/atmos/day/r1i1p1/v20161204/pr'
    fileName = 'pr_day_FGOALS-g2_historical_r1i1p1_19970101-19971231.nc'
    ds_out = xr.open_dataset(folder + '/' + fileName).sel(lat=slice(-30,30))
    regridder = xe.Regridder(ds_in.isel(time=0), ds_out, 'conservative', periodic=True)

    return regridder


def save_file(dataset, folder, filename):
    os.makedirs(folder, exist_ok=True)
    path = folder + '/' + filename

    if os.path.exists(path):
        os.remove(path)    
    
    dataset.to_netcdf(path)



# ---------------------------------------------------------------- get the data ----------------------------------------------------------------


def get_gpcp():

    path_gen = '/g/data/ia39/aus-ref-clim-data-nci/gpcp/data/day/v1-3'
    years = range(1996,2023)
    folders = [f for f in os.listdir(path_gen) if (f.isdigit() and int(f) in years)]
    folders = sorted(folders, key=int)

    path_fileList = []
    for folder in folders:
        path_folder = os.path.join(path_gen, folder)
        files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
        files = sorted(files, key=lambda x: x[x.index("y_d")+1:x.index("_c")])

        for file in files:
            path_fileList = np.append(path_fileList, os.path.join(path_folder, file))

    ds = xr.open_mfdataset(path_fileList, combine='by_coords')
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})

    precip = ds.precip.sel(lat=slice(-35,35), time=slice('1998','2021'))
    valid_range = [0, 250]
    precip = precip.where((precip >= valid_range[0]) & (precip <= valid_range[1]), np.nan)
    precip = precip.where(precip.sum(dim =('lat','lon')) != 0, np.nan)
    threshold = 0.5
    precip = precip.where(precip.isnull().sum(dim=('lat','lon'))/(precip.shape[1]*precip.shape[2]) < threshold, other=np.nan)
    precip = precip.dropna('time', how='all')
    nb_nan = precip.isnull().sum(dim=('lat', 'lon'))
    nan_days =np.nonzero(nb_nan.data)[0]
    for day in nan_days:
        time_slice = precip.isel(time=day)
        nan_indices = np.argwhere(np.isnan(time_slice.values))
        nonnan_indices = np.argwhere(~np.isnan(time_slice.values))
        interpolated_values = griddata(nonnan_indices, time_slice.values[~np.isnan(time_slice.values)], nan_indices, method='linear')
        time_slice.values[nan_indices[:, 0], nan_indices[:, 1]] = interpolated_values

    regridder = regrid_conserv_xesmf(precip) # define regridder based of grid from other model
    precip_n = regridder(precip) # conservatively interpolate to grid from other model, onto lat: -30, 30 (_n is new grid)

    ds_gpcp = xr.Dataset(
        data_vars = {'precip': precip_n},
        attrs = ds.attrs
        )

    return ds_gpcp








if __name__ == '__main__':

    import matplotlib.pyplot as plt

    ds_gpcp = get_gpcp()

    
    save_gpcp = False

    folder_save = '/g/data/k10/cb4968/data/obs/ds' 
    if save_gpcp:
        fileName = 'GPCP_precip.nc'
        save_file(ds_gpcp, folder_save, fileName)












