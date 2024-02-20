''' 
# ------------------------
#       Get data
# ------------------------
This script loads and interpolates observational data from and nci / online
The data is stored temporarily in scratch directory.
'''



# -------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import warnings
from pathlib import Path


# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars as mV
import myFuncs as mF

sys.path.insert(0, f'{os.getcwd()}/util-data')
import regrid.xesmf_regrid as rGh



# ------------------------
#      Open files
# ------------------------
# ------------------------------------------------------------------------------------- concatenate ----------------------------------------------------------------------------------------------------------#
def concat_files(path_folder, experiment):
    ''' Concatenate files between specified years '''
    files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
    year1, year2 = (1970, 1999)                      if experiment == 'historical' else (2070, 2099)                # range of years to concatenate files for
    fileYear1_charStart, fileYear1_charEnd = (13, 9) if 'Amon' in path_folder      else (17, 13)                    # character index range for starting year of file (counting from the end)
    fileYear2_charStart, fileYear2_charEnd = (6, 2)  if 'Amon' in path_folder      else (8, 4)                      #                             end
    files = sorted(files, key=lambda x: x[x.index(".nc")-fileYear1_charStart:x.index(".nc")-fileYear1_charEnd])
    files = [f for f in files if int(f[f.index(".nc")-fileYear1_charStart : f.index(".nc")-fileYear1_charEnd]) <= int(year2) and int(f[f.index(".nc")-fileYear2_charStart : f.index(".nc")-fileYear2_charEnd]) >= int(year1)]
    paths = []
    for file in files:
        paths = np.append(paths, os.path.join(path_folder, file))
    # print(paths[0])                                                                                                
    ds = xr.open_mfdataset(paths, combine='by_coords').sel(time=slice(str(year1), str(year2)),lat=slice(-35,35))     # take out a little bit wider range to not exclude data when interpolating grid
    return ds


# ------------------------
#   Remap coorindates
# ------------------------
# ---------------------------------------------------------------------------------- interpolate / mask ----------------------------------------------------------------------------------------------------------#
def regrid_hor(ds, da): 
    ''' Regrid to common horizontal grid '''
    regridder = rGh.regrid_conserv_xesmf(ds) # define regridder based of grid from other model (FGOALS-g2 from cmip5 currently)
    da = regridder(da)
    return da

def regrid_vert(da, model = ''):                                                                                # does the same thing as scipy.interp1d, but quicker (can only be applied for models with 1D pressure coordinate)
    ''' Interpolate to common pressure levels (cloud fraction is dealt with separately)'''
    da['plev'] = da['plev'].round(0)                if model in ['ACCESS-ESM1-5', 'ACCESS-CM2'] else da['plev'] # plev coordinate is specified to a higher number of significant figures in these models
    p_new = np.array([100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100])      

    warnings.filterwarnings("ignore", category=FutureWarning, module="xarray")
    da_p_new = da.interp(plev=p_new, method='linear', kwargs={'bounds_error':False, "fill_value": 0})    
    warnings.resetwarnings()
    return da_p_new
                                                             


# ------------------------
#  Observational product
# ------------------------
# ---------------------------------------------------------------------------------------- GPCP ----------------------------------------------------------------------------------------------------------#
def get_gpcp():
    ''' Observations from the Global Precipitation Climatology Project (GPCP) '''
    path_gen = '/g/data/ia39/aus-ref-clim-data-nci/gpcp/data/day/v1-3'
    years = range(1997,2022)                                                # there is a constant shift in high percentile precipitation rate trend from around (2009-01-2009-06) forward
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
    # print(ds)
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds = ds.sel(lat=slice(-35,35))
    da = ds['precip']
    valid_range = [0, 10000]                                                # There are some e+33 values in the dataset
    da = da.where((da >= valid_range[0]) & (da <= valid_range[1]), np.nan)
    da = da.dropna('time', how='all')                                       # One day all values are NaN
    da = regrid_hor(ds, da) if mV.resolutions[0] == 'regridded' else da     # horizontally interpolate
    da = pick_ocean_region(da)   if switch['ocean']             else da     # pick ocean region if needed
    return xr.Dataset(data_vars = {'pr': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)


# ---------------------------------------------------------------------------------------- ERA5 ----------------------------------------------------------------------------------------------------------#
def convert_to_era_var_name(var_name):
    if var_name == 'hur':
        return 'r' 
    if var_name == 'ta':
        return 't' 
    if var_name == 'hus':
        return 'q' 
    if var_name == 'zg':
        return 'z' 

def get_era5_monthly(var_name):
    ''' Reanalysis data from ERA5 '''
    var_nameERA = convert_to_era_var_name(var_name)
    print(var_nameERA)
    # exit()
    path_gen = f'/g/data/rt52/era5/pressure-levels/monthly-averaged/{var_nameERA}'
    years = range(1998,2022)                                                # same years as for GPCP obs
    folders = [f for f in os.listdir(path_gen) if (f.isdigit() and int(f) in years)]
    folders = sorted(folders, key=int)
    path_fileList = []
    for folder in folders:
        path_folder = os.path.join(path_gen, folder)
        files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
        files = sorted(files, key=lambda x: x[x.index("l_")+1:x.index("-")])
        for file in files:
            path_fileList = np.append(path_fileList, os.path.join(path_folder, file))
    # print(path_fileList[0])
    ds = xr.open_mfdataset(path_fileList, combine='by_coords')
    # print(ds)
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds = ds.sortby('lat').sel(lat = slice(-35,35))
    da = ds[var_nameERA]
    if 'level' in da.dims:
        da['level'] = da['level']*100                                       # convert from millibar (hPa) to Pa
        da = da.rename({'level': 'plev'})
        da = da.sortby('plev')
    da = regrid_hor(ds, da)     if mV.resolutions[0] == 'regridded' else da # horizontally interpolate
    da = regrid_vert(da)        if 'plev' in da.dims                else da # vertically interpolate
    ds = xr.Dataset(data_vars = {f'{var_name}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)
    return ds


# ---------------------------------------------------------------------------------------- NOAA ----------------------------------------------------------------------------------------------------------#
def get_NOAA(var):
    ds = xr.open_dataset('/g/data/k10/cb4968/data/sample_data/tas/obs/sst.mnmean.nc')
    ds = ds.sortby('lat').sel(lat = slice(-35,35))
    da = ds[var]
    da = regrid_hor(ds, da)     if mV.resolutions[0] == 'regridded' else da # horizontally interpolate
    ds = xr.Dataset(data_vars = {f'{var}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)
    return ds


# ---------------------------------------------------------------------------------------- ISCCP ----------------------------------------------------------------------------------------------------------#


# ---------------------------------------------------------------------------------------- CERES ----------------------------------------------------------------------------------------------------------#


# da['time'] = da['time'] - pd.Timedelta(days=14) # this observational dataset have monthly data with day specified as the middle of the month instead of the first
    




# ----------------------------
#  Run script / save variable
# ----------------------------
# --------------------------------------------------------------------------------- pick out variable data ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator(show_time = True)
def process_data(dataset, experiment, var_name):
    print(f'Processing {mV.datasets[0]} {mV.resolutions[0]} {mV.timescales[0]} {var_name} data from {mV.experiments[0]} experiment') # change to model instead of mV.datasets
    if mV.resolutions[0] == 'regridded':
        print(f'Regridding to {mV.x_res}x{mV.y_res} degrees')
    source = mF.find_source(dataset)
    da = get_gpcp()[var_name]                   if dataset == 'GPCP'        else None
    da = get_era5_monthly(var_name)[var_name]   if dataset == 'ERA5'        else da
    da = get_NOAA(var_name)[var_name]           if dataset == 'NOAA'        else da
    ds = xr.Dataset(data_vars = {var_name: da}) 
    folder = f'{mV.folder_scratch}/sample_data/{var_name}/{source}'
    filename = f'{dataset}_{var_name}_{mV.timescales[0]}_*_{experiment}_{mV.resolutions[0]}.nc'
    if mV.resolutions[0] == 'regridded':
        filename = f'{dataset}_{var_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}_{int(360/mV.x_res)}x{int(180/mV.y_res)}.nc'
    Path(folder).mkdir(parents=True, exist_ok=True)
    path = Path(f'{folder}/{filename}')
    ds.to_netcdf(path, mode="w")
    print(f'{dataset} {var_name} data saved at {path}')
    return xr.open_dataset(path, chunks= {'time': 'auto'})[var_name]


# --------------------------------------------------------------------------------- run variable ----------------------------------------------------------------------------------------------------- #
def request_process(dataset, experiment, var_name, path):
    print(f'no {dataset} {experiment} {mV.resolutions[0]} {mV.timescales[0]} {var_name} data at {mV.x_res}x{mV.y_res} deg in {path}')
    response = input(f"Do you want to process {var_name} from {dataset}? (y/n) (check folder first): ").lower()
    # respone = 'y'
    if response == 'y':
        process_data(dataset, experiment, var_name = var_name)
    if response == 'n':
        exit()
    return response

def pick_ocean_region(da):
    mask = xr.open_dataset('/home/565/cb4968/Documents/code/phd/util/ocean.nc')['ocean']
    da = da * mask
    return da

def get_obs_data(switch_var = {'pr': True}, switch = {'test_sample': False}, dataset = mV.datasets[0], experiment = mV.experiments[0]):
    source = mF.find_source(dataset)
    var_name = next((key for key, value in switch_var.items() if value), None)
    folder = f'{mV.folder_scratch}/sample_data/{var_name}/{source}'
    filename = f'{dataset}_{var_name}_{mV.timescales[0]}_*_{experiment}_{mV.resolutions[0]}.nc'
    if mV.resolutions[0] == 'regridded':
        filename = f'{dataset}_{var_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}_{int(360/mV.x_res)}x{int(180/mV.y_res)}.nc'
    path = f'{folder}/{filename}'
    if os.path.exists(path):
        da = xr.open_dataset(path, chunks= {'time': 'auto'})[var_name]
        if mV.obs_years:
            da = da.sel(time= slice(mV.obs_years[0].split('-')[0], mV.obs_years[0].split('-')[1]))
        if switch['test_sample']:
            da = da.isel(time = slice(0, 365))
        if switch['ocean_mask']:
            da = pick_ocean_region(da)
        return da 
    else:
        da = request_process(dataset, experiment, var_name, path) # process the data and save in scratch
        return da



# ------------------------
#         Test
# ------------------------
# ------------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    import myFuncs_plots    as mFp    
    switch_var = {
        'pr':       False,                                                                                       # Precipitation
        'tas':      False,                                                                                      # Temperature
        'hur':      True, 'hus' :          False,                                                              # Humidity                  
        'rlut':     False,                                                                                      # Longwave radiation
        'ws':       False,                                                                                      # weather states
        }

    switch = {
        'ocean_mask':    False, # mask
        'test_sample':   False  # save
        }

    process_all_datasets = False
    if process_all_datasets:
        for var_name in [k for k, v in switch_var.items() if v]:
            for dataset, experiment in mF.run_dataset(var_name):
                print(f'settings: {[key for key, value in switch.items() if value]}')
                process_data(dataset, experiment, var_name)
    else:
        var_name = next((key for key, value in switch_var.items() if value), None)
        da = get_obs_data(switch_var, switch)
        print(da)
        # if 'plev' in da.dims:
        #     da = da.sel(plev = 500e2)
        # mFp.get_snapshot(da, plot = True, show_type = 'cycle', cmap = 'Blues')  # show_type = [show, save_cwd, cycle] 








