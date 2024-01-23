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


# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars as mV

sys.path.insert(0, f'{os.getcwd()}/util-data')
import regrid.xesmf_regrid as rGh



# ------------------------
#   Remap coorindates
# ------------------------
# --------------------------------------------------------------------------------- interpolate / mask ----------------------------------------------------------------------------------------------------------#
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
                                                                
def pick_ocean_region(da):
    mask = xr.open_dataset('/home/565/cb4968/Documents/code/phd/util/ocean.nc')['ocean']
    da = da * mask
    return da



# ------------------------
#        Get data
# ------------------------
# ----------------------------------------------------------------------------------- open files ----------------------------------------------------------------------------------------------------------#
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

# -------------------------------------------------------------------------------------- GPCP ----------------------------------------------------------------------------------------------------------#
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


# --------------------------------------------------------------------------------------- ERA5 ----------------------------------------------------------------------------------------------------------#
def convert_to_era_var_name(var_name):
    if var_name == 'hur':
        return 'r' 
    if var_name == 'ta':
        return 't' 
    if var_name == 'hus':
        return 'q' 
    if var_name == 'zg':
        return 'z' 

def get_era5_monthly(var_name, switch = {'ocean': False}):
    ''' Reanalysis data from ERA5 '''
    var_name = convert_to_era_var_name(var_name)
    path_gen = f'/g/data/rt52/era5/pressure-levels/monthly-averaged/{var_name}'
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
    da = ds[var_name]
    if 'level' in da.dims:
        da['level'] = da['level']*100                                       # convert from millibar (hPa) to Pa
        da = da.rename({'level': 'plev'})
        da = da.sortby('plev')
    da = regrid_hor(ds, da)     if mV.resolutions[0] == 'regridded' else da # horizontally interpolate
    da = regrid_vert(da)        if 'plev' in da.dims                else da # vertically interpolate
    da = pick_ocean_region(da)  if switch['ocean']                  else da # pick ocean region (specifically for stability calculation)    
    ds = xr.Dataset(data_vars = {f'{var_name}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)
    return ds

# --------------------------------------------------------------------------------------- NOAA ----------------------------------------------------------------------------------------------------------#
def get_NOAA(var):
    ds = xr.open_dataset('/g/data/k10/cb4968/data/sample_data/tas/obs/sst.mnmean.nc')
    ds = ds.sortby('lat').sel(lat = slice(-35,35))
    da = ds[var]
    da = regrid_hor(ds, da)     if mV.resolutions[0] == 'regridded' else da # horizontally interpolate
    ds = xr.Dataset(data_vars = {f'{var}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)
    return ds



# ----------------------------
#  Run script / save variable
# ----------------------------
# ---------------------------------------------------------------------------------- save ----------------------------------------------------------------------------------------------------- #
def save_sample(source, dataset, experiment, ds, var_name):
    folder = f'{mV.folder_scratch}/sample_data/{var_name}/{source}'
    path = f'{folder}/{dataset}_{var_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'
    if mV.resolutions[0] == 'regridded':
        path = f'{path}_{mV.x_res}x{mV.y_res}'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    ds.to_netcdf(path, mode = 'w')


# -------------------------------------------------------------------------------- pick out variable data ----------------------------------------------------------------------------------------------------- #
def process_data(source, dataset, experiment, var_name, switch = {'ocean': False}):
    print(f'loading {var_name}')
    da = None
    da = get_gpcp()[var_name]                   if dataset == 'GPCP'        else da
    da = get_era5_monthly(var_name)[var_name]   if dataset == 'NOAA'        else da
    da = get_NOAA(var_name)[var_name]           if dataset == 'NOAA'        else da
    ds = xr.Dataset(data_vars = {var_name: da}) if not var_name == 'ds_cl'  else da
    # print(ds)
    # print(ds[f'{var_name}'])
    save_sample(source, dataset, experiment, ds, var_name)
    return da

def request_data(model, variable_id):
    response = input(f"Do you want to process {variable_id} from {model}? (y/n): ").lower()
    return response


# --------------------------------------------------------------------------------- run variable ----------------------------------------------------------------------------------------------------- #
def data_available(source = '', dataset = '', experiment = '', var = '', switch = {'ocean_mask': False}, resolution = 'regridded'):
    ''' Check if dataset has variable. Returning False skips the loop in calc '''
    if [source, experiment] == ['cmip5', 'ssp585'] or [source, experiment] == ['cmip6', 'rcp85']: # only run fitting scenario for cmip version
        return  False
    if not experiment and not source in ['obs', 'test']:                                          # only run obs or test data for experiment == ''
        return False
    if experiment and source in ['obs']:                                                          # only run models when experiment ~= '' 
        return False
    
    if var in ['lcf', 'hcf', 'cl', 
               'ds_cl', 'cl_p_hybrid', 'p_hybrid'] \
        and dataset in ['INM-CM5-0', 'KIOST-ESM', 'EC-Earth3', 'INM-CM4-8', 
                        'CNRM-CM6-1-HR', 'GFDL-ESM4']:                                            # Some models do not have cloud variable
        print(f'No {var} data for this dataset')
        return False                                                                   
    return True

def find_source(dataset, models_cmip5 = mV.models_cmip5, models_cmip6 = mV.models_cmip6, observations = mV.observations):
    '''Determining source of dataset '''
    source = 'obs'   if np.isin(observations, dataset).any() else None
    return source
    
def get_obs_data(switch_var = {'pr': True}, switch = {'test_sample': False}, dataset = mV.datasets[0], experiment = mV.experiments[0], x_res = mV.x_res, y_res = mV.y_res):
    source = find_source(dataset)
    variable_id = next((key for key, value in switch_var.items() if value), None)
    print(f'getting {dataset} {mV.resolutions[0]} {mV.timescales[0]} {variable_id} data from {experiment} experiment')
    if mV.resolutions[0] == 'regridded':
        print(f'Regridded to {mV.x_res}x{mV.y_res}')
    print(f'settings: {[key for key, value in switch.items() if value]}')
    folder = f'{mV.folder_scratch}/sample_data/{variable_id}/{source}'
    path = f'{folder}/{dataset}_{variable_id}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'
    if os.path.exists(path):
        if '_' in dataset: # sometimes a year_range version of the obs will be used ex: 'GPCP_2010-2022'   
            aString = dataset.split('_')
            dataset = aString[0]
            years = int(aString[1].split('-'))
            return xr.open_dataset(path)[variable_id].sel(time= slice(years[0], years[1]))
        return xr.open_dataset(path)[variable_id]
    else:
        print(f'no {dataset} {mV.resolutions[0]} {mV.timescales[0]} {variable_id} data at {x_res}x{y_res} deg in {folder} (check)')
        response = request_data(dataset, variable_id)
        if response == 'y':
            process_data(source, dataset, experiment, var_name = variable_id)



# ------------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':

    switch_var = {
        'pr':       True,                                                                                      # Precipitation
        'tas':      False, 'ta':            False,                                                              # Temperature
        'wap':      False,                                                                                      # Circulation
        'hur':      False, 'hus' :          False,                                                              # Humidity                   
        'rlds':     False, 'rlus':          False,  'rlut':     False,  'netlw':    False,                      # Longwave radiation
        'rsdt':     False, 'rsds':          False,  'rsus':     False,  'rsut':     False,  'netsw':    False,  # Shortwave radiation
        'cl':       False, 'cl_p_hybrid':   False,  'p_hybrid': False,  'ds_cl':    False,                      # Cloudfraction (ds_cl is for getting pressure levels)
        'zg':       False,                                                                                      # Height coordinates
        'hfss':     False, 'hfls':          False,                                                              # Surface fluxes
        'clwvi':    False,                                                                                      # Cloud ice and liquid water
        }

    switch = {
        'ocean':         False, # mask
        'save_sample':   False  # save
        }

    process_all = False
    if process_all:
        for dataset in mV.datasets:
            source = find_source(dataset)
            print(f'\t{dataset} ({source})')
            for experiment in mV.experiments:
                if not data_available(source, dataset, experiment):
                    continue
                print(f'\t\t {experiment}')
                process_data(switch_var, switch, source, dataset, experiment)

    da = get_obs_data(switch_var, switch)
    print(da)






