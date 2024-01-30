''' 
# ------------------------
#       Get data
# ------------------------
This script loads cmip data from nci and remaps to common grid.
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
#      Find folders
# ------------------------
# ---------------------------------------------------------------------------------------- folders ----------------------------------------------------------------------------------------------------------#
def folder_ensemble(source, model, experiment):
    ''' Some models don't have the ensemble most common amongst other models and some models don't have common ensembles for historical and warmer simulation '''
    if source == 'cmip5':
        ensemble = 'r6i1p1'    if model in ['EC-EARTH', 'CCSM4']                                                        else 'r1i1p1'
        ensemble = 'r6i1p1'    if model in ['GISS-E2-H'] and experiment == 'historical'                                 else ensemble
        ensemble = 'r2i1p1'    if model in ['GISS-E2-H'] and not experiment == 'historical'                             else ensemble
    if source == 'cmip6':
        ensemble = 'r1i1p1f2'  if model in ['CNRM-CM6-1', 'UKESM1-0-LL', 'CNRM-ESM2-1', 'CNRM-CM6-1-HR', 'MIROC-ES2L']  else 'r1i1p1f1'
        ensemble = 'r11i1p1f1' if model == 'CESM2' and not experiment == 'historical'                                   else ensemble
    return ensemble

def folder_grid(model):
    ''' Some models have a different grid folder in the path to the files (only applicable for cmip6) '''
    folder = 'gn'
    folder = 'gr'  if model in ['CNRM-CM6-1', 'EC-Earth3', 'IPSL-CM6A-LR', 'FGOALS-f3-L', 'CNRM-ESM2-1', 'CNRM-CM6-1-HR', 'KACE-1-0-G'] else folder
    folder = 'gr1' if model in ['GFDL-CM4', 'INM-CM5-0', 'KIOST-ESM', 'GFDL-ESM4', 'INM-CM4-8'] else folder           
    return folder

def folder_latestVersion(path):
    ''' Picks the latest version if there are multiple '''    
    versions = os.listdir(path)
    version = max(versions, key=lambda x: int(x[1:])) if len(versions)>1 else versions[0]
    return version

def folder_var(var, model, experiment, ensemble, project, timeInterval):
    ''' ACCESS model is stored separately '''
    if model in ['ACCESS-ESM1-5', 'ACCESS-CM2']:
        path_gen = f'/g/data/fs38/publications/CMIP6/{project}/{mV.institutes[model]}/{model}/{experiment}/{ensemble}/{timeInterval}/{var}'
        folder_to_grid = folder_grid(model)
        version = 'latest'
    else:
        path_gen = f'/g/data/oi10/replicas/CMIP6/{project}/{mV.institutes[model]}/{model}/{experiment}/{ensemble}/{timeInterval}/{var}' 
        folder_to_grid = folder_grid(model)
        version = folder_latestVersion(os.path.join(path_gen, folder_to_grid))
    return f'{path_gen}/{folder_to_grid}/{version}'

# ---------------------------------------------------------------------------------------- open files ----------------------------------------------------------------------------------------------------------#
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
#   Remap coordinates
# ------------------------
# --------------------------------------------------------------------------------- remap ----------------------------------------------------------------------------------------------------------#
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
                                                                
def get_p_hybrid(model, ds):
    ''' For hybrid-sigma pressure coordinates interpolation to pressure coordinates '''
    if model == 'IITM-ESM':               
        da = ds['plev']
    elif model == 'IPSL-CM6A-LR':         
        da = ds['presnivs']
    elif model in ['MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'CanESM5', 'CNRM-CM6-1', 'GFDL-CM4', 'CNRM-ESM2-1', 'CNRM-CM6-1-HR', 'IPSL-CM5A-MR', 'MPI-ESM-MR', 'CanESM2']: 
        da = ds['ap'] + ds['b']*ds['ps']
    elif model in ['FGOALS-g2', 'FGOALS-g3']:                                                   
        da = ds['ptop'] + ds['lev']*(ds['ps']-ds['ptop'])
    elif model in ['UKESM1-0-LL', 'KACE-1-0-G', 'ACCESS-CM2', 'ACCESS-ESM1-5', 'HadGEM2-CC']:
        da = ds['lev']+ds['b']*ds['orog']
    else:
        da = ds['a']*ds['p0'] + ds['b']*ds['ps']
    return da



# ------------------------
#        Get data
# ------------------------
# --------------------------------------------------------------------------------- Cloud fraction ----------------------------------------------------------------------------------------------------------#
def get_cmip_cl_data(var_name, model, experiment, source, ensemble, project, timeInterval): 
    ''' For hybrid-sigma pressure coordinates interpolation to pressure coordinates '''       
    if var_name in ['cl']:                                  # interpolation takes a long time, so stored locally       
        path = f'/g/data/k10/cb4968/data/sample_data/cl/{source}/{model}_cl_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'                                                                                            
        return xr.open_dataset(path) #['cl']         
    else:                                                   # for interpolation in 'cl_vert_interp' in util folder (could extend to other variables on hybrid-sigma pressure coordinates)
        path_folder = folder_var('cl', model, experiment, ensemble, project, timeInterval)
        ds = concat_files(path_folder, experiment) 
        if var_name == 'ds_cl':
                return ds                
        if var_name == 'cl_p_hybrid':
            da = ds['cl']     
        if var_name == 'p_hybrid':
            da = get_p_hybrid(model, ds)
            if model in ['IITM-ESM', 'IPSL-CM6A-LR']:
                return xr.Dataset(data_vars = {f'{var_name}': da}, attrs = ds.attrs)                # no lat, lon for vertical coordiante for these models
        da = regrid_hor(ds, da)     if mV.resolutions[0] == 'regridded' else da                     # horizontally interpolate
        ds = xr.Dataset(data_vars = {f'{var_name}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)   # if regridded it should already be lat: [-30,30]
    return ds


# --------------------------------------------------------------------------------- Other variables ----------------------------------------------------------------------------------------------------------#
def get_var_data(var_name, model, experiment, source):
    ''' concatenate files, interpolate grid, and mask '''
    ensemble = folder_ensemble(source, model, experiment)
    project = 'CMIP'        if experiment == 'historical'  else 'ScenarioMIP'
    timeInterval = 'day'    if mV.timescales[0] == 'daily' else 'Amon'         
    if var_name in ['cl', 'ds_cl', 'cl_p_hybrid', 'p_hybrid']:                                                  # cloud variable is on hybrid-sigma pressure coordiantes, so treated separately
        return get_cmip_cl_data(var_name, model, experiment, source, ensemble, project, timeInterval)
    else:                                                                                                       # all other 2D or 3D pressure level variables     
        folder_to_var = folder_var(var_name, model, experiment, ensemble, project, timeInterval)
        # path_folder = '/g/data/oi10/replicas/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3/historical/r11i1p1f1/Amon/clwvi/gr/v20200201/'
        ds = concat_files(folder_to_var, experiment)    
        da = ds[var_name]      
        da = regrid_vert(da, model)                         if 'plev' in da.dims                       else da  # vertically interpolate (some models have different number of vertical levels)
        da = regrid_hor(ds, da)                             if mV.resolutions[0] == 'regridded'         else da # horizontally interpolate
    return xr.Dataset(data_vars = {f'{var_name}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)                 # for when running original resolution. If regridded it should already be lat: [-30,30]



# ----------------------------
#  Run script / save variable
# ----------------------------
# ---------------------------------------------------------------------------------- save ----------------------------------------------------------------------------------------------------- #
def convert_units(model, da, var_name):
    if var_name == 'pr':
        da = da * 60 * 60 * 24
    if var_name == 'tas':
        da = da - 273.15  
    if var_name == 'wap':
        da = da * 60 * 60 * 24 / 100
        if model in ['IITM-ESM']: 
            da = da * 1000
    return da


@mF.timing_decorator(show_time = True)
def process_data(model, experiment, var_name):
    print(f'Processing {model} {mV.resolutions[0]} {mV.timescales[0]} {var_name} data from {experiment} experiment')
    if mV.resolutions[0] == 'regridded':
        print(f'Regridding to {mV.x_res}x{mV.y_res} degrees')
    source = mF.find_source(model)
    da = get_var_data(var_name, model, experiment, source)[var_name]
    da = convert_units(model, da, var_name)
    ds = xr.Dataset(data_vars = {var_name: da}) if not var_name == 'ds_cl' else da
    folder = f'{mV.folder_scratch}/sample_data/{var_name}/{source}'
    filename = f'{model}_{var_name}_{mV.timescales[0]}_*_{experiment}_{mV.resolutions[0]}.nc'
    if mV.resolutions[0] == 'regridded':
        filename = f'{model}_{var_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}_{int(360/mV.x_res)}x{int(180/mV.y_res)}.nc'
    Path(folder).mkdir(parents=True, exist_ok=True)
    path = Path(f'{folder}/{filename}')
    ds.to_netcdf(path, mode="w")
    print(f'{model} {var_name} data saved at {path}')
    return xr.open_dataset(path, chunks= {'time': 'auto'})[var_name]

# --------------------------------------------------------------------------------- run variable ----------------------------------------------------------------------------------------------------- #
def request_process(model, experiment, var_name, path):
    print(f'no {model} {experiment} {mV.resolutions[0]} {mV.timescales[0]} {var_name} data at {mV.x_res}x{mV.y_res} deg in {path}')
    response = input(f"Do you want to process {var_name} from {model}? (y/n) (check folder first): ").lower()
    # respone = 'y'
    if response == 'y':
        da = process_data(model, experiment, var_name = var_name)
    if response == 'n':
        exit()
    return da

def pick_ocean_region(da):
    mask = xr.open_dataset('/home/565/cb4968/Documents/code/phd/util/ocean.nc')['ocean']
    da = da * mask
    return da

def get_cmip_data(switch_var = {'pr': True}, switch = {'test_sample': False}, model = mV.datasets[0], experiment = mV.experiments[0]):
    source = mF.find_source(model)
    var_name = next((key for key, value in switch_var.items() if value), None)
    folder = f'{mV.folder_scratch}/sample_data/{var_name}/{source}'
    filename = f'{model}_{var_name}_{mV.timescales[0]}_*_{experiment}_{mV.resolutions[0]}.nc'
    if mV.resolutions[0] == 'regridded':
        filename = f'{model}_{var_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}_{int(360/mV.x_res)}x{int(180/mV.y_res)}.nc'
    path = f'{folder}/{filename}'
    if os.path.exists(path):
        ds = xr.open_dataset(path, chunks= {'time': 'auto'}) 
        da = ds[var_name]
        if switch['test_sample']:
            da = da.isel(time = slice(0, 365))
        if switch['ocean_mask']:
            da = pick_ocean_region(da)
        return da 
    else:
        da = request_process(model, experiment, var_name, path) # process the data and save in scratch
        return da



# ------------------------
#         Test
# ------------------------
# ------------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    import myFuncs_plots    as mFp     
    switch_var = {
        'pr':       False,                                                                                      # Precipitation
        'tas':      False, 'ta':            False,                                                              # Temperature
        'wap':      True,                                                                                      # Circulation
        'hur':      False,  'hus' :         False,                                                              # Humidity                   
        'rlds':     False, 'rlus':          False,  'rlut':     False,                                          # Longwave radiation
        'rsdt':     False, 'rsds':          False,  'rsus':     False,  'rsut':     False,                      # Shortwave radiation
        'cl':       False, 'cl_p_hybrid':   False,  'p_hybrid': False,  'ds_cl':    False,                      # Cloudfraction (ds_cl is for getting pressure levels)
        'zg':       False,                                                                                      # Geopotential
        'hfss':     False, 'hfls':          False,                                                              # Surface fluxes
        'clwvi':    False,                                                                                      # Cloud ice and liquid water
        }

    switch = {
        'test_sample':   False, # first year (or shorter)
        'ocean_mask':    False, # mask
        }

    process_all_datasets = True
    if process_all_datasets:
        for var_name in [k for k, v in switch_var.items() if v]:
            for model, experiment in mF.run_dataset(var_name):
                print(f'settings: {[key for key, value in switch.items() if value]}')
                process_data(model, experiment, var_name)
    else:
        var_name = next((key for key, value in switch_var.items() if value), None)
        da = get_cmip_data(switch_var, switch)
        print(da)
        # if 'plev' in da.dims:
        #     da = da.sel(plev = 500e2)
        # mFp.get_snapshot(da, plot = True, show_type = 'cycle', cmap = 'Blues')  # show_type = [show, save_cwd, cycle] 


