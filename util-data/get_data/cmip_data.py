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


# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars as mV

sys.path.insert(0, f'{os.getcwd()}/util-data')
import regrid.xesmf_regrid as rGh



# ------------------------
#      Find folders
# ------------------------
def ensemble_folder(source, model, experiment):
    ''' Some models don't have the ensemble most common amongst other models and some models don't have common ensembles for historical and warmer simulation '''
    if source == 'cmip5':
        ensemble = 'r6i1p1'    if model in ['EC-EARTH', 'CCSM4']                                                        else 'r1i1p1'
        ensemble = 'r6i1p1'    if model in ['GISS-E2-H'] and experiment == 'historical'                                 else ensemble
        ensemble = 'r2i1p1'    if model in ['GISS-E2-H'] and not experiment == 'historical'                             else ensemble
    if source == 'cmip6':
        ensemble = 'r1i1p1f2'  if model in ['CNRM-CM6-1', 'UKESM1-0-LL', 'CNRM-ESM2-1', 'CNRM-CM6-1-HR', 'MIROC-ES2L']  else 'r1i1p1f1'
        ensemble = 'r11i1p1f1' if model == 'CESM2' and not experiment == 'historical'                                   else ensemble
    return ensemble

def grid_folder(model):
    ''' Some models have a different grid folder in the path to the files (only applicable for cmip6) '''
    folder = 'gn'
    folder = 'gr'  if model in ['CNRM-CM6-1', 'EC-Earth3', 'IPSL-CM6A-LR', 'FGOALS-f3-L', 'CNRM-ESM2-1', 'CNRM-CM6-1-HR', 'KACE-1-0-G'] else folder
    folder = 'gr1' if model in ['GFDL-CM4', 'INM-CM5-0', 'KIOST-ESM', 'GFDL-ESM4', 'INM-CM4-8'] else folder           
    return folder

def latestVersion(path):
    ''' Picks the latest version if there are multiple '''    
    versions = os.listdir(path)
    version = max(versions, key=lambda x: int(x[1:])) if len(versions)>1 else versions[0]
    return version

def var_folder(var, model, experiment, ensemble, project, timeInterval):
    ''' ACCESS model is stored separately '''
    if model in ['ACCESS-ESM1-5', 'ACCESS-CM2']:
        path_gen = f'/g/data/fs38/publications/CMIP6/{project}/{mV.institutes[model]}/{model}/{experiment}/{ensemble}/{timeInterval}/{var}'
        folder_grid = grid_folder(model)
        version = 'latest'
    else:
        path_gen = f'/g/data/oi10/replicas/CMIP6/{project}/{mV.institutes[model]}/{model}/{experiment}/{ensemble}/{timeInterval}/{var}' 
        folder_grid = grid_folder(model)
        version = latestVersion(os.path.join(path_gen, folder_grid))
    return f'{path_gen}/{folder_grid}/{version}'



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

def get_cmip_cl_data(var_name, model, experiment, ensemble, project, timeInterval): 
    ''' For hybrid-sigma pressure coordinates interpolation to pressure coordinates '''       
    if var_name in ['cl']:                                  # interpolation takes a long time, so stored locally       
        path = f'/g/data/k10/cb4968/data/sample_data/cl/{find_source(model)}/{model}_cl_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'                                                                                            
        return xr.open_dataset(path) #['cl']         
    else:                                                   # for interpolation in 'cl_vert_interp' in util folder (could extend to other variables on hybrid-sigma pressure coordinates)
        path_folder = var_folder('cl', model, experiment, ensemble, project, timeInterval)
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
        da = pick_ocean_region(da)  if switch['ocean']                  else da                     # pick ocean region (specifically for stability calculation)
        ds = xr.Dataset(data_vars = {f'{var_name}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)   # if regridded it should already be lat: [-30,30]
    return ds



# ------------------------
#        Get data
# ------------------------
# ---------------------------------------------------------------------------------- open files ----------------------------------------------------------------------------------------------------------#
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

def pick_ocean_region(da):
    mask = xr.open_dataset('/home/565/cb4968/Documents/code/phd/util/ocean.nc')['ocean']
    da = da * mask
    return da

def get_var_data(var_name, source, model, experiment, switch = {'ocean': False}):
    ''' concatenate files, interpolate grid, and mask '''
    ensemble = ensemble_folder(source, model, experiment)
    project = 'CMIP'        if experiment == 'historical'  else 'ScenarioMIP'
    timeInterval = 'day'    if mV.timescales[0] == 'daily' else 'Amon'         
    if var_name in ['cl', 'ds_cl', 'cl_p_hybrid', 'p_hybrid']:                                                  # cloud variable is on hybrid-sigma pressure coordiantes, so treated separately
        return get_cmip_cl_data(var_name, model, experiment, ensemble, project, timeInterval)
    else:                                                                                                       # all other 2D or 3D pressure level variables     
        path_folder = var_folder(var_name, model, experiment, ensemble, project, timeInterval)
        # path_folder = '/g/data/oi10/replicas/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3/historical/r11i1p1f1/Amon/clwvi/gr/v20200201/'
        ds = concat_files(path_folder, experiment)    
        da = ds[var_name]      
        da = regrid_vert(da, model)                         if 'plev' in da.dims                       else da # vertically interpolate (some models have different number of vertical levels)
        da = regrid_hor(ds, da)                             if mV.resolutions[0] == 'regridded'         else da # horizontally interpolate
        da = pick_ocean_region(da)                          if switch['ocean']                          else da # pick ocean region (specifically for stability calculation)
    return xr.Dataset(data_vars = {f'{var_name}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)               # for when running original resolution. If regridded it should already be lat: [-30,30]



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
    for var_name in [k for k, v in switch_var.items() if v]:
        if not data_available(source, dataset, experiment, var = var_name):
            continue
        da = None
        da = get_var_data(var_name, source, dataset, experiment, switch)[var_name]         if var_name not in ['pr', 'tas', 'wap']         else da # no need to change units
        da = get_var_data('pr', source, dataset, experiment, switch)['pr']*60*60*24        if var_name == 'pr'                             else da # Precipitation [r'mm day$^-1$']
        da = get_var_data('tas', source, dataset, experiment, switch)['tas']-273.15        if var_name == 'tas'                            else da # Surface temperature [r'$\degree$C'] (air temperature (ta) is in K)
        da = get_var_data('wap', source, dataset, experiment, switch)['wap']*60*60*24/100  if var_name == 'wap'                            else da # Vertical pressure velocity [r'hPa day^-1']
        da = da * 1000                                                                     if var_name == 'wap' and dataset == 'IITM-ESM'  else da # one model is off by a factor of 100
        ds = xr.Dataset(data_vars = {var_name: da}) if not var_name == 'ds_cl' else da
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
    source = 'cmip5' if np.isin(models_cmip5, dataset).any() else None      
    source = 'cmip6' if np.isin(models_cmip6, dataset).any() else source         
    return source
    
def get_cmip_data(switch_var = {'pr': True}, switch = {'test_sample': False}, model = mV.datasets[0], experiment = mV.experiments[0], x_res = mV.x_res, y_res = mV.y_res):
    source = find_source(model)
    variable_id = next((key for key, value in switch_var.items() if value), None)
    print(f'getting {model} {mV.resolutions[0]} {mV.timescales[0]} {variable_id} data from {experiment} experiment')
    if mV.resolutions[0] == 'regridded':
        print(f'Regridded to {mV.x_res}x{mV.y_res}')
    folder = f'{mV.folder_scratch}/sample_data/{variable_id}/{source}'
    path = f'{folder}/{model}_{variable_id}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'
    if os.path.exists(path):
        return xr.open_dataset(path)[variable_id]
    else:
        print(f'no {model} {mV.resolutions[0]} {mV.timescales[0]} {variable_id} data at {x_res}x{y_res} deg in {path} (check)')
        response = request_data(model, variable_id)
        if response == 'y':
            process_data(source, dataset, experiment, var_name = variable_id, switch = {'ocean': False})


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
        'ocean_mask':    False, # mask
        'test_sample':   False  # save
        }

    process_all = False
    if process_all:
        print(f'getting {model} {mV.resolutions[0]} {mV.timescales[0]} {variable_id} data from {experiment} experiment')
        if mV.resolutions[0] == 'regridded':
            print(f'Regridded to {mV.x_res}x{mV.y_res}')
        print(f'settings: {[key for key, value in switch.items() if value]}')
        for dataset in mV.datasets:
            source = find_source(dataset)
            print(f'\t{dataset} ({source})')
            for experiment in mV.experiments:
                if not data_available(source, dataset, experiment):
                    continue
                print(f'\t\t {experiment}')
                process_data(switch_var, switch, source, dataset, experiment)

    da = get_cmip_data(switch_var, switch)
    print(da)





