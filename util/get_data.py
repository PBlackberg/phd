import xarray as xr
import numpy as np
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV
import myFuncs as mF



# ------------------------
#   General functions
# ------------------------
# --------------------------------------------------------------------------------- concatenate files ----------------------------------------------------------------------------------------------------------#
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
    # print(paths[0])                                                                                                 # for debugging
    ds = xr.open_mfdataset(paths, combine='by_coords').sel(time=slice(str(year1), str(year2)),lat=slice(-35,35))    # take out a little bit wider range to not exclude data when interpolating grid
    return ds


# --------------------------------------------------------------------------------- interpolate / mask ----------------------------------------------------------------------------------------------------------#
def regrid_hor(ds, da): 
    ''' Regrid to common horizontal grid '''
    import regrid_xesmf as regrid
    regridder = regrid.regrid_conserv_xesmf(ds) # define regridder based of grid from other model (FGOALS-g2 from cmip5 currently)
    da = regridder(da)
    return da

def interp_p_to_p_new_xr(da, p_new):            # does the same thing as scipy.interp1d, but quicker (can only be applied for models with 1D pressure coordinate)
    ''' Interpolate to common pressure levels (cloud fraction is dealt with separately)'''
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="xarray")
    da_p_new = da.interp(plev=p_new, method='linear', kwargs={'bounds_error':False, "fill_value": 0})    
    warnings.resetwarnings()
    return da_p_new

def pick_ocean_region(da):
    mask = xr.open_dataset('/home/565/cb4968/Documents/code/phd/util/ocean.nc')['ocean']
    da = da * mask
    return da


# ------------------------
#       CMIP data
# ------------------------
# ------------------------------------------------------------------------------------- pick folders ----------------------------------------------------------------------------------------------------------#
def latestVersion(path):
    ''' Picks the latest version if there are multiple '''    
    versions = os.listdir(path)
    version = max(versions, key=lambda x: int(x[1:])) if len(versions)>1 else versions[0]
    return version

def ensemble_folder(source, model, experiment):
    ''' Some models don't have the ensemble most common amongst other models and some experiments don't have the same ensemble as the historical simulation'''
    if source == 'cmip5':
        ensemble = 'r6i1p1'    if model in ['EC-EARTH', 'CCSM4']                            else 'r1i1p1'
        ensemble = 'r6i1p1'    if model in ['GISS-E2-H'] and experiment == 'historical'     else ensemble
        ensemble = 'r2i1p1'    if model in ['GISS-E2-H'] and not experiment == 'historical' else ensemble
    if source == 'cmip6':
        ensemble = 'r1i1p1f2'  if model in ['CNRM-CM6-1', 'UKESM1-0-LL', 'CNRM-ESM2-1', 'CNRM-CM6-1-HR', 'MIROC-ES2L'] else 'r1i1p1f1'
        ensemble = 'r11i1p1f1' if model == 'CESM2' and not experiment == 'historical' else ensemble
    return ensemble

def grid_folder(model):
    ''' Some models have a different grid folder in the path to the files (for cmip6) '''
    folder = 'gn'
    folder = 'gr'  if model in ['CNRM-CM6-1', 'EC-Earth3', 'IPSL-CM6A-LR', 'FGOALS-f3-L', 'CNRM-ESM2-1', 'CNRM-CM6-1-HR', 'KACE-1-0-G'] else folder
    folder = 'gr1' if model in ['GFDL-CM4', 'INM-CM5-0', 'KIOST-ESM', 'GFDL-ESM4', 'INM-CM4-8'] else folder           
    return folder

def var_folder(var, model, experiment, ensemble, project , timeInterval):
    if model in ['ACCESS-ESM1-5', 'ACCESS-CM2']:
        path_gen = f'/g/data/fs38/publications/CMIP6/{project}/{mV.institutes[model]}/{model}/{experiment}/{ensemble}/{timeInterval}/{var}'
        folder_grid = grid_folder(model)
        version = 'latest'
    else:
        path_gen = f'/g/data/oi10/replicas/CMIP6/{project}/{mV.institutes[model]}/{model}/{experiment}/{ensemble}/{timeInterval}/{var}' 
        folder_grid = grid_folder(model)
        version = latestVersion(os.path.join(path_gen, folder_grid))
    path_folder =  f'{path_gen}/{folder_grid}/{version}'
    return path_folder


# ------------------------------------------------------------------------------------- preprocess data ----------------------------------------------------------------------------------------------------------#
def get_p_hybrid(model, ds):
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


def get_cmip_data(var_name, source, model, experiment, switch = {'ocean': False}):
    ''' concatenates file data and interpolates grid to common grid if needed'''
    ensemble =         ensemble_folder(source, model, experiment)
    project =          'CMIP'                                 if experiment == 'historical'  else 'ScenarioMIP'
    timeInterval =     'day'                                  if mV.timescales[0] == 'daily' else 'Amon'

    if var_name in ['cl']:                                                                                                            # the interpolated data is stored locally
        da =            xr.open_dataset(f'/g/data/k10/cb4968/data/sample_data/cl/{source}/{model}_cl_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc')['cl']  

    if var_name not in ['cl', 'ds_cl', 'cl_p_hybrid', 'p_hybrid']:                                                               
        path_folder =   var_folder(var_name, model, experiment, ensemble, project, timeInterval)
        ds =            concat_files(path_folder, experiment)      
        da =            ds[var_name]      
        if 'plev' in       da.dims:
            da['plev'] =   da['plev'].round(0)                if model in ['ACCESS-ESM1-5', 'ACCESS-CM2'] else da['plev']           # plev coordinate is specified to a higher number of significant figures in these models
            p_new =        np.array([100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100])   
            da =           interp_p_to_p_new_xr(da, p_new)                                                                          # vertically interpolate (some models have different number of vertical levels)
        da =               regrid_hor(ds, da)                 if mV.resolutions[0] == 'regridded'                           else da # horizontally interpolate

    if var_name in ['ds_cl', 'cl_p_hybrid', 'p_hybrid']:                                                                            # vertical interpolation of cloud fraction dealt with in separate script                                                                                  
        path_folder =  var_folder('cl', model, experiment, ensemble, project, timeInterval)
        ds =           concat_files(path_folder, experiment) 
        if var_name == 'ds_cl':
                return ds                
        if var_name == 'cl_p_hybrid':
            da = ds['cl']     
        if var_name == 'p_hybrid':
            da =       get_p_hybrid(model, ds)
            if model in ['IITM-ESM', 'IPSL-CM6A-LR']:
                return xr.Dataset(data_vars = {f'{var_name}': da}, attrs = ds.attrs)                                                # no lat, lon for vertical coordiante for these models
        da =           regrid_hor(ds, da)                     if mV.resolutions[0] == 'regridded'                           else da # horizontally interpolate
 
    da =               pick_ocean_region(da)                  if switch['ocean']                                            else da # pick ocean region (specifically for stability calculation)
    ds =               xr.Dataset(data_vars = {f'{var_name}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)                         # if regridded it should already be lat: [-30,30]
    return ds



# -------------------------
#  Observational data (NCI)
# -------------------------
# ------------------------------------------------------------------------------------- GPCP ----------------------------------------------------------------------------------------------------------#
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
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds = ds.sel(lat=slice(-35,35))
    da = ds['precip']
    valid_range = [0, 10000]                                                # There are some e+33 values in the dataset
    da = da.where((da >= valid_range[0]) & (da <= valid_range[1]), np.nan)
    da = da.dropna('time', how='all')                                       # drop days where all values are NaN (one day)

    da, _ = regrid_hor(ds, da) if mV.resolutions[0] == 'regridded' else da  # horizontally interpolate
    da =    pick_ocean_region(da)   if switch['ocean']             else da  # pick ocean region if needed
    ds = xr.Dataset(data_vars = {'pr': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)
    return ds


# -------------------------------------------------------------------------------------- ERA5 ----------------------------------------------------------------------------------------------------------#
def get_era5_monthly(var):
    ''' Reanalysis data from ERA5 '''
    path_gen = f'/g/data/rt52/era5/pressure-levels/monthly-averaged/{var}'
    years = range(1998,2022)
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
    da = ds[var]

    if 'level' in da.dims:
        da['level'] = da['level']*100                                                       # convert from millibar (hPa) to Pa
        da = da.rename({'level': 'plev'})
        plevs = da['plevs']

    da, plevs = regrid_hor(ds, da)              if mV.resolutions[0] == 'regridded' else da # horizontally interpolate
    da =        interp_p_to_p_new_xr(da, plevs) if 'plev' in da.dims                else da # vertically interpolate (some models have different number of vertical levels)
    da =        pick_ocean_region(da)           if switch['ocean']                  else da # pick ocean region (specifically for stability calculation)    
    ds = xr.Dataset(data_vars = {f'{var}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)
    return ds



# ------------------------
#    Loading variable
# ------------------------
# -------------------------------------------------------------------------- pick out variable data array ----------------------------------------------------------------------------------------------------- #
def get_var_data(source, dataset, experiment, var_name, switch = {'ocean': False}):
    da = None

    # Precipitation
    if var_name == 'pr':
        da = get_cmip_data('pr', source, dataset, experiment, switch)['pr']*60*60*24              if source in ['cmip5', 'cmip6'] else da
        da = get_gpcp()['pr']                                                                     if dataset == 'GPCP'            else da
        da.attrs['units'] = r'mm day$^-1$'

    # precipitation efficiency
    if var_name == 'clwvi':
        da = get_cmip_data('clwvi', source, dataset, experiment, switch)['clwvi']                 if source in ['cmip5', 'cmip6'] else da
    if var_name == 'clivi':
        da = get_cmip_data('clivi', source, dataset, experiment, switch)['clivi']                 if source in ['cmip5', 'cmip6'] else da
    if var_name == 'cli':
        da = get_cmip_data('cli', source, dataset, experiment, switch)['cli']                     if source in ['cmip5', 'cmip6'] else da

    # Temperature
    if var_name == 'tas':
        da = get_cmip_data('tas', source, dataset, experiment, switch)['tas']-273.15              if source in ['cmip5', 'cmip6'] else da
        da.attrs['units'] = r'$\degree$C'
    if var_name == 'ta':
        da = get_cmip_data('ta', source, dataset, experiment, switch)['ta']                       if source in ['cmip5', 'cmip6'] else da
        da = get_era5_monthly('t')['t']                                                           if dataset == 'ERA5' else da
        da.attrs['units'] = 'K'

    # Humidity
    if var_name == 'hur':
        da = get_cmip_data('hur', source, dataset, experiment, switch)['hur']                     if source in ['cmip5', 'cmip6'] else da
        da.attrs['units'] = '%'
    if var_name == 'hus':
        da = get_cmip_data('hus', source, dataset, experiment, switch)['hus']                     if source in ['cmip5', 'cmip6'] else da
        da = get_era5_monthly('q')['q']                                                           if dataset == 'ERA5' else da
        da.attrs['units'] = ''

    # Circulation
    if var_name == 'wap':
        da = get_cmip_data('wap', source, dataset, experiment, switch)['wap']*60*60*24/100        if source in ['cmip5', 'cmip6'] else da
        da = da * 1000 if dataset == 'IITM-ESM' else da
        da.attrs['units'] = r'hPa day$^-1$'

    # Longwave radiation
    if var_name == 'rlds':
        da = get_cmip_data('rlds', source, dataset, experiment, switch)['rlds']                   if source in ['cmip5', 'cmip6'] else da
        da.attrs['units'] = r'W m$^-2$'
    if var_name == 'rlus':
        da = get_cmip_data('rlus', source, dataset, experiment, switch)['rlus']                   if source in ['cmip5', 'cmip6'] else da
        da.attrs['units'] = r'W m$^-2$'
    if var_name == 'rlut':
        da = get_cmip_data('rlut', source, dataset, experiment, switch)['rlut']                   if source in ['cmip5', 'cmip6'] else da
        da.attrs['units'] = r'W m$^-2$'

    # Shortwave radiation
    if var_name == 'rsdt':
        da = get_cmip_data('rsdt', source, dataset, experiment, switch)['rsdt']                   if source in ['cmip5', 'cmip6'] else da
        da.attrs['units'] = r'W m$^-2$'
    if var_name == 'rsds':
        da = get_cmip_data('rsds', source, dataset, experiment, switch)['rsds']                   if source in ['cmip5', 'cmip6'] else da
        da.attrs['units'] = r'W m$^-2$'
    if var_name == 'rsus':
        da = get_cmip_data('rsus', source, dataset, experiment, switch)['rsus']                   if source in ['cmip5', 'cmip6'] else da
        da.attrs['units'] = r'W m$^-2$'
    if var_name == 'rsut':
        da = get_cmip_data('rsut', source, dataset, experiment, switch)['rsut']                   if source in ['cmip5', 'cmip6'] else da
        da.attrs['units'] = r'W m$^-2$'

    # Clouds
    if var_name in ['cl', 'cl_p_hybrid', 'p_hybrid', 'ds_cl']:   
        if var_name == 'cl':            # cloud fraction on interpolated pressure levels
            da = get_cmip_data('cl', source, dataset, experiment, switch)['cl']                   if source in ['cmip5', 'cmip6'] else da
        if var_name == 'cl_p_hybrid':   # cloud fraction on original vertical grid
            da = get_cmip_data('cl_p_hybrid', source, dataset, experiment, switch)['cl_p_hybrid'] if source in ['cmip5', 'cmip6'] else da
        if var_name == 'p_hybrid':      # vertical levels for original grid  
            da = get_cmip_data('p_hybrid', source, dataset, experiment, switch)['p_hybrid']       if source in ['cmip5', 'cmip6'] else da
        if var_name == 'ds_cl':         # original dataset with data and levels     
            da = get_cmip_data('ds_cl', source, dataset, experiment, switch)                      if source in ['cmip5', 'cmip6'] else da

    # Height coords
    if var_name == 'zg':
        da = get_cmip_data('zg', source, dataset, experiment, switch)['zg']                       if source in ['cmip5', 'cmip6'] else da
        # da.attrs['units'] = ''

    # Surface fluexes
    if var_name == 'hfls':
        da = get_cmip_data('hfls', source, dataset, experiment, switch)['hfls']                   if source in ['cmip5', 'cmip6'] else da
    if var_name == 'hfss':
        da = get_cmip_data('hfss', source, dataset, experiment, switch)['hfss']                   if source in ['cmip5', 'cmip6'] else da

    return da



# ----------------------------
#  Run script / save variable
# ----------------------------
# ---------------------------------------------------------------------------------------- Get variable and save ----------------------------------------------------------------------------------------------------- #
def save_sample(source, dataset, experiment, ds, var_name):
    folder = f'{mV.folder_save[0]}/sample_data/{var_name}/{source}'
    filename = f'{dataset}_{var_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'
    mF.save_file(ds, folder, filename)

def run_var_data(switch_var, switch, source, dataset, experiment):
    for var_name in [k for k, v in switch_var.items() if v]:
        if not mV.data_available(source, dataset, experiment, var = var_name):
            continue
        da = get_var_data(source, dataset, experiment, var_name, switch)
        ds = xr.Dataset(data_vars = {var_name: da}) if not var_name == 'ds_cl' else da
        # print(ds)
        # print(ds[f'{var_name}'])
        save_sample(source, dataset, experiment, ds, var_name) if switch['save_sample'] else None

# --------------------------------------------------------------------------------------------- run variable ----------------------------------------------------------------------------------------------------- #
def run_experiment(switch_var, switch, source, dataset):
    for experiment in mV.experiments:
        if not mV.data_available(source, dataset, experiment):
            continue
        print(f'\t\t {experiment}') if experiment else print(f'\t observational dataset')
        run_var_data(switch_var, switch, source, dataset, experiment)

def run_dataset(switch_var, switch):
    for dataset in mV.datasets:
        source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        print(f'\t{dataset} ({source})')
        run_experiment(switch_var, switch, source, dataset)

@mF.timing_decorator
def run_get_data(switch_var, switch):
    print(f'Getting {mV.resolutions[0]} {mV.timescales[0]} data')
    print(f'var: {[key for key, value in switch_var.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')
    run_dataset(switch_var, switch)


# ------------------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':

    switch_var = {
        'pr':       False,                                                                                      # Precipitation
        'tas':      False, 'ta':            False,                                                               # Temperature
        'wap':      False,                                                                                      # Circulation
        'hur':      True, 'hus' :          False,                                                              # Humidity                   
        'rlds':     False, 'rlus':          False,  'rlut':     False,  'netlw':    False,                      # Longwave radiation
        'rsdt':     False, 'rsds':          False,  'rsus':     False,  'rsut':     False,  'netsw':    False,  # Shortwave radiation
        'cl':       False, 'cl_p_hybrid':   False,  'p_hybrid': False,  'ds_cl':    False,                      # Cloudfraction (ds_cl is for getting pressure levels)
        'zg':       False,                                                                                      # Height coordinates
        'hfss':     False, 'hfls':          False,                                                              # Surface fluxes
        'clwvi':    False, 'clivi':         False,  'cli':      False,                                          # precipitation efficiency variables
        }

    switch = {
        'ocean':         False, # mask
        'save_sample':   False  # save
        }

    run_get_data(switch_var, switch)





