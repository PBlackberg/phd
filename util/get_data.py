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
    year1, year2 = (1970, 1999)                      if experiment == 'historical' else (2070, 2099)            # range of years to concatenate files for
    fileYear1_charStart, fileYear1_charEnd = (13, 9) if 'Amon' in path_folder      else (17, 13)                # character index range for starting year of file (counting from the end)
    fileYear2_charStart, fileYear2_charEnd = (6, 2)  if 'Amon' in path_folder      else (8, 4)                  #                             end
    files = sorted(files, key=lambda x: x[x.index(".nc")-fileYear1_charStart:x.index(".nc")-fileYear1_charEnd])
    files = [f for f in files if int(f[f.index(".nc")-fileYear1_charStart : f.index(".nc")-fileYear1_charEnd]) <= int(year2) and int(f[f.index(".nc")-fileYear2_charStart : f.index(".nc")-fileYear2_charEnd]) >= int(year1)]
    paths = []
    for file in files:
        paths = np.append(paths, os.path.join(path_folder, file))
    print(paths[0])                                                                                              # for debugging
    ds = xr.open_mfdataset(paths, combine='by_coords').sel(time=slice(str(year1), str(year2)),lat=slice(-35,35)) # take out a little bit wider range to not exclude data when interpolating grid
    return ds


# --------------------------------------------------------------------------------- interpolate / mask ----------------------------------------------------------------------------------------------------------#
def hor_interpolate(ds, da, plevs = ''): 
    import regrid_xesmf as regrid
    regridder = regrid.regrid_conserv_xesmf(ds)  # define regridder based of grid from other model (FGOALS-g2 from cmip5 currently)
    da = regridder(da)
    if any(plevs) and len(plevs.dims) > 1:
        plevs = regridder(plevs)                 # hybrid-sigma coordinates converted to pressure coordinates have the same shape as da
    return da, plevs

def vert_interpolate(model, da, plevs): 
    new_p = [200000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100] # making sure all variables are on the same pressure coordiantes
    return da

def pick_ocean_region(da):
    mask = xr.open_dataset('/home/565/cb4968/Documents/code/phd/util/ocean.nc')['ocean']
    da = da * mask
    return da


# ------------------------
#       For CMIP
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

def var_folder(model, ensemble, project , timeInterval):
    if model in ['ACCESS-ESM1-5', 'ACCESS-CM2']:
        path_gen = f'/g/data/fs38/publications/CMIP6/{project}/{mV.institutes[model]}/{model}/{experiment}/{ensemble}/{timeInterval}/{variable}'
        folder_grid = grid_folder(model)
        version = 'latest'
    else:
        path_gen = f'/g/data/oi10/replicas/CMIP6/{project}/{mV.institutes[model]}/{model}/{experiment}/{ensemble}/{timeInterval}/{variable}' 
        folder_grid = grid_folder(model)
        version = latestVersion(os.path.join(path_gen, folder_grid))
    path_folder =  f'{path_gen}/{folder_grid}/{version}'
    return path_folder


# ------------------------------------------------------------------------------------- preprocess data ----------------------------------------------------------------------------------------------------------#
def p_coords(model, var, ds):
    ''' Covert from hybrid-sigma coordinates to pressure coordinates (mix of cmip5 and cmip6 models)'''
    if var not in 'cl':
        ds['plev'] = ds['plev'].round(0) if model in ['ACCESS-ESM1-5', 'ACCESS-CM2'] else ds['plev'] # plev coordinate is specified to a higher number of significant figures in these models
        plevs = ds['plev']                                                          
    else:
        if model == 'IITM-ESM':                                                                     # already on pressure levels
            plevs = ds['plev']                                            
        elif model == 'IPSL-CM6A-LR':                                                               # already on pressure levels
            ds = ds.rename({'presnivs':'plev'})
            plevs = ds['plev'] 
        elif model in ['MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'CanESM5', 'CNRM-CM6-1', 'GFDL-CM4', 'CNRM-ESM2-1', 'CNRM-CM6-1-HR', 'IPSL-CM5A-MR', 'MPI-ESM-MR', 'CanESM2']: 
            plevs = ds.ap + ds.b*ds.ps
            ds = ds.rename({'lev':'plev'})
        elif model in ['FGOALS-g2', 'FGOALS-g3']:
            plevs = ds.ptop + ds.lev*(ds.ps-ds.ptop)
            ds = ds.rename({'lev':'plev'})
        elif model in ['UKESM1-0-LL', 'KACE-1-0-G', 'ACCESS-CM2', 'ACCESS-ESM1-5', 'HadGEM2-CC']:
            h_hybridsigma = ds.lev + ds.b*ds.orog                                                    # in meters
            plevs = 1000e2 * (1 -  0.0065*h_hybridsigma/288.15)**(9.81*0.029)/(8.314*0.0065)         # to pressure: P = P_0 * (1- L*(h-h_0)/T_0)^(g*M/R*L) Barometric formula (approximation based on lapserate)
            # p_hybridsigma = 1000e2 * np.exp(0.029*9.82*h_hybridsigma/287*T)                        # to pressure: P = P_0 * exp(- Mgh/(RT)) Hydrostatic balance (don't have T at pressure level)
            ds = ds.rename({'lev':'plev'})
        else:
            plevs = ds.a*ds.p0 + ds.b*ds.ps
            ds = ds.rename({'lev':'plev'})
    return plevs

def get_cmip_data(source, var, model, experiment, switch = {'ocean': False}):
    ''' concatenates file data and interpolates grid to common grid if needed'''
    ensemble =     ensemble_folder(source, model, experiment)
    project =      'CMIP' if experiment == 'historical'  else 'ScenarioMIP'
    timeInterval = 'day'  if mV.timescales[0] == 'daily' else 'Amon'
    path_folder =  var_folder(model, ensemble, project , timeInterval)
    ds =           concat_files(path_folder, experiment)                                                                # picks out lat: [-35, 35]
    plevs =        p_coords(model, ds)            if any(dim in ds.dims for dim in ['plev', 'lev', 'presnivs']) else '' # particularly cloud fraction is converted to pressure coordinates
    da =           ds[var]
    da, plevs =    hor_interpolate(ds, da, plevs) if mV.resolutions[0] == 'regridded'                           else da # horizontally interpolate
    da =           vert_interpolate(da, plevs)    if 'plev' in da.dims                                          else da # vertically interpolate (some models have different number of vertical levels)
    da =           pick_ocean_region(da)          if switch['ocean']                                            else da # pick ocean region (specifically for stability calculation)
    ds = xr.Dataset(data_vars = {f'{var}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)                                # if regridded it should already be lat: [-30,30]
    return ds



# ------------------------
#  For observations (NCI)
# ------------------------
# ------------------------------------------------------------------------------------- GPCP ----------------------------------------------------------------------------------------------------------#
def get_gpcp():
    ''' Observations from the Global Precipitation Climatology Project (GPCP) '''
    path_gen = '/g/data/ia39/aus-ref-clim-data-nci/gpcp/data/day/v1-3'
    years = range(1997,2022)                                                    # there is a constant shift in high percentile precipitation rate trend from around (2009-01-2009-06) forward
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
    valid_range = [0, 10000]                                                    # There are some e+33 values in the dataset
    da = da.where((da >= valid_range[0]) & (da <= valid_range[1]), np.nan)
    da = da.dropna('time', how='all')                                           # drop days where all values are NaN (one day)

    da, _ = hor_interpolate(ds, da) if mV.resolutions[0] == 'regridded' else da # horizontally interpolate
    da =    pick_ocean_region(da)   if switch['ocean']                  else da # pick ocean region if needed
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
        da['level'] = da['level']*100                                                   # convert from millibar (hPa) to Pa
        da = da.rename({'level': 'plev'})
        plevs = da['plevs']

    da, plevs = hor_interpolate(ds, da)     if mV.resolutions[0] == 'regridded' else da # horizontally interpolate
    da =        vert_interpolate(da, plevs) if 'plev' in da.dims                else da # vertically interpolate (some models have different number of vertical levels)
    da =        pick_ocean_region(da)       if switch['ocean']                  else da # pick ocean region (specifically for stability calculation)    
    ds = xr.Dataset(data_vars = {f'{var}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)
    return ds



# ------------------------
#     Load variable
# ------------------------
# -------------------------------------------------------------------------- pick out variable data array ----------------------------------------------------------------------------------------------------- #
def get_var_data(source, dataset, experiment, var_name, switch):
    da = None

    # Precipitation
    if var_name == 'pr':
        da = get_cmip_data('pr', dataset, experiment, switch)['pr']*60*60*24 if source in ['cmip5', 'cmip6'] else da
        da = get_gpcp()['pr']                                                if dataset == 'GPCP'            else da
        da.attrs['units'] = r'mm day$^-1$'

    # Temperature
    if var_name == 'tas':
        da = get_cmip_data('tas', dataset, experiment, switch)['tas']-273.15  if source in ['cmip5', 'cmip6'] else da
        da.attrs['units'] = r'$\degree$C'
    if var_name == 'ta':
        da = get_cmip_data('ta', dataset, experiment, switch)['ta']           if source in ['cmip5', 'cmip6'] else da
        da = get_era5_monthly('t')['t']                                       if dataset == 'ERA5' else da
        da.attrs['units'] = 'K'

    # Humidity
    if var_name == 'hur':
        da = get_cmip_data('hur', dataset, experiment, switch)['hur']  if source in ['cmip5', 'cmip6'] else da
        da.attrs['units'] = '%'
    if var_name == 'hus':
        da = get_cmip_data('hus', dataset, experiment, switch)['hus'] if source in ['cmip5', 'cmip6'] else da
        da = get_era5_monthly('q')['q']                                if dataset == 'ERA5' else da
        da.attrs['units'] = ''

    # Circulation
    if var_name == 'wap':
        da = get_cmip_data('wap', dataset, experiment, switch)['wap']*60*60*24/100 if source in ['cmip5', 'cmip6'] else da
        da = da * 1000 if dataset == 'IITM-ESM' else da
        da.attrs['units'] = r'hPa day$^-1$'

    # Longwave radiation
    if var_name == 'rlds':
        da = get_cmip_data('rlds', dataset, experiment, switch)['rlds'] if source in ['cmip5', 'cmip6'] else da
        da.attrs['units'] = r'W m$^-2$'
    if var_name == 'rlus':
        da = get_cmip_data('rlus', dataset, experiment, switch)['rlus'] if source in ['cmip5', 'cmip6'] else da
        da.attrs['units'] = r'W m$^-2$'
    if var_name == 'rlut':
        da = get_cmip_data('rlut', dataset, experiment, switch)['rlut'] if source in ['cmip5', 'cmip6'] else da
        da.attrs['units'] = r'W m$^-2$'

    # Shortwave radiation
    if var_name == 'rsdt':
        da = get_cmip_data('rsdt', dataset, experiment, switch)['rsdt'] if source in ['cmip5', 'cmip6'] else da
        da.attrs['units'] = r'W m$^-2$'
    if var_name == 'rsds':
        da = get_cmip_data('rsds', dataset, experiment, switch)['rsds'] if source in ['cmip5', 'cmip6'] else da
        da.attrs['units'] = r'W m$^-2$'
    if var_name == 'rsus':
        da = get_cmip_data('rsus', dataset, experiment, switch)['rsus'] if source in ['cmip5', 'cmip6'] else da
        da.attrs['units'] = r'W m$^-2$'
    if var_name == 'rsut':
        da = get_cmip_data('rsut', dataset, experiment, switch)['rsut'] if source in ['cmip5', 'cmip6'] else da
        da.attrs['units'] = r'W m$^-2$'

    # Clouds
    if var_name == 'cl':
        da = get_cmip_data('cl', dataset, experiment, switch)['cl'] if source in ['cmip5', 'cmip6'] else da
        da.attrs['units'] = '%'

    # Height coords
    if var_name == 'zg':
        da = get_cmip_data('zg', dataset, experiment, switch)['zg'] if source in ['cmip5', 'cmip6'] else da
        # da.attrs['units'] = ''

    # Surface fluexes
    if var_name == 'hfls':
        da = get_cmip_data('hfls', dataset, experiment, switch)['hfls'] if source in ['cmip5', 'cmip6'] else da
    if var_name == 'hfss':
        da = get_cmip_data('hfss', dataset, experiment, switch)['hfss'] if source in ['cmip5', 'cmip6'] else da

    return da



# ------------------------
#   Run / save variable
# ------------------------
# ---------------------------------------------------------------------------------------- Get variable and save ----------------------------------------------------------------------------------------------------- #
def save_sample(source, dataset, experiment, ds, var_name):
    folder = f'{mV.folder_save[0]}/sample_data/{var_name}/{source}'
    filename = f'{dataset}_{var_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'
    mF.save_file(ds, folder, filename)

def run_var_data(switch_var, switch, source, dataset, experiment):
    for var_name in [k for k, v in switch_var.items() if v] : # loop over true keys
        ds = xr.Dataset(data_vars = {var_name: get_var_data(source, dataset, experiment, var_name, switch)})
        save_sample(source, dataset, experiment, ds, var_name) if switch['save_sample'] and ds[var_name].any() else None


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
        'pr':   False,                                                                          # Precipitation
        'tas':  False, 'ta':            False,                                                  # Temperature
        'wap':  True,                                                                           # Circulation
        'hur':  False, 'hus' :          False,                                                  # Humidity                   
        'rlds': False, 'rlus':          False, 'rlut': False, 'netlw': False,                   # Longwave radiation
        'rsdt': False, 'rsds':          False, 'rsus': False, 'rsut':  False, 'netsw': False,   # Shortwave radiation
        'cl':   False, 'p_hybridsigma': False,                                                  # Cloudfraction
        'zg':   False,                                                                          # Height coordinates
        'hfss': False, 'hfls':          False,                                                  # Surface fluxes
        }

    switch = {
        'ocean':         False,                                                                 # mask
        'save_sample':   True                                                                   # save
        }

    run_get_data(switch_var, switch)





