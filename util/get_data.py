import xarray as xr
import numpy as np
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV
import myFuncs as mF



# ------------------------
#   Concatenate files
# ------------------------
# ------------------------------------------------------------------------------ concatenate files ----------------------------------------------------------------------------------------------------------#
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



# ------------------------
#       For CMIP
# ------------------------
# ------------------------------------------------------------------------------------ general funcs ----------------------------------------------------------------------------------------------------------#
def latestVersion(path):
    ''' Picks the latest version if there are multiple '''    
    versions = os.listdir(path)
    version = max(versions, key=lambda x: int(x[1:])) if len(versions)>1 else versions[0]
    return version

def choose_ensemble(source, model, experiment):
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
    else
        path_gen = f'/g/data/oi10/replicas/CMIP6/{project}/{mV.institutes[model]}/{model}/{experiment}/{ensemble}/{timeInterval}/{variable}' 
        folder_grid = grid_folder(model)
        version = latestVersion(os.path.join(path_gen, folder_grid))
    path_folder =  f'{path_gen}/{folder_grid}/{version}'
    return path_folder

def hor_interpolate(da): 
    import regrid_xesmf as regrid
    return da

def convert_coordinates(ds):
    ''
    return da

def vert_interpolate(model, da): 
    da['plev'] =   da['plev'].round(0)   if model in ['ACCESS-ESM1-5', 'ACCESS-CM2'] and 'plev' in da.dims else da['plev'] # plev coordinate is specified to a higher number of significant figures
    import regrid_xesmf as regrid
    regridder = regrid.regrid_conserv_xesmf(ds)  # define regridder based of grid from other model (FGOALS-g2 from cmip5 currently)
    da = regridder(da)
    return da

def pick_ocean_region(da):
    mask = xr.open_dataset('/home/565/cb4968/Documents/code/phd/util/ocean.nc')['ocean']
    da = da * mask
    return da



# ------------------------------------------------------------------------------------- get variable ----------------------------------------------------------------------------------------------------------#
def get_cmip_data(variable, model, experiment, switch = {'ocean': False}):
    ''' concatenates file data and interpolates grid to common grid if needed'''
    ensemble =     choose_ensemble(source, model, experiment)
    project =      'CMIP' if experiment == 'historical'  else 'ScenarioMIP'
    timeInterval = 'day'  if mV.timescales[0] == 'daily' else 'Amon'
    path_folder =  var_folder(model, ensemble, project , timeInterval)
    ds =           concat_files(path_folder, experiment)                                                                   # picks out lat: [-35, 35]
    da =           ds[variable]
    da =           hor_interpolate(da)   if mV.resolutions[0] == 'regridded'               else da                         # horizontally interpolate



    
    da =           vert_interpolate(da)  if any(dim in da.dims for dim in ['plev', 'lev']) else da                         # vertically interpolate (some models have different number of vertical levels)
    da =           pick_ocean_region(da) if switch['ocean']                                else da                         # pick ocean region (specifically for stability calculation)

    ds = xr.Dataset(data_vars = {f'{variable}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)                              # if regridded it should already be lat: [-30,30]
    return ds


# ------------------------------------------------------------------------------- For most variables ----------------------------------------------------------------------------------------------------------#
def get_cmip5_data(variable, model, experiment, switch = {'ocean': False}):
    ''' concatenates file data and interpolates grid to common grid if needed '''    
    ensemble = choose_cmip5_ensemble(model, experiment)
    timeInterval = ('day', 'day') if mV.timescales[0] == 'daily' else ('mon', 'Amon')
    path_gen = f'/g/data/al33/replicas/CMIP5/combined/{mV.institutes[model]}/{model}/{experiment}/{timeInterval[0]}/atmos/{timeInterval[1]}/{ensemble}'
    version = latestVersion(path_gen)
    path_folder = f'{path_gen}/{version}/{variable}'
    ds = concat_files(path_folder, variable, model, experiment)
    da= ds[variable]
    if mV.resolutions[0] == 'regridded':
        import regrid_xesmf as regrid
        regridder = regrid.regrid_conserv_xesmf(ds)
        da = regridder(da)
    ds = xr.Dataset(data_vars = {f'{variable}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)
    return ds


# --------------------------------------------------------------------------------- For cloudfraction ----------------------------------------------------------------------------------------------------------#
def get_cmip5_cl(variable, model, experiment, switch = {'ocean': False}):
    ds_cl, ds_p_hybridsigma = None, None
    ''' Cloud pressure on hybrid-sigma vertical levels '''    
    ensemble = choose_cmip5_ensemble(model, experiment)
    timeInterval = ('day', 'day') if mV.timescales[0] == 'daily' else ('mon', 'Amon')
    path_gen = f'/g/data/al33/replicas/CMIP5/combined/{mV.institutes[model]}/{model}/{experiment}/{timeInterval[0]}/atmos/{timeInterval[1]}/{ensemble}'
    version = latestVersion(path_gen)
    path_folder = f'{path_gen}/{version}/{variable}'
    ds = concat_files(path_folder, variable, model, experiment)
        
    if model == 'IPSL-CM5A-MR' or model == 'MPI-ESM-MR' or model=='CanESM2': # different models have different conversions from height coordinate to pressure coordinate.
        p_hybridsigma = ds.ap + ds.b*ds.ps
    elif model == 'FGOALS-g2':
        p_hybridsigma = ds.ptop + ds.lev*(ds.ps-ds.ptop)
    elif model == 'HadGEM2-CC':
        p_hybridsigma = ds.lev+ds.b*ds.orog
    else:
        p_hybridsigma = ds.a*ds.p0 + ds.b*ds.ps
    
    if mV.resolutions[0] == 'orig':
        ds_cl = ds # units in % on sigma pressure coordinates
        ds_p_hybridsigma = xr.Dataset(data_vars = {'p_hybridsigma': p_hybridsigma}, attrs = ds.lev.attrs)
    if mV.resolutions[0] == 'regridded':
        import regrid_xesmf as regrid
        cl = ds['cl'] # units in % on sigma pressure coordinates
        regridder = regrid.regrid_conserv_xesmf(ds)
        cl = regridder(cl)
        p_hybridsigma_n = regridder(p_hybridsigma)
        ds_cl = xr.Dataset(data_vars = {'cl': cl}, attrs = ds.attrs)
        ds_p_hybridsigma = xr.Dataset(data_vars = {'p_hybridsigma': p_hybridsigma_n}, attrs = ds.lev.attrs)
    return ds_cl, ds_p_hybridsigma





# --------------------------------------------------------------------------------- For cloudfraction ----------------------------------------------------------------------------------------------------------#
def get_cmip6_cl(variable, model, experiment, switch = {'ocean': False}):
    ''' Cloud pressure on hybrid-sigma vertical levels '''
    ensemble = choose_cmip6_ensemble(model, experiment)
    project = 'CMIP' if experiment == 'historical' else 'ScenarioMIP'
    timeInterval = 'day' if mV.timescales[0] == 'daily' else 'Amon'

    if model in ['ACCESS-ESM1-5', 'ACCESS-CM2']:
        path_gen = f'/g/data/fs38/publications/CMIP6/{project}/{mV.institutes[model]}/{model}/{experiment}/{ensemble}/{timeInterval}/{variable}'
    else:
        path_gen = f'/g/data/oi10/replicas/CMIP6/{project}/{mV.institutes[model]}/{model}/{experiment}/{ensemble}/{timeInterval}/{variable}'

    # print(path_gen)
    folder_grid = grid_folder(model)
    version = latestVersion(os.path.join(path_gen, folder_grid)) if not model in ['ACCESS-ESM1-5', 'ACCESS-CM2'] else 'latest'
    path_folder =  f'{path_gen}/{folder_grid}/{version}'
    ds = concat_files(path_folder, experiment) # picks out lat: [-35, 35]
    if model == 'IITM-ESM': # different models have different conversions from height coordinate to pressure coordinate.
        ds = ds.rename({'plev':'lev'})
        p_hybridsigma = ds['lev'] # already on pressure levels
    elif model == 'IPSL-CM6A-LR':
        ds = ds.rename({'presnivs':'lev'})
        p_hybridsigma = ds['lev'] # already on pressure levels
    elif model in ['MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'CanESM5', 'CNRM-CM6-1', 'GFDL-CM4', 'CNRM-ESM2-1', 'CNRM-CM6-1-HR']: 
        p_hybridsigma = ds.ap + ds.b*ds.ps
    elif model == 'FGOALS-g3':
        p_hybridsigma = ds.ptop + ds.lev*(ds.ps-ds.ptop)
    elif model in ['UKESM1-0-LL', 'KACE-1-0-G', 'ACCESS-CM2', 'ACCESS-ESM1-5']:
        h_hybridsigma = ds.lev + ds.b*ds.orog                                                      # in meters
        p_hybridsigma = 1000e2 * (1 -  0.0065*h_hybridsigma/288.15)**(9.81*0.029)/(8.314*0.0065) # to pressure: P = P_0 * (1- L*(h-h_0)/T_0)^(g*M/R*L) Barometric formula (approximation based on lapserate)
        # p_hybridsigma = 1000e2 * np.exp(0.029*9.82*h_hybridsigma/287*T)                        # to pressure: P = P_0 * exp(- Mgh/(RT)) Hydrostatic balance (don't have T at pressure level)
    else:
        p_hybridsigma = ds.a*ds.p0 + ds.b*ds.ps

    if mV.resolutions[0] == 'orig':
        ds_cl = ds # units in % on sigma pressure coordinates
        ds_p_hybridsigma = xr.Dataset(data_vars = {'p_hybridsigma': p_hybridsigma}, attrs = ds.lev.attrs)
    if mV.resolutions[0] == 'regridded':
        import regrid_xesmf as regrid
        cl = ds['cl'] # units in % on sigma pressure coordinates
        regridder = regrid.regrid_conserv_xesmf(ds)
        cl = regridder(cl)
        ds_cl = xr.Dataset(data_vars = {'cl': cl}, attrs = ds.attrs)
        p_hybridsigma = regridder(p_hybridsigma) if not model in ['IITM-ESM', 'IPSL-CM6A-LR'] else p_hybridsigma
        ds_p_hybridsigma = xr.Dataset(data_vars = {'p_hybridsigma': p_hybridsigma}, attrs = ds.lev.attrs)
    return ds_cl, ds_p_hybridsigma



# ------------------------
#  For observations (NCI)
# ------------------------
# ------------------------------------------------------------------------------------- GPCP ----------------------------------------------------------------------------------------------------------#
def get_gpcp():
    ''' Observations from the Global Precipitation Climatology Project (GPCP) '''
    path_gen = '/g/data/ia39/aus-ref-clim-data-nci/gpcp/data/day/v1-3'
    years = range(1997,2022) # there is a constant shift in high percentile precipitation rate trend from around (2009-01-2009-06) forward
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
    valid_range = [0, 10000] # There are some e+33 values in the dataset
    da = da.where((da >= valid_range[0]) & (da <= valid_range[1]), np.nan)
    da = da.dropna('time', how='all') # drop days where all values are NaN (one day)

    if mV.resolutions[0] == 'regridded':
        import regrid_xesmf as regrid
        regridder = regrid.regrid_conserv_xesmf(ds)
        da = regridder(da)
    
    ds = xr.Dataset(data_vars = {'pr': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)
    return ds


# -------------------------------------------------------------------------------------- ERA5 ----------------------------------------------------------------------------------------------------------#
def get_era5_monthly(variable):
    ''' Reanalysis data from ERA5 '''
    path_gen = f'/g/data/rt52/era5/pressure-levels/monthly-averaged/{variable}'
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
    da = ds[variable]

    da['level'] = da['level']*100 # convert from millibar (hPa) to Pa
    da = da.rename({'level': 'plev'})

    if mV.resolutions[0] == 'regridded':
        import regrid_xesmf as rD
        regridder = rD.regrid_conserv_xesmf(ds)
        da = regridder(da)
    
    ds = xr.Dataset(data_vars = {f'{variable}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)
    return ds



# ------------------------
#     Load variable
# ------------------------
# -------------------------------------------------------------------------- pick out variable data array ----------------------------------------------------------------------------------------------------- #
def get_var_data(source, dataset, experiment, var_name, switch):
    da = None

    # Checking changes in organization and precipitation extremes with warming (convection based on precipitation threshold)
    if var_name == 'pr':
        da = get_cmip5_data('pr', dataset, experiment, switch)['pr']*60*60*24 if source == 'cmip5' else da
        da = get_cmip6_data('pr', dataset, experiment, switch)['pr']*60*60*24 if source == 'cmip6' else da
        da = get_gpcp()['pr']                                                 if dataset == 'GPCP' else da
        da.attrs['units'] = r'mm day$^-1$'

    if var_name == 'tas':
        da = get_cmip5_data('tas', dataset, experiment, switch)['tas']-273.15  if source == 'cmip5' else da
        da = get_cmip6_data('tas', dataset, experiment, switch)['tas']-273.15  if source == 'cmip6' else da
        da.attrs['units'] = r'$\degree$C'


    # Checking domain mean drying with changes in organization
    if var_name == 'wap':
        da = get_cmip5_data('wap', dataset, experiment, switch)['wap']*60*60*24/100 if source == 'cmip5' else da
        da = get_cmip6_data('wap', dataset, experiment, switch)['wap']*60*60*24/100 if source == 'cmip6' else da
        da = da * 1000 if dataset == 'IITM-ESM' else da
        da.attrs['units'] = r'hPa day$^-1$'

    if var_name == 'hur':
        da = get_cmip5_data('hur', dataset, experiment, switch)['hur']  if source == 'cmip5' else da
        da = get_cmip6_data('hur', dataset, experiment, switch)['hur']  if source == 'cmip6' else da
        da.attrs['units'] = '%'


    # Radiation budget
        # longwave
    if var_name == 'rlds':
        da = get_cmip5_data('rlds', dataset, experiment, switch)['rlds'] if source == 'cmip5' else da
        da = get_cmip6_data('rlds', dataset, experiment, switch)['rlds'] if source == 'cmip6' else da
        da.attrs['units'] = r'W m$^-2$'

    if var_name == 'rlus':
        da = get_cmip5_data('rlus', dataset, experiment, switch)['rlus'] if source == 'cmip5' else da
        da = get_cmip6_data('rlus', dataset, experiment, switch)['rlus'] if source == 'cmip6' else da
        da.attrs['units'] = r'W m$^-2$'

    if var_name == 'rlut':
        da = get_cmip5_data('rlut', dataset, experiment, switch)['rlut'] if source == 'cmip5' else da
        da = get_cmip6_data('rlut', dataset, experiment, switch)['rlut'] if source == 'cmip6' else da
        da.attrs['units'] = r'W m$^-2$'

        # shortwave
    if var_name == 'rsdt':
        da = get_cmip5_data('rsdt', dataset, experiment, switch)['rsdt'] if source == 'cmip5' else da
        da = get_cmip6_data('rsdt', dataset, experiment, switch)['rsdt'] if source == 'cmip6' else da
        da.attrs['units'] = r'W m$^-2$'

    if var_name == 'rsds':
        da = get_cmip5_data('rsds', dataset, experiment, switch)['rsds'] if source == 'cmip5' else da
        da = get_cmip6_data('rsds', dataset, experiment, switch)['rsds'] if source == 'cmip6' else da
        da.attrs['units'] = r'W m$^-2$'

    if var_name == 'rsus':
        da = get_cmip5_data('rsus', dataset, experiment, switch)['rsus'] if source == 'cmip5' else da
        da = get_cmip6_data('rsus', dataset, experiment, switch)['rsus'] if source == 'cmip6' else da
        da.attrs['units'] = r'W m$^-2$'

    if var_name == 'rsut':
        da = get_cmip5_data('rsut', dataset, experiment, switch)['rsut'] if source == 'cmip5' else da
        da = get_cmip6_data('rsut', dataset, experiment, switch)['rsut'] if source == 'cmip6' else da
        da.attrs['units'] = r'W m$^-2$'


    # Cloud types
    if var_name == 'cl':
        da, _ = get_cmip5_cl('cl', dataset, experiment, switch) if source == 'cmip5' else [da, None]
        da, _ = get_cmip6_cl('cl', dataset, experiment, switch) if source == 'cmip6' else [da, None] # hybrid-sigma coords
        da = da['cl'] 
        da.attrs['units'] = '%'

    if var_name == 'p_hybridsigma':
        _, da = get_cmip5_cl('p_hybridsigma', dataset, experiment, switch) if source == 'cmip5' else [None, da]
        _, da = get_cmip6_cl('cl', dataset, experiment, switch)            if source == 'cmip6' else [None, da] # hybrid-sigma coords
        da = da['p_hybridsigma']
        da.attrs['units'] = ''


    # Moist static energy (remaning componentns)
    if var_name == 'ta':
        da = get_cmip5_data('ta', dataset, experiment, switch)['ta'] if source == 'cmip5' else da
        da = get_cmip6_data('ta', dataset, experiment, switch)['ta'] if source == 'cmip6' else da
        da = get_era5_monthly('t')['t']                              if dataset == 'ERA5' else da
        da.attrs['units'] = 'K'

    if var_name == 'hus':
        da = get_cmip5_data('hus', dataset, experiment, switch)['hus'] if source == 'cmip5' else da
        da = get_cmip6_data('hus', dataset, experiment, switch)['hus'] if source == 'cmip6' else da
        da = get_era5_monthly('q')['q']                                if dataset == 'ERA5' else da
        da.attrs['units'] = ''

    if var_name == 'zg':
        da = get_cmip5_data('zg', dataset, experiment, switch)['zg'] if source == 'cmip5' else da
        da = get_cmip6_data('zg', dataset, experiment, switch)['zg'] if source == 'cmip6' else da
        # da.attrs['units'] = ''

    # Moist static energy budget (remaning componentns)
    if var_name == 'hfls':
        da = get_cmip5_data('hfls', dataset, experiment, switch)['hfls'] if source == 'cmip5' else da
        da = get_cmip6_data('hfls', dataset, experiment, switch)['hfls'] if source == 'cmip6' else da

    if var_name == 'hfss':
        da = get_cmip5_data('hfss', dataset, experiment, switch)['hfss'] if source == 'cmip5' else da
        da = get_cmip6_data('hfss', dataset, experiment, switch)['hfss'] if source == 'cmip6' else da


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
        'tas':  False, 'ta':            False                                                   # Temperature
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





