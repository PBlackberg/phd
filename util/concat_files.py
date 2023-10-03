import xarray as xr
import numpy as np
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV



# ------------------------
#      General funcs
# ------------------------
# ----------------------------------------------------------------------------- concatenate files ----------------------------------------------------------------------------------------------------------#
def concat_files(path_folder, experiment):
    ''' Concatenates files of monthly or daily data between specified years
    (takes out a little bit wider range to not exclude data when interpolating grid) '''
    files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
    year1, year2 = (1970, 1999) if experiment == 'historical' else (2070, 2099) # range of years to concatenate files for
    fileYear1_charStart, fileYear1_charEnd = (13, 9) if 'Amon' in path_folder else (17, 13) # indicates between which characters in the filename the first fileyear is described (count starting from the end)
    fileYear2_charStart, fileYear2_charEnd = (6, 2) if 'Amon' in path_folder else (8, 4) # where the last fileyear is described
    files = sorted(files, key=lambda x: x[x.index(".nc")-fileYear1_charStart:x.index(".nc")-fileYear1_charEnd])
    files = [f for f in files if int(f[f.index(".nc")-fileYear1_charStart : f.index(".nc")-fileYear1_charEnd]) <= int(year2) and int(f[f.index(".nc")-fileYear2_charStart : f.index(".nc")-fileYear2_charEnd]) >= int(year1)]
    # for f in files:  # one model from warming scenario from cmip5 have a file that needs to be removed (creates duplicate data otherwise)
    #     files.remove(f) if f[f.index(".nc")-fileYear1_charStart : f.index(".nc")-fileYear1_charEnd]=='19790101' and f[f.index(".nc")-fileYear2_charStart : f.index(".nc")]=='20051231' else None
    paths = []
    for file in files:
        paths = np.append(paths, os.path.join(path_folder, file))
    # print(paths[0])
    ds = xr.open_mfdataset(paths, combine='by_coords').sel(time=slice(str(year1), str(year2)),lat=slice(-35,35)) # take out a little bit wider range to not exclude data when interpolating grid
    return ds


# ---------------------------------------------------------------------------------- pick folder ----------------------------------------------------------------------------------------------------------#
def latestVersion(path):
    ''' Picks the latest version if there are multiple '''
    versions = os.listdir(path)
    version = max(versions, key=lambda x: int(x[1:])) if len(versions)>1 else versions[0]
    return version

def grid_folder(model):
    ''' Some models have a different grid folder in the path to the files'''
    folder = 'gn'
    folder = 'gr' if model in ['CNRM-CM6-1', 'EC-Earth3', 'IPSL-CM6A-LR'] else folder
    folder = 'gr1' if model in ['GFDL-CM4', 'INM-CM5-0', 'KIOST-ESM'] else folder           
    return folder



# ------------------------
#          CMIP5
# ------------------------
# ------------------------------------------------------------------------------- For most variables ----------------------------------------------------------------------------------------------------------#
def choose_cmip5_ensemble(model, experiment):
    ''' Some models don't have the ensemble most common amongst other models 
    and some experiments don't have the same ensemble as the historical simulation'''
    ensemble = 'r6i1p1' if model in ['EC-EARTH', 'CCSM4'] else 'r1i1p1'
    ensemble = 'r6i1p1' if model == 'GISS-E2-H' and experiment == 'historical' else ensemble
    ensemble = 'r2i1p1' if model == 'GISS-E2-H' and not experiment == 'historical' else ensemble
    return ensemble

def get_cmip5_data(variable, model, experiment):
    ''' concatenates file data and interpolates grid to common grid if needed '''    
    ensemble = choose_cmip5_ensemble(model, experiment)
    timeInterval = ('day', 'day') if mV.timescales[0] == 'daily' else ('mon', 'Amon')
    path_gen = f'/g/data/al33/replicas/CMIP5/combined/{mV.institutes[model]}/{model}/{experiment}/{timeInterval[0]}/atmos/{timeInterval[1]}/{ensemble}'
    version = latestVersion(path_gen)
    path_folder = f'{path_gen}/{version}/{variable}'
    ds = concat_files(path_folder, variable, model, experiment)
    da= ds[variable]
    if mV.resolutions[0] == 'regridded':
        import xesmf_regrid as regrid
        regridder = regrid.regrid_conserv_xesmf(ds)
        da = regridder(da)
    ds = xr.Dataset(data_vars = {f'{variable}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)
    return ds


# --------------------------------------------------------------------------------- For cloudfraction ----------------------------------------------------------------------------------------------------------#
def get_cmip5_cl(variable, model, experiment):
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
        import xesmf_regrid as regrid
        cl = ds['cl'] # units in % on sigma pressure coordinates
        regridder = regrid.regrid_conserv_xesmf(ds)
        cl = regridder(cl)
        p_hybridsigma_n = regridder(p_hybridsigma)
        ds_cl = xr.Dataset(data_vars = {'cl': cl}, attrs = ds.attrs)
        ds_p_hybridsigma = xr.Dataset(data_vars = {'p_hybridsigma': p_hybridsigma_n}, attrs = ds.lev.attrs)
    return ds_cl, ds_p_hybridsigma



# ------------------------
#         CMIP6
# ------------------------
# ------------------------------------------------------------------------------- For most variables ----------------------------------------------------------------------------------------------------------#
def choose_cmip6_ensemble(model, experiment):
    ''' Some models don't have the ensemble most common amongst other models 
    and some experiments don't have the same ensemble as the historical simulation'''
    ensemble = 'r1i1p1f2' if model in ['CNRM-CM6-1', 'UKESM1-0-LL'] else 'r1i1p1f1'
    ensemble = 'r11i1p1f1' if model == 'CESM2' and not experiment == 'historical' else ensemble
    return ensemble

def get_cmip6_data(variable, model, experiment, switch = ''):
    ''' concatenates file data and interpolates grid to common grid if needed'''
    ensemble = choose_cmip6_ensemble(model, experiment)
    project = 'CMIP' if experiment == 'historical' else 'ScenarioMIP'
    timeInterval = 'day' if mV.timescales[0] == 'daily' else 'Amon'
    path_gen = f'/g/data/oi10/replicas/CMIP6/{project}/{mV.institutes[model]}/{model}/{experiment}/{ensemble}/{timeInterval}/{variable}'
    folder_grid = grid_folder(model)
    version = latestVersion(os.path.join(path_gen, folder_grid))
    path_folder =  f'{path_gen}/{folder_grid}/{version}'
    ds = concat_files(path_folder, experiment) # picks out lat: [-35, 35]
    da = ds[variable]

    if switch['ocean_mask']:
        ''

    if mV.resolutions[0] == 'regridded': # conservatively interpolate
        import xesmf_regrid as regrid
        regridder = regrid.regrid_conserv_xesmf(ds) # define regridder based of grid from other model
        da = regridder(da) # conservatively interpolate data onto grid from other model
    ds = xr.Dataset(data_vars = {f'{variable}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs) # if regridded it should already be lat: [-30,30]
    return ds
    

# '/g/data/oi10/replicas/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r1i1p1f1/fx/sftlf/gr/v20180803/sftlf_fx_IPSL-CM6A-LR_historical_r1i1p1f1_gr.nc'


# --------------------------------------------------------------------------------- For cloudfraction ----------------------------------------------------------------------------------------------------------#
def get_cmip6_cl(variable, model, experiment):
    ''' Cloud pressure on hybrid-sigma vertical levels '''
    ensemble = choose_cmip6_ensemble(model, experiment)
    project = 'CMIP' if experiment == 'historical' else 'ScenarioMIP'
    timeInterval = 'day' if mV.timescales[0] == 'daily' else 'Amon'
    path_gen = f'/g/data/oi10/replicas/CMIP6/{project}/{mV.institutes[model]}/{model}/{experiment}/{ensemble}/{timeInterval}/{variable}'
    folder_grid = grid_folder(model)
    version = latestVersion(os.path.join(path_gen, folder_grid))
    path_folder =  f'{path_gen}/{folder_grid}/{version}'
    ds = concat_files(path_folder, experiment) # picks out lat: [-35, 35]
    if model == 'IITM-ESM': # different models have different conversions from height coordinate to pressure coordinate.
        ds = ds.rename({'plev':'lev'})
        p_hybridsigma = ds['lev'] # already on pressure levels
    elif model == 'IPSL-CM6A-LR':
        ds = ds.rename({'presnivs':'lev'})
        p_hybridsigma = ds['lev'] # already on pressure levels
    elif model in ['MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'CanESM5', 'CNRM-CM6-1', 'GFDL-CM4']: 
        p_hybridsigma = ds.ap + ds.b*ds.ps
    elif model == 'FGOALS-g3':
        p_hybridsigma = ds.ptop + ds.lev*(ds.ps-ds.ptop)
    elif model == 'UKESM1-0-LL': # (not used)
        p_hybridsigma = ds.lev+ds.b*ds.orog # in meters
        p_hybridsigma = 1000e2 * (1 -  0.0065*p_hybridsigma/288.15)**(9.81*0.029)/(8.314*0.0065) # to pressure: P = P_0 * (1- L*(h-h_0)/T_0)^(g*M/R*L) Barometric formula
    else:
        p_hybridsigma = ds.a*ds.p0 + ds.b*ds.ps

    if mV.resolutions[0] == 'orig':
        ds_cl = ds # units in % on sigma pressure coordinates
        ds_p_hybridsigma = xr.Dataset(data_vars = {'p_hybridsigma': p_hybridsigma}, attrs = ds.lev.attrs)
    if mV.resolutions[0] == 'regridded':
        import xesmf_regrid as regrid
        cl = ds['cl'] # units in % on sigma pressure coordinates
        regridder = regrid.regrid_conserv_xesmf(ds)
        cl = regridder(cl)
        ds_cl = xr.Dataset(data_vars = {'cl': cl}, attrs = ds.attrs)
        p_hybridsigma = regridder(p_hybridsigma) if not model in ['IITM-ESM', 'IPSL-CM6A-LR'] else p_hybridsigma
        ds_p_hybridsigma = xr.Dataset(data_vars = {'p_hybridsigma': p_hybridsigma}, attrs = ds.lev.attrs)
    return ds_cl, ds_p_hybridsigma



# ------------------------
#   Observations (NCI)
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
        import xesmf_regrid as regrid
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
        import xesmf_regrid as rD
        regridder = rD.regrid_conserv_xesmf(ds)
        da = regridder(da)
    
    ds = xr.Dataset(data_vars = {f'{variable}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)
    return ds










