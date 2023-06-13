import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import os


def latestVersion(path):
    ''' Picks the latest version if there are multiple 
    '''
    versions = os.listdir(path)
    version = max(versions, key=lambda x: int(x[1:])) if len(versions)>1 else versions[0]
    return version

def concat_files(path_folder, experiment):
    ''' Concatenates files of monthly or daily data between specified years
    (takes out a little bit wider range to not exclude data when interpolating grid) 
    '''
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
    ds = xr.open_mfdataset(paths, combine='by_coords').sel(time=slice(str(year1), str(year2)),lat=slice(-35,35)) # take out a little bit wider range to not exclude data when interpolating grid
    return ds


# ----------------------------------------------------------------------------------------- for cmip5 data ----------------------------------------------------------------------------------------------------------#

def choose_cmip5_ensemble(model, experiment):
    ''' Some models don't have the ensemble most common amongst other models 
    and some experiments don't have the same ensemble as the historical simulation'''
    ensemble = 'r6i1p1' if model in ['EC-EARTH', 'CCSM4'] else 'r1i1p1'

    ensemble = 'r6i1p1' if model == 'GISS-E2-H' and experiment == 'historical' else ensemble
    ensemble = 'r2i1p1' if model == 'GISS-E2-H' and not experiment == 'historical' else ensemble
    return ensemble

def get_cmip5_data(variable, institute, model, experiment, timescale, resolution, data_exists=True):
    ''' concatenates file data and interpolates grid to common grid if needed '''
    if not data_exists:
        print(f'there is no {variable} data for experiment: {experiment} for model: {model}')
        return
    
    ensemble = choose_cmip5_ensemble(model, experiment)
    timeInterval = ('day', 'day') if timescale == 'daily' else ('mon', 'Amon')
    path_gen = f'/g/data/al33/replicas/CMIP5/combined/{institute}/{model}/{experiment}/{timeInterval[0]}/atmos/{timeInterval[1]}/{ensemble}'
    version = latestVersion(path_gen)
    path_folder = f'{path_gen}/{version}/{variable}'

    ds = concat_files(path_folder, variable, model, experiment)
    da= ds[variable]

    if resolution == 'regridded':
        import xesmf_regrid as regrid
        regridder = regrid.regrid_conserv_xesmf(ds)
        da = regridder(da)

    da.attrs['units']= 'mm day' + mF.super('-1')
    ds_n = xr.Dataset(data_vars = {'pr': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)
    return ds_n

def get_cmip5_cl(variable, institute, model, experiment, timescale, resolution, data_exists=True):
    ''' Cloud pressure on hybrid-sigma vertical levels '''
    if not data_exists:
        print(f'there is no {variable} data for experiment: {experiment} for model: {model}')
        return
    
    ensemble = choose_cmip5_ensemble(model, experiment)
    timeInterval = ('day', 'day') if timescale == 'daily' else ('mon', 'Amon')
    path_gen = f'/g/data/al33/replicas/CMIP5/combined/{institute}/{model}/{experiment}/{timeInterval[0]}/atmos/{timeInterval[1]}/{ensemble}'
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
    
    if resolution == 'orig':
        ds_cl = ds # units in % on sigma pressure coordinates
        ds_p_hybridsigma = xr.Dataset(data_vars = {'p_hybridsigma': p_hybridsigma}, attrs = ds.lev.attrs)

    if resolution == 'regridded':
        import xesmf_regrid as regrid
        cl = ds['cl'] # units in % on sigma pressure coordinates
        regridder = regrid.regrid_conserv_xesmf(ds)
        cl = regridder(cl)
        p_hybridsigma_n = regridder(p_hybridsigma)

        ds_cl = xr.Dataset(data_vars = {'cl': cl}, attrs = ds.attrs)
        ds_p_hybridsigma = xr.Dataset(data_vars = {'p_hybridsigma': p_hybridsigma_n}, attrs = ds.lev.attrs)
    return ds_cl, ds_p_hybridsigma



# ---------------------------------------------------------------------------------------- for cmip6 data ----------------------------------------------------------------------------------------------------------#

def choose_cmip6_ensemble(model, experiment):
    ''' Some models don't have the ensemble most common amongst other models 
    and some experiments don't have the same ensemble as the historical simulation
    '''
    ensemble = 'r1i1p1f2' if model in ['CNRM-CM6-1', 'UKESM1-0-LL'] else 'r1i1p1f1'
    ensemble = 'r11i1p1f1' if model == 'CESM2' and not experiment == 'historical' else ensemble
    return ensemble

def grid_folder(model):
    ''' Some models have a different grid folder in the path to the files'''
    folder = 'gn'
    folder = 'gr' if model == 'CNRM-CM6-1' else folder
    folder = 'gr1' if model == 'GFDL-CM4' else folder           
    return folder

def get_cmip6_data(variable, institute, model, experiment, timescale, resolution, data_exists = True):
    ''' concatenates file data and interpolates grid to common grid if needed
    '''
    if not data_exists:
        print(f'there is no {variable} data for experiment: {experiment} for model: {model}')
        return
    ensemble = choose_cmip6_ensemble(model, experiment)
    project = 'CMIP' if experiment == 'historical' else 'ScenarioMIP'
    timeInterval = 'day' if timescale == 'daily' else 'Amon'
    path_gen = f'/g/data/oi10/replicas/CMIP6/{project}/{institute}/{model}/{experiment}/{ensemble}/{timeInterval}/{variable}'
    folder_grid = grid_folder(model)
    version = latestVersion(os.path.join(path_gen, folder_grid))
    path_folder =  f'{path_gen}/{folder_grid}/{version}'

    ds = concat_files(path_folder, experiment) # picks out lat: [-35, 35]
    da = ds[variable]

    if resolution == 'regridded': # conservatively interpolate
        import xesmf_regrid as regrid
        regridder = regrid.regrid_conserv_xesmf(ds) # define regridder based of grid from other model
        da = regridder(da) # conservatively interpolate data onto grid from other model

    ds_n = xr.Dataset(data_vars = {'pr': da.sel(lat=slice(-30,30))}, attrs = ds.attrs) # if regridded it should already be lat: [-30,30]
    return ds_n
    
def get_cmip6_cl(variable, institute, model, experiment, timescale, resolution, data_exists=True):
    ''' Cloud pressure on hybrid-sigma vertical levels '''
    if not data_exists:
        print(f'there is no {variable} data for experiment: {experiment} for model: {model}')
        return
    
    ensemble = choose_cmip6_ensemble(model, experiment)
    project = 'CMIP' if experiment == 'historical' else 'ScenarioMIP'
    timeInterval = 'day' if timescale == 'daily' else 'Amon'
    path_gen = f'/g/data/oi10/replicas/CMIP6/{project}/{institute}/{model}/{experiment}/{ensemble}/{timeInterval}/{variable}'
    folder_grid = grid_folder(model)
    version = latestVersion(os.path.join(path_gen, folder_grid))
    path_folder =  f'{path_gen}/{folder_grid}/{version}'
    
    ds = concat_files(path_folder, experiment) # picks out lat: [-35, 35]
        
    if model in ['MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'CanESM5', 'CNRM-CM6-1', 'GFDL-CM4']: # different models have different conversions from height coordinate to pressure coordinate.
        p_hybridsigma = ds.ap + ds.b*ds.ps
    elif model == 'FGOALS-g3':
        p_hybridsigma = ds.ptop + ds.lev*(ds.ps-ds.ptop)
    elif model == 'UKESM1-0-LL':
        p_hybridsigma = ds.lev+ds.b*ds.orog
    else:
        p_hybridsigma = ds.a*ds.p0 + ds.b*ds.ps
    
    if resolution == 'orig':
        ds_cl = ds # units in % on sigma pressure coordinates
        ds_p_hybridsigma = xr.Dataset(data_vars = {'p_hybridsigma': p_hybridsigma}, attrs = ds.lev.attrs)

    if resolution == 'regridded':
        import xesmf_regrid as regrid
        cl = ds['cl'] # units in % on sigma pressure coordinates
        regridder = regrid.regrid_conserv_xesmf(ds)
        cl = regridder(cl)
        p_hybridsigma_n = regridder(p_hybridsigma)

        ds_cl = xr.Dataset(data_vars = {'cl': cl}, attrs = ds.attrs)
        ds_p_hybridsigma = xr.Dataset(data_vars = {'p_hybridsigma': p_hybridsigma_n}, attrs = ds.lev.attrs)
    return ds_cl, ds_p_hybridsigma



# ----------------------------------------------------------------------------------------- OBS: GPCP data ----------------------------------------------------------------------------------------------------------#

def get_gpcp(resolution):
    ''' Observations from the Global Precipitation Climatology Project (GPCP) '''
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
    da = ds.precip.sel(lat=slice(-35,35), time=slice('1998','2021'))

    # Linearly interpolate where there is missing data or outliers (I have set valied range to [0, 250] mm/day)
    valid_range = [0, 250] 
    da = da.where((da >= valid_range[0]) & (da <= valid_range[1]), np.nan)
    da = da.where(da.sum(dim =('lat','lon')) != 0, np.nan)
    threshold = 0.5
    da = da.where(da.isnull().sum(dim=('lat','lon'))/(da.shape[1]*da.shape[2]) < threshold, other=np.nan)
    da = da.dropna('time', how='all')
    nb_nan = da.isnull().sum(dim=('lat', 'lon'))
    nan_days =np.nonzero(nb_nan.data)[0]
    for day in nan_days:
        time_slice = da.isel(time=day)
        nan_indices = np.argwhere(np.isnan(time_slice.values))
        nonnan_indices = np.argwhere(~np.isnan(time_slice.values))
        interpolated_values = griddata(nonnan_indices, time_slice.values[~np.isnan(time_slice.values)], nan_indices, method='linear')
        time_slice.values[nan_indices[:, 0], nan_indices[:, 1]] = interpolated_values

    if resolution == 'regridded':
        import xesmf_regrid as regrid
        regridder = regrid.regrid_conserv_xesmf(ds)
        da = regridder(da)
    
    da.attrs['units']= 'mm day' + mF.super('-1')
    ds_n = xr.Dataset(data_vars = {'pr': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)
    return ds_n




# ----------------------------------------------------------------------------------------- OBS: IMERGE data ----------------------------------------------------------------------------------------------------------#

def get_imerge(resolution):
    return


































































