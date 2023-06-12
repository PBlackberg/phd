import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import timeit

import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/functions')
import myFuncs as mF # imports common operator
import myVars as mV # imports common variables
import concat_files as cfiles

# make one function for cmip6, one for cmip5, and one or more for obs, then one function that calls them depending on switch

def prData_exist(model, experiment):
    ''' Check if model/project has data
    (for precipitation a model is not included if it does not have daily precipitation data)'''
    data_exist = 'True'
    return data_exist


def get_pr_cmip6(institute, model, experiment, timescale, resolution):
    ''' Surfae precipitation from cmip6 (only using daily data so far)'''
    variable = 'pr'
    if not prData_exist(model, experiment):
        print(f'there is no precipitation data for experiment: {experiment} for model: {model}')
        return
    
    ensemble = cfiles.choose_cmip6_ensemble(model, experiment)
    project = 'CMIP' if experiment == 'historical' else 'ScenarioMIP'
    timeInterval = 'day' if timescale == 'daily' else 'Amon'
    path_gen = f'/g/data/oi10/replicas/CMIP6/{project}/{institute}/{model}/{experiment}/{ensemble}/{timeInterval}/{variable}'

    ds = cfiles.concat_files(path_gen, variable, model, experiment) # picks out lat: [-35, 35]
    precip = ds['pr']*60*60*24 # convert to mm/day

    if resolution == 'regridded': # conservatively interpolate
        import xesmf_regrid as regrid
        regridder = regrid.regrid_conserv_xesmf(ds) # define regridder based of grid from other model
        precip = regridder(precip) # conservatively interpolate data onto grid from other model
    
    precip.attrs['units']= 'mm day' + mF.super('-1') # assign new units
    ds_pr = xr.Dataset(data_vars = {'precip': precip.sel(lat=slice(-30,30))}, attrs = ds.attrs) # if regridded it should already be lat: [-30,30]
    return ds_pr


def get_pr_cmip5(institute, model, experiment, timescale, resolution):
    ''' Surfae precipitation from cmip6 (only using daily data so far) '''
    variable = 'pr'
    if not prData_exist(model, experiment):
        print(f'there is no precipitation data for experiment: {experiment} for model: {model}')
        return
    
    ensemble = cfiles.choose_cmip5_ensemble(model, experiment)
    timeInterval = 'day' if timescale == 'daily' else 'Amon'
    path_gen = f'/g/data/al33/replicas/CMIP5/combined/{institute}/{model}/{experiment}/{timeInterval}/atmos/{timeInterval}/{ensemble}'
    version = cfiles.latestVersion(path_gen)
    path_folder = f'{path_gen}/{version}/{variable}'

    ds = cfiles.concat_files(path_folder, variable, model, experiment)
    precip = ds['pr']*60*60*24

    if resolution == 'regridded':
        import xesmf_regrid as regrid
        regridder = regrid.regrid_conserv_xesmf(ds)
        precip = regridder(precip)

    precip.attrs['units']= 'mm day' + mF.super('-1')
    ds_pr = xr.Dataset(data_vars = {'precip': precip.sel(lat=slice(-30,30))}, attrs = ds.attrs)
    return ds_pr


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

    precip = ds.precip.sel(lat=slice(-35,35), time=slice('1998','2021'))

    # Linearly interpolate where there is missing data or outliers (I have set valied range to [0, 250] mm/day)
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

    if resolution == 'regridded':
        import xesmf_regrid as regrid
        regridder = regrid.regrid_conserv_xesmf(ds)
        precip = regridder(precip)
    
    precip.attrs['units']= 'mm day' + mF.super('-1')
    ds_gpcp = xr.Dataset(data_vars = {'precip': precip.sel(lat=slice(-30,30))}, attrs = ds.attrs)
    return ds_gpcp



def get_imerge(resolution):
    return


def get_pr(source, dataset, experiment, timescale, resolution):
    if source == 'cmip5':
        ds_pr = get_pr_cmip5(mV.institutes[dataset], dataset, experiment, timescale, resolution)

    if source == 'cmip6':
        ds_pr = get_pr_cmip6(mV.institutes[dataset], dataset, experiment, timescale, resolution)

    if dataset == 'GPCP':
        ds_pr = get_gpcp(resolution)
    return ds_pr


def run_get_pr(switch, datasets, experiments, timescale = 'daily', resolution= 'regridded', folder_save=''):
    print(f'running {resolution} {timescale} data')
    for dataset in datasets:
        print(f'{dataset}..')
        source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)

        for experiment in experiments:
            if experiment and source in ['cmip5', 'cmip6']:
                print(f'\t {experiment}') if prData_exist(dataset, experiment) \
                    else print(f'\t no {experiment} data')
            elif not experiment and source == 'obs':
                print( '\t obserational dataset')
            else:
                continue
            
            ds_pr = get_pr(source, dataset, experiment, timescale, resolution)['pr']
            mV.save_sample_data(ds_pr, folder_save, source, dataset, 'pr', timescale, experiment, resolution) if switch['save'] else None    



if __name__ == '__main__':

    start = timeit.default_timer()

    # choose which metrics to calculate
    switch = {
        'save': False
        }

    # compute and save the metrics
    ds_metric = run_get_pr(switch=switch,
                           datasets = mV.datasets, 
                           experiments = mV.experiments,
                           folder_save = '/g/data/k10/cb4968/data/pr'
                           )
    
    stop = timeit.default_timer()
    print(f'Finshed, script finished in {round((stop-start)/60, 2)} minutes.')












