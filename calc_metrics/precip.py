import numpy as np
import xarray as xr
import timeit
import skimage.measure as skm
import cartopy

import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/functions')
import myFuncs as mF # imports common operators
import myVars as mV # imports common variables
import constructed_fields as cF # imports fields for testing
import get_data as gD # imports functions to get data from gadi


# ------------------------------------------------------------------------------------- Calculating metric from data array ----------------------------------------------------------------------------------------------------- #

def get_scene(da):
    ''' Snapshot to visualize the calculation by the precipitation metrics.
    '''
    return da.isel(time=0)

def calc_rx1day(da):
    ''' Most extreme daily gridpoint value locally over time (1 year here)
    '''
    rx1day = da.resample(time='Y').max(dim='time')
    aWeights = np.cos(np.deg2rad(da.lat))
    rx1day_tMean = rx1day.mean(dim='time')
    rx1day_sMean = rx1day.weighted(aWeights).mean(dim=('lat', 'lon'))
    return rx1day_tMean, rx1day_sMean

def calc_rx5day(da):
    ''' Most extreme 5-day average gridpoint value locally over time (1 year here)
    '''
    da5day = da.resample(time='5D').mean(dim='time')
    rx5day = da5day.resample(time='Y').max(dim='time')
    aWeights = np.cos(np.deg2rad(da.lat))
    rx5day_tMean = rx5day.mean(dim='time')
    rx5day_sMean = rx5day.weighted(aWeights).mean(dim=('lat', 'lon'))
    return rx5day_tMean, rx5day_sMean

def find_percentile(da, percentile):
    ''' Spatial percentile of the scene
    '''
    percentile_value = da.quantile(percentile, dim=('lat', 'lon'), keep_attrs=True)
    return percentile_value

def calc_meanInPercentile(da, percentile):
    ''' Mean precipitation rate of the gridboxes included in the percentile of each scene
    '''
    da_snapshot = da.isel(time=0)
    percentile_snapshot = da_snapshot.where(da_snapshot>=find_percentile(da, percentile).isel(time=0))
    aWeights = np.cos(np.deg2rad(da.lat))
    meanInPercentile = da.where(da >= find_percentile(da, percentile)).weighted(aWeights).mean(dim=('lat', 'lon'), keep_attrs=True)
    return percentile_snapshot, meanInPercentile

def calc_F_pr10(da):
    ''' Frequency of gridboxes exceeding 10 mm/day
    '''
    da = mF.resample_timeMean(da, 'M')
    mask = xr.where(da>10,1,0)
    F_pr10 = mask.sum(dim=('lat','lon'))
    da_snapshot = da.isel(time=0)
    F_pr10_snapshot = da_snapshot.where(mask.isel(time=0) > 0)
    return F_pr10_snapshot, F_pr10

def calc_o_pr(da, conv_threshold):
    ''' Precipitation rate in each contigous convective region (object)
    '''
    lat = da['lat'].data
    lon = da['lon'].data
    lonm,latm = np.meshgrid(lon,lat)
    dlat = (lat[1]-lat[0])
    dlon = (lon[1]-lon[0])
    R = 6371 # km
    aream = np.cos(np.deg2rad(latm))*np.float64(dlon*dlat*R**2*(np.pi/180)**2)
    aream3d = np.expand_dims(aream,axis=2) # used for broadcasting
    o_pr = []
    for day in np.arange(0,len(da.time.data)):
        pr_day = da.isel(time=day)
        pr_day3d = np.expand_dims(pr_day,axis=2)
        L = skm.label(pr_day.where(pr_day>=conv_threshold,0)>0, background=0,connectivity=2)
        mF.connect_boundary(L)
        labels = np.unique(L)[1:]
        obj3d = np.stack([(L==label) for label in labels],axis=2)*1 # 3d matrix with each object in a scene being a binary 2d slice
        o_prScene = np.sum(obj3d * pr_day3d * aream3d, axis=(0,1)) / np.sum(obj3d * aream3d, axis=(0,1))
        o_pr = np.append(o_pr, o_prScene)
    return o_pr


# ------------------------------------------------------------------------------------ Organize metric into dataset and save ----------------------------------------------------------------------------------------------------- #

def calc_metrics(switch, da, source, dataset, experiment, folder_save):
    if switch['rxday']:
        rx1day_tMean, rx1day_sMean = calc_rx1day(da)
        rx5day_tMean, rx5day_sMean = calc_rx5day(da)
        ds_rxday_tMean = xr.Dataset({'rx1day':rx1day_tMean , 'rx5day': rx5day_tMean})
        ds_rxday = xr.Dataset({'rx1day': rx1day_sMean , 'rx5day': rx5day_sMean}) 
        mV.save_metric(ds_rxday_tMean, folder_save, 'rxday_pr_tMean', source, dataset, experiment) if switch['save'] else None
        mV.save_metric(ds_rxday,       folder_save, 'rxday_pr',       source, dataset, experiment) if switch['save'] else None

    if switch['percentiles']:
        percentiles = [0.95, 0.97, 0.99]
        ds_percentile_value = xr.Dataset()
        for percentile in percentiles:
            ds_percentile_value[f'pr{int(percentile*100)}'] = find_percentile(da, percentile)
        mV.save_metric(ds_percentile_value, folder_save, 'percentiles_pr', source, dataset, experiment) if switch['save'] else None

    if switch['meanInPercentiles']:
        percentiles = [0.95, 0.97, 0.99]
        ds_percentile_snapshot = xr.Dataset()
        ds_meanInPercentiles = xr.Dataset()
        for percentile in percentiles:
            percentile_snapshot, meanInPercentile = calc_meanInPercentile(da, percentile)
            ds_percentile_snapshot[f'pr{int(percentile*100)}'] = percentile_snapshot
            ds_meanInPercentiles[f'pr{int(percentile*100)}'] = meanInPercentile
        mV.save_metric(ds_percentile_snapshot, folder_save, 'percentiles_pr_snapshot', source, dataset, experiment) if switch['save'] else None    
        mV.save_metric(ds_meanInPercentiles,   folder_save, 'meanInPercentiles_pr',    source, dataset, experiment) if switch['save'] else None

    if switch['F_pr10']:
        F_pr10_snapshot, F_pr10 = calc_F_pr10(da)
        ds_F_pr10_snapshot = xr.Dataset({'F_pr10': F_pr10_snapshot})
        ds_F_pr10 = xr.Dataset({'F_pr10': F_pr10})
        mV.save_metric(ds_F_pr10_snapshot, folder_save, 'F_pr10_snapshot', source, dataset, experiment) if switch['save'] else None
        mV.save_metric(ds_F_pr10,          folder_save, 'F_pr10',          source, dataset, experiment) if switch['save'] else None

    if switch['o_pr']:
        conv_percentile = 0.97
        conv_threshold = find_percentile(da, conv_percentile).mean(dim='time')
        o_pr = xr.DataArray(data = calc_o_pr(da, conv_threshold), dims = 'object',
                                attrs = {'units':'mm day' + mF.get_super('-1'),
                                         'descrption': 'area weighted mean precipitation in contiguous convective region (object)'})
        ds_o_pr = xr.Dataset({'o_pr': o_pr})
        mV.save_metric(ds_o_pr, folder_save, 'o_pr', source, dataset, experiment) if switch['save'] else None



# -------------------------------------------------------------------------------- Get the data from the dataset / experiment and run ----------------------------------------------------------------------------------------------------- #

def load_data(switch, source, dataset, experiment, timescale, resolution, folder_save):
    if switch['constructed_fields']:
        return cF.var2D
    elif switch['sample_data']:
        return mV.load_sample_data(folder_save, dataset, 'pr', timescale, experiment, resolution)['pr']
    else:
        return gD.get_pr(source, dataset, experiment, timescale, resolution)
    

def run_experiment(switch, source, dataset, experiments, timescale, resolution, folder_save):
    for experiment in experiments:
        if experiment and source in ['cmip5', 'cmip6']:
            print(f'\t {experiment}') if mV.data_exist(dataset, experiment) else print(f'\t no {experiment} data')
        print( '\t obserational dataset') if not experiment and source == 'obs' else None

        if mV.no_data(source, experiment, mV.data_exist(dataset, experiment)):
            continue

        da = load_data(switch, source, dataset, experiment, timescale, resolution, folder_save)
        calc_metrics(switch, da, source, dataset, experiment, folder_save)


def run_precip_metrics(switch, datasets, experiments, timescale = 'daily', resolution= 'regridded', folder_save = f'{mV.folder_save}/pr'):
    print(f'Running precip metrics with {resolution} {timescale} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')

    for dataset in datasets:
        source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        print(f'{dataset} ({source})')

        run_experiment(switch, source, dataset, experiments, timescale, resolution, folder_save)



# -------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':

    start = timeit.default_timer()

    # choose which metrics to calculate
    switch = {
        'constructed_fields': False, 
        'sample_data':        True,

        'rxday':              True, 
        'percentiles':        False, 
        'meanInPercentiles':  False, 
        'F_pr10':             False,
        'o_pr':               False,
        
        'save':               False
        }

    # choose which datasets and experiments to run, and where to save the metric
    ds_metric = run_precip_metrics(switch=switch,
                                   datasets = mV.datasets, 
                                   experiments = mV.experiments,
                                #    folder_save = f'{mV.folder_save_gadi}/pr'
                                   )
    

    stop = timeit.default_timer()
    print(f'Finshed, script finished in {round((stop-start)/60, 2)} minutes.')
    














