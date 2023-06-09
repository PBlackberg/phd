import numpy as np
import xarray as xr
import timeit

import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/functions')
import myFuncs as mF # imports common operators
import myVars as mV # imports common variables
import constructed_fields as cF # imports fields for testing
import get_data.pr_data as pD


def calc_rx1day(da):
    rx1day = da.resample(time='Y').max(dim='time')
    aWeights = np.cos(np.deg2rad(da.lat))
    rx1day_tMean = rx1day.mean(dim='time')
    rx1day_sMean = rx1day.weighted(aWeights).mean(dim=('lat', 'lon'))
    return rx1day_tMean, rx1day_sMean

def calc_rx5day(da):
    da5day = da.resample(time='5D').mean(dim='time')
    rx5day = da5day.resample(time='Y').max(dim='time')
    aWeights = np.cos(np.deg2rad(da.lat))
    rx5day_tMean = rx5day.mean(dim='time')
    rx5day_sMean = rx5day.weighted(aWeights).mean(dim=('lat', 'lon'))
    return rx5day_tMean, rx5day_sMean

def find_percentile(da, percentile):
    percentile_value = da.quantile(percentile, dim=('lat', 'lon'), keep_attrs=True)
    return percentile_value

def calc_meanInPercentile(da, percentile):
    da_snapshot = da.isel(time=0)
    percentile_snapshot = da_snapshot.where(da_snapshot>=find_percentile(da, percentile).isel(time=0))
    aWeights = np.cos(np.deg2rad(da.lat))
    meanInPercentile = da.where(da >= find_percentile(da, percentile)).weighted(aWeights).mean(dim=('lat', 'lon'), keep_attrs=True)
    return percentile_snapshot, meanInPercentile

def calc_F_pr10(da):
    da = mF.resample_timeMean(da, 'M')
    mask = xr.where(da>10,1,0)
    F_pr10 = mask.sum(dim=('lat','lon'))
    da_snapshot = da.isel(time=0)
    F_pr10_snapshot = da_snapshot.where(mask.isel(time=0) > 0)
    return F_pr10_snapshot, F_pr10


def calc_metrics(switch, da, folder_save, source, dataset, experiment):
    if switch['rxday']:
        rx1day_tMean, rx1day_sMean = calc_rx1day(da)
        rx5day_tMean, rx5day_sMean = calc_rx5day(da)
        ds_rxday_tMean = xr.Dataset({'rx1day':rx1day_tMean , 'rx5day': rx5day_tMean})
        ds_rxday = xr.Dataset({'rx1day': rx1day_sMean , 'rx5day': rx5day_sMean}) 
        mF.save_metric('rxday_tMean', ds_rxday_tMean, folder_save, source, dataset, experiment) if switch['save'] else None
        mF.save_metric('rxday', ds_rxday, folder_save, source, dataset, experiment) if switch['save'] else None

    if switch['percentiles']:
        percentiles = [0.95, 0.97, 0.99]
        ds_percentile_value = xr.Dataset()
        for percentile in percentiles:
            ds_percentile_value[f'pr{int(percentile*100)}'] = find_percentile(da, percentile)
        mF.save_metric('prPercentiles', ds_percentile_value, folder_save, source, dataset, experiment) if switch['save'] else None

    if switch['meanInPercentiles']:
        percentiles = [0.95, 0.97, 0.99]
        ds_percentile_snapshot = xr.Dataset()
        ds_meanInPercentiles = xr.Dataset()
        for percentile in percentiles:
            percentile_snapshot, meanInPercentile = calc_meanInPercentile(da, percentile)
            ds_percentile_snapshot[f'pr{int(percentile*100)}'] = percentile_snapshot
            ds_meanInPercentiles[f'pr{int(percentile*100)}'] = meanInPercentile
        mF.save_metric('prPercentiles_snapshot', ds_percentile_snapshot, folder_save, source, dataset, experiment) if switch['save'] else None    
        mF.save_metric('prMeanInPercentiles', ds_meanInPercentiles, folder_save, source, dataset, experiment) if switch['save'] else None

    if switch['F_pr10']:
        F_pr10_snapshot, F_pr10 = calc_F_pr10(da)
        ds_F_pr10_snapshot = xr.Dataset(data_vars = {'F_pr10': F_pr10_snapshot})
        ds_F_pr10 = xr.Dataset(data_vars = {'F_pr10': F_pr10})
        mF.save_metric('F_pr10_snapshot', ds_F_pr10_snapshot, folder_save, source, dataset, experiment) if switch['save'] else None
        mF.save_metric('F_pr10', ds_F_pr10, folder_save, source, dataset, experiment) if switch['save'] else None



def run_precip_metrics(switch, datasets, experiments, timescale = 'daily', resolution= 'regridded', folder_save = f'{home}/Documents/data/pr'):
    for dataset in datasets:
        print(f'running {dataset}')
        source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)

        for experiment in experiments:
            print(experiment) if pD.prData_exist(dataset, experiment) else print(f'no {experiment} data')
            if not pD.prData_exist(dataset, experiment):
                continue

            if switch['constructed_fields']:
                da = cF.var2D
            elif switch['sample_data']:
                da = mF.load_sample_data('pr', dataset, experiment, timescale, resolution)['precip']
            else:
                da = pD.get_pr('pr', dataset, experiment, timescale)['pr']
            
            calc_metrics(switch, da, folder_save, source, dataset, experiment)
                



if __name__ == '__main__':

    start = timeit.default_timer()

    # choose which metrics to calculate
    switch = {
        'constructed_fields': False, 
        'sample_data': True,

        'rxday': True, 
        'percentiles': False, 
        'meanInPercentiles': False, 
        'F_pr10': False,
        
        'save': True
        }

    # compute and save the metrics
    ds_metric = run_precip_metrics(switch=switch,
                                   datasets = mV.datasets, 
                                   experiments = mV.experiments,
                                #    folder_save = '/g/data/k10/cb4968/data/pr'
                                   )
    

    stop = timeit.default_timer()
    print(f'Finshed, script finished in {round((stop-start)/60, 2)} minutes.')
    











