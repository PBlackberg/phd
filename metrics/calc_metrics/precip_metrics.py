import numpy as np
import xarray as xr
import scipy

import timeit
import os
import sys
run_on_gadi = False
if run_on_gadi:
    home = '/g/data/k10/cb4968'
    sys.path.insert(0, '{}/phd/metrics/get_variables'.format(home))
else:
    home = os.path.expanduser("~") + '/Documents'
sys.path.insert(0, '{}/code/phd/functions'.format(home))
from myFuncs import *
# import constructed_fields as cf


def snapshot(var):
    return var.isel(time=0)

def tMean(var):
    return var.mean(dim='time', keep_attrs=True)

def sMean(var):
    aWeights = np.cos(np.deg2rad(var.lat))
    return var.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)

def in_descent(var, dataset, experiment):
    wap500 = get_dsvariable('wap', dataset, experiment, resolution=resolutions[0])['wap'].sel(plev = 5e4)

    if len(var)<1000:
        wap500 = resample_timeMean(wap500, 'monthly')
        wap500 = wap500.assign_coords(time=data.time)
    return var.where(wap500>0, np.nan)

def rxday(precip):
    rx1day = precip.resample(time='Y').max(dim='time')
    rx1day.attrs['units']= 'mm day' + chr(0x207B) + chr(0x00B9)

    precip5day = precip.resample(time='5D').mean(dim='time')
    rx5day = precip5day.resample(time='Y').max(dim='time')
    rx5day.attrs['units']= 'mm day' + chr(0x207B) + chr(0x00B9)

    ds_rxday = xr.Dataset(
        data_vars = {'rx1day': rx1day, 
                     'rx5day': rx5day}
        )
    return ds_rxday

def pr_percentiles(precip):

    pr95 = precip.quantile(0.95,dim=('lat','lon'),keep_attrs=True)
    pr95 = xr.DataArray(
        data = pr95.data,
        dims = ['time'],
        coords = {'time': precip.time.data}, 
        attrs = {'units':'mm day' + chr(0x207B) + chr(0x00B9)}
        )

    pr97 = precip.quantile(0.97,dim=('lat','lon'),keep_attrs=True)
    pr97 = xr.DataArray(
        data = pr97.data,
        dims = ['time'],
        coords = {'time': precip.time.data},
        attrs = {'units':'mm day' + chr(0x207B) + chr(0x00B9)}
        )

    pr99 = precip.quantile(0.99,dim=('lat','lon'),keep_attrs=True)
    pr99 = xr.DataArray(
        data = pr99.data,
        dims = ['time'],
        coords = {'time': precip.time.data},
        attrs = {'units':'mm day' + chr(0x207B) + chr(0x00B9)}
        )
        
    ds_prPercentiles = xr.Dataset(
        data_vars = {'pr95': pr95, 
                     'pr97': pr97, 
                     'pr99': pr99}
        ) 
    return ds_prPercentiles

def pr_MeanPercentiles(precip):

    pr95 = precip.quantile(0.95,dim=('lat','lon'),keep_attrs=True)
    aWeights = np.cos(np.deg2rad(precip.lat))
    pr95Mean = precip.where(precip>= pr95).weighted(aWeights).mean(dim=('lat', 'lon'))
    pr95Mean = xr.DataArray(
        data = pr95Mean.data,
        dims = ['time'],
        coords = {'time': precip.time.data}, 
        attrs = {'units':'mm day' + chr(0x207B) + chr(0x00B9)}
        )

    pr97 = precip.quantile(0.97,dim=('lat','lon'),keep_attrs=True)
    aWeights = np.cos(np.deg2rad(precip.lat))
    pr97Mean = precip.where(precip>= pr97).weighted(aWeights).mean(dim=('lat', 'lon'))
    pr97Mean = xr.DataArray(
        data = pr97Mean.data,
        dims = ['time'],
        coords = {'time': precip.time.data}, 
        attrs = {'units':'mm day' + chr(0x207B) + chr(0x00B9)}
        )
    
    pr99 = precip.quantile(0.99,dim=('lat','lon'),keep_attrs=True)
    aWeights = np.cos(np.deg2rad(precip.lat))
    pr99Mean = precip.where(precip>= pr99).weighted(aWeights).mean(dim=('lat', 'lon'))
    pr99Mean = xr.DataArray(
        data = pr99Mean.data,
        dims = ['time'],
        coords = {'time': precip.time.data}, 
        attrs = {'units':'mm day' + chr(0x207B) + chr(0x00B9)}
        )
    
    ds_prPercentiles = xr.Dataset(
        data_vars = {'pr95': pr95Mean, 
                     'pr97': pr97Mean, 
                     'pr99': pr99Mean}
        ) 
    return ds_prPercentiles

def F_pr10(precip):
    precip = precip.resample(time='M').mean(dim='time', keep_attrs=True)
    mask = xr.where(precip>10,1,0)
    F_pr10 = (mask).sum(dim=('lat','lon'))
    F_pr10.attrs['units'] = 'Nb'

    ds_F_pr10 = xr.Dataset(
    data_vars = {'F_pr10': F_pr10},
    attrs = {'description': 'Number of gridboxes in daily scene exceeding 10 mm/day'}
        )
    return ds_F_pr10


def data_exist(dataset, experiment):
    data_exitsts = True
    if (experiment == 'abrupt-4xCO2') and (dataset == 'TaiESM1' or dataset == 'BCC-CSM2-MR' or dataset == 'CanESM5' or dataset == 'CMCC-ESM2' or dataset == 'TaiESM1' or dataset == 'NESM3'):
        data_exitsts = False
    return data_exitsts


if __name__ == '__main__':

    models_cmip5 = [
        # 'IPSL-CM5A-MR', # 1
        # 'GFDL-CM3',     # 2
        # 'GISS-E2-H',    # 3
        # 'bcc-csm1-1',   # 4
        # 'CNRM-CM5',     # 5
        # 'CCSM4',        # 6
        # 'HadGEM2-AO',   # 7
        # 'BNU-ESM',      # 8
        # 'EC-EARTH',     # 9
        # 'FGOALS-g2',    # 10
        # 'MPI-ESM-MR',   # 11
        # 'CMCC-CM',      # 12
        # 'inmcm4',       # 13
        # 'NorESM1-M',    # 14
        # 'CanESM2',      # 15
        # 'MIROC5',       # 16
        # 'HadGEM2-CC',   # 17
        # 'MRI-CGCM3',    # 18
        # 'CESM1-BGC'     # 19
        ]
    
    models_cmip6 = [
        'TaiESM1',        # 1
        'BCC-CSM2-MR',    # 2
        'FGOALS-g3',      # 3
        'CNRM-CM6-1',     # 4
        'MIROC6',         # 5
        'MPI-ESM1-2-HR',  # 6
        'NorESM2-MM',     # 7
        'GFDL-CM4',       # 8
        'CanESM5',        # 9
        'CMCC-ESM2',      # 10
        'UKESM1-0-LL',    # 11
        'MRI-ESM2-0',     # 12
        'CESM2',          # 13
        'NESM3'           # 14
        ]
    
    observations = [
        # 'GPCP'
        ]
    
    datasets = models_cmip5 + models_cmip6 + observations

    resolutions = [
        # 'orig',
        'regridded'
        ]
    
    experiments = [
        'historical',
        # 'rcp85',
        'ssp585',
        ]

    for dataset in datasets:
        print(dataset)
        for experiment in experiments:
            if not data_exist(dataset, experiment):
                print(f'no {experiment} data')
            else:
                print(experiment)

                # load data
                if run_on_gadi:
                    if dataset == 'GPCP':
                        from obs_variables import *
                        precip = get_GPCP(institutes[model], model, experiment)['precip']
                    
                    if np.isin(models_cmip5, dataset).any():
                        from cmip5_variables import *
                        precip = get_pr(institutes[model], model, experiment)['precip']
                    
                    if run_on_gadi and np.isin(models_cmip6, dataset).any():
                        from cmip6_variables import *
                        precip = get_pr(institutes[model], model, experiment)['precip']
                else:
                    precip = get_dsvariable('precip', dataset, experiment, timescale = 'daily')['precip']


                # Calculate diagnostics and put into dataset
                # ds_rxday = rxday(precip)
                # ds_prPercentiles = pr_percentiles(precip)
                ds_prMeanPercentiles = pr_MeanPercentiles(precip)
                # ds_F_pr10 = F_pr10(precip)


                # save
                save_rxday = False
                save_prPercentiles = False
                save_prMeanPercentiles = True
                save_F_pr10 = False


                if np.isin(models_cmip5, dataset).any():
                    project = 'cmip5'
                elif np.isin(models_cmip6, dataset).any():
                    project = 'cmip6'
                elif np.isin(observations, dataset).any():
                    project = 'obs'
                folder_save = home + '/data/' + project + '/' + 'metrics_' + project + '_' + resolutions[0] + '/' + dataset 


                if save_rxday:
                    fileName = dataset + '_rxday_' + experiment + '_' + resolutions[0] + '.nc'
                    save_file(ds_rxday, folder_save, fileName)

                if save_prPercentiles:
                    fileName = dataset + '_prPercentiles_' + experiment + '_' + resolutions[0] + '.nc'
                    save_file(ds_prPercentiles, folder_save, fileName)

                if save_prMeanPercentiles:
                    fileName = dataset + '_prMeanPercentiles_' + experiment + '_' + resolutions[0] + '.nc'
                    save_file(ds_prMeanPercentiles, folder_save, fileName)

                if save_F_pr10 :
                    fileName = dataset + '_F_pr10_' + experiment + '_' + resolutions[0] + '.nc'
                    save_file(ds_F_pr10, folder_save, fileName)
                
                print('finished')















