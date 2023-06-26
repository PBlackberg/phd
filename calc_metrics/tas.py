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
    aWeights = np.cos(np.deg2rad(var['lat']))
    return var.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)

def in_descent(var, dataset, experiment):
    wap500 = get_dsvariable('wap', dataset, experiment, resolution=resolutions[0])['wap'].sel(plev = 5e4)

    if len(var)<1000 and len(var)>100:
        wap500 = resample_timeMean(wap500, 'monthly')
        wap500 = wap500.assign_coords(time=data['time'])
    return var.where(wap500>0, np.nan)


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

    datasets = models_cmip5 + models_cmip6

    resolutions = [
        # 'orig',
        'regridded'
        ]
    
    experiments = [
        # 'historical',
        # 'rcp85'
        'ssp585'
        ]


    for dataset in datasets:
        print('dataset:', dataset) 
        start = timeit.default_timer()

        for experiment in experiments:
            print(experiment) 

            # load data
            # ds = cf.matrix3d
            # ds = get_tas(institutes[dataset], dataset, experiment)
            ds = get_dsvariable('tas', dataset, experiment, home, resolutions[0])
            data = ds['tas']

            in_descent_regions = False
            if in_descent_regions:
                data = in_descent(data)
            
            tas_snapshot = snapshot(data)
            tas_tMean = tMean(data)
            tas_sMean = sMean(data)


            # organize into dataset
            ds_snapshot = xr.Dataset({'tas_snapshot':tas_snapshot})
            ds_tMean = xr.Dataset({'tas_tMean':tas_tMean})
            ds_sMean = xr.Dataset({'tas_sMean':tas_sMean})

            # save
            if np.isin(models_cmip5, dataset).any():
                project = 'cmip5'
            elif np.isin(models_cmip6, dataset).any():
                project = 'cmip6'
            folder_save = home + '/data/' + project + '/' + 'metrics_' + project + '_' + resolutions[0] + '/' + dataset 
            
            save = True
            if save:
                if in_descent_regions:
                    fileName = dataset + '_tas_snapshot_d_' + experiment + '_' + resolutions[0] + '.nc'
                    save_file(ds_snapshot, folder_save, fileName)

                    fileName = dataset + '_tas_tMean_d_' + experiment + '_' + resolutions[0] + '.nc'
                    save_file(ds_tMean, folder_save, fileName)
                    
                    fileName = dataset + '_tas_sMean_d_' + experiment + '_' + resolutions[0] + '.nc'
                    save_file(ds_sMean, folder_save, fileName)

                else:
                    fileName = dataset + '_tas_snapshot_' + experiment + '_' + resolutions[0] + '.nc'
                    save_file(ds_snapshot, folder_save, fileName)

                    fileName = dataset + '_tas_tMean_' + experiment + '_' + resolutions[0] + '.nc'
                    save_file(ds_tMean, folder_save, fileName)
                    
                    fileName = dataset + '_tas_sMean_' + experiment + '_' + resolutions[0] + '.nc'
                    save_file(ds_sMean, folder_save, fileName)
            




























































