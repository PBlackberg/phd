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
sys.path.insert(0, '{}/phd/functions'.format(home))
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
        # 'TaiESM1',        # 1
        # 'BCC-CSM2-MR',    # 2
        # 'FGOALS-g3',      # 3
        # 'CNRM-CM6-1',     # 4
        # 'MIROC6',         # 5
        # 'MPI-ESM1-2-HR',  # 6
        # 'NorESM2-MM',     # 7
        # 'GFDL-CM4',       # 8
        # 'CanESM5',        # 9
        # 'CMCC-ESM2',      # 10
        # 'UKESM1-0-LL',    # 11
        # 'MRI-ESM2-0',     # 12
        # 'CESM2',          # 13
        # 'NESM3'           # 14
        ]

    datasets = models_cmip5 + models_cmip6

    resolutions = [
        'orig',
        # 'regridded'
        ]
    
    experiments = [
        'historical',
        # 'rcp85'
        # 'abrupt-4xCO2'
        ]

    in_descent_regions = True


    for dataset in datasets:
        print('dataset:', dataset) 
        start = timeit.default_timer()

        for experiment in experiments:
            print(experiment) 

            # load data
            # ds, ds_p_hybridsigma = get_cl(institutes[dataset], dataset, experiment)
            # ds, ds_p_hybridsigma = cf.ds_matrix4d, cf.ds_matrix4d
            ds, ds_p_hybridsigma = get_dsvariable('cl', dataset, experiment, resolution=resolutions[0]), get_dsvariable('p_hybridsigma', dataset, experiment, resolution=resolutions[0])

            data = ds['cl']
            p_hybridsigma = ds['p_hybridsigma']


            # find low clouds and high clouds
            data_low = data.where((p_hybridsigma<=1500e2) & (p_hybridsigma>=600e2)).max(dim='plev') 
            data_high = data.where((p_hybridsigma<=250e2) & (p_hybridsigma>=0)).max(dim='plev')

            if in_descent_regions:
                data_low = in_descent(data_low)
                data_high = in_descent(data_high)

            # calculate diagnositcs
            cl_snapshot = snapshot(data)
            cl_tMean = tMean(data)
            cl_sMean = sMean(data)

            cl_low_snapshot = snapshot(data_low)
            cl_low_tMean = tMean(data_low)
            cl_low_sMean = sMean(data_low)

            cl_high_snapshot = snapshot(data_high)
            cl_high_tMean = tMean(data_high)
            cl_high_sMean = sMean(data_high)


            # organize into dataset
            ds_snapshot = xr.Dataset(
                data_vars ={'cl_snapshot':cl_snapshot,
                            'cl_low_snapshot': cl_low_snapshot,
                            'cl_high_snapshot': cl_high_snapshot}
                )
            
            ds_tMean = xr.Dataset(
                data_vars ={'cl_tMean':cl_tMean,
                            'cl_low_tMean':cl_low_tMean,
                            'cl_high_tMean':cl_high_tMean}
                )
            
            
            ds_sMean = xr.Dataset(
                data_vars ={'cl_sMean':cl_low_sMean,
                            'cl_low_sMean':cl_low_sMean,
                            'cl_high_sMean':cl_low_sMean}
                )


            # save
            if np.isin(models_cmip5, dataset).any():
                folder_save = '{}/data/cmip5/metrics_cmip5_{}'.format(resolutions[0])
            if np.isin(models_cmip6, dataset).any():
                folder_save = '{}/data/cmip6/metrics_cmip6_{}'.format(resolutions[0])

            save = True
            if save:
                if in_descent_regions():
                    fileName = dataset + '_cl_snapshot_d_' + experiment + '.nc'
                    save_file(ds_snapshot, folder_save, fileName)

                    fileName = dataset + '_cl_tMean_d_' + experiment + '.nc'
                    save_file(ds_tMean, folder_save, fileName)
                    
                    fileName = dataset + '_cl_sMean_d_' + experiment + '.nc'
                    save_file(ds_sMean, folder_save, fileName)

                else:
                    fileName = dataset + '_cl_snapshot_' + experiment + '.nc'
                    save_file(ds_snapshot, folder_save, fileName)

                    fileName = dataset + '_cl_tMean_' + experiment + '.nc'
                    save_file(ds_tMean, folder_save, fileName)
                    
                    fileName = dataset + '_cl_sMean_' + experiment + '.nc'
                    save_file(ds_sMean, folder_save, fileName)
                    





        stop = timeit.default_timer()
        print('finished: in {} minutes'.format((stop-start)/60))


























































































