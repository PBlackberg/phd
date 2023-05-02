import numpy as np
import xarray as xr
import skimage.measure as skm


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



# --------------------------------------------------------------------------------- apply regridding ------------------------------------------------------------------------------- #

if __name__ == '__main__':
    import matplotlib.pyplot as plt

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

    models_cmip6 =[
        # 'TaiESM1',        # 1 rcp monthly
        # 'BCC-CSM2-MR',    # 2 rcp monthly   
        # 'FGOALS-g3',      # 3 rcp 0463 - 0614
        'CNRM-CM6-1',     # 4 rcp 1850-1999
        'MIROC6',         # 5 rcp 3200 - 3340
        'MPI-ESM1-2-HR',  # 6 rcp 1850 - 2014
        'NorESM2-MM',     # 7 rcp 0001 - 0141
        'GFDL-CM4',       # 8 rcp 0001 - 0141 (gr2)
        # 'CanESM5',        # 9 rcp 1850 - 2000
        # 'CMCC-ESM2',      # 10 rcp monthly
        'UKESM1-0-LL',    # 11 rcp 1850 - 1999
        'MRI-ESM2-0',     # 12 rcp 1850 - 2000
        'CESM2',          # 13 rcp 0001 - 0990  (multiple fill values (check if all get converted to NaN), for historical)
        # 'NESM3',          # 14 rcp 1850-2014
        ]


    observations = [
        # 'GPCP'
        ]

    datasets = models_cmip5 + models_cmip6 + observations


    resolutions = [
        'orig', 
        'regridded'
        ]
        
    experiments = [
                # 'historical',
                # 'rcp85',
                'abrupt-4xCO2'
                ]


    for dataset in datasets:
        print(dataset, 'started') 
        start = timeit.default_timer()
        for experiment in experiments:
            print(experiment) 

            # precip
            if dataset == 'GPCP':
                # ds = cf.matrix3d
                # ds = get_pr(institutes[dataset], dataset, experiment)  
                ds = get_dsvariable('precip', dataset, experiment, home, resolutions[0])
            else:
                # ds = cf.matrix3d
                # ds = get_pr(institutes[dataset], dataset, experiment)  
                ds = get_dsvariable('precip', dataset, experiment, home, resolutions[0])
            
            data = ds['precip']


            # regridding
            print('loaded data')
            data_regrid = regrid_conserv(data)
            print('finished regridding')


            # save if necessary
            save_pr = True

            # select folder
            if np.isin(models_cmip5, dataset).any():
                project = 'cmip5'
            elif np.isin(models_cmip6, dataset).any():
                project = 'cmip6'
            elif np.isin(observations, dataset).any():
                project = 'obs'
            
            folder_save = home + '/data/' + project + '/' + 'ds_' + project + '_' + resolutions[1] + '/' + dataset 

            # save
            if save_pr:
                fileName = dataset + '_precip_' + experiment + '_' + resolutions[1] + '.nc'
                save_file(xr.Dataset(data_vars = {'precip': data_regrid}, attrs = ds.attrs), folder_save, fileName)


        stop = timeit.default_timer()
        print('script took: {} minutes to finsih'.format((stop-start)/60))
























































