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
import get_data as gD # imports functions to get data from gadi


# --------------------------------------------------------------------------------- Find subsidence/ascent regions ----------------------------------------------------------------------------------------------------- #

def load_wap_data(switch, source, dataset, experiment, timescale, resolution, folder_load):
    if  switch['constructed_fields']:
        return cF.var3D
    elif switch['sample_data']:
        return mV.load_sample_data(folder_load, source, dataset, 'wap', timescale, experiment, resolution)['wap']
    else:
        return gD.get_wap(source, dataset, experiment, timescale, resolution)

def pick_wap_region(switch, da, source, dataset, experiment, timescale, resolution, folder_load):
    ''' Pick out data in regions of ascent/descent based on 500 hPa vertical pressure velocity (wap)'''
    if not switch['ascent'] and not switch['descent']:
        region = ''
        return da, region
    wap = load_wap_data(switch, source, dataset, experiment, timescale, resolution, folder_load)
    wap500 = wap.sel(plev = 500e2)
    if switch['descent']:
        region = '_d'
        da = da.where(wap500>0)
    elif switch['ascent']:
        region = '_a'
        da = da.where(wap500<0)
    return da, region

def calc_vertical_mean(da):
    da = da.sel(plev=slice(850e2,0)) # free troposphere (most values at 1000 hPa over land are NaN)
    return (da * da.plev).sum(dim='plev') / da.plev.sum(dim='plev')


# ------------------------------------------------------------------------------------ Calculate metrics and save ----------------------------------------------------------------------------------------------------- #

def calc_metrics(switch, da, region, source, dataset, experiment, resolution, folder_save):
    if switch['snapshot']:
        ds_snapshot = xr.Dataset({f'hur{region}_snapshot' : mF.get_scene(da)})
        mV.save_metric(ds_snapshot, folder_save, f'hur{region}_snapshot', source, dataset, experiment, resolution) if switch['save'] else None

    if switch['sMean']:
        ds_sMean = xr.Dataset({f'hur{region}_sMean' : mF.calc_sMean(da)})
        mV.save_metric(ds_sMean, folder_save, f'hur{region}_sMean', source, dataset, experiment, resolution) if switch['save'] else None

    if switch['tMean']:
        ds_tMean = xr.Dataset({f'hur{region}_tMean' : mF.calc_tMean(da)})
        mV.save_metric(ds_tMean, folder_save, f'hur{region}_tMean', source, dataset, experiment, resolution) if switch['save'] else None


# ---------------------------------------------------------------------------------- Get the data, pick regions, and run ----------------------------------------------------------------------------------------------------- #

def load_hur_data(switch, source, dataset, experiment, timescale, resolution, folder_load):
    if  switch['constructed_fields']:
        return cF.var3D, cF.var3D
    elif switch['sample_data']:
        return mV.load_sample_data(folder_load, source, dataset, 'hur', timescale, experiment, resolution)['hur']
    else:
        return gD.get_hus(source, dataset, experiment, timescale, resolution)
    
def run_experiment(switch, source, dataset, experiments, timescale, resolution, folder_save):
    for experiment in experiments:
        if experiment and source in ['cmip5', 'cmip6']:
            print(f'\t {experiment}') if mV.data_exist(dataset, experiment) else print(f'\t no {experiment} data')
        print( '\t obserational dataset') if not experiment and source == 'obs' else None

        if mV.no_data(source, experiment, mV.data_exist(dataset, experiment)):
            continue

        da = load_hur_data(switch, source, dataset, experiment, timescale, resolution, folder_load = folder_save)
        da = calc_vertical_mean(da)
        da, region = pick_wap_region(switch, da, source, dataset, experiment, timescale, resolution, folder_load = f'{mV.folder_save}/wap')
        calc_metrics(switch, da, region, source, dataset, experiment, resolution, folder_save)

def run_hus_metrics(switch, datasets, experiments, timescale, resolution, folder_save = f'{mV.folder_save}/hur'):
    print(f'Running hur metrics with {resolution} {timescale} data')
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

        'ascent':             False,
        'descent':            False,

        'snapshot':           True, 
        'sMean':              False, 
        'tMean':              False, 
        
        'save':               True
        }

    # choose which datasets and experiments to run, and where to save the metric
    ds_metric = run_hus_metrics(switch = switch,
                               datasets =    mV.datasets, 
                               experiments = mV.experiments,
                               timescale =   mV.timescales[0],
                               resolution =  mV.resolutions[0],
                                # folder_save = f'{mV.folder_save_gadi}/cl'
                                )


    stop = timeit.default_timer()
    print(f'Finshed, script finished in {round((stop-start)/60, 2)} minutes.')
    































# import numpy as np
# import xarray as xr
# import scipy

# import timeit
# import os
# import sys
# run_on_gadi = False
# if run_on_gadi:
#     home = '/g/data/k10/cb4968'
#     sys.path.insert(0, '{}/phd/metrics/get_variables'.format(home))
# else:
#     home = os.path.expanduser("~") + '/Documents'
# sys.path.insert(0, '{}/phd/functions'.format(home))
# from myFuncs import *
# # import constructed_fields as cf


# def snapshot(var):
#     return var.isel(time=0)

# def tMean(var):
#     return var.mean(dim='time', keep_attrs=True)

# def sMean(var):
#     aWeights = np.cos(np.deg2rad(var.lat))
#     return var.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)

# def in_descent(var, dataset, experiment):
#     wap500 = get_dsvariable('wap', dataset, experiment, resolution=resolutions[0])['wap'].sel(plev = 5e4)

#     if len(var)<1000:
#         wap500 = resample_timeMean(wap500, 'monthly')
#         wap500 = wap500.assign_coords(time=data.time)
#     return var.where(wap500>0, np.nan)





# if __name__ == '__main__':

#     models_cmip5 = [
#         # 'IPSL-CM5A-MR', # 1
#         # 'GFDL-CM3',     # 2
#         # 'GISS-E2-H',    # 3
#         # 'bcc-csm1-1',   # 4
#         # 'CNRM-CM5',     # 5
#         # 'CCSM4',        # 6
#         # 'HadGEM2-AO',   # 7
#         # 'BNU-ESM',      # 8
#         # 'EC-EARTH',     # 9
#         # 'FGOALS-g2',    # 10
#         # 'MPI-ESM-MR',   # 11
#         # 'CMCC-CM',      # 12
#         # 'inmcm4',       # 13
#         # 'NorESM1-M',    # 14
#         # 'CanESM2',      # 15 
#         # 'MIROC5',       # 16
#         # 'HadGEM2-CC',   # 17
#         # 'MRI-CGCM3',    # 18
#         # 'CESM1-BGC'     # 19
#             ]
    

#     models_cmip6 = [
#         # 'TaiESM1',        # 1
#         # 'BCC-CSM2-MR',    # 2
#         # 'FGOALS-g3',      # 3
#         # 'CNRM-CM6-1',     # 4
#         # 'MIROC6',         # 5
#         # 'MPI-ESM1-2-HR',  # 6
#         # 'NorESM2-MM',     # 7
#         # 'GFDL-CM4',       # 8
#         # 'CanESM5',        # 9
#         # 'CMCC-ESM2',      # 10
#         # 'UKESM1-0-LL',    # 11
#         # 'MRI-ESM2-0',     # 12
#         # 'CESM2',          # 13
#         # 'NESM3'           # 14
#             ]

#     datasets = models_cmip5 + models_cmip6

#     resolutions = [
#         # 'orig',
#         'regridded'
#         ]
    
#     experiments = [
#         'historical',
#         # 'rcp85'
#         # 'abrupt-4xCO2'
#         ]

#     in_descent_regions = True


#     for dataset in datasets:
#         print('dataset:', dataset) 
#         start = timeit.default_timer()

#         for experiment in experiments:
#             print(experiment) 

#             # load data
#             # ds = cf.matrix4d
#             # ds = get_tas(institutes[dataset], dataset, experiment)
#             ds = get_dsvariable('hur', dataset, experiment, resolution=resolutions[0])
            
#             data = ds['hur']
#             data = vertical_mean(data)

#             if in_descent_regions:
#                 data = in_descent(data)
            
#             hur_snapshot = snapshot(data)
#             hur_tMean = tMean(data)
#             hur_sMean = sMean(data)

#             # organize into dataset
#             ds_snapshot = xr.Dataset({'hur_snapshot':hur_snapshot})
#             ds_tMean = xr.Dataset({'hur_tMean':hur_tMean})
#             ds_sMean = xr.Dataset({'hur_sMean':hur_sMean})


#             # save
#             if np.isin(models_cmip5, dataset).any():
#                 folder_save = '{}/data/cmip5/metrics_cmip5_{}'.format(resolutions[0])
#             if np.isin(models_cmip6, dataset).any():
#                 folder_save = '{}/data/cmip6/metrics_cmip6_{}'.format(resolutions[0])

#             save = True
#             if save:
#                 if in_descent_regions():
#                     fileName = dataset + '_hur_snapshot_d_' + experiment + '_' + resolutions[0] + '.nc'
#                     save_file(ds_snapshot, folder_save, fileName)

#                     fileName = dataset + '_hur_tMean_d_' + experiment + '_' + resolutions[0] + '.nc'
#                     save_file(ds_tMean, folder_save, fileName)
                    
#                     fileName = dataset + '_hur_sMean_d_' + experiment + '_' + resolutions[0] + '.nc'
#                     save_file(ds_sMean, folder_save, fileName)

#                 else:
#                     fileName = dataset + '_hur_snapshot_' + experiment + '_' + resolutions[0] + '.nc'
#                     save_file(ds_snapshot, folder_save, fileName)

#                     fileName = dataset + '_hur_tMean_' + experiment + '_' + resolutions[0] + '.nc'
#                     save_file(ds_tMean, folder_save, fileName)
                    
#                     fileName = dataset + '_hur_sMean_' + experiment + '_' + resolutions[0] + '.nc'
#                     save_file(ds_sMean, folder_save, fileName)
























































