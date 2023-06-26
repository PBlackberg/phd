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


# ------------------------------------------------------------------------------------- Calculating metric from data array ----------------------------------------------------------------------------------------------------- #

def get_scene(da):
    ''' Snapshot to visualize the calculation by the precipitation metrics.
    '''
    return da.isel(time=0)

def calc_sMean(da):
    ''' Calculate area-weighted spatial mean '''
    aWeights = np.cos(np.deg2rad(da.lat))
    return da.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)


# ------------------------------------------------------------------------------------ Organize metric into dataset and save ----------------------------------------------------------------------------------------------------- #

def calc_metrics(switch, da, cloud_type, region, source, dataset, experiment, folder_save):
    if switch['var_snapshot']:
        ds_snapshot = xr.Dataset({'var_snapshot' : get_scene(da)})
        mV.save_metric(ds_snapshot, folder_save, f'{cloud_type}_snapshot{region}', source, dataset, experiment) if switch['save'] else None

    if switch['sMean']:
        ds_sMean = xr.Dataset({'cl_sMean' : calc_sMean(da)})
        mV.save_metric(ds_sMean, folder_save, f'{cloud_type}_sMean{region}', source, dataset, experiment) if switch['save'] else None


# -------------------------------------------------------------------------------- Get the data from the dataset / experiment and run ----------------------------------------------------------------------------------------------------- #

def load_cl_data(switch, source, dataset, experiment, timescale, resolution, folder_load):
    if  switch['constructed_fields']:
        return cF.var3D, cF.var3D
    elif switch['sample_data']:
        return mV.load_sample_data(folder_load, dataset, 'cl', timescale, experiment, resolution)['cl'], mV.load_sample_data(folder_load, dataset, 'p_hybridsigma', timescale, experiment, resolution)['p_hybridsigma']
    else:
        return gD.get_cl(source, dataset, experiment, timescale, resolution), gD.get_p_hybridsigma(source, dataset, experiment, timescale, resolution)
    

def pick_cloud_type(switch, da, p_hybridsigma):
    if  switch['low_clouds']:
        cloud_type = 'lcf' # Low cloud fraction
        return da.where((p_hybridsigma <= 1500e2) & (p_hybridsigma >= 600e2), 0).max(dim='lev'), cloud_type
    elif switch['high_clouds']:
        cloud_type = 'hcf' # High cloud fraction
        return da.where((p_hybridsigma <= 250e2) & (p_hybridsigma >= 0), 0).max(dim='lev'), cloud_type
    else:
        cloud_type = 'cf'  # cloud fraction
        return da, cloud_type
    

def load_wap_data(switch, source, dataset, experiment, timescale, resolution, folder_load):
    if  switch['constructed_fields']:
        return cF.var3D
    elif switch['sample_data']:
        return mV.load_sample_data(folder_load, dataset, 'wap', timescale, experiment, resolution)['wap']
    else:
        return gD.get_wap(source, dataset, experiment, timescale, resolution)
    

def pick_wap_region(switch, da, source, dataset, experiment, folder_save, timescale = 'monthly', resolution = 'regridded'):
    ''' Pick out data in regions of ascent/descent based on 500 hPa vertical pressure velocity (wap)'''
    if not switch['ascent'] or not switch['descent']:
        region = ''
        return da, region
    wap = load_wap_data(switch, source, dataset, experiment, timescale, resolution, folder_save, variable = 'wap')
    wap500 = wap.sel(plev = 500e2)
    if switch['descent']:
        region = '_descent'
        da = da.where(wap500>0)
    elif switch['ascent']:
        region = '_ascent'
        da = da.where(wap500<0)
    return da, region


def run_experiment(switch, source, dataset, experiments, timescale, resolution, folder_save):
    for experiment in experiments:
        if experiment and source in ['cmip5', 'cmip6']:
            print(f'\t {experiment}') if mV.data_exist(dataset, experiment) else print(f'\t no {experiment} data')
        print( '\t obserational dataset') if not experiment and source == 'obs' else None

        if mV.no_data(source, experiment, mV.data_exist(dataset, experiment)):
            continue

        da, p_hybridsigma = load_cl_data(switch, source, dataset, experiment, timescale, resolution, folder_save)
        da, cloud_type = pick_cloud_type(switch, da, p_hybridsigma)
        da, region = pick_wap_region(switch, da, source, dataset, experiment, folder_save)
        calc_metrics(switch, da, cloud_type, region, source, dataset, experiment, folder_save)


def run_cl_metrics(switch, datasets, experiments, timescale = 'monthly', resolution= 'regridded', folder_save = f'{mV.folder_save}/wap'):
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
        'sample_data':        False,

        'low_clouds':         True,
        'high_clouds':        False,
        'ascent':             False,
        'descent':            False,

        'var_snapshot':       True, 
        'sMean':              False, 
        
        'save':               True
        }

    # choose which datasets and experiments to run, and where to save the metric
    ds_metric = run_cl_metrics(switch = switch,
                                datasets = mV.datasets, 
                                experiments = mV.experiments,
                                folder_save = f'{mV.folder_save_gadi}/pr'
                                )
    

    stop = timeit.default_timer()
    print(f'Finshed, script finished in {round((stop-start)/60, 2)} minutes.')
    




































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
#         ]

#     datasets = models_cmip5 + models_cmip6

#     resolutions = [
#         'orig',
#         # 'regridded'
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
#             # ds, ds_p_hybridsigma = get_cl(institutes[dataset], dataset, experiment)
#             # ds, ds_p_hybridsigma = cf.ds_matrix4d, cf.ds_matrix4d
#             ds, ds_p_hybridsigma = get_dsvariable('cl', dataset, experiment, resolution=resolutions[0]), get_dsvariable('p_hybridsigma', dataset, experiment, resolution=resolutions[0])

#             data = ds['cl']
#             p_hybridsigma = ds['p_hybridsigma']


#             # find low clouds and high clouds
#             data_low = data.where((p_hybridsigma<=1500e2) & (p_hybridsigma>=600e2)).max(dim='plev') 
#             data_high = data.where((p_hybridsigma<=250e2) & (p_hybridsigma>=0)).max(dim='plev')

#             if in_descent_regions:
#                 data_low = in_descent(data_low)
#                 data_high = in_descent(data_high)

#             # calculate diagnositcs
#             cl_snapshot = snapshot(data)
#             cl_tMean = tMean(data)
#             cl_sMean = sMean(data)

#             cl_low_snapshot = snapshot(data_low)
#             cl_low_tMean = tMean(data_low)
#             cl_low_sMean = sMean(data_low)

#             cl_high_snapshot = snapshot(data_high)
#             cl_high_tMean = tMean(data_high)
#             cl_high_sMean = sMean(data_high)


#             # organize into dataset
#             ds_snapshot = xr.Dataset(
#                 data_vars ={'cl_snapshot':cl_snapshot,
#                             'cl_low_snapshot': cl_low_snapshot,
#                             'cl_high_snapshot': cl_high_snapshot}
#                 )
            
#             ds_tMean = xr.Dataset(
#                 data_vars ={'cl_tMean':cl_tMean,
#                             'cl_low_tMean':cl_low_tMean,
#                             'cl_high_tMean':cl_high_tMean}
#                 )
            
            
#             ds_sMean = xr.Dataset(
#                 data_vars ={'cl_sMean':cl_low_sMean,
#                             'cl_low_sMean':cl_low_sMean,
#                             'cl_high_sMean':cl_low_sMean}
#                 )


#             # save
#             if np.isin(models_cmip5, dataset).any():
#                 folder_save = '{}/data/cmip5/metrics_cmip5_{}'.format(resolutions[0])
#             if np.isin(models_cmip6, dataset).any():
#                 folder_save = '{}/data/cmip6/metrics_cmip6_{}'.format(resolutions[0])

#             save = True
#             if save:
#                 if in_descent_regions():
#                     fileName = dataset + '_cl_snapshot_d_' + experiment + '_' + resolutions[0] + '.nc'
#                     save_file(ds_snapshot, folder_save, fileName)

#                     fileName = dataset + '_cl_tMean_d_' + experiment + '_' + resolutions[0] + '.nc'
#                     save_file(ds_tMean, folder_save, fileName)
                    
#                     fileName = dataset + '_cl_sMean_d_' + experiment + '_' + resolutions[0] + '.nc'
#                     save_file(ds_sMean, folder_save, fileName)

#                 else:
#                     fileName = dataset + '_cl_snapshot_' + experiment + '_' + resolutions[0] + '.nc'
#                     save_file(ds_snapshot, folder_save, fileName)

#                     fileName = dataset + '_cl_tMean_' + experiment + '_' + resolutions[0] + '.nc'
#                     save_file(ds_tMean, folder_save, fileName)
                    
#                     fileName = dataset + '_cl_sMean_' + experiment + '_' + resolutions[0] + '.nc'
#                     save_file(ds_sMean, folder_save, fileName)
                    





#         stop = timeit.default_timer()
#         print('finished: in {} minutes'.format((stop-start)/60))


























































































