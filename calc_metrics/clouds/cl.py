import numpy as np
import xarray as xr

import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/util')
import myFuncs as mF # imports common operators
import myVars as mV # imports common variables
import constructed_fields as cF # imports fields for testing
import get_data as gD # imports functions to get data from gadi


# -------------------------------------------------------------------- Find low/high clouds and subsidence/ascent regions ----------------------------------------------------------------------------------------------------- #

def pick_cloud_type(switch, da, p_hybridsigma):
    if  switch['low_clouds']:
        cloud_type = 'lcf' # Low cloud fraction
        da = da.where((p_hybridsigma <= 1500e2) & (p_hybridsigma >= 600e2), 0).max(dim='lev')
        return da, cloud_type
    elif switch['high_clouds']:
        cloud_type = 'hcf' # High cloud fraction
        da = da.where((p_hybridsigma <= 250e2) & (p_hybridsigma >= 0), 0).max(dim='lev')
        return da, cloud_type
    else:
        cloud_type = 'cf'  # cloud fraction
        return da, cloud_type
    
def pick_wap_region(switch, da, source, dataset, experiment):
    ''' Pick out data in regions of ascent/descent based on 500 hPa vertical pressure velocity (wap)'''
    if not switch['ascent'] and not switch['descent']:
        return da, ''
    wap500 = load_wap_data(switch, source, dataset, experiment).sel(plev = 500e2)
    if switch['descent']:
        return da.where(wap500>0), '_d'
    if switch['ascent']:
        return da.where(wap500<0), '_a'

def calc_sMean(da):
    ''' Calculate area-weighted spatial mean '''
    aWeights = np.cos(np.deg2rad(da.lat))
    return da.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)

# ------------------------------------------------------------------------------------ Calculate metrics and save ----------------------------------------------------------------------------------------------------- #

def calc_metrics(switch, da, cloud_type, region, source, dataset, experiment):
    if switch['snapshot']:
        metric_name =f'cl{cloud_type}{region}_snapshot' 
        ds_snapshot = xr.Dataset({metric_name: da.isel(time=0)})
        folder = f'{mV.folder_save[0]}/cl/metrics/{metric_name}/{source}'
        filename = f'{dataset}_{metric_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'
        mF.save_file(ds_snapshot, folder, filename) if switch['save'] else None

    if switch['sMean']:
        metric_name =f'cl{cloud_type}{region}_sMean' 
        ds_sMean = xr.Dataset({metric_name: calc_sMean(da)})
        folder = f'{mV.folder_save[0]}/cl/metrics/{metric_name}/{source}'
        filename = f'{dataset}_{metric_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'
        mF.save_file(ds_sMean, folder, filename) if switch['save'] else None

    if switch['tMean']:
        metric_name =f'cl{cloud_type}{region}_tMean' 
        ds_tMean = xr.Dataset({metric_name: da.mean(dim='time', keep_attrs=True)})
        folder = f'{mV.folder_save[0]}/cl/metrics/{metric_name}/{source}'
        filename = f'{dataset}_{metric_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'
        mF.save_file(ds_tMean, folder, filename) if switch['save'] else None


# ---------------------------------------------------------------------------------- Get the data, pick regions, and run ----------------------------------------------------------------------------------------------------- #

def load_cl_data(switch, source, dataset, experiment):
    if  switch['constructed_fields']:
        return cF.var3d, cF.var3d
    elif switch['sample_data']:
        path_cl = f'/Users/cbla0002/Documents/data/cl/sample_data/{source}/{dataset}_cl_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'
        path_lev = f'/Users/cbla0002/Documents/data/cl/sample_data/{source}/{dataset}_p_hybridsigma_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'
        return xr.open_dataset(path_cl)['cl'], xr.open_dataset(path_lev)['p_hybridsigma']
    else:
        return gD.get_cl(source, dataset, experiment, mV.timescales[0], mV.resolutions[0]), gD.get_p_hybridsigma(source, dataset, experiment, mV.timescales[0], mV.resolutions[0])

def load_wap_data(switch, source, dataset, experiment):
    if  switch['constructed_fields']:
        return cF.var3D
    elif switch['sample_data']:
        path = f'/Users/cbla0002/Documents/data/wap/sample_data/{source}/{dataset}_wap_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'
        return xr.open_dataset(path)['wap']
    else:
        return gD.get_wap(source, dataset, mV.timescales[0], experiment, mV.resolutions[0])

def run_experiment(switch, source, dataset):
    for experiment in mV.experiments:
        if experiment and source in ['cmip5', 'cmip6']:
            print(f'\t {experiment}') if mF.data_exist(dataset, experiment) else print(f'\t no {experiment} data')
        print( '\t obserational dataset') if not experiment and source == 'obs' else None

        if mF.no_data(source, experiment, mF.data_exist(dataset, experiment)):
            continue

        da, p_hybridsigma = load_cl_data(switch, source, dataset, experiment)
        da, cloud_type = pick_cloud_type(switch, da, p_hybridsigma)
        da, region = pick_wap_region(switch, da, source, dataset, experiment)
        calc_metrics(switch, da, cloud_type, region, source, dataset, experiment)

@mF.timing_decorator
def run_cl_metrics(switch):
    print(f'Running tas metrics with {mV.resolutions[0]} {mV.timescales[0]} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')

    for dataset in mV.datasets:
        source = mF.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        print(f'{dataset} ({source})')

        run_experiment(switch, source, dataset)

# -------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
    run_cl_metrics(switch = {
        # choose data to calculate metric on
        'constructed_fields': False, 
        'sample_data':        True,

        # choose metrics to calculate
        'snapshot':           True, 
        'sMean':              True, 
        'tMean':              True, 

        # choose type of cloud
        'low_clouds':         True,
        'high_clouds':        False,

        # mask by
        'ascent':             False,
        'descent':            False,

        # save
        'save':               True
        }
    )
    










