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


# ------------------------------------------------------------------------------------ Calculate metrics and save ----------------------------------------------------------------------------------------------------- #

def calc_metrics(switch, da, cloud_type, region, source, dataset, experiment, resolution, folder_save):
    if switch['snapshot']:
        ds_snapshot = xr.Dataset({f'{cloud_type}{region}_snapshot' : mF.get_scene(da)})
        mV.save_metric(ds_snapshot, folder_save, f'{cloud_type}{region}_snapshot', source, dataset, experiment, resolution) if switch['save'] else None

    if switch['sMean']:
        ds_sMean = xr.Dataset({f'{cloud_type}{region}_sMean' : mF.calc_sMean(da)})
        mV.save_metric(ds_sMean, folder_save, f'{cloud_type}{region}_sMean', source, dataset, experiment, resolution) if switch['save'] else None

    if switch['tMean']:
        ds_tMean = xr.Dataset({f'{cloud_type}{region}_tMean' : mF.calc_tMean(da)})
        mV.save_metric(ds_tMean, folder_save, f'{cloud_type}{region}_tMean', source, dataset, experiment, resolution) if switch['save'] else None


# ---------------------------------------------------------------------------------- Get the data, pick regions, and run ----------------------------------------------------------------------------------------------------- #

def load_cl_data(switch, source, dataset, experiment, timescale, resolution, folder_load):
    if  switch['constructed_fields']:
        return cF.var3D, cF.var3D
    elif switch['sample_data']:
        da = mV.load_sample_data(folder_load, source, dataset, 'cl', timescale, experiment, resolution)['cl']
        p_hybridsigma = mV.load_sample_data(folder_load, source, dataset, 'p_hybridsigma', timescale, experiment, resolution)['p_hybridsigma']
        return da, p_hybridsigma 
    else:
        da = gD.get_cl(source, dataset, experiment, timescale, resolution)
        p_hybridsigma = gD.get_p_hybridsigma(source, dataset, experiment, timescale, resolution)
        return da, p_hybridsigma
    

def run_experiment(switch, source, dataset, experiments, timescale, resolution, folder_save):
    for experiment in experiments:
        if experiment and source in ['cmip5', 'cmip6']:
            print(f'\t {experiment}') if mV.data_exist(dataset, experiment) else print(f'\t no {experiment} data')
        print( '\t obserational dataset') if not experiment and source == 'obs' else None

        if mV.no_data(source, experiment, mV.data_exist(dataset, experiment)):
            continue

        da, p_hybridsigma = load_cl_data(switch, source, dataset, experiment, timescale, resolution, folder_load = folder_save)
        da, cloud_type = pick_cloud_type(switch, da, p_hybridsigma)
        da, region = pick_wap_region(switch, da, source, dataset, experiment, timescale, resolution, folder_load = f'{mV.folder_save}/wap')
        calc_metrics(switch, da, cloud_type, region, source, dataset, experiment, resolution, folder_save)


def run_cl_metrics(switch, datasets, experiments, timescale, resolution, folder_save = f'{mV.folder_save}/cl'):
    print(f'Running cl metrics with {resolution} {timescale} data')
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

        'low_clouds':         True,
        'high_clouds':        False,
        'ascent':             False,
        'descent':            False,

        'snapshot':           False, 
        'sMean':              False, 
        'tMean':              True, 
        
        'save':               True
        }

    # choose which datasets and experiments to run, and where to save the metric
    ds_metric = run_cl_metrics(switch = switch,
                               datasets =    mV.datasets, 
                               experiments = mV.experiments,
                               timescale =   mV.timescales[0],
                               resolution =  mV.resolutions[0],
                                # folder_save = f'{mV.folder_save_gadi}/cl'
                                )
    

    stop = timeit.default_timer()
    print(f'Finshed, script finished in {round((stop-start)/60, 2)} minutes.')
    










