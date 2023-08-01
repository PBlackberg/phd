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

def load_wap_data(switch, source, dataset, timescale, experiment, resolution, folder_load):
    if  switch['constructed_fields']:
        return cF.var3D
    elif switch['sample_data']:
        return mV.load_sample_data(folder_load, source, dataset, 'wap', timescale, experiment, resolution)['wap']
    else:
        return gD.get_wap(source, dataset, timescale, experiment, resolution)

def pick_wap_region(switch, da, source, dataset, timescale, experiment, resolution, folder_load):
    ''' Pick out data in regions of ascent/descent based on 500 hPa vertical pressure velocity (wap)'''
    if not switch['ascent'] and not switch['descent']:
        region = ''
        return da, region
    wap = load_wap_data(switch, source, dataset, timescale, experiment, resolution, folder_load)
    wap500 = wap.sel(plev = 500e2)
    if switch['descent']:
        region = '_d'
        da = da.where(wap500>0)
    elif switch['ascent']:
        region = '_a'
        da = da.where(wap500<0)
    return da, region

# ------------------------------------------------------------------------------------ Calculate metrics and save ----------------------------------------------------------------------------------------------------- #

def calc_metrics(switch, da, region, source, dataset, timescale, experiment, resolution, folder_save):
    if switch['snapshot']:
        ds_snapshot = xr.Dataset({f'rlut{region}_snapshot' : mF.get_scene(da)})
        mV.save_metric(ds_snapshot, folder_save, f'rlut{region}_snapshot', source, dataset, timescale, experiment, resolution) if switch['save'] else None

    if switch['sMean']:
        ds_sMean = xr.Dataset({f'rlut{region}_sMean' : mF.calc_sMean(da)})
        mV.save_metric(ds_sMean, folder_save, f'rlut{region}_sMean', source, dataset, timescale, experiment, resolution) if switch['save'] else None

    if switch['tMean']:
        ds_tMean = xr.Dataset({f'rlut{region}_tMean' : mF.calc_tMean(da)})
        mV.save_metric(ds_tMean, folder_save, f'rlut{region}_tMean', source, dataset, timescale, experiment, resolution) if switch['save'] else None


# ---------------------------------------------------------------------------------- Get the data, pick regions, and run ----------------------------------------------------------------------------------------------------- #

def load_rlut_data(switch, source, dataset, timescale, experiment, resolution, folder_load):
    if  switch['constructed_fields']:
        return cF.var3D, cF.var3D
    elif switch['sample_data']:
        var = 'rlut' if not dataset == 'CERES' else 'toa_lw_all_mon'
        da = mV.load_sample_data(folder_load, source, dataset, 'rlut', timescale, experiment, resolution)[var]
        return da
    else:
        return gD.get_rlut(source, dataset, timescale, experiment, resolution)
    

def run_experiment(switch, source, dataset, timescale, experiments, resolution, folder_save):
    for experiment in experiments:
        if experiment and source in ['cmip5', 'cmip6']:
            print(f'\t {experiment}') if mV.data_exist(dataset, experiment) else print(f'\t no {experiment} data')
        print( '\t obserational dataset') if not experiment and source == 'obs' else None

        if mV.no_data(source, experiment, mV.data_exist(dataset, experiment)):
            continue

        da = load_rlut_data(switch, source, dataset, timescale, experiment, resolution, folder_load = folder_save)
        da, region = pick_wap_region(switch, da, source, dataset, timescale, experiment, resolution, folder_load = f'{mV.folder_save}/wap')
        calc_metrics(switch, da, region, source, dataset, timescale, experiment, resolution, folder_save)


def run_rlut_metrics(switch, datasets, timescale, experiments, resolution, folder_save = f'{mV.folder_save}/lw'):
    print(f'Running lw metrics with {resolution} {timescale} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')

    for dataset in datasets:
        source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        print(f'{dataset} ({source})')

        run_experiment(switch, source, dataset, timescale, experiments, resolution, folder_save)



# -------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
    run_rlut_metrics(switch = {
        'constructed_fields': False, 
        'sample_data':        False,

        'ascent':             False,
        'descent':            False,

        'snapshot':           True, 
        'sMean':              True, 
        'tMean':              True, 
        
        'save':               True
        }
    )
    









