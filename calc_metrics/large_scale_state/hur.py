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

def calc_vertical_mean(da):
    da = da.sel(plev=slice(0,850e2)) # free troposphere (most values at 1000 hPa over land are NaN)
    return (da * da.plev).sum(dim='plev') / da.plev.sum(dim='plev')


# ------------------------------------------------------------------------------------ Calculate metrics and save ----------------------------------------------------------------------------------------------------- #

def calc_metrics(switch, da, region, source, dataset, timescale, experiment, resolution, folder_save):
    if switch['snapshot']:
        ds_snapshot = xr.Dataset({f'hur{region}_snapshot' : mF.get_scene(da)})
        mV.save_metric(ds_snapshot, folder_save, f'hur{region}_snapshot', source, dataset, timescale, experiment, resolution) if switch['save'] else None

    if switch['sMean']:
        ds_sMean = xr.Dataset({f'hur{region}_sMean' : mF.calc_sMean(da)})
        mV.save_metric(ds_sMean, folder_save, f'hur{region}_sMean', source, dataset, timescale, experiment, resolution) if switch['save'] else None

    if switch['tMean']:
        ds_tMean = xr.Dataset({f'hur{region}_tMean' : mF.calc_tMean(da)})
        mV.save_metric(ds_tMean, folder_save, f'hur{region}_tMean', source, dataset, timescale, experiment, resolution) if switch['save'] else None


# ---------------------------------------------------------------------------------- Get the data, pick regions, and run ----------------------------------------------------------------------------------------------------- #

def load_hur_data(switch, source, dataset, timescale, experiment, resolution, folder_load):
    if  switch['constructed_fields']:
        return cF.var3D, cF.var3D
    elif switch['sample_data']:
        return mV.load_sample_data(folder_load, source, dataset, 'hur', timescale, experiment, resolution)['hur']
    else:
        return gD.get_hur(source, dataset, timescale, experiment, resolution)
    
def run_experiment(switch, source, dataset, timescale, experiments, resolution, folder_save):
    for experiment in experiments:
        if experiment and source in ['cmip5', 'cmip6']:
            print(f'\t {experiment}') if mV.data_exist(dataset, experiment) else print(f'\t no {experiment} data')
        print( '\t obserational dataset') if not experiment and source == 'obs' else None

        if mV.no_data(source, experiment, mV.data_exist(dataset, experiment)):
            continue
            
        if dataset == 'ERA5':
            r = gD.get_hus(source, dataset, timescale, experiment, resolution) # unitless (kg/kg)
            t = gD.get_ta(source, dataset, timescale, experiment, resolution) + 273.15 # convert to degrees Kelvin
            p = t['plev'] # Pa
            e_s = 611.2 * np.exp(17.67*(t-273.15)/(t-29.66)) # saturation water vapor pressure
            r_s = 0.622 * e_s/p
            da = (r/r_s)*100 # relative humidity
        else:
            da = load_hur_data(switch, source, dataset, timescale, experiment, resolution, folder_load = folder_save)

        da = calc_vertical_mean(da)
        da, region = pick_wap_region(switch, da, source, dataset, timescale, experiment, resolution, folder_load = f'{mV.folder_save}/wap')
        calc_metrics(switch, da, region, source, dataset, timescale, experiment, resolution, folder_save)

@mF.timing_decorator
def run_hur_metrics(switch):
    print(f'Running hur metrics with {resolution} {timescale} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')

    for dataset in datasets:
        source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        print(f'{dataset} ({source})')

        run_experiment(switch, source, dataset, timescale, experiments, resolution, folder_save)


# -------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
    run_hur_metrics(switch = {
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
    
























