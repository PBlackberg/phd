import numpy as np
import xarray as xr

import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util')
import myFuncs as mF # imports common operators
import constructed_fields as cF # imports fields for testing
import get_data as gD # imports functions to get data from gadi
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV                             # imports common variables


# -------------------------------------------------------------------- Find low/high clouds and subsidence/ascent regions ----------------------------------------------------------------------------------------------------- #

def calc_sMean(da):
    ''' Calculate area-weighted spatial mean '''
    aWeights = np.cos(np.deg2rad(da.lat))
    return da.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)

# ---------------------------------------------------------------------------------------- load data ----------------------------------------------------------------------------------------------------- #

def load_wap_data(switch, source, dataset, experiment):
    da = cF.var3D if  switch['constructed_fields'] else None
    da = xr.open_dataset(f'{mV.folder_save[0]}/wap/sample_data/{source}/{dataset}_wap_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc')['wap'] if switch['sample_data'] else da
    da = gD.get_var_data(source, dataset, experiment, 'wap') if switch['gadi_data'] else da
    return da

def pick_wap_region(switch, source, dataset, experiment, da):
    ''' Pick out data in regions of ascent/descent based on 500 hPa vertical pressure velocity (wap)'''
    if not switch['ascent'] and not switch['descent']:
        return da, ''
    wap500 = load_wap_data(switch, source, dataset, experiment).sel(plev = 500e2)
    if switch['descent']:
        return da.where(wap500>0), '_d'
    if switch['ascent']:
        return da.where(wap500<0), '_a'

def pick_cloud_type(switch, da, p_hybridsigma):
    if  switch['low_clouds']:
        return da.where((p_hybridsigma <= 1500e2) & (p_hybridsigma >= 600e2), 0).max(dim='lev'), 'lcf' # Low cloud fraction
    elif switch['high_clouds']:
        return da.where((p_hybridsigma <= 250e2) & (p_hybridsigma >= 0), 0).max(dim='lev'), 'hcf' # High cloud fraction
    else:
        return da, 'cf'  # cloud fraction
    
def load_data(switch, source, dataset, experiment):
    if switch['constructed_fields']:
        da_cl, da_p = cF.var3D
    if switch['sample_data']:
        da_cl = xr.open_dataset(f'{mV.folder_save[0]}/cl/sample_data/{source}/{dataset}_cl_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc')['cl']
        da_p = xr.open_dataset(f'{mV.folder_save[0]}/cl/sample_data/{source}/{dataset}_p_hybridsigma_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc')['p_hybridsigma']
    if switch['gadi_data']:
        da_cl, da_p = gD.get_var_data(source, dataset, experiment, 'cl')
    return da_cl, da_p

# ------------------------------------------------------------------------------------- Put metric in dataset ----------------------------------------------------------------------------------------------------- #

def get_metric(da, cloud_type, region, metric):
    da_calc, metric_name = None, None
    if metric == 'snapshot':
        metric_name =f'{cloud_type}{region}_snapshot' 
        da_calc = da.isel(time=0)

    if metric == 'sMean':
        metric_name =f'{cloud_type}{region}_sMean' 
        da_calc = calc_sMean(da)

    if metric == 'tMean':
        metric_name =f'{cloud_type}{region}_tMean'
        da_calc = da.mean(dim='time', keep_attrs=True)

    return xr.Dataset(data_vars = {metric_name: da_calc}), metric_name

# ---------------------------------------------------------------------------------- Get the data, pick regions, and run ----------------------------------------------------------------------------------------------------- #

def save_metric(source, dataset, experiment, ds, metric_name):
    folder = f'{mV.folder_save[0]}/cl/metrics/{metric_name}/{source}'
    filename = f'{dataset}_{metric_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'
    mF.save_file(ds, folder, filename)

def run_metric(switch, source, dataset, experiment):
    da, p_hybridsigma = load_data(switch, source, dataset, experiment)
    da, cloud_type = pick_cloud_type(switch, da, p_hybridsigma)
    da, region = pick_wap_region(switch, source, dataset, experiment, da)

    for metric in [k for k, v in switch.items() if v] : # loop over true keys
        ds, metric_name = get_metric(da, cloud_type, region, metric)
        save_metric(source, dataset, experiment, ds, metric_name) if switch['save'] and ds[metric_name].any() else None

def run_experiment(switch, source, dataset):
    for experiment in mV.experiments:
        if not mF.data_available(source, dataset, experiment):
            continue
        print(f'\t {experiment}') if experiment else print(f'\t observational dataset')
        run_metric(switch, source, dataset, experiment)

def run_dataset(switch):
    for dataset in mV.datasets:
        source = mF.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        print(f'{dataset} ({source})')
        run_experiment(switch, source, dataset)

@mF.timing_decorator
def run_cl_metrics(switch):
    print(f'Running {os.path.basename(__file__)} with {mV.resolutions[0]} {mV.timescales[0]} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')
    run_dataset(switch)



# -------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
    run_cl_metrics(switch = {
        # choose data to calculate metric on
        'constructed_fields': False, 
        'sample_data':        True,
        'gadi_data':          False,

        # choose metrics to calculate
        'snapshot':           True, 
        'sMean':              True, 
        'tMean':              True, 

        # choose type of cloud
        'low_clouds':         True,
        'high_clouds':        False,

        # mask by
        'ascent':             False,
        'descent':            True,

        # save
        'save':               True
        }
    )
    










