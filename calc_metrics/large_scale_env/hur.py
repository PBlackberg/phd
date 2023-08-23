import numpy as np
import xarray as xr

import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util')
import myFuncs as mF                            # imports common operators
import constructed_fields as cF                 # imports fields for testing
import get_data as gD                           # imports functions to get data from gadi
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV                             # imports common variables

# ------------------------------------------------------------------------------------ Calculate metric ----------------------------------------------------------------------------------------------------- #

def calc_sMean(da):
    ''' Calculate area-weighted spatial mean '''
    aWeights = np.cos(np.deg2rad(da.lat))
    return da.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)

def calc_vertical_mean_era(da):
    da = da.sel(plev=slice(0,850e2)) # free troposphere (most values at 1000 hPa over land are NaN)
    return (da * da.plev).sum(dim='plev') / da.plev.sum(dim='plev')

def calc_vertical_mean(da):
    da = da.sel(plev=slice(850e2, 0)) # free troposphere (most values at 1000 hPa over land are NaN)
    return (da * da.plev).sum(dim='plev') / da.plev.sum(dim='plev')

# ---------------------------------------------------------------------------------------- load data ----------------------------------------------------------------------------------------------------- #

def load_wap_data(switch, source, dataset, experiment):
    da = cF.var3D if switch['constructed fields'] else None
    da = xr.open_dataset(f'{mV.folder_save[0]}/wap/sample_data/{source}/{dataset}_wap_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc')['wap'] if switch['sample data'] else da
    da = gD.get_var_data(source, dataset, experiment, 'wap') if switch['gadi data'] else da
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
    
def pick_hur_region(switch, dataset, da):
    da = calc_vertical_mean(da) if not dataset == 'ERA5' else calc_vertical_mean_era(da)
    return da

def load_data(switch, source, dataset, experiment):
    if  switch['constructed fields']:
        return cF.var3D
    if switch['sample data']:
        path = f'{mV.folder_save[0]}/hur/sample_data/{source}/{dataset}_hur_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'
        return xr.open_dataset(path)['hur']
    if switch['gadi data']:
        if dataset == 'ERA5': # the relative humidity needs to be calculated for ERA
            r = gD.get_var_data(source, dataset, experiment, 'hus') # unitless (kg/kg)
            T = gD.get_var_data(source, dataset, experiment, 'ta') # degrees Kelvin
            p = T['plev'] # Pa
            e_s = 611.2 * np.exp(17.67*(T-273.15)/(T-29.66)) # saturation water vapor pressure
            r_s = 0.622 * e_s/p
            da = (r/r_s)*100 # relative humidity
        else:
            da = gD.get_var_data(source, dataset, experiment, 'hur')
        # could calculate humidity for better comparison with ERA5
        # r = gD.get_hus(source, dataset, experiment, mV.timescales[0], mV.resolutions[0]) # unitless (kg/kg)
        # t = gD.get_ta(source, dataset, experiment, mV.timescales[0], mV.resolutions[0]) + 273.15 # convert to degrees Kelvin
        # print(t.height)
        # p = t['plev'] # Pa
        # e_s = 611.2 * np.exp(17.67*(t-273.15)/(t-29.66)) # saturation water vapor pressure
        # r_s = 0.622 * e_s/p
        # da = (r/r_s)*100 # relative humidity
        return  da
    
# ------------------------------------------------------------------------------------- Put metric in dataset ----------------------------------------------------------------------------------------------------- #

def get_metric(da, region, metric):
    da_calc, metric_name = None, None
    if metric == 'snapshot':
        metric_name =f'hur{region}_snapshot' 
        da_calc = da.isel(time=0)

    if metric == 'sMean':
        metric_name =f'hur{region}_sMean' 
        da_calc = calc_sMean(da)

    if metric == 'tMean':
        metric_name =f'hur{region}_tMean'
        da_calc = da.mean(dim='time', keep_attrs=True)

    return xr.Dataset(data_vars = {metric_name: da_calc}), metric_name

# --------------------------------------------------------------------------------------- run metric and save ----------------------------------------------------------------------------------------------------- #

def save_metric(source, dataset, experiment, ds, metric_name):
    folder = f'{mV.folder_save[0]}/hur/metrics/{metric_name}/{source}'
    filename = f'{dataset}_{metric_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'
    mF.save_file(ds, folder, filename)

def run_metric(switch, source, dataset, experiment):
    da = load_data(switch, source, dataset, experiment)
    da = pick_hur_region(switch, dataset, da)
    da, region = pick_wap_region(switch, source, dataset, experiment, da)

    for metric in [k for k, v in switch.items() if v] : # loop over true keys
        ds, metric_name = get_metric(da, region, metric)
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
def run_hur_metrics(switch):
    print(f'Running {os.path.basename(__file__)} with {mV.resolutions[0]} {mV.timescales[0]} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')
    run_dataset(switch)


# -------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
    run_hur_metrics(switch = {
        # choose data to calculate metric on
        'constructed fields': False, 
        'sample data':        True,
        'gadi data':          False,

        # choose metrics to calculate
        'sMean':              True, 
        'tMean':              True, 
        'snapshot':           True, 
        
        # mask by
        'ascent':             False,
        'descent':            True,

        # save
        'save':               True
        }
    )
    


