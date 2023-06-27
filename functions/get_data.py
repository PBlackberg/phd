import xarray as xr
import numpy as np
import timeit

import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/functions')
import myFuncs as mF # imports common operator
import myVars as mV # imports common variables
import concat_files as cfiles



# --------------------------------------------
# Getting variable based on source and dataset
# --------------------------------------------

# ------------------------------------------------------------------------------------ Surface precipitation ----------------------------------------------------------------------------------------------------- #

def get_pr(source, dataset, experiment, timescale, resolution):
    ''' Surface precipitation
    '''
    if source == 'cmip5':
        ds = cfiles.get_cmip5_data('pr', mV.institutes[dataset], dataset, experiment, timescale, resolution)
        da = ds['pr']*60*60*24 # convert to mm/day
        da.attrs['units']= 'mm day' + mF.get_super('-1')

    if source == 'cmip6':
        ds = cfiles.get_cmip6_data('pr', mV.institutes[dataset], dataset, experiment, timescale, resolution)
        da = ds['pr']*60*60*24 # convert to mm/day
        da.attrs['units']= 'mm day' + mF.get_super('-1')

    if dataset == 'GPCP':
        ds = cfiles.get_gpcp(resolution)
        da = ds['pr']*60*60*24 # already units of mm/day
        da.attrs['units']= 'mm day' + mF.get_super('-1')
    return da


# ----------------------------------------------------------------------------------- Vertical pressure velocity ----------------------------------------------------------------------------------------------------- #

def get_wap(source, dataset, experiment, timescale, resolution):
    ''' Vertical pressure velocity
    '''
    if source == 'cmip5':
        ds = cfiles.get_cmip5_data('wap', mV.institutes[dataset], dataset, experiment, timescale, resolution)
        da = ds['wap']*60*60*24/100 # convert to hPa/day   
        da.attrs['units']= 'hPa day' + mF.get_super('-1')

    if source == 'cmip6':
        ds = cfiles.get_cmip6_data('wap', mV.institutes[dataset], dataset, experiment, timescale, resolution)
        da = ds['wap']*60*60*24/100 # convert to hPa/day   
        da.attrs['units']= 'hPa day' + mF.get_super('-1')
    return da


# -------------------------------------------------------------------------------------- Surface temperature ----------------------------------------------------------------------------------------------------- #

def get_tas(source, dataset, experiment, timescale, resolution):
    ''' Surface temperature
    '''
    if source == 'cmip5':
        ds = cfiles.get_cmip5_data('tas', mV.institutes[dataset], dataset, experiment, timescale, resolution)
        da = ds['tas']-273.15 # convert to degrees Celsius
        da.attrs['units']= mF.get_super('o') + 'C'

    if source == 'cmip6':
        ds = cfiles.get_cmip6_data('tas', mV.institutes[dataset], dataset, experiment, timescale, resolution)
        da = ds['tas']-273.15 # convert to degrees Celsius
        da.attrs['units']= mF.get_super('o') + 'C'
    return da


# ---------------------------------------------------------------------------------------- Cloud fraction ----------------------------------------------------------------------------------------------------- #

def get_cl(source, dataset, experiment, timescale, resolution):
    ''' Cloud fraction
    '''
    if source == 'cmip5':
        ds, _ = cfiles.get_cmip5_cl('cl', mV.institutes[dataset], dataset, experiment, timescale, resolution)
        da = ds['cl'] # units in % on sigma pressure coordinates
        da.attrs['units']= '%'

    if source == 'cmip6':
        ds, _ = cfiles.get_cmip6_cl('cl', mV.institutes[dataset], dataset, experiment, timescale, resolution)
        da = ds['cl'] # units in % on sigma pressure coordinates
        da.attrs['units']= '%'
    return da


# ---------------------------------------------------------------------------------- Hybrid-sgima pressure coordinates ----------------------------------------------------------------------------------------------------- #

def get_p_hybridsigma(source, dataset, experiment, timescale, resolution):
    ''' Pressure values on hybrid-sigma pressure coordinates
    '''
    if source == 'cmip5':
        _, ds = cfiles.get_cmip5_cl('cl', mV.institutes[dataset], dataset, experiment, timescale, resolution)
        da = ds['p_hybrid_sigma'] # units in hPa/day

    if source == 'cmip6':
        _, ds = cfiles.get_cmip6_cl('cl', mV.institutes[dataset], dataset, experiment, timescale, resolution)
        da = ds['p_hybridsigma'] # units in hPa/day
    return da


# ------------------------------------------------------------------------------------------ Relative humidity ----------------------------------------------------------------------------------------------------- #

def get_hur(source, dataset, experiment, timescale, resolution):
    ''' Relative humidity
    '''
    if source == 'cmip5':
        ds = cfiles.get_cmip5_data('hur', mV.institutes[dataset], dataset, experiment, timescale, resolution)
        da = ds['hur'] # units in %

    if source == 'cmip6':
        ds = cfiles.get_cmip6_data('hur', mV.institutes[dataset], dataset, experiment, timescale, resolution)
        da = ds['hur'] # units in %
    return da


# ------------------------------------------------------------------------------------------- Specific humidity ----------------------------------------------------------------------------------------------------- #

def get_hus(source, dataset, experiment, timescale, resolution):
    ''' Specific humidity
    '''
    if source == 'cmip5':
        ds = cfiles.get_cmip5_data('hus', mV.institutes[dataset], dataset, experiment, timescale, resolution)
        da = ds['hus'] # unitless kg/kg

    if source == 'cmip6':
        ds = cfiles.get_cmip6_data('hus', mV.institutes[dataset], dataset, experiment, timescale, resolution)
        da = ds['hus'] # unitless kg/kg
    return da


# --------------------------------------------------------------------------------------- Outgoing longwave radiation ----------------------------------------------------------------------------------------------------- #

def get_rlut(source, dataset, experiment, timescale, resolution):
    ''' Outgoing longwave radiation
    '''
    if source == 'cmip5':
        ds = cfiles.get_cmip5_data('rlut', mV.institutes[dataset], dataset, experiment, timescale, resolution)
        da = ds['rlut'] # W/m^2

    if source == 'cmip6':
        ds = cfiles.get_cmip6_data('rlut', mV.institutes[dataset], dataset, experiment, timescale, resolution)
        da = ds['rlut'] # W/m^2
    return da



# ----------------------------------
# Calling function based on variable
# ----------------------------------

def get_var_data(switch, source, dataset, experiment, timescale, resolution, folder_save):
    da = None

    if switch['pr']:
        da = get_pr(source, dataset, experiment, timescale, resolution)
        mV.save_sample_data(xr.Dataset({'pr': da}), f'{folder_save}/pr', source, dataset, 'pr', timescale, experiment, resolution='daily') if switch['save'] else None

    if switch['wap']:
        da = get_wap(source, dataset, experiment, timescale, resolution)
        mV.save_sample_data(xr.Dataset({'wap': da}), f'{folder_save}/wap', source, dataset, 'wap', timescale, experiment, resolution) if switch['save'] else None
    if switch['tas']:
        da = get_tas(source, dataset, experiment, timescale, resolution)
        mV.save_sample_data(xr.Dataset({'tas': da}), f'{folder_save}/tas', source, dataset, 'tas', timescale, experiment, resolution) if switch['save'] else None

    if switch['cl']:
        da = get_cl(source, dataset, experiment, timescale, resolution)
        mV.save_sample_data(xr.Dataset({'cl': da}), f'{folder_save}/cl', source, dataset, 'cl', timescale, experiment, resolution) if switch['save'] else None

    if switch['p_hybridsigma']:
        da = get_p_hybridsigma(source, dataset, experiment, timescale, resolution)
        mV.save_sample_data(xr.Dataset({'p_hybridsigma': da}), f'{folder_save}/cl', source, dataset, 'p_hybridsigma', timescale, experiment, resolution) if switch['save'] else None

    if switch['hur']:
        da = get_hur(source, dataset, experiment, timescale, resolution)
        mV.save_sample_data(xr.Dataset({'hur': da}), f'{folder_save}/hur', source, dataset, 'hur', timescale, experiment, resolution) if switch['save'] else None

    if switch['hus']:
        da = get_hus(source, dataset, experiment, timescale, resolution)
        mV.save_sample_data(xr.Dataset({'hus': da}), f'{folder_save}/hus', source, dataset, 'hus', timescale, experiment, resolution) if switch['save'] else None

    if switch['rlut']:
        da = get_rlut(source, dataset, experiment, timescale, resolution)
        mV.save_sample_data(xr.Dataset({'rlut': da}), f'{folder_save}/rlut', source, dataset, 'rlut', timescale, experiment, resolution) if switch['save'] else None
    return
        


# ---------------
# Check variable
# ---------------

def run_experiment(switch, source, dataset, experiments, timescale, resolution, folder_save):
    for experiment in experiments:
        if experiment and source in ['cmip5', 'cmip6']:
            print(f'\t {experiment}') if mV.data_exist(dataset, experiment) else print(f'\t no {experiment} data')
        print( '\t obserational dataset') if not experiment and source == 'obs' else None

        if mV.no_data(source, experiment, mV.data_exist(dataset, experiment)):
            continue
    return get_var_data(switch, source, dataset, experiment, timescale, resolution, folder_save)


def run_get_data(switch, datasets, experiments, timescale, resolution, folder_save):
    print(f'Getting variable from {resolution} {timescale} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')

    for dataset in datasets:
        source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        print(f'{dataset} ({source})')
    return run_experiment(switch, source, dataset, experiments, timescale, resolution, folder_save)



if __name__ == '__main__':

    start = timeit.default_timer()
    switch = {
        'pr'  :          False,
        'wap' :          True,
        'tas' :          False,
        'cl'  :          False,
        'p_hybridsigma': False,
        'hus' :          False,
        'hur' :          False,
        'rlut':          False,

        'save':          True
        }

    # get and save sample data if needed
    run_get_data(switch = switch,
                 datasets =    mV.datasets, 
                 experiments = mV.experiments,
                 timescale =   mV.timescales[0],
                 resolution =  mV.resolutions[0],
                 folder_save = mV.folder_save_gadi
                 )
    
    stop = timeit.default_timer()
    print(f'Finshed, script finished in {round((stop-start)/60, 2)} minutes.')























































































































