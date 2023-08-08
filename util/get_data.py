import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/util')
import myFuncs as mF # imports common operator
import myVars as mV # imports common variables
import concat_files as cfiles



# ------------------
# Loading variable 
# ------------------

# ------------------------------------------------------------------------------------ Surface precipitation ----------------------------------------------------------------------------------------------------- #

def get_pr(source, dataset, experiment, timescale = mV.timescales[0], resolution = mV.resolutions[0]):
    ''' Surface precipitation '''
    if source == 'cmip5':
        ds = cfiles.get_cmip5_data('pr', mV.institutes[dataset], dataset, timescale, experiment, resolution)
        da = ds['pr']*60*60*24 # convert to mm/day
        da.attrs['units']= 'mm day' + mF.get_super('-1')

    if source == 'cmip6':
        ds = cfiles.get_cmip6_data('pr', mV.institutes[dataset], dataset, timescale, experiment, resolution)
        da = ds['pr']*60*60*24 # convert to mm/day
        da.attrs['units']= 'mm day' + mF.get_super('-1')

    if dataset == 'GPCP':
        ds = cfiles.get_gpcp(resolution)
        da = ds['pr'] # already units of mm/day
        da.attrs['units']= 'mm day' + mF.get_super('-1')
    return da

# ----------------------------------------------------------------------------------- Vertical pressure velocity ----------------------------------------------------------------------------------------------------- #

def get_wap(source, dataset, experiment, timescale = mV.timescales[0], resolution = mV.resolutions[0]):
    ''' Vertical pressure velocity '''
    if source == 'cmip5':
        ds = cfiles.get_cmip5_data('wap', mV.institutes[dataset], dataset, timescale, experiment, resolution)
        da = ds['wap']*60*60*24/100 # convert to hPa/day   
        da.attrs['units']= 'hPa day' + mF.get_super('-1')

    if source == 'cmip6':
        ds = cfiles.get_cmip6_data('wap', mV.institutes[dataset], dataset, timescale, experiment, resolution)
        da = ds['wap']*60*60*24/100 # convert to hPa/day   
        da.attrs['units']= 'hPa day' + mF.get_super('-1')
    return da

# -------------------------------------------------------------------------------------- Surface temperature ----------------------------------------------------------------------------------------------------- #

def get_tas(source, dataset, experiment, timescale = mV.timescales[0], resolution = mV.resolutions[0]):
    ''' Surface temperature '''
    if source == 'cmip5':
        ds = cfiles.get_cmip5_data('tas', mV.institutes[dataset], dataset, timescale, experiment, resolution)
        da = ds['tas']-273.15 # convert to degrees Celsius
        da.attrs['units']= mF.get_super('o') + 'C'

    if source == 'cmip6':
        ds = cfiles.get_cmip6_data('tas', mV.institutes[dataset], dataset, timescale, experiment, resolution)
        da = ds['tas']-273.15 # convert to degrees Celsius
        da.attrs['units']= mF.get_super('o') + 'C'
    return da

# ----------------------------------------------------------------------------------------- temperature ----------------------------------------------------------------------------------------------------- #

def get_ta(source, dataset, experiment, timescale = mV.timescales[0], resolution = mV.resolutions[0]):
    ''' temperature '''
    if source == 'cmip5':
        ds = cfiles.get_cmip5_data('tas', mV.institutes[dataset], dataset, timescale, experiment, resolution)
        da = ds['tas']-273.15 # convert to degrees Celsius
        da.attrs['units']= mF.get_super('o') + 'C'

    if source == 'cmip6':
        ds = cfiles.get_cmip6_data('tas', mV.institutes[dataset], dataset, timescale, experiment, resolution)
        da = ds['tas']-273.15 # convert to degrees Celsius
        da.attrs['units']= mF.get_super('o') + 'C'

    if dataset == 'ERA5':
        ds = cfiles.get_era5_monthly('t', resolution)
        da = ds['t']-273.15 # convert to degrees Celsius
        da.attrs['units']= mF.get_super('o') + 'C'
    return da

# ---------------------------------------------------------------------------------------- Cloud fraction ----------------------------------------------------------------------------------------------------- #

def get_cl(source, dataset, experiment, timescale = mV.timescales[0], resolution = mV.resolutions[0]):
    if dataset in ['EC-Earth3', 'INM-CM5-0', 'KIOST-ESM']:
        return None
    ''' Cloud fraction '''
    if source == 'cmip5':
        ds, _ = cfiles.get_cmip5_cl('cl', mV.institutes[dataset], dataset, timescale, experiment, resolution)
        da = ds['cl'] # units in % on sigma pressure coordinates
        da.attrs['units']= '%'

    if source == 'cmip6':
        ds, _ = cfiles.get_cmip6_cl('cl', mV.institutes[dataset], dataset, timescale, experiment, resolution)
        da = ds['cl'] # units in % on sigma pressure coordinates
        da.attrs['units']= '%'
    return da

# ---------------------------------------------------------------------------------- Hybrid-sgima pressure coordinates ----------------------------------------------------------------------------------------------------- #

def get_p_hybridsigma(source, dataset, experiment,  timescale = mV.timescales[0], resolution = mV.resolutions[0]):
    if dataset in ['EC-Earth3', 'INM-CM5-0', 'KIOST-ESM']:
        return None
    ''' Pressure values on hybrid-sigma pressure coordinates '''
    if source == 'cmip5':
        _, ds = cfiles.get_cmip5_cl('cl', mV.institutes[dataset], dataset, timescale, experiment, resolution)
        da = ds['p_hybridsigma'] # units in hPa/day

    if source == 'cmip6':
        _, ds = cfiles.get_cmip6_cl('cl', mV.institutes[dataset], dataset, timescale, experiment, resolution)
        da = ds['p_hybridsigma'] # units in hPa/day
    return da

# ------------------------------------------------------------------------------------------ Relative humidity ----------------------------------------------------------------------------------------------------- #

def get_hur(source, dataset, experiment, timescale = mV.timescales[0], resolution = mV.resolutions[0]):
    ''' Relative humidity '''
    if source == 'cmip5':
        ds = cfiles.get_cmip5_data('hur', mV.institutes[dataset], dataset, timescale, experiment, resolution)
        da = ds['hur'] # units in %

    if source == 'cmip6':
        ds = cfiles.get_cmip6_data('hur', mV.institutes[dataset], dataset, timescale, experiment, resolution)
        da = ds['hur'] # units in %
    return da

# ------------------------------------------------------------------------------------------- Specific humidity ----------------------------------------------------------------------------------------------------- #

def get_hus(source, dataset, experiment, timescale = mV.timescales[0], resolution = mV.resolutions[0]):
    ''' Specific humidity '''
    if source == 'cmip5':
        ds = cfiles.get_cmip5_data('hus', mV.institutes[dataset], dataset, timescale, experiment, resolution)
        da = ds['hus'] # unitless kg/kg

    if source == 'cmip6':
        ds = cfiles.get_cmip6_data('hus', mV.institutes[dataset], dataset, timescale, experiment, resolution)
        da = ds['hus'] # unitless kg/kg

    if dataset == 'ERA5':
        ds = cfiles.get_era5_monthly('q', resolution)
        da = ds['q'] # unitless kg/kg
    return da

# --------------------------------------------------------------------------------------- Outgoing longwave radiation ----------------------------------------------------------------------------------------------------- #

def get_rlut(source, dataset, experiment, timescale = mV.timescales[0], resolution = mV.resolutions[0]):
    ''' Outgoing longwave radiation '''
    if source == 'cmip5':
        ds = cfiles.get_cmip5_data('rlut', mV.institutes[dataset], dataset, timescale, experiment, resolution)
        da = ds['rlut'] # W/m^2

    if source == 'cmip6':
        ds = cfiles.get_cmip6_data('rlut', mV.institutes[dataset], dataset, timescale, experiment, resolution)
        da = ds['rlut'] # W/m^2
    return da


# ---------------------------------------
# Calling function and saving sample data
# ---------------------------------------

def get_var_data(switch, source, dataset, experiment):
    da = None

    if switch['pr']:
        da = get_pr(source, dataset, experiment)
        folder = f'{mV.folder_save[0]}/pr/sample_data/{source}'
        filename = f'{dataset}_pr_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(da, folder, filename) if switch['save'] else None

    if switch['wap']:
        da = get_wap(source, dataset, experiment)
        folder = f'{mV.folder_save[0]}/wap/sample_data/{source}'
        filename = f'{dataset}_wap_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(da, folder, filename) if switch['save'] else None

    if switch['tas']:
        da = get_tas(source, dataset, experiment)
        folder = f'{mV.folder_save[0]}/tas/sample_data/{source}'
        filename = f'{dataset}_tas_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(da, folder, filename) if switch['save'] else None

    if switch['ta']:
        da = get_ta(source, dataset, experiment)
        folder = f'{mV.folder_save[0]}/ta/sample_data/{source}'
        filename = f'{dataset}_ta_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(da, folder, filename) if switch['save'] else None

    if switch['cl']:
        da = get_cl(source, dataset, experiment)
        folder = f'{mV.folder_save[0]}/cl/sample_data/{source}'
        filename = f'{dataset}_cl_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(da, folder, filename) if switch['save'] else None

    if switch['p_hybridsigma']:
        da = get_p_hybridsigma(source, dataset, experiment)
        folder = f'{mV.folder_save[0]}/cl/sample_data/{source}'
        filename = f'{dataset}_p_hybridsigma_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(da, folder, filename) if switch['save'] else None

    if switch['hur']:
        da = get_hur(source, dataset, experiment)
        folder = f'{mV.folder_save[0]}/hur/sample_data/{source}'
        filename = f'{dataset}_hur_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(da, folder, filename) if switch['save'] else None

    if switch['hus']:
        da = get_hus(source, dataset, experiment)
        folder = f'{mV.folder_save[0]}/hus/sample_data/{source}'
        filename = f'{dataset}_hus_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(da, folder, filename) if switch['save'] else None

    if switch['rlut']:
        da = get_rlut(source, dataset, experiment)
        folder = f'{mV.folder_save[0]}/lw/sample_data/{source}'
        filename = f'{dataset}_rlut_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(da, folder, filename) if switch['save'] else None
    return
        


# ------------------------------
# Getting dataset and experiment
# ------------------------------

def run_experiment(switch, source, dataset):
    for experiment in mV.experiments:
        if experiment and source in ['cmip5', 'cmip6']:
            print(f'\t {experiment}') if mF.data_exist(dataset, experiment) else print(f'\t no {experiment} data')
        print( '\t observational dataset') if not experiment and source == 'obs' else None

        if mF.no_data(source, experiment, mF.data_exist(dataset, experiment)):
            continue
        get_var_data(switch, source, dataset, experiment)

@mF.timing_decorator
def run_get_data(switch):
    print(f'Getting variable from {mV.resolutions[0]} {mV.timescales[0]} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')

    for dataset in mV.datasets:
        source = mF.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        print(f'{dataset} ({source})')
        run_experiment(switch, source, dataset)


if __name__ == '__main__':
    run_get_data(switch = {
        'pr'  :          True,
        'wap' :          False,
        'tas' :          False,
        'ta' :           False,
        'cl'  :          False,
        'p_hybridsigma': False,
        'hus' :          False,
        'hur' :          False,
        'rlut':          False,

        'save':          False
        }
    )






