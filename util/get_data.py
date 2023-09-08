import xarray as xr
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util')
import myFuncs as mF # imports common operator
import concat_files as cfiles
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV # imports common variables


# -----------------
# Loading variable 
# -----------------

def get_var_data(source, dataset, experiment, var_name):
    da = None
# ------------------------------------------------------------------------------- For precipitation and organization----------------------------------------------------------------------------------------------------- #
    if var_name == 'pr':
        da = cfiles.get_cmip5_data('pr', dataset, experiment)['pr']*60*60*24 if source == 'cmip5' else da
        da = cfiles.get_cmip6_data('pr', dataset, experiment)['pr']*60*60*24 if source == 'cmip6' else da
        da = cfiles.get_gpcp()['pr']                                         if dataset == 'GPCP' else da
        da.attrs['units'] = 'mm day' + mF.get_super('-1')


# -------------------------------------------------------------------------------- Large-scale environmental state ----------------------------------------------------------------------------------------------------- #
    if var_name == 'tas':
        da = cfiles.get_cmip5_data('tas', dataset, experiment)['tas']-273.15  if source == 'cmip5' else da
        da = cfiles.get_cmip6_data('tas', dataset, experiment)['tas']-273.15  if source == 'cmip6' else da
        da.attrs['units'] = mF.get_super('o') + 'C'

    if var_name == 'hur':
        da = cfiles.get_cmip5_data('hur', dataset, experiment)['hur']  if source == 'cmip5' else da
        da = cfiles.get_cmip6_data('hur', dataset, experiment)['hur']  if source == 'cmip6' else da
        da.attrs['units'] = '%'

    if var_name == 'wap':
        da = cfiles.get_cmip5_data('wap', dataset, experiment)['wap']*60*60*24/100 if source == 'cmip5' else da
        da = cfiles.get_cmip6_data('wap', dataset, experiment)['wap']*60*60*24/100 if source == 'cmip6' else da
        da = da * 1000 if dataset == 'IITM-ESM' else da
        da.attrs['units'] = 'hPa day' + mF.get_super('-1')


# ---------------------------------------------------------------------------------------------- Radation ----------------------------------------------------------------------------------------------------- #
    if var_name == 'rlut':
        da = cfiles.get_cmip5_data('rlut', dataset, experiment)['rlut'] if source == 'cmip5' else da
        da = cfiles.get_cmip6_data('rlut', dataset, experiment)['rlut'] if source == 'cmip6' else da
        da.attrs['units'] = 'W m' + mF.get_super('-2')


# ----------------------------------------------------------------------------------------------- Clouds ----------------------------------------------------------------------------------------------------- #
    if var_name == 'cl':
        # if dataset in ['INM-CM5-0', 'EC-Earth3', 'KIOST-ESM']:
        #     return da
        # da, _ = cfiles.get_cmip5_cl('cl', dataset, experiment) if source == 'cmip5' else da
        da, _ = cfiles.get_cmip6_cl('cl', dataset, experiment) #if source == 'cmip6' else da # hybrid-sigma coords
        da = da['cl'] 
        da.attrs['units'] = '%'

    if var_name == 'p_hybridsigma':
        # if dataset in ['EC-Earth3', 'INM-CM5-0', 'KIOST-ESM']:
        #     return da
        # _, da = cfiles.get_cmip5_cl('p_hybridsigma', dataset, experiment) if source == 'cmip5' else da
        _, da = cfiles.get_cmip6_cl('cl', dataset, experiment) #if source == 'cmip6' else da # hybrid-sigma coords
        da = da['p_hybridsigma']
        da.attrs['units'] = '%'

# -------------------------------------------------------------------------------------------- Moist static energy ----------------------------------------------------------------------------------------------------- #
    if var_name == 'ta':
        da = cfiles.get_cmip5_data('ta', dataset, experiment)['ta'] if source == 'cmip5' else da
        da = cfiles.get_cmip6_data('ta', dataset, experiment)['ta'] if source == 'cmip6' else da
        da = cfiles.get_era5_monthly('t')['t']                      if dataset == 'ERA5' else da
        da.attrs['units'] = 'K'

    if var_name == 'hus':
        da = cfiles.get_cmip5_data('hus', dataset, experiment)['hus'] if source == 'cmip5' else da
        da = cfiles.get_cmip6_data('hus', dataset, experiment)['hus'] if source == 'cmip6' else da
        da = cfiles.get_era5_monthly('q')['q']                        if dataset == 'ERA5' else da
        da.attrs['units'] = ' '
    return da
        

# ----------------------------------------
# Loading variable from different datasets
# ----------------------------------------

def save_sample(source, dataset, experiment, ds, var_name):
    folder = f'{mV.folder_save[0]}/{var_name}/sample_data/{source}'
    filename = f'{dataset}_{var_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
    mF.save_file(ds, folder, filename)

def run_var_data(switch, source, dataset, experiment):
    for var_name in [k for k, v in switch.items() if v] : # loop over true keys
        ds = xr.Dataset(data_vars = {var_name: get_var_data(source, dataset, experiment, var_name)})
        save_sample(source, dataset, experiment, ds, var_name) if switch['save_sample'] and ds[var_name].any() else None

def run_experiment(switch, source, dataset):
    for experiment in mV.experiments:
        if not mF.data_available(source, dataset, experiment):
            continue
        print(f'\t {experiment}') if experiment else print(f'\t observational dataset')
        run_var_data(switch, source, dataset, experiment)

def run_dataset(switch):
    for dataset in mV.datasets:
        source = mF.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        print(f'{dataset} ({source})')
        run_experiment(switch, source, dataset)

@mF.timing_decorator
def run_get_data(switch):
    print(f'Getting variable from {mV.resolutions[0]} {mV.timescales[0]} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')
    run_dataset(switch)


if __name__ == '__main__':
    run_get_data(switch = {
        # ---------
        # Variables
        # ---------
            # for precipitation and organization
            'pr'  :          False,

            # large scale environemal state
            'tas' :          False,
            'hur' :          False,
            'wap' :          False,

            # radiation        
            'rlut':          False,

            # clouds
            'cl'  :          False,
            'p_hybridsigma': False,

            # moist static energy
            'ta' :           True,
            'hus' :          False,

        'save_sample':   False
        }
    )






