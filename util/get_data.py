import xarray as xr
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util')
import concat_files as cfiles
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV
import myFuncs as mF

# ------------------------
#     Load variable
# ------------------------
def get_var_data(source, dataset, experiment, var_name):
    da = None
# -------------------------------------------------------------------------------- Precipitation and organization----------------------------------------------------------------------------------------------------- #
    if var_name == 'pr':
        da = cfiles.get_cmip5_data('pr', dataset, experiment)['pr']*60*60*24 if source == 'cmip5' else da
        da = cfiles.get_cmip6_data('pr', dataset, experiment)['pr']*60*60*24 if source == 'cmip6' else da
        da = cfiles.get_gpcp()['pr']                                         if dataset == 'GPCP' else da
        da.attrs['units'] = r'mm day$^-1$'

# -------------------------------------------------------------------------------- Large-scale environmental state ----------------------------------------------------------------------------------------------------- #
    if var_name == 'tas':
        da = cfiles.get_cmip5_data('tas', dataset, experiment)['tas']-273.15  if source == 'cmip5' else da
        da = cfiles.get_cmip6_data('tas', dataset, experiment)['tas']-273.15  if source == 'cmip6' else da
        da.attrs['units'] = r'$\degree$C'

    if var_name == 'hur':
        da = cfiles.get_cmip5_data('hur', dataset, experiment)['hur']  if source == 'cmip5' else da
        da = cfiles.get_cmip6_data('hur', dataset, experiment)['hur']  if source == 'cmip6' else da
        da.attrs['units'] = '%'

    if var_name == 'wap':
        da = cfiles.get_cmip5_data('wap', dataset, experiment)['wap']*60*60*24/100 if source == 'cmip5' else da
        da = cfiles.get_cmip6_data('wap', dataset, experiment)['wap']*60*60*24/100 if source == 'cmip6' else da
        da = da * 1000 if dataset == 'IITM-ESM' else da
        da.attrs['units'] = r'hPa day$^-1$'

    if var_name == 'rlut':
        da = cfiles.get_cmip5_data('rlut', dataset, experiment)['rlut'] if source == 'cmip5' else da
        da = cfiles.get_cmip6_data('rlut', dataset, experiment)['rlut'] if source == 'cmip6' else da
        da.attrs['units'] = r'W m$^-2$'


# --------------------------------------------------------------------------------------------- Clouds ----------------------------------------------------------------------------------------------------- #
    if var_name == 'cl':
        da, _ = cfiles.get_cmip5_cl('cl', dataset, experiment) if source == 'cmip5' else [da, None]
        da, _ = cfiles.get_cmip6_cl('cl', dataset, experiment) if source == 'cmip6' else [da, None] # hybrid-sigma coords
        da = da['cl'] 
        da.attrs['units'] = '%'

    if var_name == 'p_hybridsigma':
        _, da = cfiles.get_cmip5_cl('p_hybridsigma', dataset, experiment) if source == 'cmip5' else [None, da]
        _, da = cfiles.get_cmip6_cl('cl', dataset, experiment)            if source == 'cmip6' else [None, da] # hybrid-sigma coords
        da = da['p_hybridsigma']
        da.attrs['units'] = '%'

# ---------------------------------------------------------------------------------------- Moist static energy ----------------------------------------------------------------------------------------------------- #
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



# ------------------------
#    Run / save data
# ------------------------
# ----------------------------------------------------------------------------------------- Get variable and save ----------------------------------------------------------------------------------------------------- #
def save_sample(source, dataset, experiment, ds, var_name):
    folder = f'{mV.folder_save[0]}/sample_data/{var_name}/{source}'
    filename = f'{dataset}_{var_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'
    mF.save_file(ds, folder, filename)

def run_var_data(switch, source, dataset, experiment):
    for var_name in [k for k, v in switch.items() if v] : # loop over true keys
        ds = xr.Dataset(data_vars = {var_name: get_var_data(source, dataset, experiment, var_name)})
        save_sample(source, dataset, experiment, ds, var_name) if switch['save_sample'] and ds[var_name].any() else None



# --------------------------------------------------------------------------------------------- pick dataset ----------------------------------------------------------------------------------------------------- #
def run_experiment(switch, source, dataset):
    for experiment in mV.experiments:
        if not mV.data_available(source, dataset, experiment):
            continue
        print(f'\t {experiment}') if experiment else print(f'\t observational dataset')
        run_var_data(switch, source, dataset, experiment)

def run_dataset(switch):
    for dataset in mV.datasets:
        source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        print(f'{dataset} ({source})')
        run_experiment(switch, source, dataset)

@mF.timing_decorator
def run_get_data(switch):
    print(f'Getting variable from {mV.resolutions[0]} {mV.timescales[0]} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')
    run_dataset(switch)



# ------------------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    run_get_data(switch = {
        # for precipitation and organization
        'pr'  :          False,

        # large scale state variables
        'tas' :          False,
        'hur' :          False,
        'wap' :          False,    
        'rlut':          False,

        # clouds
        'cl'  :          False,
        'p_hybridsigma': False,

        # moist static energy
        'ta' :           True,
        'hus' :          False,

        # save
        'save_sample':   True
        }
    )






