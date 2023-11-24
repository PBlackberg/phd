import xarray as xr
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import detrend
from eofs.xarray import Eof
import pandas as pd
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util')
import constructed_fields as cF
import get_data as gD
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV
import myClasses as mC
import myFuncs as mF



# ------------------------
#       Get data
# ------------------------
# ---------------------------------------------------------------------------------------- load data ----------------------------------------------------------------------------------------------------- #
def load_data(switch, source, dataset, experiment):
    if switch['constructed_fields']:
        da = cF.var3D
    if switch['sample_data']:
        da = xr.open_dataset(f'{mV.folder_save[0]}/sample_data/tas/{source}/{dataset}_tas_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc')['tas']
    if switch['gadi_data']:
        da = gD.get_var_data(source, dataset, experiment, 'tas', switch)
    return da


def load_obs_data(switch, source, dataset, experiment):
    ds = xr.open_dataset(f'{mV.folder_save[0]}/sample_data/tas/obs/sst.mnmean.nc')['sst']
    da = ds.sel(time=slice('1998', '2022'))
    return da



# ------------------------
#    Calculate metrics
# ------------------------
def calc_oni(da):
    da.load()
    sst_clim = da.groupby('time.month').mean(dim='time')
    sst_anom = da.groupby('time.month') - sst_clim
    sst_anom_detrended = xr.apply_ufunc(detrend, sst_anom.fillna(0), kwargs={'axis': 0}).where(~sst_anom.isnull())
    # sst_anom_nino34 = sst_anom_detrended.sel(lat=slice(5, -5), lon=slice(190, 240))
    sst_anom_nino34 = sst_anom_detrended.sel(lat=slice(-5, 5), lon=slice(190, 240))
    sst_anom_nino34_mean = sst_anom_nino34.mean(dim=('lon', 'lat'))
    oni = sst_anom_nino34_mean.rolling(time=3, center=True).mean()

    plot = False
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(-oni.data)
        plt.axhline(0.5, linestyle = '--')
        plt.axhline(-0.5, linestyle = '--')
        plt.grid()
        plt.legend()
        plt.show()
    return oni


def calc_eof(da):
    da.load()
    sst_clim = da.groupby('time.month').mean(dim='time')
    sst_anom = da.groupby('time.month') - sst_clim
    sst_anom_detrended = xr.apply_ufunc(detrend, sst_anom.fillna(0), kwargs={'axis': 0}).where(~sst_anom.isnull())
    sst_anom_detrended = sst_anom_detrended.drop_vars(['lat', 'lon', 'month'])
    weights = np.cos(np.deg2rad(da.lat)).where(~sst_anom[0].isnull())
    weights /= weights.mean()
    solver = Eof(sst_anom_detrended, weights=np.sqrt(weights))
    eof1 = solver.eofsAsCorrelation(neofs=1)
    pc1 = solver.pcs(npcs=1, pcscaling=1).sel(mode=0)
    return pc1



# ------------------------
#   Run / save metric
# ------------------------
# -------------------------------------------------------------------------------------------- Get metric and save ----------------------------------------------------------------------------------------------------- #
def get_metric(switch, source, dataset, experiment, da, metric):
    da_calc, metric_name = None, None
    if metric == 'oni':
        metric_name =f'{metric}' 
        da_calc = calc_oni(da)

    if metric == 'eof':
        metric_name =f'{metric}'
        da_calc = calc_eof(da)

    mF.save_in_structured_folders(da_calc, f'{mV.folder_save[0]}/metrics', 'tas', metric_name, source, dataset, mV.timescales[0], experiment, mV.resolutions[0])                       if switch['save']            else None
    mF.save_file(xr.Dataset(data_vars = {metric_name: da_calc}), f'{home}/Desktop/{metric_name}', f'{dataset}_{metric_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc') if switch['save_to_desktop'] else None


# -------------------------------------------------------------------------------------------- Get metric and save ----------------------------------------------------------------------------------------------------- #
def run_metric(switch_var, switch, source, dataset, experiment, da):
    for metric in [k for k, v in switch_var.items() if v]:
        get_metric(switch, source, dataset, experiment, da, metric)
            
def run_experiment(switch_var, switch, source, dataset):
    for experiment in mV.experiments:
        if not mV.data_available(source, dataset, experiment, var = '', switch = switch):   # skips invalid experiment combinations (like obs, or cmip5 model with ssp585)
            continue
        print(f'\t\t {experiment}') if experiment else print(f'\t observational dataset')
        da = load_data(switch, source, dataset, experiment) if source not in ['obs'] else load_obs_data(switch, source, dataset, experiment)
        run_metric(switch_var, switch, source, dataset, experiment, da)

def run_dataset(switch_var, switch):
    for dataset in mV.datasets:
        source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        print(f'\t{dataset} ({source})')
        run_experiment(switch_var, switch, source, dataset)

@mF.timing_decorator
def run_nino_metrics(switch_var, switch):
    print(f'Running {mV.resolutions[0]} {mV.timescales[0]} data')
    print(f'metric: {[key for key, value in switch_var.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')
    run_dataset(switch_var, switch)



# --------------------------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':

    switch_var = {                                                                                   # choose variable (can choose multiple)
        'oni':    True, 'eof':    False,
        }

    switch = {                                                                                       # choose data to use and mask
        'constructed_fields': False, 'sample_data': True, 'gadi_data': False,                        # data to use
        'save_to_desktop':    False, 'save':        True                                             # save
        }

    run_nino_metrics(switch_var, switch)
























