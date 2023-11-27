import xarray as xr
import numpy as np
from scipy.signal import detrend
from eofs.xarray import Eof
import pandas as pd
import datetime
import cftime
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
from matplotlib import pyplot as plt



# ------------------------
#       Get data
# ------------------------
# ---------------------------------------------------------------------------------------- load data ----------------------------------------------------------------------------------------------------- #
def load_model_data(switch, source, dataset, experiment):
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
    da = da.sortby("lat")                   # rearrange data such that latitude is in ascending order
    # da = ds.sel(lat=slice(-30, 30))
    return da



# ------------------------
#    Calculate metrics
# ------------------------
def convert_to_datetime(cftime_dates):
    return [datetime.datetime(date.year, date.month, date.day) for date in cftime_dates]

def calc_oni(da):
    ''' Monthly anomalies in surface temperature for the nino3.4 region '''
    da.load()                                                                                                       # loads all data into memory which makes calculation quicker
    sst_clim = da.groupby('time.month').mean(dim='time')
    sst_anom = da.groupby('time.month') - sst_clim
    sst_anom_detrended = xr.apply_ufunc(detrend, sst_anom.fillna(0), kwargs={'axis': 0}).where(~sst_anom.isnull())    
    sst_anom_nino34 = sst_anom_detrended.sel(lat=slice(-5, 5), lon=slice(190, 240))
    sst_anom_nino34_mean = sst_anom_nino34.mean(dim=('lon', 'lat'))
    oni = sst_anom_nino34_mean.rolling(time=3, center=True).mean()

    plot = True
    if plot:
        time_values = convert_to_datetime(oni.time.values) if isinstance(oni.time.values[0], cftime.datetime) else oni.time.values
        plt.figure()
        plt.plot(time_values, oni.data)
        plt.axhline(0.5, linestyle = '--')
        plt.axhline(-0.5, linestyle = '--')
        plt.show()
    return oni

def calc_eof(da):
    ''' This uses the whole surface temperature field instead of the nino3.4 '''
    da.load()
    weights = np.cos(np.deg2rad(da.lat)).where(~sst_anom[0].isnull())
    weights /= weights.mean()
    sst_clim = da.groupby('time.month').mean(dim='time')
    sst_anom = da.groupby('time.month') - sst_clim
    sst_anom_detrended = xr.apply_ufunc(detrend, sst_anom.fillna(0), kwargs={'axis': 0}).where(~sst_anom.isnull())
    sst_anom_detrended = sst_anom_detrended.drop_vars(['lat', 'lon', 'month'])                                          # the eof package cannot handle multiple dimensions
    solver = Eof(sst_anom_detrended, weights=np.sqrt(weights))
    eof1 = solver.eofsAsCorrelation(neofs=1)
    pc1 = solver.pcs(npcs=1, pcscaling=1).sel(mode=0)
    return pc1

def compare_with_noaa_website(oni):
    ''' plots noaa website oni index for nino3.4'''
    df = pd.read_csv('/Users/cbla0002/Documents/data/sample_data/tas/obs/oni_index.csv', delim_whitespace=True)
    season_to_month = {
        'DJF': '01-01', 
        'JFM': '02-01', 
        'FMA': '03-01', 
        'MAM': '04-01',
        'AMJ': '05-01', 
        'MJJ': '06-01', 
        'JJA': '07-01', 
        'JAS': '08-01', 
        'ASO': '09-01', 
        'SON': '10-01',
        'OND': '11-01',
        'NDJ': '12-01'
        }
    df['time'] = pd.to_datetime(df['YR'].astype(str) + '-' + df['SEAS'].map(season_to_month))   # put time dimesions in the same format as the calculated metric
    ds = df.set_index('time').to_xarray()
    ds = ds.rename({'ANOM': 'sst_anom'})
    ds = ds.sel(time=slice('1998', '2022'))

    plt.figure()
    ds.sst_anom.plot(label='ONI website')
    oni.plot(label='ONI calc')
    plt.axhline(0.5, linestyle = '--')
    plt.axhline(-0.5, linestyle = '--')
    plt.legend()
    plt.show()



# ------------------------
#   Run / save metric
# ------------------------
# -------------------------------------------------------------------------------------------- Get metric and save ----------------------------------------------------------------------------------------------------- #
def get_metric(switch, source, dataset, experiment, da, metric):
    da_calc, metric_name = None, None
    if metric == 'oni':
        metric_name =f'{metric}' 
        da_calc = calc_oni(da)
        compare_with_noaa_website(da_calc) if switch['compare'] else None

    if metric == 'eof':
        metric_name =f'{metric}'
        da_calc = calc_eof(da)

    mF.save_in_structured_folders(da_calc, f'{mV.folder_save[0]}/metrics', 'tas', metric_name, source, dataset, mV.timescales[0], experiment, mV.resolutions[0])                     if switch['save']            else None
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
        da = load_model_data(switch, source, dataset, experiment) if source not in ['obs'] else load_obs_data(switch, source, dataset, experiment)
        run_metric(switch_var, switch, source, dataset, experiment, da)

def run_dataset(switch_var, switch):
    for dataset in mV.datasets:
        source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        print(f'\t{dataset} ({source})')
        run_experiment(switch_var, switch, source, dataset)

@mF.timing_decorator
def run_nino_metric(switch_var, switch):
    print(f'Running {mV.resolutions[0]} {mV.timescales[0]} data')
    print(f'metric: {[key for key, value in switch_var.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')
    run_dataset(switch_var, switch)


# --------------------------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':

    switch_var = {                                                                                   # choose variable (can choose multiple)
        'oni': True, 'eof': False,
        }

    switch = {                                                                                       # choose data to use and mask
        'constructed_fields': False, 'sample_data':       True, 'gadi_data': False,                  # data to use
        'compare':            False,                                                                  # compare calculation to saved variable
        'save_to_desktop':    False, 'save':              False                                      # save
        }

    run_nino_metric(switch_var, switch)
























