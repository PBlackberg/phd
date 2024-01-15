'''
# ------------------------
#       myFuncs
# ------------------------
This script has functions that are communly used in other scripts, including

Operations -    (ex: looping through datasets/experiments, load/save data, time functions)
Calculation -   (ex: connect lon boundary, calculate spherical distance)
Plots -         (ex: plot figure / subplots, move column and rows)
'''


# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from functools import wraps 
import cftime
import datetime
import time

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


# ------------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")

sys.path.insert(0, f'{os.getcwd()}/util-data')
import constructed_fields as cF                 # manually created scripts for testing
import get_data as gD                           # for loading data from supercomp   

sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars as mV                             # list of datasets to use    
import myClasses as mC                          # list of metrics to choose from



# ------------------------
#       Operations
# ------------------------
# -------------------------------------------------------------------------- Identify source of dataset ----------------------------------------------------------------------------------------------------- #
def find_source(dataset, models_cmip5 = mV.models_cmip5, models_cmip6 = mV.models_cmip6, observations = mV.observations):
    '''Determining source of dataset '''
    source = 'cmip5' if np.isin(models_cmip5, dataset).any() else None      
    source = 'cmip6' if np.isin(models_cmip6, dataset).any() else source         
    source = 'test'  if np.isin(mV.test_fields, dataset).any()        else source     
    source = 'obs'   if np.isin(observations, dataset).any() else source
    return source

def find_list_source(datasets, models_cmip5 = mV.models_cmip5, models_cmip6 = mV.models_cmip6, observations = mV.observations):
    ''' Determining source of dataset list (for plots) '''
    sources = set()
    for dataset in datasets:
        sources.add('cmip5') if dataset in models_cmip5 else None
        sources.add('cmip6') if dataset in models_cmip6 else None
        sources.add('obs')   if dataset in observations else None
    list_source = 'cmip5' if 'cmip5' in sources else 'test'
    list_source = 'cmip6' if 'cmip6' in sources else list_source
    list_source = 'obs'   if 'obs'   in sources else list_source
    list_source = 'mixed' if 'cmip5' in sources and 'cmip6' in sources else list_source
    return list_source

def find_ifWithObs(datasets, observations= mV.observations):
    ''' Indicate if there is observations in the dataset list (for plots) '''
    for dataset in datasets:
        if dataset in observations:
            return '_withObs'
    return ''


# --------------------------------------------------------------------------------------- Loading --------------------------------------------------------------------------------------------------- #
def data_available(var = '', dataset = mV.datasets[0], experiment = mV.experiments[0]):
    ''' Check if dataset has variable. Returning False skips the loop in calc '''
    source = find_source(dataset)
    if [source, experiment] == ['cmip5', 'ssp585'] or [source, experiment] == ['cmip6', 'rcp85']: # only run fitting scenario for cmip version
        return  False
    if not experiment and not source in ['obs', 'test']:                                          # only run obs or test data for experiment == ''
        return False
    if experiment and source in ['obs']:                                                          # only run models when experiment ~= '' 
        return False
    if var in ['lcf', 'hcf', 'cl', 
               'ds_cl', 'cl_p_hybrid', 'p_hybrid'] \
        and dataset in ['INM-CM5-0', 'KIOST-ESM', 'EC-Earth3', 'INM-CM4-8', 
                        'CNRM-CM6-1-HR', 'GFDL-ESM4']:                                            # Some models do not have cloud variable
        print(f'No {var} data for this dataset')
        return False                                                                   
    return True

def load_variable(switch = {'constructed_fields': False, 'sample_data': True, 'gadi_data': False}, var = 'pr', 
                    dataset = 'random', experiment = mV.experiments[0], timescale = mV.timescales[0]):
    ''' Loading variable data.

        Sometimes sections of years of a dataset will be used instead of the full data ex: if dataset = GPCP_1998-2010 (for obsservations) 
        (There is a double trend in high percentile precipitation rate for the first 12 years of the data (that affects the area picked out by the time-mean percentile threshold)'''
    source = find_source(dataset)
    dataset_alt = dataset.split('_')[0] if '_' in dataset else dataset    
    var = 'pr'  if var == 'var_2d' else var # for testing
    var = 'hur' if var == 'var_3d' else var # for testing
                                                   
    da = cF.get_cF_var(dataset_alt, var)                                                                                                                    if switch['constructed_fields']             else None
    da = xr.open_dataset(f'{mV.folder_save[0]}/sample_data/{var}/{source}/{dataset_alt}_{var}_{timescale}_{experiment}_{mV.resolutions[0]}.nc')[f'{var}']   if switch['sample_data']                    else da  
    da = gD.get_var_data(source, dataset_alt, experiment, var)                                                                                              if switch['gadi_data']                      else da
    if '_' in dataset:                                                  
        start_year, end_year = dataset.split('_')[1].split('-')
        da = da.sel(time= slice(start_year, end_year))
    return da

def load_metric(metric_class, dataset = mV.datasets[0], experiment = mV.experiments[0], timescale = mV.timescales[0], resolution = mV.resolutions[0], folder_load = mV.folder_save[0]):
    source = find_source(dataset)
    experiment = ''             if source in ['obs'] else experiment
    if source in ['obs'] and dataset not in ['GPCP', 'GPCP_1998-2009', 'GPCP_2010-2022']:
        # dataset = 'GPCP_1998-2009'  if source == 'obs' and metric_class.var in ['pr', 'org'] else dataset  # for comparing with other obs datasets
        dataset = 'GPCP_2010-2022'  if source == 'obs' and metric_class.var in ['pr', 'org'] else dataset   # for comparing with other obs datasets
        # dataset = 'GPCP'            if source == 'obs' and metric_class.var in ['pr', 'org'] else dataset  # for comparing with other obs datasets
    timescale = 'daily'         if metric_class.var in ['pr', 'org', 'hus', 'ws'] else timescale            # some metrics are only on daily
    timescale = 'monthly'         if dataset == 'NOAA'    else timescale                                      # some datasets are only on daily
    timescale = 'daily'         if dataset == 'ISCCP'   else timescale                                      # some datasets are only on daily

    path = f'{folder_load}/metrics/{metric_class.var}/{metric_class.name}/{source}/{dataset}_{metric_class.name}_{timescale}_{experiment}_{resolution}.nc'
    ds = xr.open_dataset(path)     
    da = ds[f'{metric_class.name}']
    if dataset == 'CERES':
        da['time'] = da['time'] - pd.Timedelta(days=14) # this observational dataset have monthly data with day specified as the middle of the month instead of the first
    return da

def convert_to_datetime(dates, data):
    ''' Some models have cftime datetime format which isn't very nice for plotting.

        Further, models have different calendars. When converting to datetime objects only standard calendar dates are valid.
        In some models the extra day in leap-years is missing. In some models all months have 30 days.
        Calling this function removes the extra days with invalid datetime days. '''
    if isinstance(dates[0], cftime.datetime):
        dates_new, data_new = [], []
        for i, date in enumerate(dates):
            year, month, day = date.year, date.month, date.day
            if month == 2 and day > 29:                                                     # remove 30th of feb                                                
                continue                                                                
            if not (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) and day > 28:   # remove 29th of feb in non-leap years
                continue
            dates_new.append(datetime.datetime(year, month, day))
            data_new.append(data[i])
        data = xr.DataArray(data_new, coords={'time': dates_new}, dims=['time'])
    else:
        dates_new = pd.to_datetime(dates)
    return dates_new, data

def run_experiment(var, dataset):
    for experiment in mV.experiments:
        if not data_available(var, dataset, experiment):
            continue
        print(f'\t\t {dataset} ({find_source(dataset)}) {experiment}') if experiment else print(f'\t {dataset} ({find_source(dataset)}) observational dataset')
        yield dataset, experiment
        
def run_dataset(var = ''):
    for dataset in mV.datasets:
        print(f'\t{dataset} ({find_source(dataset)})')
        yield from run_experiment(var, dataset)


# --------------------------------------------------------------------------------------- Saving --------------------------------------------------------------------------------------------------- #
def save_file(data, folder=f'{home}/Documents/phd', filename='test.nc', path = ''):
    ''' Basic saving function '''
    if folder and filename:
        path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    data.to_netcdf(path)

def save_metric(switch, var_name, dataset, experiment, metric, metric_name, folder):
    ''' Saves in variable/metric specific folders '''
    filename = f'{dataset}_{metric_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'
    folder = f'{mV.folder_save[0]}/{folder}/{var_name}/{metric_name}/{find_source(dataset)}'
    for save_type in [k for k, v in switch.items() if v]:
        save_file(xr.Dataset({metric_name: metric}), folder, filename)                          if save_type == 'save'                  else None                                        
        save_file(xr.Dataset({metric_name: metric}), f'{home}/Desktop/{metric_name}', filename) if save_type == 'save_folder_desktop'   else None


# --------------------------------------------------------------------------------------- Timing --------------------------------------------------------------------------------------------------- #
def timing_decorator(show_time = False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ''' prints start and optionally end of function call, with time taken for the function to finish '''
            print(f'{func.__name__} started')
            start_time = time.time()
            result = func(*args, **kwargs)
            if show_time:
                end_time = time.time()
                time_taken = end_time - start_time
                print(f'{func.__name__} took {time_taken/60:.2f} minutes')
            return result
        return wrapper
    return decorator



# ------------------------
#      Calculations
# ------------------------
# ----------------------------------------------------------------- For objects (contiguous convective regions) --------------------------------------------------------------------------------------------------- #
def connect_boundary(da):
    ''' Connect objects across boundary 
    Objects that touch across lon=0, lon=360 boundary are the same object.
    Takes array(lat, lon)) '''
    s = np.shape(da)
    for row in np.arange(0,s[0]):
        if da[row,0]>0 and da[row,-1]>0:
            da[da==da[row,0]] = min(da[row,0],da[row,-1])
            da[da==da[row,-1]] = min(da[row,0],da[row,-1])
    return da

def haversine_dist(lat1, lon1, lat2, lon2):
    '''Great circle distance (from Haversine formula) (used for distance between objects)
    h = sin^2(phi_1 - phi_2) + (cos(phi_1)cos(phi_2))sin^2(lambda_1 - lambda_2)
    (1) h = sin(theta/2)^2
    (2) theta = d_{great circle} / R    (central angle, theta)
    (1) in (2) and rearrange for d gives
    d = R * sin^-1(sqrt(h))*2 

    where 
    phi -latitutde
    lambda - longitude
    (Takes vectorized input) '''
    R = 6371 # radius of earth in km
    lat1 = np.deg2rad(lat1)                       
    lon1 = np.deg2rad(lon1-180) # function requires lon [-180 to 180]
    lat2 = np.deg2rad(lat2)                       
    lon2 = np.deg2rad(lon2-180)
    
    h = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2) * np.sin((lon2 - lon1)/2)**2 # Haversine formula
    return 2 * R * np.arcsin(np.sqrt(h))


# ---------------------------------------------------------------------------------- Temporal resampleing --------------------------------------------------------------------------------------------------- #
def monthly_clim(da):
    ''' Creates a data array with the climatology of each month  '''
    year = da.time.dt.year
    month = da.time.dt.month
    da = da.assign_coords(year=("time", year.data), month=("time", month.data))
    return da.set_index(time=("year", "month")).unstack("time") # reshape the array to ("month", "year")

def resample_timeMean(da, timeMean_option=''):
    ''' Resample data to specified timescale [annual, seasonal, monthly, daily] '''
    if timeMean_option == 'annual' and len(da) >= 100:
        da = da.resample(time='Y').mean(dim='time', keep_attrs=True)
    elif timeMean_option == 'seasonal' and len(da) >= 100:
        da = da.resample(time='QS-DEC').mean(dim="time")
        da = monthly_clim(da)
        da = da.rename({'month':'season'})
        da = da.assign_coords(season=["MAM", "JJA", "SON", "DJF"])
        da = da.isel(year=slice(1, None))
    elif timeMean_option == 'monthly' and len(da) > 30:
        da = da.resample(time='1MS').mean(dim='time')
    elif timeMean_option == 'daily' or not timeMean_option:
        pass
    else:
        pass
    return da







