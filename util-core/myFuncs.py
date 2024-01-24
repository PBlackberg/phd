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
import time
from pathlib import Path
import glob

from functools import wraps 
from subprocess import run, PIPE
from distributed import(Client, progress, wait)
import psutil
import multiprocessing
from dask.utils import format_bytes # check size of variable: print(format_bytes(da.nbytes))

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


# ------------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")

sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars as mV                             # list of datasets to use    

sys.path.insert(0, f'{os.getcwd()}/util-data')
import get_data.test_data   as gD_test
import get_data.cmip_data   as gD_cmip
import get_data.obs_data    as gD_obs
import get_data.icon_data   as gD_icon
# import get_data.metric_data as gD_metric


# ------------------------
#   Script operations
# ------------------------
# -------------------------------------------------------------------------- Identify source of dataset ----------------------------------------------------------------------------------------------------- #
def find_source(dataset, models_cmip5 = mV.models_cmip5, models_cmip6 = mV.models_cmip6, observations = mV.observations, models_dyamond = mV.models_dyamond, models_nextgems = mV.models_nextgems):
    '''Determining source of dataset '''
    source = 'test'     if np.isin(mV.test_fields, dataset).any()   else None     
    source = 'cmip5'    if np.isin(models_cmip5, dataset).any()     else source      
    source = 'cmip6'    if np.isin(models_cmip6, dataset).any()     else source         
    source = 'obs'      if np.isin(observations, dataset).any()     else source
    source = 'dyamond'  if np.isin(models_dyamond, dataset).any()   else source        
    source = 'nextgems' if np.isin(models_nextgems, dataset).any()  else source   
    return source

# ----------------------------------------------------------------------------- Checking available data --------------------------------------------------------------------------------------------------- #
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

# ---------------------------------------------------------------------- Loop through datasets and experiments  --------------------------------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------------- Load data --------------------------------------------------------------------------------------------------- #
def load_variable(switch_var = {'pr': True}, switch = {'constructed_fields': False},
                    dataset = 'random', experiment = mV.experiments[0], timescale = mV.timescales[0]):
    ''' Loading variable data.
        Sometimes sections of years of a dataset will be used instead of the full data ex: if dataset = GPCP_1998-2010 (for obsservations) 
        (There is a double trend in high percentile precipitation rate for the first 12 years of the data (that affects the area picked out by the time-mean percentile threshold)'''
    source = find_source(dataset)
    if source in ['test']:
        da = gD_test.get_cF_var(switch_var, dataset)    
    if source in ['cmip5', 'cmip6']:
        da = gD_cmip.get_cmip_data(switch_var, switch, dataset, experiment)   
    if source in ['obs']: 
        da = gD_obs.get_obs_data(switch_var, switch, dataset, experiment)         
    if source in ['nextgems'] and dataset == 'ICON-ESM_ngc2013': 
        da = gD_icon.get_icon_data(switch_var, switch, dataset)
    # file_pattern = "/Users/cbla0002/Desktop/pr/ICON-ESM_ngc2013/ICON-ESM_ngc2013_pr_daily_*.nc" 
    # paths = sorted(glob.glob(file_pattern))
    # da = xr.open_mfdataset(paths, combine='by_coords', parallel=True) # chunks="auto"
    return da

# def load_metric(metric_class, dataset = mV.datasets[0], experiment = mV.experiments[0], timescale = mV.timescales[0], resolution = mV.resolutions[0], folder_load = mV.folder_save[0]):
#     source = find_source(dataset)
#     da = gD_metric.get_metric(source, dataset)
#     # file_pattern = "/Users/cbla0002/Desktop/pr/ICON-ESM_ngc2013/ICON-ESM_ngc2013_pr_daily_*.nc" 
#     # paths = sorted(glob.glob(file_pattern))
#     # da = xr.open_mfdataset(paths, combine='by_coords', parallel=True)
#     return da


# --------------------------------------------------------------------------------------- Saving --------------------------------------------------------------------------------------------------- #
def save_file(data, folder=f'{home}/Documents/phd', filename='test.nc', path = ''):
    ''' Basic saving function '''
    if folder and filename:
        path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    data.to_netcdf(path, mode = 'w')

def save_metric(switch, met_type, dataset, experiment, metric, metric_name):
    ''' Saves in variable/metric specific folders '''
    filename = f'{dataset}_{metric_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'
    if mV.resolutions[0] == 'regridded':
        filename = f'{filename}_{int(360/mV.x_res)}x{int(180/mV.y_res)}'
    folder = f'/metrics/{met_type}/{metric_name}/{find_source(dataset)}'
    for save_type in [k for k, v in switch.items() if v]:
        save_file(xr.Dataset({metric_name: metric}), f'{mV.folder_save}{folder}', filename)     if save_type == 'save'                  else None      
        save_file(xr.Dataset({metric_name: metric}),f'{mV.folder_scratch}/{folder}', filename)  if save_type == 'save_scratch'          else None                                   
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


# ------------------------------------------------------------------ Gives dimension specs ----------------------------------------------------------------------------------------------------- #
class dims_class():
    R = 6371        # radius of earth
    g = 9.81        # gravitaional constant
    c_p = 1.005     # specific heat capacity
    L_v = 2.256e6   # latent heat of vaporization
    def __init__(self, da):
        self.lat, self.lon       = da['lat'].data, da['lon'].data
        self.lonm, self.latm     = np.meshgrid(self.lon, self.lat)
        self.dlat, self.dlon     = da['lat'].diff(dim='lat').data[0], da['lon'].diff(dim='lon').data[0]
        self.aream               = np.cos(np.deg2rad(self.latm))*np.float64(self.dlon*self.dlat*self.R**2*(np.pi/180)**2) # area of domain
        self.latm3d, self.lonm3d = np.expand_dims(self.latm,axis=2), np.expand_dims(self.lonm,axis=2)                     # used for broadcasting
        self.aream3d             = np.expand_dims(self.aream,axis=2)


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
    elif timeMean_option == 'daily' and len(da) > 11000:
        pass
    else:
        pass
    return da



# ------------------------
#   Terminal operations
# ------------------------
def run_cmd(cmd, path_extra=Path(sys.exec_prefix) / "bin"):
    """Run a bash command."""
    env_extra = os.environ.copy()
    env_extra["PATH"] = str(path_extra) + ":" + env_extra["PATH"]
    status = run(cmd, check=False, stderr=PIPE, stdout=PIPE, env=env_extra)
    if status.returncode != 0:
        error = f"""{' '.join(cmd)}: {status.stderr.decode('utf-8')}"""
        raise RuntimeError(f"{error}")
    return status.stdout.decode("utf-8")



# ------------------------
#     Dask related
# ------------------------
def create_client(ncpus = 'all', nworkers = 2, switch = {'dashboard': False}):
    if ncpus == 'all':
        if 'SLURM_JOB_ID' in os.environ:
            ncpus = int(os.environ.get('SLURM_CPUS_ON_NODE', multiprocessing.cpu_count()))
        else:
            ncpus = multiprocessing.cpu_count()
    total_memory = psutil.virtual_memory().total
    ncpu = multiprocessing.cpu_count() if ncpus == 'all' else ncpus
    threads_per_worker = ncpu // nworkers
    mem_per_worker = threads_per_worker     # 1 GB per worker (standard capacity is 940 MB / worker, giving some extra for overhead operations)
    processes = False                       # False: Workers share memory (True: Each worker mostly deal with sub-tasks separately)
    print(f'Dask client')
    print(f'\tSpecs: {format_bytes(total_memory)}, {ncpu} CPUs')
    print(f'\tCluster: {nworkers} workers, {threads_per_worker} cpus/worker, {mem_per_worker} GiB/worker, processes: {processes}')
    client = Client(n_workers = nworkers, memory_limit = f"{mem_per_worker}GB", threads_per_worker = threads_per_worker, processes = processes)         
    if switch['dashboard']:         # tunnel IP to local by: ssh -L 8787:localhost:8787 b382628@136.172.124.4 (on local terminal)
        import webbrowser
        webbrowser.open('http://localhost:8787/status') 
    return client

def get_MiB(da):
    ''' 
    print(f'{mFd.get_MiB(da)} MiB')
    '''
    return  round(da.nbytes / 1024**2, 2)
    
def get_GiB(da):
    ''' 
    print(f'{mFd.get_GiB(da)} GiB')
    '''
    return  round(da.nbytes / 1024**3, 2)

def persist_process(client, task, task_des, persistIt = True, computeIt = False, progressIt = True):
    if persistIt:
        task = client.persist(task)        # can also do .compute()
        print(f'{task_des} started')
        if progressIt:
            progress(task)
        print(f'{task_des} finished')
    if computeIt:    
        task = task.compute()
    return task


# ------------------------------------------------------------------------------------- test function ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    da = load_variable(switch_var = {'pr': True}, switch = {'test_sample': False}, dataset = mV.datasets[0])
    print(da)





