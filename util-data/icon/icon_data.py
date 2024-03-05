'''
# ------------------------
#   get_data - ICON
# ------------------------
- This script should be able to handle a sample (one file) and the full dataset
- It will first check if the dataset is in scratch. If not, it will generate and process the data, save it in scratch, and then access it from scratch.
- If this script is called from a different script a client will need to have been defined

# path_targetGrid = '/pool/data/ICON/grids/public/mpim/0015/icon_grid_0015_R02B09_G.nc' # for cycle 1
'''



# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import xarray as xr
from pathlib import Path
import numpy as np
import multiprocessing
from dask.utils import format_bytes # check size of variable: print(format_bytes(da.nbytes))
from distributed import(Client, progress, wait)
import psutil
import glob
import functools
import time


# ------------------------------------------------------------------------------------ imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-core')
import choose_datasets as cD

sys.path.insert(0, f'{os.getcwd()}/util-data')
import get_data.icon.icon_regrid as rGi
import get_data.icon.xesmf_regrid as rGh
import organize_files.save_folders as sF
import get_data.missing_data as mD



# ------------------------
#     dask related
# ------------------------
# ----------------------------------------------------------------------------------- Set cpu distribution --------------------------------------------------------------------------------------------------- #
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



# ------------------------
#       General
# ------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def find_source(dataset):
    '''Determining source of dataset '''
    source = 'dyamond'  if np.isin(cD.models_dyamond, dataset).any()   else None        
    source = 'nextgems' if np.isin(cD.models_nextgems, dataset).any()  else source   
    return source

def pick_ocean_region(da):
    mask = xr.open_dataset(f'{sF.folder_save}/util-data/get_data/cmip/ocean_mask.nc')['ocean_mask']
    da = da * mask
    return da

def get_target_grid():
    path_targetGrid = '/pool/data/ICON/grids/public/mpim/0033/icon_grid_0033_R02B08_G.nc'
    ds_grid = xr.open_dataset(path_targetGrid, chunks="auto", engine="netcdf4").rename({"cell": "ncells"})    # ~ 18 GB
    return ds_grid, path_targetGrid 

def retry(exception_to_check, tries=4, delay=3, backoff=2, logger=None):
    """
    A retry decorator that allows a function to retry if a specific exception is caught.

    Parameters:
    - exception_to_check: Exception to check for retry.
    - tries: Maximum number of attempts.
    - delay: Initial delay between retries in seconds.
    - backoff: Multiplier applied to delay each retry.
    - logger: Logging.Logger instance for logging retries.

    Usage:
    @retry(ValueError, tries=5, delay=2, backoff=3)
    # @retry(ValueError, tries=5, delay=1, backoff=2)
    def some_function_that_might_fail():
        ...
    """
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except exception_to_check as e:
                    if logger:
                        logger.warning(f"Retry {tries-mtries+1}, waiting {mdelay} seconds: {e}")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return func(*args, **kwargs)  # Last attempt
        return wrapper_retry
    return decorator_retry



# ------------------------
#   Resample and remap
# ------------------------
def load_data(path_files, year=2020, month=3, pattern_file='ngc2013_atm_2d_3h_mean'):
    # [print(file) for file in path_files]
    # print(type(month))
    # print('executes')
    # exit()
    paths_month = [path for path in path_files if f"{pattern_file}_{year}{month:02d}" in path]
    if not paths_month:
        raise ValueError(f"No files found for {pattern_file} in year {year}, month {month:02d}")
    ds = xr.open_mfdataset(paths_month, combine = "by_coords", chunks = {'time': 'auto'}, engine = "netcdf4", parallel = True)
    return ds

def regrid_hor(ds, da): 
    ''' Regrid to common horizontal grid '''
    regridder = rGh.regrid_conserv_xesmf(ds) # define regridder based of grid from other model (FGOALS-g2 from cmip5 currently)
    da = regridder(da)
    return da

def remap_data(var_name, x_res, y_res, pattern_file, client, path_dataRaw, ds_grid, path_targetGrid, calc_weights, year, month, path_month):
    ds = load_data(path_dataRaw, year, month, pattern_file)
    da = ds[var_name]
    da = da.resample(time="1D", skipna=True).mean()
    da = persist_process(client, task = da, task_des = 'resample', persistIt = True, computeIt = True, progressIt = True)
    da = da.assign_coords(clon=("ncells", ds_grid.clon.data * 180 / np.pi), clat=("ncells", ds_grid.clat.data * 180 / np.pi))
    if calc_weights:
        path_gridDes, path_weights = rGi.gen_weights(ds_in = da.isel(time = 0).compute(), x_res = 0.1, y_res = 0.1, path_targetGrid = path_targetGrid, path_scratch = sF.folder_scratch)
    print('remapping started')
    ds_month = rGi.remap(ds_in = da, path_gridDes = path_gridDes, path_weights = path_weights, path_targetGrid = path_targetGrid, path_scratch = sF.folder_scratch)
    print('remapping finished')
    print('regridding started')
    da_month = regrid_hor(ds_month, ds_month[var_name])
    print('regridding finished')
    ds_month = xr.Dataset({var_name: da_month.sel(lat = slice(-30, 30))*24*60*60})
    ds_month = ds_month.load()
    ds_month.to_netcdf(path_month)

def process_months(var_name, x_res, y_res, client, pattern_file, path_dataRaw, ds_grid, path_targetGrid, calc_weights = True, year = '', path_year = ''):
    path_months = []
    month_list = np.arange(1, 13)
    if year == 2050:
        month_list = np.arange(1, 3)
    for month in month_list:
        path_month = os.path.join(f'{sF.folder_scratch}', f'data_month_{month}.nc')
        if os.path.exists(path_month):
            path_months.append(path_month)
            print(f"File for month {month} already exists. Skipping...")
        else:
            print(f"Processing month: {month}")
            remap_data(var_name, x_res, y_res, pattern_file, client, path_dataRaw, ds_grid, path_targetGrid, calc_weights, year, month, path_month)
            path_months.append(path_month)
    ds = xr.open_mfdataset(path_months, combine="by_coords", chunks = {'time': 'auto'}, engine="netcdf4", parallel=True)
    ds = ds.load()
    # ds.coords['lon'] = ds.coords['lon'] + 180
    ds.to_netcdf(path_year, mode="w")
    for path_month in path_months:
        os.remove(path_month)


# --------------------------------------------------------------------------------------- process data --------------------------------------------------------------------------------------------------- #
def process_data(var_name, model, experiment, resolution, timescale, folder, x_res, y_res, years_range):
    simulation_id = model.split('_')[1]    # 'ngc2013'
    realm = 'atm'
    frequency = '3hour'
    pattern_file = 'ngc2013_atm_2d_3h_mean' # could potentially do this with splitting the name of the file, and looking for dates in last split (instead of pattern search)
    print(f'Processing {frequency} {realm} {var_name} data from {model}')
    print(f'Remapping to daily {x_res}x{y_res} deg data')
    client = create_client(ncpus = 'all', nworkers = 2, switch = {'dashboard': False})
    import intake_tools as iT
    path_dataRaw = iT.get_files_intake(simulation_id, frequency, var_name, realm)
    ds_grid, path_targetGrid = get_target_grid()
    show = False
    if show:
        print('example of file structure:')
        [print(file) for file in path_dataRaw[0:2]]
    if not os.path.exists(folder):
        Path(folder).mkdir(parents=True, exist_ok=True)
    path_years = []
    calc_weights = True
    for year in np.arange(int(years_range[0].split('-')[0]), int(years_range[0].split('-')[1])+1):
        filename = f'{model}_{var_name}_{timescale}_{experiment}_{year}_{resolution}'
        if resolution == 'regridded':
            filename = f'{filename}_{int(360/x_res)}x{int(180/y_res)}'
        path_year = f'{folder}/{filename}.nc'
        if os.path.exists(path_year):
            path_years.append(path_year)
            print(f"File for year {year} already exists. Skipping...")
        else:
            print(f'processing year {year}')
            path_year = process_months(var_name, x_res, y_res, client, pattern_file, path_dataRaw, ds_grid, path_targetGrid, calc_weights, year, path_year)
            path_years.append(path_year)
    filename = f'{model}_{var_name}_{timescale}_{experiment}_*_{resolution}'
    if resolution == 'regridded':
        filename = f'{filename}_{int(360/x_res)}x{int(180/y_res)}'
    path_pattern = f'{folder}/{filename}.nc'
    paths = glob.glob(path_pattern)
    ds = xr.open_mfdataset(paths, combine="by_coords", chunks="auto", engine="netcdf4", parallel=True)                                 # open all years
    # print(ds)
    return ds


# ----------------------------------------------------------------------------- Check if variable is in scratch --------------------------------------------------------------------------------------------------- #
def request_process(var_name, model, experiment, resolution, timescale, source, folder, x_res, y_res, years_range):
    print(f'no {model} {experiment} {resolution} {timescale} {var_name} data at {x_res}x{y_res} deg in \n {folder}')
    # response = input(f"Do you want to process {var_name} from {model}? (y/n) (check folder first): ").lower()
    response = 'y'
    if response == 'y':
        da = process_data(var_name, model, experiment, resolution, timescale, folder, x_res, y_res, years_range)
        print('requested dataset is processed and saved in scratch')
    if response == 'n':
        print('exiting')
        exit()
    return da

def check_scratch(var_name, model, experiment, resolution, timescale, source, years_range, x_res, y_res):
    folder = f'{sF.folder_scratch}/sample_data/{var_name}/{source}/{model}'
    if not os.path.exists(folder):
        return folder, False
    filename = f'{model}_{var_name}_{timescale}_{experiment}_*_{resolution}'
    if resolution == 'regridded':
        filename = f'{filename}_{int(360/x_res)}x{int(180/y_res)}'
    path_pattern = f'{folder}/{filename}.nc'
    paths = glob.glob(path_pattern)
    paths.sort()

    start_year, end_year = [int(y) for y in years_range[0].split('-')]
    all_years_covered = True
    for year in np.arange(start_year, end_year + 1):
        year_str = str(year)
        year_in_paths = any(year_str in os.path.basename(path) for path in paths)
        if not year_in_paths:
            all_years_covered = False
            break
    
    if not all_years_covered:
        paths = False
    # print(paths)
    return folder, paths


def get_icon_data(switch_var = {'pr': True}, switch = {'test_sample': False, 'ocean_mask': False}, model = cD.models_nextgems[0], experiment = cD.experiments[0], 
                  resolution = cD.resolutions[0], timescale = cD.timescales[0], years_range = ['2020-2050'], x_res = cD.x_res, y_res = cD.y_res):
    print(model)
    source = find_source(model)
    var_name = next((key for key, value in switch_var.items() if value), None)
    folder, paths = check_scratch(var_name, model, experiment, resolution, timescale, source, years_range, x_res, y_res)
    if paths:
        ds = xr.open_mfdataset(paths, combine="by_coords", chunks= {'time': 'auto'}, parallel=True)
        da = ds[var_name]
        if switch.get('test_sample', False):
            da = da.isel(time = slice(0, 365))  if switch['test_sample']   else da
        if switch.get('ocean_mask', False):
            da = pick_ocean_region(da)          if switch['ocean_mask']    else da
        return da
    else:
        ds = request_process(var_name, model, experiment, resolution, timescale, source, folder, x_res, y_res, years_range)
        da = ds[var_name]
        return da



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    switch_var = {
        'pr':   True,
        }

    switch = {
        'test_sample':    False, 
        'ocean_mask':     False
        }
    
    switch_test = {
        'process':                  True,
        'delete_previous_plots':    False,
        'plot':                     False
        }

    if switch_test['process']:
        da = get_icon_data(switch_var = {'pr': True}, switch = {'test_sample': False, 'ocean_mask': False}, model = cD.models_nextgems[0], experiment = cD.experiments[0], 
                            resolution = cD.resolutions[0], timescale = cD.timescales[0], years_range = ['2020-2050'], x_res = cD.x_res, y_res = cD.y_res)
    
    if switch_test['plot']:
        sys.path.insert(0, f'{os.getcwd()}/util-plot')
        import map_plot as mP
        mP.remove_test_plots() if switch_test['delete_previous_plots'] else None


        # da = get_icon_data(switch_var = {'pr': True}, switch = {'test_sample': True, 'ocean_mask': False})
        # print(da)

        # import xarray as xr
        # ds = xr.open_dataset('/scratch/b/b382628/sample_data/pr/nextgems/ICON-ESM_ngc2013/ICON-ESM_ngc2013_pr_daily_historical_2024_regridded_144x72.nc')
        # da = ds['pr']

        model = cD.models_nextgems[0]
        experiment = cD.experiments[0]
        var_name = 'pr'

        ds = xr.Dataset()
        ds[f'{model}-{experiment}'] = da.mean(dim = 'time').compute() #isel(time=0)
        filename = f'{var_name}_{model}-{experiment}_2025.png'
        vmin = None
        vmax = None
        label = '[mm/day]'
        fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = 'Blues', variable_list = list(ds.data_vars.keys()))
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)











