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

# ------------------------------------------------------------------------------------ imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars as mV

sys.path.insert(0, f'{os.getcwd()}/util-data')
import regrid.icon_regrid as rGi


# ------------------------
#   Resample and remap
# ------------------------
# ------------------------------------------------------------------------------ load raw data and target grid file --------------------------------------------------------------------------------------------------- #
def load_month(path_files, year=2020, month=3, pattern_file='ngc2013_atm_2d_3h_mean'):
    paths_month = [path for path in path_files if f"{pattern_file}_{year}{month:02d}" in path]
    # [print(file) for file in paths_month]
    if not paths_month:
        raise ValueError(f"No files found for {pattern_file} in year {year}, month {month:02d}")
    ds = xr.open_mfdataset(paths_month, combine="by_coords", chunks="auto", engine="netcdf4", parallel=True)
    return ds

def get_target_grid():
    path_targetGrid = '/pool/data/ICON/grids/public/mpim/0033/icon_grid_0033_R02B08_G.nc'
    ds_grid = xr.open_dataset(path_targetGrid, chunks="auto", engine="netcdf4").rename({"cell": "ncells"})    # ~ 18 GB
    return ds_grid, path_targetGrid 


# -------------------------------------------------------------------------------------- Perform remapping --------------------------------------------------------------------------------------------------- #
def process_month(variable_id, x_res, y_res, pattern_file, client, path_dataRaw, ds_grid, path_targetGrid, calc_weights, year):
    path_months = []
    month_list = np.arange(1, 13)
    if year == 2050:
        month_list = np.arange(1, 3)
    for month in month_list:
        path_month = os.path.join(mV.folder_scratch, f'data_month_{month}.nc')
        if os.path.exists(path_month):
            path_months.append(path_month)
            print(f"File for month {month} already exists. Skipping...")
        else:
            print(f"Processing month: {month}")
            ds = load_month(path_dataRaw, year, month, pattern_file)
            da = ds[variable_id]
            da = da.resample(time="1D", skipna=True).mean()
            da = persist_process(client, task = da, task_des = 'resample', persistIt = True, computeIt = True, progressIt = True)
            da = da.assign_coords(clon=("ncells", ds_grid.clon.data * 180 / np.pi), clat=("ncells", ds_grid.clat.data * 180 / np.pi))
            if calc_weights:
                path_gridDes, path_weights = rI.gen_weights(ds_in = da.isel(time = 0).compute(), x_res = x_res, y_res = y_res, path_targetGrid = path_targetGrid, path_scratch = mV.folder_scratch)
                calc_weights = False
            ds_month = rGi.remap(ds_in = da, path_gridDes = path_gridDes, path_weights = path_weights, path_targetGrid = path_targetGrid, path_scratch = mV.folder_scratch)
            ds_month.to_netcdf(path_month)
            path_months.append(path_month)
    return path_months

def process_year(model, variable_id, years_range, x_res, y_res, pattern_file, client, path_dataRaw, ds_grid, path_targetGrid, calc_weights = True):
    folder_variableParent = Path(mV.folder_scratch) / variable_id
    folder_variableParent.mkdir(exist_ok=True)
    folder_variable = Path(folder_variableParent) / model
    folder_variable.mkdir(exist_ok=True)
    path_years = []
    for year in np.arange(years_range[0], years_range[1]+1):
        path_year = Path(folder_variable) / f"{model}_{variable_id}_daily_{year}_regridded_{int(360/x_res)}x{int(180/y_res)}.nc"
        if os.path.exists(path_year):
            path_years.append(path_year)
            print(f"File for year {year} already exists. Skipping...")
        else:
            print(f'processing year {year}')
            path_months = process_month(variable_id, x_res, y_res, pattern_file, client, path_dataRaw, ds_grid, path_targetGrid, calc_weights, year)
            ds = xr.open_mfdataset(path_months, combine="by_coords", chunks="auto", engine="netcdf4", parallel=True).sel(lat = slice(-30, 30))
            ds.to_netcdf(path_year, mode="w")
            path_years.append(path_year)
            for path_month in path_months:
                os.remove(path_month)
    return path_years



# ------------------------
#         Run
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

# --------------------------------------------------------------------------------------- process data --------------------------------------------------------------------------------------------------- #
def process_data(model, variable_id, x_res, y_res, years_range):
    simulation_id = model.split('_')[1]    # 'ngc2013'
    realm = 'atm'
    frequency = '3hour'
    pattern_file = 'ngc2013_atm_2d_3h_mean'                                             # could potentially do this with splitting the name of the file, and looking for dates in last split (instead of pattern search)
    print(f'Processing {frequency} {realm} {variable_id} data from {mV.datasets[0]}')   # this happens if processing data is requested
    print(f'Remapping to daily {x_res}x{y_res} deg data')
    client = create_client(ncpus = 'all', nworkers = 2, switch = {'dashboard': False})
    import intake_tools as iT
    path_dataRaw = iT.get_files_intake(simulation_id, frequency, variable_id, realm)
    ds_grid, path_targetGrid = get_target_grid()
    show = False
    if show:
        print('example of file structure:')
        [print(file) for file in path_dataRaw[0:2]]
    path_dataProcessed = process_year(model, variable_id, years_range, x_res, y_res, pattern_file, client, path_dataRaw, ds_grid, path_targetGrid)     # process and save processed data in folder
    ds = xr.open_mfdataset(path_dataProcessed, combine="by_coords", chunks="auto", engine="netcdf4", parallel=True)                     # open all years
    print(ds)

def request_data(model, variable_id, years_list):
    response = input(f"Do you want to process {variable_id} from {model} for years: {years_list[0]}:{years_list[1]}? (y/n): ").lower()
    return response


# ---------------------------------------------------------------------------- Choose dataset, time-peroiod and resolution --------------------------------------------------------------------------------------------------- #
def find_source(dataset, models_cmip5 = mV.models_cmip5, models_cmip6 = mV.models_cmip6, observations = mV.observations, models_dyamond = mV.models_dyamond, models_nextgems = mV.models_nextgems):
    '''Determining source of dataset '''
    source = 'test'     if np.isin(mV.test_fields, dataset).any()   else None     
    source = 'cmip5'    if np.isin(models_cmip5, dataset).any()     else source      
    source = 'cmip6'    if np.isin(models_cmip6, dataset).any()     else source         
    source = 'obs'      if np.isin(observations, dataset).any()     else source
    source = 'dyamond'  if np.isin(models_dyamond, dataset).any()   else source        
    source = 'nextgems' if np.isin(models_nextgems, dataset).any()  else source   
    return source

def get_icon_data(switch_var = {'pr': True}, switch = {'test_sample': False}, model = mV.datasets[0], years_range = [2020, 2050], x_res = mV.x_res, y_res = mV.y_res):
    source = find_source(model)
    variable_id = next((key for key, value in switch_var.items() if value), None)
    if switch['test_sample']:
        years_range = [2020, 2020]
    folder_dataProcessed = f'{mV.folder_scratch}/sample_data/{variable_id}/{source}/{model}'
    if os.path.exists(folder_dataProcessed):
        print(f'getting {model} daily {variable_id} data at {x_res}x{y_res} deg, between {years_range[0]}:{years_range[1]}')
        path_dataProcessed = [os.path.join(folder_dataProcessed, f) for f in os.listdir(folder_dataProcessed) if os.path.isfile(os.path.join(folder_dataProcessed, f))]
        if path_dataProcessed == []:
            print(f'no {model}: daily {variable_id} data at {x_res}x{y_res} deg in {folder_dataProcessed} (check)')
        path_dataProcessed.sort()
        if switch['test_sample']:
            ds.coords['lon'] = ds.coords['lon'] + 180
            return xr.open_mfdataset(path_dataProcessed[0], combine="by_coords", chunks="auto", engine="netcdf4", parallel=True).sel(lat = slice(-30, 30))*24*60*60 # open first processed year       
        ds = xr.open_mfdataset(path_dataProcessed, combine="by_coords", chunks="auto", engine="netcdf4", parallel=True)        # open all processed years
        years_ds = ds["time"].dt.year.values
        if years_ds[0] != years_range[0] or years_ds[-1] != years_range[1]:
            print(f'years requested: {years_range[0]}:{years_range[-1]}')
            print(f'time range of processed {variable_id} data is {years_ds[0]}:{years_ds[-1]}')
            response = request_data(model, variable_id, years_range)
            if response == 'y':
                del ds
                process_data(model, variable_id, x_res, y_res, years_range)
            if response == "n":
                print('returning existing years')
                return ds[variable_id]
            
        else:
            ds.coords['lon'] = ds.coords['lon'] + 180
            return ds[variable_id].sel(lat = slice(-30, 30))*24*60*60
    else:
        print(f'no {model}: daily {variable_id} data in {folder_dataProcessed}')
        response = request_data(model, variable_id, years_range)
        if response == 'y':
            process_data(model, variable_id, x_res, y_res, years_range)




# --------------------------------------------------------------------------------------------- Test here --------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    switch_var = {
        'pr':   True,
        }

    switch = {
        'test_sample':  False
        }
    
    da = get_icon_data(switch_var, switch)
    print(da)




