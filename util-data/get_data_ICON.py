'''
# ------------------------
#   get_data - ICON
# ------------------------
- This script should be able to handle a sample (one file) and the full dataset
- It will first check if the dataset is in scratch. If not, it will generate and process the data, save it in scratch, and then access it from scratch.
- If this script is called from a different script a client will need to have been defined

    # path_targetGrid = '/pool/data/ICON/grids/public/mpim/0015/icon_grid_0015_R02B09_G.nc' # cycle 1
'''



# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import xarray as xr
from pathlib import Path
import numpy as np


# ------------------------------------------------------------------------------------ imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-data')
import intake_tools as iT
import regrid_ICON as rI

sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars as mV
import myFuncs_dask as mFd



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
            da = mFd.persist_process(client, task = da, task_des = 'resample', persistIt = True, computeIt = True, progressIt = True)
            da = da.assign_coords(clon=("ncells", ds_grid.clon.data * 180 / np.pi), clat=("ncells", ds_grid.clat.data * 180 / np.pi))
            if calc_weights:
                path_gridDes, path_weights = rI.gen_weights(ds_in = da.isel(time = 0).compute(), x_res = x_res, y_res = y_res, path_targetGrid = path_targetGrid, path_scratch = mV.folder_scratch)
                calc_weights = False
            ds_month = rI.remap(ds_in = da, path_gridDes = path_gridDes, path_weights = path_weights, path_targetGrid = path_targetGrid, path_scratch = mV.folder_scratch)
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
# --------------------------------------------------------------------------------------- process data --------------------------------------------------------------------------------------------------- #
def process_data(model, variable_id, x_res, y_res, years_range):
    simulation_id = model.split('_')[1]    # 'ngc2013'
    realm = 'atm'
    frequency = '3hour'
    pattern_file = 'ngc2013_atm_2d_3h_mean'                                             # could potentially do this with splitting the name of the file, and looking for dates in last split (instead of pattern search)
    print(f'Processing {frequency} {realm} {variable_id} data from {mV.datasets[0]}')   # this happens if processing data is requested
    print(f'Remapping to daily {x_res}x{y_res} deg data')
    client = mFd.create_client(ncpus = 'all', nworkers = 2, switch = {'dashboard': False})
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
def get_data_ICON(switch_var = {'pr': True}, switch = {'test_sample': False}, years_range = [2020, 2050], model = mV.models_nextgems[0], x_res = mV.x_res, y_res = mV.y_res):
    variable_id = next((key for key, value in switch_var.items() if value), None)
    if switch['test_sample']:
        years_range = [2020, 2020]
    folder_dataProcessed = f'{mV.folder_scratch}/{variable_id}/{model}'
    if os.path.exists(folder_dataProcessed):
        print(f'getting {model} daily {variable_id} data at {x_res}x{y_res} deg, between {years_range[0]}:{years_range[1]}')
        path_dataProcessed = [os.path.join(folder_dataProcessed, f) for f in os.listdir(folder_dataProcessed) if os.path.isfile(os.path.join(folder_dataProcessed, f))]
        if path_dataProcessed == []:
            print(f'no {model}: daily {variable_id} data at {x_res}x{y_res} deg')
        path_dataProcessed.sort()
        ds = xr.open_mfdataset(path_dataProcessed, combine="by_coords", chunks="auto", engine="netcdf4", parallel=True) # open all processed years
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
            return ds[variable_id]
    else:
        print(f'no {model}: daily {variable_id} data')
        response = request_data(model, variable_id, years_range)
        if response == 'y':
            process_data(model, variable_id, x_res, y_res, years_range)


if __name__ == '__main__':
    switch_var = {
        'pr':   True,
        }

    switch = {
        'test_sample':  False
        }
    
    da = get_data_ICON(switch_var, switch)
    print(da)




