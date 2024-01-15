''' 
# -------------------
#   NextGEMS process
# -------------------
This script processes NextGEMS simulations data, and saves data in scratch directory.
The original data is temporally and spatially remapped to daily data on lat-lon grid

NextGEMS overview:
Cycle 1 (Like DYAMOND but for 1 year)
ICON - 1 year 5 km
IFS -  1 year 5 km

Cycle 2:
ICON - 10 years 10 km resolution (ngc2012)
       30 years 10 km resolution (ngc2013
IFS -  1 year

Cycle 3: 
ICON - 5 years coupled, 5 km   resolution (ngc3028) use intake module (daily)
IFS  - 5 years          4.4 km resolution


grids:
horizontal grid: /pool/data/ICON/grids/public/mpim/0015/icon_grid_0015_R02B09_G.nc
vertical grid: /work/bm1235/b380952/experiments/ngc2009_OXF_10day_irad33/run_20200822T000000-20200831T235920/ngc2009_OXF_10day_irad33_atm_vgrid_ml.nc
'''



# -------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import pandas as pd
import dask
from dask.utils import format_bytes # check size of variable: print(format_bytes(da.nbytes))
# import multiprocessing

from tempfile import (NamedTemporaryFile, TemporaryDirectory)

import intake
dask.config.set(**{"array.slicing.split_large_chunks": True})
import outtake



# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
from pathlib import Path
import os
import sys
home = os.path.expanduser("~")                            
sys.path.insert(0, f'{os.getcwd()}/util_core')
import myVars as mV                                 
import myFuncs_dask as mFd      



# --------------------
#   Remap functions
# --------------------
# --------------------------------------------------------------------------------------- raw data --------------------------------------------------------------------------------------------------- #
def get_raw_variable(variable):
    print(f'get_raw_variable started for {variable}')
    catalog_file = "/work/ka1081/Catalogs/dyamond-nextgems.json"
    cat = intake.open_esm_datastore(catalog_file)
    hits = cat.search(simulation_id="ngc2013", variable_id=variable, frequency="3hour")
    dataset_dict = hits.to_dataset_dict(cdf_kwargs={"chunks": {"time": 1}})
    keys = list(dataset_dict.keys())
    ds = dataset_dict[keys[0]]
    # print('dataset size', format_bytes(ds.nbytes))
    # nb_sections = 12
    # for section in range(nb_sections):
    #     idx_start = int(len(ds.time) / nb_sections * section) + 1
    #     idx_end = int(len(ds.time) / nb_sections * (section+1))
    #     print(idx_start)
    #     print(idx_end)
    idx_start = 1
    idx_end = 250
    da = ds[variable].isel(time = slice(idx_start, idx_end))
    print(da)
    print('data array size', format_bytes(da.nbytes))
    return da


# ------------------------------------------------------------------------------------ temporal regridding --------------------------------------------------------------------------------------------------- #
def chunk_by_time(da):
    ''' Each cpu has a limit of 940 MB '''
    mem_slice = format_bytes(da.isel(time = 0)).nbytes
    print(f'memory per slice: {mem_slice}')
    mem_slice = float(mem_slice.split()[0])
    print(f'memory per slice: {mem_slice}')

    mem_cpu = 900 # 900 MB to bytes
    print(f'memory per cpu: {mem_cpu}')
    nb_slice_per_cpu = int(mem_cpu / mem_slice)
    # nb_slice_per_cpu = max(1, nb_slice_per_cpu)
    exit()
    
    print(f'Number of time slices per cpu: {nb_slice_per_cpu}')
    da = da.chunk({'time': nb_slice_per_cpu})

    print(da)
    return da


def fix_timeaxis(da, time_period):
    # time_days = pd.to_datetime(da.time.data, format="%Y%m%d")                                                           
    # hours = (da.time.values % 1) * 24                                                              # picks out the comma, and multiplied the fractional day with 24 hours
    # time_hours = pd.to_datetime(hours, format="%H")
    # time_dt = pd.to_datetime(pd.to_numeric(time_days) + pd.to_numeric(time_hours - time_hours[0])) # initall hour is zero, so not really necessary to subtract
    # da['time'] = time_dt
    if time_period == 'daily':
        da = da.resample(time="1D", skipna=True).mean()
    return da


# -------------------------------------------------------------------------------------- spatial regridding --------------------------------------------------------------------------------------------------- #
def get_gridDes(x_res, y_res, x_first = -180, y_first = -90):
    """ Create a description for a regular global grid at given x, y resolution."""
    xsize = 360 / x_res
    ysize = 180 / y_res
    xfirst = x_first + x_res / 2
    yfirst = y_first + x_res / 2
    return f"""
#
# gridID 1
#
gridtype  = lonlat
gridsize  = {int(xsize * ysize)}
xsize     = {int(xsize)}
ysize     = {int(ysize)}
xname     = lon
xlongname = "longitude"
xunits    = "degrees_east"
yname     = lat
ylongname = "latitude"
yunits    = "degrees_north"
xfirst    = {xfirst}
xinc      = {x_res}
yfirst    = {yfirst}
yinc      = {y_res}
    """
    
@dask.delayed
def gen_weights(ds_in, xres, yres, path_targetGrid):
    """ Create distance weights using cdo. """
    with TemporaryDirectory(dir = mV.folder_scratch, prefix = "weights_") as td: # tempdir adds random ID after weights_
        path_dsIn, path_gridDes, path_weights = Path(td) / "ds_in.nc", Path(td) / "gridDescription.txt",  Path(td) / "weights.nc"
        ds_in.to_netcdf(path_dsIn, mode="w")
        with path_gridDes.open("w") as f:
            f.write(get_gridDes(xres, yres))    
        mFd.run_cmd(("cdo", "-O", f"gendis,{path_gridDes}", f"-setgrid,{path_targetGrid}", str(path_dsIn), str(path_weights)))
        ds_weights = xr.open_dataset(path_weights).load()
        mFd.wait(ds_weights) # wait for ds_weights to be fully computed / loaded before returning
        return ds_weights

@dask.delayed
def remap(ds_in, x_res, y_res, weights, path_targetGrid):
    """Perform a weighted remapping."""
    ds_in = xr.Dataset(data_vars={ds_in.name: ds_in}) if isinstance(ds_in, xr.DataArray) else ds_in    # If a dataArray is given create a dataset
    with TemporaryDirectory(dir= mV.folder_scratch, prefix="remap_") as td:
        path_dsOut, path_dsIn, path_gridDes, path_weights = Path(td) / "ds_out.nc", Path(td) / "ds_in.nc", Path(td) / "gridDescription.txt", Path(td) / "weights.nc"
        ds_in.to_netcdf(path_dsIn, mode="w")  # Write the file to a temorary netcdf file        
        weights.to_netcdf(path_weights, mode="w")
        with path_gridDes.open("w") as f:
            f.write(get_gridDes(x_res, y_res))
        mFd.run_cmd(("cdo", "-O", f"remap,{path_gridDes},{path_weights}", f"-setgrid,{path_targetGrid}", str(path_dsIn), str(path_dsOut)))
        return xr.open_dataset(path_dsOut).load()

def hor_interp_cdo(ds, x_res, y_res):
    path_targetGrid = '/pool/data/ICON/grids/public/mpim/0033/icon_grid_0033_R02B08_G.nc'
    ds_grid = xr.open_dataset(path_targetGrid, chunks="auto", engine="netcdf4").rename({"cell": "ncells"})    # ~ 18 GB
    ds = ds.assign_coords(clon=("ncells", ds_grid.clon.data * 180 / np.pi), clat=("ncells", ds_grid.clat.data * 180 / np.pi))
    weights = gen_weights(ds, x_res, y_res, path_targetGrid)
    ds = remap(ds, x_res, y_res, weights, path_targetGrid)
    return ds



# ------------------------
#         Run
# ------------------------
def run_test(switch_var, switch):
    print(f'{os.path.basename(__file__)} started')
    print(f'Getting {mV.resolutions[0]} {mV.timescales[0]} data')
    print(f'var: {[key for key, value in switch_var.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')

    for variable in [k for k, v in switch_var.items() if v]:
        da = get_raw_variable(variable)
        client = mFd.create_client(ncpus = 'all', nworkers = 2, switch = switch)
        da = chunk_by_time(da)
        exit()

        da = fix_timeaxis(da, time_period = 'daily')
        da = mFd.persist_process(client, da = da, task = 'resample', persistIt = True, loadIt = True, progressIt = True)
        print('data array size after resampling', format_bytes(da.nbytes))


        da = hor_interp_cdo(ds = da, x_res = 2.5, y_res = 2.5)
        da = mFd.persist_process(client, da = da, task = 'remap', persistIt = True, loadIt = True, progressIt = True) # daily data

        print('data array size after interpolating', format_bytes(da.nbytes))
        path_file = Path(mV.folder_scratch) / "pr_22x144.nc"
        da.to_netcdf(path_file, mode="w")
        da = xr.open_mfdataset(path_file, combine="by_coords", chunks="auto", engine="netcdf4", parallel=True)
        print(da.info)
        client.close()
        print('finsihed')



# ----------------------------------------------------------------------------------- Choose settings --------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    switch_var = {
        'pr':   True
        }

    switch = {
        'dashboard':   False,                               # show client calculation process
        'test_sample': False,                                # pick out the first file
        }


    run_test(switch_var, switch)
























