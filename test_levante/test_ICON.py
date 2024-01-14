
print('starting dask testing') # user: b382628, project/account: bb1153
# ------------------------------------------------------------------------------------ Packages --------------------------------------------------------------------------------------------------------- #
import numpy as np
import xarray as xr
import pandas as pd
import sys
import os

from distributed import(Client, progress, wait)
import dask
from dask.utils import format_bytes # check size of variable: print(format_bytes(da.nbytes))
import dask.distributed
from dask_jobqueue import SLURMCluster

import psutil
import multiprocessing
from subprocess import run, PIPE
from tempfile import (NamedTemporaryFile, TemporaryDirectory)

from getpass import getuser
from pathlib import Path
import warnings
warnings.filterwarnings(action="ignore")



# -------------------
#     Functions
# -------------------
# ------------------------------------------------------------------------------------ Regridding --------------------------------------------------------------------------------------------------------- #
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
def gen_weights(ds_in, xres, yres, path_grid):
    """ Create distance weights using cdo. """
    scratch_dir = (Path("/scratch") / getuser()[0] / getuser())  # /scratch/b/b382628
    with TemporaryDirectory(dir = scratch_dir, prefix = "weights_") as td: # tempdir adds random ID after weights_
        path_gridDes = Path(td) / "gridDescription.txt"
        path_dsIn = Path(td) / "ds_in.nc"
        path_weights = Path(td) / "weights.nc"
        with path_gridDes.open("w") as f:
            f.write(get_gridDes(xres, yres))    
        ds_in.to_netcdf(path_dsIn, mode="w")
        cmd = ("cdo", "-O", f"gendis,{path_gridDes}", f"-setgrid,{path_grid}", str(path_dsIn), str(path_weights))
        run_cmd(cmd)
        ds_weights = xr.open_dataset(path_weights).load()
        wait(ds_weights) # wait for ds_weights to be fully loaded / computed
        return ds_weights

def run_cmd(cmd, path_extra=Path(sys.exec_prefix) / "bin"):
    """Run a bash command."""
    env_extra = os.environ.copy()
    env_extra["PATH"] = str(path_extra) + ":" + env_extra["PATH"]
    status = run(cmd, check=False, stderr=PIPE, stdout=PIPE, env=env_extra)
    if status.returncode != 0:
        error = f"""{' '.join(cmd)}: {status.stderr.decode('utf-8')}"""
        raise RuntimeError(f"{error}")
    return status.stdout.decode("utf-8")

@dask.delayed
def remap(ds_in, x_res, y_res, weights, path_grid):
    """Perform a weighted remapping.
    """
    if isinstance(ds_in, xr.DataArray):# If a dataArray is given create a dataset
        ds_in = xr.Dataset(data_vars={ds_in.name: ds_in})
    scratch_dir = (Path("/scratch") / getuser()[0] / getuser())  # Define the users scratch dir
    with TemporaryDirectory(dir=scratch_dir, prefix="remap_") as td:
        path_gridDes = Path(td) / "gridDescription.txt"
        path_dsIn = Path(td) / "ds_in.nc"
        path_weights = Path(td) / "weights.nc"
        path_dsOut = Path(td) / "ds_out.nc"
        with path_gridDes.open("w") as f:
            f.write(get_gridDes(x_res, y_res))
        ds_in.to_netcdf(path_dsIn, mode="w")  # Write the file to a temorary netcdf file
        weights.to_netcdf(path_weights, mode="w")
        cmd = ("cdo", "-O", f"remap,{path_gridDes},{path_weights}", f"-setgrid,{path_grid}", str(path_dsIn), str(path_dsOut))
        run_cmd(cmd)
        return xr.open_dataset(path_dsOut).load()



# -------------------
#    Calculation
# -------------------
# ------------------------------------------------------------------------------ Setting up parallelization --------------------------------------------------------------------------------------------------------- #
total_memory = psutil.virtual_memory().total    # 501.94 GB
ncpu = multiprocessing.cpu_count()              # 256 logical cpus (each has 940 MB memory, or 1940 MB if interactive terminal)
nworker = 2                                     # typically number of physical cores (128) for heavy numerical computations (less if tasks require a lot of memory, or few intensive tasks) (performance based)
threads_per_worker = ncpu // nworker            # threads
mem_per_worker = threads_per_worker             # memory per worker (GB) (also needs a bit of overhead memory allocation, so doing 1 instead of 0.94*)
processes = False                               # Workers share memory (True means each worker mostly deal with sub-tasks separately)
print(f"memory: {format_bytes(total_memory)}, number of CPUs: {ncpu}, number of workers: {nworker}, processes: {processes}")
client = Client(n_workers = nworker, memory_limit = f"{mem_per_worker}GB", threads_per_worker = threads_per_worker, processes = processes)      # create client (128GB was used as memory initially)
open_dashboard = True # tunnel IP to local by: ssh -L 8787:localhost:8787 b382628@136.172.124.4 (on local terminal)
if open_dashboard:  
    import webbrowser
    webbrowser.open('http://localhost:8787/status') 


# --------------------------------------------------------------------------- Get data (atmosphere 2 day varriables) --------------------------------------------------------------------------------------------------------- #
print('loading variable meta data finished')
file_glob_pattern = "atm_2d_ml_2020"                                                                          
folder = Path("/work/mh0287/k203123/Dyamond++/icon-aes-dyw2/experiments/dpp0029")
filename = sorted([str(f) for f in folder.rglob(f"*{file_glob_pattern}*.nc")])[:]
ds = xr.open_mfdataset(filename[0:50], combine="by_coords", chunks="auto", engine="netcdf4", parallel=True)
da = ds['pr']   # ~ 17 GB
del ds          # ~ 500 GB (36 variables)
print('loading variable meta data finished')

# -------------------------------------------------------------------------------- Coonvert time-coordinates  --------------------------------------------------------------------------------------------------------- #
time_days = pd.to_datetime(da.time.data, format="%Y%m%d")                                                           
hours = (da.time.values % 1) * 24                                                              # picks out the comma, and multiplied the fractional day with 24 hours
time_hours = pd.to_datetime(hours, format="%H")
time_dt = pd.to_datetime(pd.to_numeric(time_days) + pd.to_numeric(time_hours - time_hours[0])) # initall hour is zero, so not really necessary to subtract
da['time'] = time_dt

da = da.resample(time="1D", skipna=True).mean().mean(dim='time')
da = client.persist(da)        # can also do .compute()
print('resampling and mean calculation started')
progress(da)
print('resampling and mean calculation finished')
da = da.compute()


# ------------------------------------------------------------------------------ Interpolate to lower resolution  --------------------------------------------------------------------------------------------------------- #
path_targetGrid = "/pool/data/ICON/grids/public/mpim/0015/icon_grid_0015_R02B09_G.nc"                                  
ds_grid = xr.open_dataset(path_targetGrid, chunks="auto", engine="netcdf4").rename({"cell": "ncells"})    # ~ 18 GB
da = da.assign_coords(clon=("ncells", ds_grid.clon.data * 180 / np.pi), clat=("ncells", ds_grid.clat.data * 180 / np.pi))
x_res, y_res = 0.1, 0.1
weights = gen_weights(da, x_res, y_res, path_targetGrid)
da_mapped = remap(da, x_res, y_res, weights, path_targetGrid)
da = client.persist(da_mapped)
print('calculating weights and interpolating started')
progress(da)
print('calculating weights and interpolating finished')
result = da.compute()


# ------------------------------------------------------------------------------------------ Save metric --------------------------------------------------------------------------------------------------------- #
scratch_dir = (Path("/scratch") / getuser()[0] / getuser())  # if it has not been defined before
out_file = Path(scratch_dir) / "OutfileName.nc"
result.to_netcdf(out_file, mode="w")
da = xr.open_dataset(out_file, chunks="auto", engine="netcdf4")
print(da.info)
print('finsihed')
























































