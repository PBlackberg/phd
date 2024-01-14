'''
# ------------------------
#       myFuncs2
# ------------------------
This script contains functions related to Levante operations including
setting up client for parallelization
cdo operations (Climate Data Operations)

'''



# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import numpy as np
import xarray as xr
import pandas as pd

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



# ------------------------------------------------------------------------------------ imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV                             # list of datasets to use    



# ------------------------
#          Dask
# ------------------------
# --------------------------------------------------------------------------------------- Dask client --------------------------------------------------------------------------------------------------- #
def create_client(ncpus = 'all', nworkers = 2, switch = {'dashboard': False}):
    total_memory = psutil.virtual_memory().total
    ncpu = multiprocessing.cpu_count() if ncpus == 'all' else ncpus
    threads_per_worker = ncpu // nworkers
    mem_per_worker = threads_per_worker     # 1 GB per worker (standard capacity is 940 MB / worker, giving some extra for overhead operations)
    processes = False                       # False: Workers share memory (True: Each worker mostly deal with sub-tasks separately)
    print(f'Specs: {format_bytes(total_memory)}, {ncpu} CPUs')
    print(f'Client: {nworkers} workers, {threads_per_worker} cpus/worker, processes: {processes}')
    client = Client(n_workers = nworkers, memory_limit = f"{mem_per_worker}GB", threads_per_worker = threads_per_worker, processes = processes)         
    if switch['dashboard']:         # tunnel IP to local by: ssh -L 8787:localhost:8787 b382628@136.172.124.4 (on local terminal)
        import webbrowser
        webbrowser.open('http://localhost:8787/status') 
    return client

def slurm_cluster(scratch_dir):
    dask_tmp_dir = TemporaryDirectory(dir=scratch_dir, prefix='PostProc')
    cluster = SLURMCluster(memory='500GiB',
                        cores=72,
                        project='xz0123',
                        walltime='1:00:00',
                        queue='compute',
                        name='PostProc',
                        scheduler_options={'dashboard_address': ':12435'},
                        local_directory=dask_tmp_dir.name,
                        job_extra=[f'-J PostProc',
                                    f'-D {dask_tmp_dir.name}',
                                    f'--begin=now',
                                    f'--output={dask_tmp_dir.name}/LOG_cluster.%j.o',
                                    f'--output={dask_tmp_dir.name}/LOG_cluster.%j.o'
                                    ],
                        interface='ib0')
    return cluster

def persist_process(client, da, task, persistIt = True, loadIt = False, progressIt = True):
    if persistIt:
        da = client.persist(da)        # can also do .compute()
        print(f'{task} started')
        if progressIt:
            progress(da)
        print(f'{task} finished')
    if loadIt:    
        da = da.compute()
    return da







# ------------------------
#      Operations
# ------------------------
# --------------------------------------------------------------------------------------- load data --------------------------------------------------------------------------------------------------- #
def load_variable(switch, folder, pattern_file):                            
    path = sorted([str(f) for f in folder.rglob(f"*{pattern_file}*.nc")])[:]
    # print(path[0])
    ds = xr.open_mfdataset(path[0], combine="by_coords", chunks="auto", engine="netcdf4", parallel=True)

    if not switch['test_sample']:
        ds = xr.open_mfdataset(path[0:50], combine="by_coords", chunks="auto", engine="netcdf4", parallel=True)
    return ds


# ---------------------------------------------------------------------------------- convert_time_coordinates --------------------------------------------------------------------------------------------------- #
def convert_time_coordinates(da):
    time_days = pd.to_datetime(da.time.data, format="%Y%m%d")                                                           
    hours = (da.time.values % 1) * 24                                                              # picks out the comma, and multiplied the fractional day with 24 hours
    time_hours = pd.to_datetime(hours, format="%H")
    time_dt = pd.to_datetime(pd.to_numeric(time_days) + pd.to_numeric(time_hours - time_hours[0])) # initall hour is zero, so not really necessary to subtract
    da['time'] = time_dt
    return da


# --------------------------------------------------------------------------------------- Interpolation --------------------------------------------------------------------------------------------------- #
def run_cmd(cmd, path_extra=Path(sys.exec_prefix) / "bin"):
    """Run a bash command."""
    env_extra = os.environ.copy()
    env_extra["PATH"] = str(path_extra) + ":" + env_extra["PATH"]
    status = run(cmd, check=False, stderr=PIPE, stdout=PIPE, env=env_extra)
    if status.returncode != 0:
        error = f"""{' '.join(cmd)}: {status.stderr.decode('utf-8')}"""
        raise RuntimeError(f"{error}")
    return status.stdout.decode("utf-8")

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
        run_cmd(("cdo", "-O", f"gendis,{path_gridDes}", f"-setgrid,{path_targetGrid}", str(path_dsIn), str(path_weights)))
        ds_weights = xr.open_dataset(path_weights).load()
        wait(ds_weights) # wait for ds_weights to be fully computed / loaded before returning
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
        run_cmd(("cdo", "-O", f"remap,{path_gridDes},{path_weights}", f"-setgrid,{path_targetGrid}", str(path_dsIn), str(path_dsOut)))
        return xr.open_dataset(path_dsOut).load()
    
def regrid_dataset(ds, x_res, y_res):
    path_targetGrid = "/pool/data/ICON/grids/public/mpim/0015/icon_grid_0015_R02B09_G.nc"    
    ds_grid = xr.open_dataset(path_targetGrid, chunks="auto", engine="netcdf4").rename({"cell": "ncells"})    # ~ 18 GB
    ds = ds.assign_coords(clon=("ncells", ds_grid.clon.data * 180 / np.pi), clat=("ncells", ds_grid.clat.data * 180 / np.pi))
    weights = gen_weights(ds, x_res, y_res, path_targetGrid)
    ds = remap(ds, x_res, y_res, weights, path_targetGrid)
    return ds





# ---------------------------------------------------------------------------------- Easy-gems helper functions --------------------------------------------------------------------------------------------------- #
def fix_time_axis(data):
    """Turn icon's yyyymmdd.f time axis into actual datetime format.

    This will fail for extreme values, but should be fine for a few centuries around today.
    """
    if (data.time.dtype != "datetime64[ns]") and (
        data["time"].units == "day as %Y%m%d.%f"
    ):
        data["time"] = pd.to_datetime(
            ["%8i" % x for x in data.time], format="%Y%m%d"
        ) + pd.to_timedelta([x % 1 for x in data.time.values], unit="d")


def get_from_cat(catalog, columns):
    """A helper function for inspecting an intake catalog.

    Call with the catalog to be inspected and a list of columns of interest."""
    import pandas as pd

    pd.set_option("max_colwidth", None)  # makes the tables render better

    if type(columns) == type(""):
        columns = [columns]
    return (
        catalog.df[columns]
        .drop_duplicates()
        .sort_values(columns)
        .reset_index(drop=True)
    )

def get_list_from_cat(catalog, column):
    """A helper function for getting the contents of a column in an intake catalog.

    Call with the catalog to be inspected and the column of interest."""
    return sorted(catalog.unique(column)[column]["values"])

def make_tempdir(name):
    """Creates a temporary directory in your /scratch/ and returns its path as string"""

    uid = getuser()
    temppath = os.path.join("/scratch/", uid[0], uid, name)
    os.makedirs(temppath, exist_ok=True)
    return temppath

def find_grids(dataset):
    """Generic ICON Grid locator

    This function checks an xarray dataset for attributes that contain "grid_file_uri", and checks if it can map them to a local path.
    It also checks for "grid_file_name"

    It returns a list of paths on disk that are readable (os.access(x, os.R_OK)).
    """
    uris = [
        dataset.attrs[x] for x in dataset.attrs if "grid_file_uri" in x
    ]  # this thing might come in one of various names...
    search_paths = [
        re.sub("http://icon-downloads.mpimet.mpg.de", "/pool/data/ICON", x)
        for x in uris
    ] + [
        os.path.basename(x) for x in uris
    ]  # plausible mappings on mistral.
    if "grid_file_path" in dataset.attrs:
        search_paths.append(dataset.attrs["grid_file_path"])
        search_paths.append(
            os.path.basename(dataset.attrs["grid_file_path"])
        )  # also check the current dir.
    paths = [
        x for x in search_paths if (os.access(x, os.R_OK))
    ]  # remove things that don't exist.
    if not paths:
        message = "Could not determine grid file!"
        if search_paths:
            message = message + "\nI looked in\n" + "\n".join(search_paths)
        if uris:
            message = message + (
                "\nPlease check %s for a possible grid file" % (" or ").join(uris)
            )
        raise Exception(message)
    if len(set(paths)) > 1:
        print(
            "Found multiple conflicting grid files. Using the first one.",
            file=sys.stderr,
        )
        print("Files found:", file=sys.stderr)
        print("\n".join(paths), file=sys.stderr)
    return paths

def add_grid(dataset):
    """Generic icon grid adder.

    Calls find_grids to locate a grid file, and - if it finds one - adds this grid file to a Dataset.

    also tries to ensure that clon has the same dimensions as the data variables.
    """
    paths = find_grids(dataset)
    grid = xr.open_dataset(paths[0])
    rename = (
        {}
    )  # icon uses different dimension names in the output than in the grid file. (whyever...)
    if "ncells" in dataset.dims:
        grid_ncells = grid.clon.dims[0]
        rename = {grid_ncells: "ncells"}
    drops = set(
        [x for x in grid.coords if x in dataset.data_vars or x in dataset.coords]
        + [x for x in grid.data_vars if x in dataset.data_vars or x in dataset.coords]
    )
    return xr.merge((dataset.drop(drops), grid.rename(rename)))
















