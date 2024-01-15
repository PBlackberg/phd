'''
# ------------------------
#     myFuncs - Dask
# ------------------------
General functions utilizing dask

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
sys.path.insert(0, f'{os.getcwd()}/util_core')
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


    















