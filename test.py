
import numpy as np
import xarray as xr
import dask
import multiprocessing

import os
import sys
home = os.path.expanduser("~")                            
sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars as mV                                 
import myFuncs_dask as mFd      

def get_bytes(da):
    return  da.nbytes / 1024**2

ncpus = multiprocessing.cpu_count()
nworkers = 2
threads_per_worker = ncpus // nworkers

da = xr.open_dataset('/Users/cbla0002/Documents/data/sample_data/pr/cmip6/CanESM5_pr_daily_historical_regridded.nc')['pr']
mFd.create_client(ncpus = 'all', nworkers = 2, switch = {'dashboard': False})

mem_cpu = 900 # MiB
slice_time = da.isel(time=0)
print(f'data array size: {get_bytes(da)} MiB')
print(f'timestep size: {get_bytes(slice_time)} MiB')
print(f'mem per cpu: {mem_cpu} MiB')

nb_slices_per_cpu = int(mem_cpu // get_bytes(slice_time))
print(f'max number of time slices per cpu: {nb_slices_per_cpu}')
print(f'number of timesteps in data array: {len(da.time)}')

if len(da.time) // nb_slices_per_cpu < threads_per_worker: # if not all cpus in a worker is assigned a chunk (for smaller arrays or when many cpus are used)
    print('Need to allocate sections smaller than max size to utilize all cpus')
    time_chunk = np.ceil(len(da.time) / threads_per_worker)
else:
    print('Max sections used')
    time_chunk = nb_slices_per_cpu

print(f'time chunk is: {time_chunk} out of {len(da.time)} timesteps')

da = da.chunk({"time": time_chunk})
print(da)
print(f"The Dask array has {da.data.npartitions} chunks.")











