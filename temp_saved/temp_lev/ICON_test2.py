import dask
import multiprocessing
from dask.utils import format_bytes # check size of variable: print(format_bytes(da.nbytes))
import dask.diagnostics
progress_bar = dask.diagnostics.ProgressBar()

import intake
# dask.config.set(**{"array.slicing.split_large_chunks": True})
import outtake


import os
import sys
home = os.path.expanduser("~")                            
sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars as mV                                 
import myFuncs_dask as mFd      
sys.path.insert(0, f'{os.getcwd()}/util-data')
import regrid_ICON as rI


variable = 'pr'
catalog_file = "/work/ka1081/Catalogs/dyamond-nextgems.json"
cat = intake.open_esm_datastore(catalog_file)
hits = cat.search(simulation_id="ngc2013", variable_id=variable, frequency="3hour")
dataset_dict = hits.to_dataset_dict(cdf_kwargs={"chunks": {"time": 1}})
keys = list(dataset_dict.keys())
ds = dataset_dict[keys[0]]
da = ds[variable].sel(time='2020') #.isel(time = slice(1, 8*365))
print(da)
print(format_bytes(da.nbytes))

ncpus = multiprocessing.cpu_count()
nworkers = 2
threads_per_worker = ncpus // nworkers
client = mFd.create_client(ncpus = ncpus, nworkers = nworkers, switch = {'dashboard': False})


# da_resampled = da.resample(time="1D", skipna=True).mean()
# print(format_bytes(da_resampled.nbytes))

da = da.isel(time = 1)
da = da.load()
x_res, y_res = 2.5, 2.5
ds = rI.hor_interp_cdo(da, x_res, y_res)
ds = mFd.persist_process(client, da = ds, task = 'remap', persistIt = True, loadIt = True, progressIt = True) # daily data
print(da)







# x_res, y_res = 2.5, 2.5
# rI.hor_interp_cdo(ds, x_res, y_res)






























