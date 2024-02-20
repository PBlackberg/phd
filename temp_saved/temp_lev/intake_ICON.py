''' 
# -------------------
#   ICON process
# -------------------
This script processes 30-year, 3 hourly ICON data on 10kmx10km grid to daily data on 2.5x2.5 degree grid 
ICON Cycle 2:
ICON - 30 years 10 km resolution (ngc2013)

'''



# -------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import pandas as pd
import dask
from dask.utils import format_bytes # check size of variable: print(format_bytes(da.nbytes))
import multiprocessing

from tempfile import (NamedTemporaryFile, TemporaryDirectory)


dask.config.set(**{"array.slicing.split_large_chunks": True})




# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
from pathlib import Path
import os
import sys
home = os.path.expanduser("~")                            
sys.path.insert(0, f'{os.getcwd()}/util-core')
import temp_saved.myVars_saved as mV                                 
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
    da = ds[variable].isel(time = slice(1, 8)) #.isel(time = slice(1, 8*365))
    print(da)
    print('data array size', format_bytes(da.nbytes))
    return da

def get_bytes(da):
    return  da.nbytes / 1024**2 # MiB


# ------------------------------------------------------------------------------------ temporal regridding --------------------------------------------------------------------------------------------------- #
def chunk_by_time(da, threads_per_worker):
    ''' Each cpu has a limit of 940 MB '''
    mem_slice = get_bytes(da.isel(time = 0))
    print(f'memory per slice: {mem_slice} MiB') # one slice is 20 MiB (so can do 45 time steps per cpu)

    mem_cpu = 900 # MiB
    print(f'memory per cpu: {mem_cpu} MiB')
    nb_slices_per_cpu = int(mem_cpu / mem_slice)

    if len(da.time) // nb_slices_per_cpu < threads_per_worker: # if not all cpus in a worker is assigned a chunk (for smaller arrays or when many cpus are used)
        print('Need to allocate sections smaller than max size to utilize all cpus')
        time_chunk = np.ceil(len(da.time) / threads_per_worker)
    else:
        print('Max size sections used')
        time_chunk = nb_slices_per_cpu
    print(f'time chunk is: {time_chunk} out of {len(da.time)} timesteps')
    da = da.chunk({"time": time_chunk})

    print(f"The Dask array has {da.data.npartitions} chunks.")
    print(da)    
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


# ------------------------
#         Run
# ------------------------
def run_test(switch_var, switch):
    print(f'{os.path.basename(__file__)} started')
    print(f'Getting {mV.resolutions[0]} {mV.timescales[0]} data')
    print(f'var: {[key for key, value in switch_var.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')

    path_targetGrid = '/pool/data/ICON/grids/public/mpim/0033/icon_grid_0033_R02B08_G.nc'
    ds_grid = xr.open_dataset(path_targetGrid, chunks="auto", engine="netcdf4").rename({"cell": "ncells"})    # ~ 18 GB

    for variable in [k for k, v in switch_var.items() if v]:
        
        da.isel(time = slice(1, 8)) #.isel(time = slice(1, 8*365))

        da = get_raw_variable(variable)
        print(da)
        ncpus = multiprocessing.cpu_count()
        nworkers = 2
        threads_per_worker = ncpus // nworkers
        client = mFd.create_client(ncpus = ncpus, nworkers = nworkers, switch = switch)
        # da = chunk_by_time(da, threads_per_worker)
        print(da)
        da = da.resample(time="1D", skipna=True).mean()
        da = mFd.persist_process(client, da = da, task = 'resample',persistIt = True, loadIt = True, progressIt = True)
        print(da)
        da = da.assign_coords(clon=("ncells", ds_grid.clon.data * 180 / np.pi), clat=("ncells", ds_grid.clat.data * 180 / np.pi))
        x_res, y_res = 2.5, 2.5
        weights = gen_weights(da, x_res, y_res, path_targetGrid) # only on first loop
        ds = remap(da, x_res, y_res, weights, path_targetGrid)
        ds = mFd.persist_process(client, da = ds, task = 'remap', persistIt = True, loadIt = True, progressIt = True) # daily data

        print('data array size after interpolating', format_bytes(da.nbytes))
        path_file = Path(mV.folder_scratch) / "pr_22x144.nc"
        ds.to_netcdf(path_file, mode="w")
        ds = xr.open_mfdataset(path_file, combine="by_coords", chunks="auto", engine="netcdf4", parallel=True)
        print(ds.info)


        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        da = ds['pr'].sel(lat = slice(-30, 30))*60*60*24
        print(format_bytes(da.nbytes))
        plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()  # Add coastlines
        da.plot(ax=ax, transform=ccrs.PlateCarree(), vmin=0, vmax=20)
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        plt.savefig("my_plot.png", bbox_inches='tight', dpi=300)
        plt.close()

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







        # could do a for loop here
        # for i in range(len(da.time)):        


# def hor_interp_cdo(ds, x_res, y_res):
#     path_targetGrid = '/pool/data/ICON/grids/public/mpim/0033/icon_grid_0033_R02B08_G.nc'
#     ds_grid = xr.open_dataset(path_targetGrid, chunks="auto", engine="netcdf4").rename({"cell": "ncells"})    # ~ 18 GB
#     ds = ds.assign_coords(clon=("ncells", ds_grid.clon.data * 180 / np.pi), clat=("ncells", ds_grid.clat.data * 180 / np.pi))
#     weights = gen_weights(ds, x_res, y_res, path_targetGrid)
#     ds = remap(ds, x_res, y_res, weights, path_targetGrid)
#     return ds












