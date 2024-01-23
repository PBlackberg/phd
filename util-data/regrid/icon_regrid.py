'''
# ------------------------
#   regrid - ICON
# ------------------------
This script uses a cdo (climate data operator) to perform weighted remapping to regrid the ICON simulations
'''



# -------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
from pathlib import Path
from tempfile import (NamedTemporaryFile, TemporaryDirectory)
from subprocess import run, PIPE
import os
import sys



def run_cmd(cmd, path_extra=Path(sys.exec_prefix) / "bin"):
    ''' Run a bash command (terminal command) and capture output (this function is also available in myFuncs_dask)'''
    env_extra = os.environ.copy()
    env_extra["PATH"] = str(path_extra) + ":" + env_extra["PATH"]
    status = run(cmd, check=False, stderr=PIPE, stdout=PIPE, env=env_extra)
    if status.returncode != 0:
        error = f"""{' '.join(cmd)}: {status.stderr.decode('utf-8')}"""
        raise RuntimeError(f"{error}")
    return status.stdout.decode("utf-8")

def get_gridDes(x_res, y_res, x_first = -180, y_first = -90):
    ''' Create a description for a regular global grid at given x, y resolution '''
    xsize = 360 / x_res
    ysize = 180 / y_res
    xfirst = x_first + x_res / 2
    yfirst = y_first + x_res / 2
    return f'''
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
            '''

def gen_weights(ds_in, x_res, y_res, path_targetGrid, path_scratch):
    ''' Create remap weights using cdo (only if they are not already saved) '''
    suffix = f'{int(360/x_res)}x{int(180/y_res)}'
    folder = Path(path_scratch) / f'weights_{suffix}'
    if folder.exists():
        path_gridDes = folder / "gridDescription.txt"        
        path_weights = folder / "weights.nc"
        return path_gridDes, path_weights
    else:
        print('calculating weights')
        folder.mkdir(exist_ok=True)
        path_gridDes = folder / "gridDescription.txt"
        with path_gridDes.open("w") as f:
            f.write(get_gridDes(x_res, y_res))    
        path_weights = folder / "weights.nc"
        with NamedTemporaryFile() as temp_file:
            path_dsIn = Path(temp_file.name)
            ds_in.to_netcdf(path_dsIn, mode="w")
            run_cmd(("cdo", "-O", f"gendis,{path_gridDes}", f"-setgrid,{path_targetGrid}", str(path_dsIn), str(path_weights)))
        print('calculating weights finished')
    return path_gridDes, path_weights

# @dask.delayed
def remap(ds_in, path_gridDes, path_weights, path_targetGrid, path_scratch):
    ''' Perform a weighted remapping (takes dataset and data array) '''
    ds_in = xr.Dataset(data_vars = {ds_in.name: ds_in}) if isinstance(ds_in, xr.DataArray) else ds_in    # If a dataArray is given create a dataset
    with TemporaryDirectory(dir= path_scratch, prefix="remap_") as temp_dir:
        path_dsOut, path_dsIn = Path(temp_dir) / "ds_out.nc", Path(temp_dir) / "ds_in.nc"
        ds_in.to_netcdf(path_dsIn, mode="w")                                                             # Write the file to a temorary netcdf file        
        run_cmd(("cdo", "-O", f"remap,{path_gridDes},{path_weights}", f"-setgrid,{path_targetGrid}", str(path_dsIn), str(path_dsOut)))
        return xr.open_dataset(path_dsOut, chunks={'time': 'auto'}).load()






