''' 
Data on disk:
/work/bm1235/b380952/experiments/ngc2009_OXF_10day_irad33/

grids:
horizontal grid: /pool/data/ICON/grids/public/mpim/0015/icon_grid_0015_R02B09_G.nc
vertical grid: /work/bm1235/b380952/experiments/ngc2009_OXF_10day_irad33/run_20200822T000000-20200831T235920/ngc2009_OXF_10day_irad33_atm_vgrid_ml.nc

Cycle 1:
Like DYAMOND but for 1 year
ICON - 1 year 5 km  dpp0066  /work/mh0287/k203123/Dyamond++/icon-aes-dyw2/experiments/dpp0029 (prob made for dyamond but has 1 year 6h) /work/mh0287/m300083/experiments/dpp0066 (depreciated) /work/ka1081/NextGEMS/MPIM-DWD-DKRZ/ICON-SAP-5km/Cycle1/atm (only pr on 30 min)
IFS -  1 year 5 km

Cycle 2:
ICON - 10 years 10 km resolution ngc2012    /work/bm1235/k203123/experiments/ngc2012/outdata/atm (monthly only?)
       30 years 10 km resolution ngc2013    /work/bm1235/k203123/experiments/ngc2013/outdata/atm (monthly only?)    (also at rthk001)
IFS -  1 year

Cycle 3: 
ICON - 5 years coupled, 5 km   resolution (ngc3028) use intake module (daily)
IFS  - 5 years          4.4 km resolution
'''



# -------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
from pathlib import Path
from dask.utils import format_bytes # check size of variable: print(format_bytes(da.nbytes))


# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")                            
sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars as mV                                 
import myFuncs2 as mF2      



# --------------------
#      Get data
# --------------------
def get_variable():
    if mV.datasets[0] == 'ICON cycle 1 - 5km':
        pattern_file = "atm_2d_ml_2020"   
        folder = Path("/work/mh0287/k203123/Dyamond++/icon-aes-dyw2/experiments/dpp0029")
        ds = mF2.load_variable(switch, folder, pattern_file)
    if mV.datasets[0] == 'ICON cycle 2 - 10km':
        pattern_file = "atm_2d_ml_1mth_2020"   
        folder = Path('/work/bm1235/k203123/experiments/ngc2013/outdata/atm')
        ds = mF2.load_variable(switch, folder, pattern_file)
    if mV.datasets[0] == 'ICON cycle 3 - 5km':
        ds = ''
    return ds



# ------------------------
#         Run
# ------------------------
def run_test(switch):
    print(f'{os.path.basename(__file__)} started')
    client = mF2.create_client(ncpus = 'all', nworkers = 2, switch = switch)

    print(f'Testing {mV.datasets[0]} data')
    ds = get_variable()
    print('dataset size', format_bytes(ds.nbytes))
    da = ds['pr']
    print('data array size', format_bytes(da.nbytes))
    da = mF2.convert_time_coordinates(da)



    da = da.resample(time="1D", skipna=True).mean()
    print('data array size, daily', format_bytes(da.nbytes))







    da = mF2.persist_process(client, da = da, task = 'resample',persistIt = True, loadIt = True, progressIt = True)
    da = mF2.regrid_dataset(ds = da, x_res = 0.1, y_res = 0.1)
    da = mF2.persist_process(client, da = da, task = 'remap', persistIt = True, loadIt = True, progressIt = True) # daily data




    path_file = Path(mV.folder_scratch) / "test.nc"
    da.to_netcdf(path_file, mode="w")
    da = xr.open_dataset(path_file, chunks="auto", engine="netcdf4")
    print(da.info)
    print('finsihed')






# ----------------------------------------------------------------------------------- Choose settings --------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    switch = {
        'dashboard':   False,                               # show client calculation process
        'test_sample': True,                                # pick out the first file
        }

    # print(ds)
    # run_test(switch)
























