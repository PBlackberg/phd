
import xarray as xr
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/switch')
import myFuncs as mF


a = xr.open_dataset('/g/data/oi10/replicas/CMIP6/CMIP/BCC/BCC-CSM2-MR/historical/r1i1p1f1/Amon/cl/gn/v20181126/cl_Amon_BCC-CSM2-MR_historical_r1i1p1f1_gn_197001-198412.nc')
print(a)


# mF.save_file(a, folder='/g/data/k10/cb4968/data/sample_data/cl/cmip6', filename='BCC-CSM2-MR_cl_monthly_historical_orig_vert.nc', path = '')










