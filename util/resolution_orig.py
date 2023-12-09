''' 
# ----------------------------
#   Find original resolution
# ----------------------------
Find the original resolution of the datasets 
'''



# ------------------------------------------------------------------------------------ Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr


# --------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV



# ------------------------
#     Get resolution
# -----------------------
# ---------------------------------------------------------------------------------- Get resolution --------------------------------------------------------------------------------------------------- #
def get_orig_res(dataset, source):
    da = xr.open_dataset(f'/Users/cbla0002/Documents/data/sample_data/pr/{source}/{dataset}_pr_daily_historical_orig.nc')['pr']
    dlat, dlon = da['lat'].diff(dim='lat').data[0], da['lon'].diff(dim='lon').data[0]
    resolution = dlat * dlon
    return dlat, dlon, resolution



# ------------------------
#         Run
# -----------------------
# ---------------------------------------------------------------------------------- Get resolution --------------------------------------------------------------------------------------------------- #
if __name__=='__main__':
    for dataset in mV.datasets:
        source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        print(dataset)
        dlat, dlon, resolution = get_orig_res(dataset, source) 
        print('dlat', dlat)
        print('dlon', dlon)
        print('resolution', resolution)











