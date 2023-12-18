
import xarray as xr
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
home = os.path.expanduser("~")                           
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV                                 
import myFuncs as mF     
import myClasses as mC

# a = xr.open_dataset('/Users/cbla0002/Documents/data/metrics/ws/ws_hc_snapshot/obs/ISCCP_ws_hc_snapshot_daily__regridded.nc')
# print(a)

# mF.plot_scene(a['ws_hc_snapshot'])
# plt.show()



def a_func(a = 5):
    if a == 6:
        return True

if a_func(a = 6):
    print('executes')



