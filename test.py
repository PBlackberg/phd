
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



my_list = ['string1', 'string2', 'string3', 'string4', 'string5', 'string6', 'string7', 'string8', 'string9', 'string10', 'string11', 'string12', 'string13', 'string14', 'string15']
chunk_size = 4

for i in range(0, len(my_list), chunk_size):
    current_chunk = my_list[i:i + chunk_size]
    print(current_chunk)


