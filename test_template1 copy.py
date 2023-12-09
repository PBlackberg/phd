'''
# ------------------------
#      test script
# ------------------------
Copy this file for easier testing of variables and metrics
myVars script determine if random, constructed, or sample from model
'''



# ------------------------------------------------------------------------------------ Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")               
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV                                 
import myClasses as mC
import myFuncs as mF     



# ----------------------
#     Get test data
# ----------------------
# ------------------------------------------------------------------------------- manually created data --------------------------------------------------------------------------------------------------- #
def get_man_np_array(switch):
    # ----------------
    # 1D data array 
    # ----------------
    # (shape is (layers, rows, columns))
    da = np.array(   
        [2, 4, np.nan, 6, 7],         # row 0
        )



    # # ----------------
    # # 2D data array 
    # # ----------------
    # # (shape is (layers, rows, columns))
    # da = np.array([
    #     [1, 2, 3],              # row 0
    #     [4, np.nan, 6],         # ...
    #     [1, 2, 3],
    #     [1, np.nan, 3],
    #     [1, 2, 3]
    #     ])



    # # ----------------
    # # 3D data array 
    # # ----------------
    # # (layers, rows, columns)
    # da = np.array([
    #     [                   # layer 0
    #     [1, 2, 3],              # row 0
    #     [4, np.nan, 6],         # ...
    #     [1, 2, 3],
    #     [1, np.nan, 3],
    #     [1, 2, 3]
    #     ],

    #     [
    #     [7, 8, 9],          # layer 1
    #     [10, np.nan, 12],       # row 0
    #     [1, 2, 3],              # ...
    #     [1, 2, 3],
    #     [1, 2, 3]
    #     ],

    #     [                   # ...
    #     [13, 14, 15], 
    #     [16, 17, 18],
    #     [1, 2, 3],
    #     [1, 2, 3],
    #     [1, 2, 3]
    #     ],

    #     [
    #     [13, 14, 15], 
    #     [16, 17, 18],
    #     [1, 2, 3],
    #     [1, 2, 3],
    #     [1, 2, 3]
    #     ]
    #     ])
    # # print(my_3d_array)
    # # print(my_3d_array[0, :, 1])



    # # ----------------
    # # 4D data array 
    # # ----------------
    # # (groups, layers, rows, columns)
    # da = np.array([
    #     [                   # group 0
    #     [                       # layer 0
    #     [1, 2, 3],                  # row 0
    #     [4, np.nan, 6],             # row 1
    #     ],

    #     [                       # layer 1
    #     [7, 8, 9],                  # row 0
    #     [10, 11, 12],               # row 1
    #     ],
    #     ],


    #     [                   # group 1
    #     [                       # layer 0
    #     [13, 14, 15],               # row 0
    #     [16, 17, 18],               # row 1
    #     ],

    #     [                       # layer 1
    #     [19, 20, 21],               # row 0
    #     [22, 23, 24],               # row 1
    #     ],
    #     ],
    #     ])
    # # print(my_4d_array)
    # # print(my_4d_array[0,1,1,:])
    
    da = xr.DataArray(da)
    return da


# ------------------------------------------------------------------------------- get test data --------------------------------------------------------------------------------------------------- #
def get_test_data(switch, switch_var):
    var = next(key for key, value in switch_var.items() if value)
    if var == 'manual':
        return get_man_np_array(switch)                                                                                     # manually created data in this script

    if switch['path']:                                                                                                      # path to data
        path = ''
        print(f'data from path: {path}')
        da = xr.open_dataset(path)

    if switch['variable']:
        print(f'{var} data from dataset: {mV.datasets[0]} ({mV.find_source(mV.datasets[0])})')
        print(f'experiment: {mV.experiments[0]}')           if mV.find_source(mV.datasets[0]) not in ['test']   else None   # only print experiment for model data
        da = mF.load_variable(switch, var, mV.datasets[0])                                                                  # default is precipitation data

    if switch['metric']:
        print(f'ROME time series')
        da = mF.load_metric()                                                                                               # default is ROME timeseries
    return da



# ----------------------
#   Choose test data
# ----------------------
# -------------------------------------------------------------------------------- choose test data --------------------------------------------------------------------------------------------------- #
switch = {                                                                        # type of data (dataset chosen in myVars)
    'variable':             True, 'metric':        False,  'path':      False,   # variables can be created, metric are saved metrics, path if for copied path
    'constructed_fields':   True, 'sample_data':  False, 'gadi_data':  False,   # from created or saved data
    }

switch_var = {                                                  # variable type
    'var_1d':   False, 'var_2d':    True,  'var_3d':   False,  # basic
    'conv':     False,                                          # like convective regions
    'manual':   False                                           # matrices that can be edited in this script
    }



# ----------------------
#      Testing
# ----------------------
# da = get_test_data(switch, switch_var)
# print(da)

# mF.plot_one_scene(a['obj_snapshot_95thprctile'])


# a = xr.open_dataset('/Users/cbla0002/Desktop/o_area_95thprctile/TaiESM1_o_area_95thprctile_daily_historical_regridded.nc')['o_area_95thprctile'].data
# # print(a)





