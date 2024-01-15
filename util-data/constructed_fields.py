''' 
# ------------------------
#   Constructed fields
# ------------------------
This script creates random fields or manually created fields (specified in myVars).
The random field is always the same for reproducibility
'''



# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import pandas as pd 
np.random.seed(0)

# ------------------------------------------------------------------------------------- Imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")               
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV                                 



# ------------------------
#      data specs
# ------------------------
# ---------------------------------------------------------------------------------------- Dimensions --------------------------------------------------------------------------------------------------- #
def get_time_range():
    time_range = pd.date_range("1970/01/01", "2000/01/01", freq='D', inclusive='left') if mV.timescales[0] == 'daily'       else None
    time_range = pd.date_range("1970/01/01", "2000/01/01", freq='MS', inclusive='left') if mV.timescales[0] == 'monthly'    else time_range
    return time_range

def get_lat_lon():
    lat, lon = [np.linspace(-30, 30, 22), np.linspace(0, 359, 128)] if mV.resolutions[0] == 'regridded'   else [None, None]
    lat, lon = [np.linspace(-30, 30, 90), np.linspace(0, 359, 360)] if mV.resolutions[0] == 'orig'        else [lat, lon]
    return lat, lon


# ------------------------------------------------------------------------------------------- data --------------------------------------------------------------------------------------------------- #
def get_conv_scenes():
    conv_scenes = np.zeros(shape = (4, 22, 128))
    time_range = pd.date_range("1970/01/01", "1970/01/05", freq='D', inclusive='left')
    conv_scenes = xr.DataArray(data = conv_scenes, dims=['time','lat','lon'], coords={'time': time_range,'lat': np.linspace(-30, 30, 22),'lon': np.linspace(0, 359, 128)})

    # 0. one object
    conv_scenes[0, :, :] = 0
    conv_scenes[0, 10:15, 60:70] = 2

    # 1. two objects (across boundary)
    conv_scenes[1, :, :] = 0
    conv_scenes[1, 10:15, 60:70] = 2

    conv_scenes[1, 5:8, 0:10] = 2
    conv_scenes[1, 5:10, 125:] = 2

    # 2. two objects (do not cross boundary, but distance between objects across boundary is closer)
    conv_scenes[2, :, :] = 0
    conv_scenes[2, 5:8, 2:10] = 2

    conv_scenes[2, 10:15, 120:-5] = 2

    # 3. multiple objects (including crossing boundary multiple times) (9 objects)
    conv_scenes[3, :, :] = 0
    conv_scenes[3, 10:15, 60:70] = 2

    conv_scenes[3, 5:8, 0:10] = 2
    conv_scenes[3, 5:10, 125:] = 2

    conv_scenes[3, 17:20, 0:3] = 2
    conv_scenes[3, 17:19, 125:] = 2

    conv_scenes[3, 16:18, 15:19] = 2

    conv_scenes[3, 3:5, 30:40] = 2

    conv_scenes[3, 10:17, 92:95] = 2

    conv_scenes[3, 6:7, 105:106] = 2

    conv_scenes[3, 2:4, 80:85] = 2

    conv_scenes[3, 18:20, 35:39] = 2
    return conv_scenes

def var_to_exampleVar(var):
    ''' Fpr if a random or constructed fields are generated for testing calculation of metric'''
    var = 'var_2d' if var in ['pr', 'tas', 'rlut']      else var
    var = 'var_3d' if var in ['hur', 'hus', 'cl', 'ta'] else var
    return var


# ------------------------
#     Create data
# ------------------------
# -------------------------------------------------------------------------------------- data + dimensions --------------------------------------------------------------------------------------------------- #
def get_cF_var(dataset, var = 'var_2d'):
    var = var_to_exampleVar(var)
    time_range = get_time_range()
    lat, lon = get_lat_lon()
    plev = np.array([100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100])

    if dataset == 'random':
        da = xr.DataArray(np.random.rand(len(time_range)),                                  dims = ['time'],                        coords = {'time': time_range})                                          if var in ['var_1d'] else None
        da = xr.DataArray(np.random.rand(len(time_range), len(lat), len(lon)),              dims = ['time', 'lat', 'lon'],          coords = {'time': time_range, 'lat': lat, 'lon': lon})                  if var in ['var_2d'] else da
        da = xr.DataArray(np.random.rand(len(time_range), len(plev), len(lat), len(lon)),   dims = ['time', 'plev', 'lat', 'lon'],  coords = {'time': time_range, 'plev': plev, 'lat': lat, 'lon': lon})    if var in ['var_3d'] else da

    elif dataset == 'constructed':
        da = get_conv_scenes()   if var == 'conv'     else None

    else:
        print('dataset needs to be random or constructed')
    return da


# ------------------------------------------------------------------------------------- Choose what to create --------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    print(f'dataset: {mV.datasets[0]} ({mV.find_source(mV.datasets[0])})')
    # var = 'var_1d'
    var = 'var_2d'
    # var = 'var_3d'
    # var = 'conv'
    # var = 'manual'


    da = get_cF_var(dataset = mV.datasets[0], var = var)
    print(da)
    # print(da.isel(time = slice(0,5)).data)


    plot = False
    if plot:
        import matplotlib.pyplot as plt
        if not var == 'manual':
            da.plot() if var in ['var_1d'] else da.isel(time=0).plot()
        else:
            da.plot()        
        plt.show()











