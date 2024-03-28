'''
# ------------------------
#   Conv_obj_metrics
# ------------------------
This script calculates metrics describing convective object characteristics
Functions:
    get_prop_obj_id(prop_name, da_prop, da_area, conv_obj, obj_id)

Inputs:
    prop            - dict                              {string : boolean}          options given after if __main__
    conv_obj        - integer matrix indifying objects  dim: (time, lat, lon)       get from: util-data/var_calc/conv_obj_var.py
    obj_id          - included objects                  dim: (time, obj)            get from: util-data/var_calc/conv_obj_var.py
    dim             - dimensions (object)                                           get from: util-data/dimensions_data.py
    folder_temp     - folder in scratch to save partial calc

Outputs:
    o_prop          - matrix                            dim: (time, obj)
'''


# -------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
from distributed import get_client


# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")                
sys.path.insert(0, f'{os.getcwd()}/util-core')
import choose_datasets as cD  

sys.path.insert(0, f'{os.getcwd()}/util-files')
import save_folders as sF           # base folder to save metrics to

sys.path.insert(0, f'{os.getcwd()}/util-data')
import missing_data     as mD
import dimensions_data  as dD       
import variable_calc    as vC       # for loading variable to calculate metric from
import metric_data      as metD     # for saving metric

sys.path.insert(0, f'{os.getcwd()}/util-calc')
                                    # necessary scripts are imported in get_metric()


# -------------
#   Calculate
# -------------
# -------------------------------------------------------------------------------- Get metric and metric name ----------------------------------------------------------------------------------------------------- #
def adjust_metric_name(switchM, switch, var_name, region, metric_name):
    metric_name = f'{metric_name}{region}_{var_name}_{cD.conv_percentiles[0]}thprctile'
    if switch.get('fixed_area', False):
        metric_name = f'{metric_name}_fixed_area'
    return metric_name

def get_metric(switchM, switch, distance_matrix, conv_obj, obj_id, dim, var_name, da_var, region, dataset, experiment):
    for metric_name in [k for k, v in switchM.items() if v]:
        print(f'calculating {metric_name} metric ..')
        metric = None
        if metric_name == 'o_prop':
            import conv_obj.o_prop as oP
            metric = oP.get_prop_obj_id(prop_name = var_name, da_prop = da_var, da_area = dim.aream, conv_obj = conv_obj, obj_id = obj_id)
        metric_name = adjust_metric_name(switchM, switch, var_name, region, metric_name)
        yield metric, metric_name


# ------------------------------------------------------------------------------------ Get dataset and save metric ----------------------------------------------------------------------------------------------------- #
def get_position_matrix(var_name, conv_obj, dim):
    if var_name == 'area':
        da = dim.aream
        ntimes, lat, lon = conv_obj.shape
        da_replicated = np.repeat(da[np.newaxis, :, :], ntimes, axis=0)
        da_var = xr.DataArray(da_replicated, dims=conv_obj.dims, coords=conv_obj.coords)
    if var_name == 'lat':
        da = dim.latm
        ntimes, lat, lon = conv_obj.shape
        da_replicated = np.repeat(da[np.newaxis, :, :], ntimes, axis=0)
        da_var = xr.DataArray(da_replicated, dims=conv_obj.dims, coords=conv_obj.coords)
    return da_var

def run_conv_obj_metrics(switch_var, switchM, switch, resolution = cD.resolutions[0], timescale = cD.timescales[0]):
    print(f'variable: {resolution} {timescale} data \n {[key for key, value in switch_var.items() if value]}')
    print(f'metric: {[key for key, value in switchM.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')
    distance_matrix = xr.open_dataset(f'{sF.folder_scratch}/sample_data/distance_matrix/distance_matrix_{int(360/cD.x_res)}x{int(180/cD.y_res)}.nc')['distance_matrix']
    distance_matrix.load()
    for var_name in [k for k, v in switch_var.items() if v]:
        for dataset, experiment in mD.run_dataset(var = var_name, datasets = cD.datasets, experiments = cD.experiments):
            print('loading object variables ..')
            conv_obj, region = vC.get_variable(switch_var = {'conv_obj': True}, switch = switch, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = timescale)
            obj_id, _ = vC.get_variable(switch_var = {'obj_id': True}, switch = switch, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = timescale)
            conv_obj.load()
            obj_id.load()
            dim = dD.dims_class(conv_obj)
            # print(distance_matrix)
            # print(conv_obj)
            # print(obj_id)
            # exit()
            print(f'loading {var_name} variable ..')
            if var_name in ['lat', 'lon', 'area']:
                da_var = get_position_matrix(var_name, conv_obj, dim)
            else:
                da_var, _ = vC.get_variable(switch_var = {var_name: True}, switch = switch, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = timescale)
                da_var.load()
            # print(da_var)
            # print(region)
            # exit()
            for metric, metric_name in get_metric(switchM, switch, distance_matrix, conv_obj, obj_id, dim, var_name, da_var, region, dataset = dataset, experiment = experiment):       
                # print(metric)      
                # print(metric_name)      
                # exit()
                metD.save_metric(switch, met_type = 'conv_obj', dataset = dataset, experiment = experiment, metric = metric, metric_name = metric_name)



# -------------
#     Run
# -------------
# ------------------------------------------------------------------------------------------- Choose metric ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    print('starting conv_obj_metrics')
    switch_var = {                                                                                              # Choose variable (can choose multiple)
        'lat':          True                                                                                    # position
        }
    
    switchM = {                                                                                                 # choose metric type (can choose multiple)
        'o_prop':       True                                                                                    # object property
        }

    switch = {                                                                                                  # choose data to use and mask
        'test_sample':          False,                                                                          # test calcualtion on a sample of the data
        'fixed_area':           False,                                                                          # pr_convective_threshold_type
        'from_scratch':         True,   're_process':       False,                                              # if taking data from scratch or processing data
        '700hpa':               False,  '500hpa':           False,  '250hpa':   False,  'vMean':    False,      # vertical mask (3D variables are: wap, hur, ta, zg, hus)
        'ocean_mask':           False,                                                                          # horizontal mask
        'ascent_fixed':         False,  'descent_fixed':    False,  'ascent':   False,  'descent':  False,      # horizontal mask
        'save_folder_desktop':  False,  'save_scratch':     True,   'save':     False,                          # Save
        'run_in_parallel':      False                                                                           # if script is submitted individually for all models
        }    
    
    if switch['run_in_parallel']:
        run_conv_obj_metrics(switch_var, switchM, switch, datasets = [os.environ.get('MODEL_IDENTIFIER')])
    else:
        run_conv_obj_metrics(switch_var, switchM, switch) #, datasets = [cD.datasets[0]])

    try:
        client = get_client()
        client.close()
        print('conv_obj_metrics finished')
    except:
        print('No client defined')
        print('conv_obj_metrics finished')




