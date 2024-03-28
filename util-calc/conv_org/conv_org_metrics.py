''' 
# ------------------------
#  Organization metrics
# ------------------------
Summarizes organization metric calculations
Function:
    get_conv_org_metric(conv_obj, obj_id, distance_matrix, dim, folder_temp = f'{dataset}_{experiment}') dim: (time)

Input:
    conv_obj        - integer matrix indifying objects dim: (time, lat, lon)        get from: util-data/var_calc/conv_obj_var.py
    obj_id          - included objects                 dim: (time, obj)             get from: util-data/var_calc/conv_obj_var.py
    distance_matrix - distance calculator              dim: (n_gridbox, lat, lon)   get from: util-data/var_calc/distance_matrix.py
    dim             - dimensions (object)                                           get from: util-data/dimensions_data.py
    folder_temp     - folder in scratch to save partial calc

Output:
    No output       - saves metric to file                                          filestructure in util-data/metric_data

Call from different script:
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-calc')
import conv_org.rome_calc as rC
rome = rC.get_rome(conv_obj, obj_id, distance_matrix, dim, folder_temp = f'{dataset}_{experiment}') dim: (time)
'''


# -------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr


# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")                                        
sys.path.insert(0, f'{os.getcwd()}/util-core')
import choose_datasets as cD        # settings from util-core/choose_datasets.py                                                     

sys.path.insert(0, f'{os.getcwd()}/util-files')
import save_folders as sF           # base folder to save metrics to

sys.path.insert(0, f'{os.getcwd()}/util-data')
import missing_data     as mD       # to skip datasets with missing variable
import variable_calc    as vC       # for loading variable to calculate metric
import dimensions_data  as dD       # getting dimensions and constants for dataset
import metric_data      as metD     # for saving metric

sys.path.insert(0, f'{os.getcwd()}/util-calc') 
                                    # necessary scripts are imported in get_metric()



# ------------------------
#   Run / save metrics
# ------------------------
# ----------------------------------------------------------------------------------- Get metric and metric name ----------------------------------------------------------------------------------------------------- #
def adjust_metric_name(switch_metric, switch, metric_name):
    metric_name = f'{metric_name}_{cD.conv_percentiles[0]}thprctile'
    if switch.get('fixed_area', False):
        metric_name = f'{metric_name}_fixed_area'
    return metric_name

def get_metric(switch_metric, switch, conv_obj, obj_id, distance_matrix, dim, dataset, experiment):
    ''' Calls organization metric calculation script on convective regions '''
    print('calculating metrics ..')
    for metric_name in [k for k, v in switch_metric.items() if v]:
        metric = None
        if metric_name == 'rome':
            import conv_org.rome_calc as rC
            metric = rC.get_rome(conv_obj, obj_id, distance_matrix, dim, folder_temp = f'{dataset}_{experiment}')
        metric_name = adjust_metric_name(switch_metric, switch, metric_name)
        yield metric, metric_name


# ------------------------------------------------------------------------------------ Get dataset and save metric ----------------------------------------------------------------------------------------------------- #
def run_conv_org_metrics(switch_metric, switch, datasets = cD.datasets, experiments = cD.experiments):
    print('Getting conv_org metrics')
    timescale = 'daily'
    print(f'objects based on {cD.conv_percentiles[0]}th percetile {cD.resolutions[0]} {timescale} precipitation data')
    if cD.resolutions[0] == 'regridded':
        print(f'\t at {cD.x_res}x{cD.y_res} deg resolution')
    print(f'metrics: {[key for key, value in switch_metric.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')
    distance_matrix = xr.open_dataset(f'{sF.folder_scratch}/sample_data/distance_matrix/distance_matrix_{int(360/cD.x_res)}x{int(180/cD.y_res)}.nc')['distance_matrix']
    distance_matrix.load()
    for dataset, experiment in mD.run_dataset(var = 'pr', datasets = datasets, experiments = experiments):
        print('loading variables ..')
        conv_obj, _ = vC.get_variable(switch_var = {'conv_obj': True}, switch = switch, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = timescale)
        obj_id, _ = vC.get_variable(switch_var = {'obj_id': True}, switch = switch, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = timescale)
        conv_obj.load()
        obj_id.load()
        dim = dD.dims_class(conv_obj)
        # print(distance_matrix)
        # print(conv_obj)
        # print(obj_id)
        # exit()
        for metric, metric_name in get_metric(switch_metric, switch, conv_obj, obj_id, distance_matrix, dim = dim, dataset = dataset, experiment = experiment):      
            # print(metric)    
            # print(metric_name)      
            # exit()
            metD.save_metric(switch, met_type = 'conv_org', dataset = dataset, experiment = experiment, metric = metric, metric_name = metric_name)



# -------------
#     Run
# -------------
# ---------------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    print('starting conv_org_metrics')

    switch_metric = {                                                                   # metric
        'rome':             True,                                                       # ROME
        'ni':               False,  'areafraction':     False,                          # Number of objects
        'area_mean':        False,                                                      # mean area of objects             
        'o_heatmap':        False,                                                      # rough object location
        }

    switch = {                                                                              # settings
        'test_sample':      False,                                                           # for quick run
        'fixed_area':       False,                                                          # conv_threshold type (fixed area instead of fixed precipitation rate threshold)
        'from_scratch':     True,   're_process':   False,                                  # if taking data from scratch or calculating
        'save':             False,  'save_scratch': True, 'save_folder_desktop':  False,    # save
        'run_in_parallel':  False                                                            # if script is submitted individually for all models
        }
    

    if switch['run_in_parallel']:
        run_conv_org_metrics(switch_metric, switch, datasets = [os.environ.get('MODEL_IDENTIFIER')])
    else:
        run_conv_org_metrics(switch_metric, switch) #, datasets = [cD.datasets[0]])
    

    # could add 
    # switchM = {
    # 'o_area_mask': True, 'o_area_mask_threshold': 1000,   # km^2
    # }
    #
    #
    #

    # could run scripts from their origin, and just concatenate the files and save here
    #
    #
    #
        
    # Could give years as an argument, and parallelize based on years
    # 
    #