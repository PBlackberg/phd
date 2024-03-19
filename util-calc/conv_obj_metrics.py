'''
# ------------------------
#   Large-scale state
# ------------------------
This script calculates metrics describing convective object characteristics
'''


# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")                
sys.path.insert(0, f'{os.getcwd()}/util-core')
import choose_datasets as cD  

sys.path.insert(0, f'{os.getcwd()}/util-data')
import missing_data     as mD
import dimensions_data  as dD
import variable_calc    as vC       # for loading variable to calculate metric from
import metric_data      as metD     # for saving metric


sys.path.insert(0, f'{os.getcwd()}/util-calc')
import ls_state.in_obj          as in_obj



# -------------
#   Calculate
# -------------
# -------------------------------------------------------------------------------- Get metric and metric name ----------------------------------------------------------------------------------------------------- #
def calc_metric(switchM, var_name, da, region):
    dims = dD.dims_class(da)
    for metric_name in [k for k, v in switchM.items() if v]:
        metric = None
        metric = in_obj.get_in_obj(da, dims)                    if metric_name == 'in_obj'              else metric
        metric_name =f'{var_name}{region}_{metric_name}' 
        yield metric, metric_name


# ------------------------------------------------------------------------------------ Get dataset and save metric ----------------------------------------------------------------------------------------------------- #
def run_ls_metrics(switch_var, switchM, switch, resolution = cD.resolutions[0], timescale = cD.timescales[0]):
    print(f'variable: {resolution} {timescale} data \n {[key for key, value in switch_var.items() if value]}')
    print(f'metric: {[key for key, value in switchM.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')
    for var_name in [k for k, v in switch_var.items() if v]:
        for dataset, experiment in mD.run_dataset(var = var_name, datasets = cD.datasets, experiments = cD.experiments):
            da, region = vC.get_variable_data(switch, var_name, dataset, experiment, resolution, timescale)
            for metric, metric_name in calc_metric(switchM, var_name, da, region):      
                # print(metric_name)      
                # print(metric)      
                path = metD.save_metric(switch, var_name, dataset, experiment, metric, metric_name)
                print(f'Metric saved at: {path}')
    return metric # returns last metric (for testing one metric)



# -------------
#     Run
# -------------
# ------------------------------------------------------------------------------------------- Choose metric ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    switch_var = {                                                                                              # Choose variable (can choose multiple)
        'pr':       False,  'clwvi':        False,   'pe':          True,                                       # Precipitation
        'wap':      False,                                                                                      # Circulation
        'hur':      False,  'hus':          False,                                                              # Humidity                             
        'tas':      False,  'ta':           False,  'stability':    False,                                      # Temperature
        'rlut':     False,  'rlds':         False,  'rlus':         False,  'netlw':    False,                  # Longwave radiation
        'rsut':     False,  'rsdt':         False,  'rsds':         False,  'rsus':     False, 'netsw': False,  # Shortwave radiation
        'lcf':      False,  'hcf':          False,                                                              # Cloud fraction
        'zg':       False,                                                                                      # Geopotential height
        'hfss':     False,  'hfls':         False,                                                              # Surface fluxes
        'h':        False,  'h_anom2':      False,                                                              # Moist Static Energy
        }
    
    switchM = {                                                                                                 # choose metric type (can choose multiple)
        'eof':          False,                                                                                  # ENSO
        'in_obj':       True                                                                                    # precipitation efficiency in objects
        }

    switch = {                                                                                                  # choose data to use and mask
        'constructed_fields':   False,  'test_sample':      False,                                              # data to use (test_sample uses first file (usually first year))
        '700hpa':               False,  '500hpa':           False,  '250hpa':   False,  'vMean':    False,      # vertical mask (3D variables are: wap, hur, ta, zg, hus)
        'ocean_mask':           False,                                                                          # horizontal mask
        'ascent_fixed':         False,  'descent_fixed':    False,  'ascent':   False,  'descent':  False,      # horizontal mask
        'save_folder_desktop':  False,  'save_scratch':     True,    'save':     False                           # Save
        }
    
    metric = run_ls_metrics(switch_var, switchM, switch)                                                        # metric here shows last metric calc













































































