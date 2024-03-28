'''
# ------------------------
#   Large-scale state
# ------------------------
This script calculates metrics describing the large scale state of key variables
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
import variable_calc    as vC       # for loading variable to calculate metric
import metric_data      as metD     # for saving metric


sys.path.insert(0, f'{os.getcwd()}/util-calc')
import ls_state.means_calc      as mean_calc
import ls_state.itcz_width_wap  as itcz_wap



# -------------
#   Calculate
# -------------
# -------------------------------------------------------------------------------- Get metric and metric name ----------------------------------------------------------------------------------------------------- #
def calc_metric(switchM, var_name, da, region):
    dims = dD.dims_class(da)
    for metric_name in [k for k, v in switchM.items() if v]:
        metric = None
        metric = mean_calc.get_tMean(da)                        if metric_name == 'tMean'                                       else metric
        metric = mean_calc.get_sMean(da)                        if metric_name == 'sMean'                                       else metric
        metric = itcz_wap.itcz_width(da)                        if metric_name == 'itcz_width'          and var_name == 'wap'   else metric
        metric = itcz_wap.itcz_width_sMean(da)                  if metric_name == 'itcz_width_sMean'    and var_name == 'wap'   else metric
        metric = itcz_wap.get_fraction_descent(da, dims)        if metric_name == 'decent_fraction'     and var_name == 'wap'   else metric
        metric_name =f'{var_name}{region}_{metric_name}' 
        yield metric, metric_name


# ------------------------------------------------------------------------------------ Get dataset and save metric ----------------------------------------------------------------------------------------------------- #
def run_ls_metrics(switch_var, switchM, switch, resolution = cD.resolutions[0], timescale = cD.timescales[0]):
    print('getting ls-state_metrics')
    print(f'from variable: {resolution} {timescale} data \n {[key for key, value in switch_var.items() if value]}')
    print(f'metric: {[key for key, value in switchM.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')
    for var_name in [k for k, v in switch_var.items() if v]:
        for dataset, experiment in mD.run_dataset(var = var_name, datasets = cD.datasets, experiments = cD.experiments):
            da, region = vC.get_variable_data(switch, var_name, dataset, experiment, resolution, timescale)
            for metric, metric_name in calc_metric(switchM, var_name, da, region):      
                # print(metric_name)      
                # print(metric)      
                path = metD.save_metric(switch, met_type = var_name, dataset = dataset, experiment = experiment, metric = metric, metric_name = metric_name)
                print(f'Metric saved at: {path}')
    return metric # returns last metric (for testing one metric)



# -------------
#     Run
# -------------
# ------------------------------------------------------------------------------------------- Choose metric ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    switch_var = {                                                                                              # Choose variable (can choose multiple)
        'pr':       False,  'clwvi':        False,   'pe':          False,                                       # Precipitation
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
        'tMean':        False,  'sMean':            False,                                                      # means 
        'eof':          False,                                                                                  # ENSO
        'itcz_width':   False,   'itcz_width_tMean': False,   'descent_fraction': False,                        # ITCZ width (+ fraction of descent)
        'in_obj':       True                                                                                    # precipitation efficiency in objects
        }

    switch = {                                                                                                  # choose data to use and mask
        'constructed_fields':   False,  'test_sample':      False,                                              # data to use (test_sample uses first file (usually first year))
        '700hpa':               False,  '500hpa':           False,  '250hpa':   False,  'vMean':    False,      # vertical mask (3D variables are: wap, hur, ta, zg, hus)
        'ocean_mask':           False,                                                                          # horizontal mask
        'ascent_fixed':         False,  'descent_fixed':    False,  'ascent':   False,  'descent':  False,      # horizontal mask
        'save_folder_desktop':  False,  'save_scratch':     True,    'save':     False                          # Save
        }
    
    metric = run_ls_metrics(switch_var, switchM, switch)                                                        # metric here shows last metric calc



# use this:
# if switch.get('ocean_mask', False):
# that way a single metric and setting can be given (for example in a plotting script)
# Also, could have a similar check for checking if the metric is in the folder, and otherwise ask if it should be calculated
# might need a separate script for this








