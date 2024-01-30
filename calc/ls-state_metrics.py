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
sys.path.insert(0, f'{os.getcwd()}/util-data')
import get_data.variable_data as vD

sys.path.insert(0, f'{os.getcwd()}/util-calc')
import ls_state.means_calc      as mean_calc
import ls_state.itcz_width_calc as itcz_calc

sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars               as mV                                 
import myFuncs              as mF     
import myFuncs_plots        as mFp     
import get_data.metric_data as mD


# ------------------------
#   Run / save metric
# ------------------------
# ------------------------------------------------------------------------------------ Get metric and metric name ----------------------------------------------------------------------------------------------------- #
def calc_metric(switchM, var_name, da, region):
    dim = mF.dims_class(da)
    for metric_name in [k for k, v in switchM.items() if v]:
        metric = None
        metric = mFp.get_snapshot(da)           if metric_name == 'snapshot'            else metric
        metric = mean_calc.get_tMean(da)        if metric_name == 'tMean'               else metric
        metric = mean_calc.get_sMean(da)        if metric_name == 'sMean'               else metric
        metric = itcz_calc.itcz_width(da)       if metric_name == 'itcz_width'          else metric
        metric = itcz_calc.itcz_width_sMean(da) if metric_name == 'itcz_width_sMean'    else metric
        metric = itcz_calc.itcz_width(da)       if metric_name == 'area_pos'            else metric
        metric_name =f'{var_name}{region}_{metric_name}' 
        yield metric, metric_name


# ------------------------------------------------------------------------------------ Get dataset and save metric ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator(show_time = True)
def run_ls_metrics(switch_var, switchM, switch):
    print(f'variable: {mV.resolutions[0]} {mV.timescales[0]} data \n {[key for key, value in switch_var.items() if value]}')
    print(f'metric: {[key for key, value in switchM.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')
    for var_name in [k for k, v in switch_var.items() if v]:
        for dataset, experiment in mF.run_dataset(var_name):
            da, region = vD.get_variable_data(switch, var_name, dataset, experiment)
            for metric, metric_name in calc_metric(switchM, var_name, da, region):      
                # print(metric_name)      
                # print(metric)      
                path = mD.save_metric(switch, var_name, dataset, experiment, metric, metric_name)
                print(f'Metric saved at: {path}')

# ------------------------------------------------------------------------------------------- Choose metric ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    switch_var = {                                                                                              # Choose variable (can choose multiple)
        'pr':       False,  'clwvi':        False,   'pe':          False,                                      # Precipitation
        'wap':      True,                                                                                       # Circulation
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
        'snapshot':     False,                                                                                  # visualization
        'tMean':        False,  'sMean':            False,                                                      # means 
        'eof':          False,                                                                                  # ENSO
        'itcz_width':   True,   'itcz_width_sMean': True,   'area_pos': False,   'area_pos': False,            # ITCZ width (+ fraction of descent)
        }

    switch = {                                                                                                  # choose data to use and mask
        'constructed_fields':   False,  'test_sample':      False,                                              # data to use (test_sample uses first file (usually first year))
        '700hpa':               False,  '500hpa':           True,   '250hpa':   False,  'vMean':    False,      # vertical mask (3D variables are: wap, hur, ta, zg, hus)
        'ocean_mask':           False,                                                                          # horizontal mask
        'ascent_fixed':         False,  'descent_fixed':    False,  'ascent':   False,  'descent':  False,      # horizontal mask
        'save_folder_desktop':  False,   'save_scratch':    True,  'save':     False                            # Save
        }
    
    run_ls_metrics(switch_var, switchM, switch)

















