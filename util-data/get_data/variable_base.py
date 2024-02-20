'''
# ------------------------
#  Getting base variable
# ------------------------
These variables are at times saved in scratch for quicker access

To use:
sys.path.insert(0, f'{os.getcwd()}/util-data')
import get_data.variable_base as vB
da = vB.load_variable({'pr': True}, switch, dataset, experiment, resolution, timescale = 'daily')

'''



# ---------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import numpy as np


# ------------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util-core')
import choose_datasets as cD  

sys.path.insert(0, f'{os.getcwd()}/util-data')
# import get_data.constructed_fields      as constF
import get_data.cmip.cmip_data          as cmipD
# import get_data.observations.obs_data   as obsD
# import get_data.icon.icon_data          as iconD



# ------------------------
#       Get data
# ------------------------
# ------------------------------------------------------------------------------------- get model / obs data --------------------------------------------------------------------------------------------------- #
def find_source(dataset):
    '''Determining source of dataset '''
    source = 'test'     if np.isin(cD.test_fields, dataset).any()      else None     
    source = 'cmip5'    if np.isin(cD.models_cmip5, dataset).any()     else source      
    source = 'cmip6'    if np.isin(cD.models_cmip6, dataset).any()     else source         
    source = 'dyamond'  if np.isin(cD.models_dyamond, dataset).any()   else source        
    source = 'nextgems' if np.isin(cD.models_nextgems, dataset).any()  else source   
    source = 'obs'      if np.isin(cD.observations, dataset).any()     else source
    return source

def load_variable(switch_var = {'pr': True}, switch = {'test_sample': False, 'ocean_mask': False}, dataset = '', experiment = '', 
                  resolution = cD.resolutions[0], timescale = cD.timescales[0]):
    source = find_source(dataset)
    # if source in ['test']:
    #     da = constF.get_cF_var(switch_var, dataset)    
    if source in ['cmip5', 'cmip6']:
        da = cmipD.get_cmip_data(switch_var, switch, model = dataset, experiment = experiment, resolution = resolution, timescale = timescale)
    # if source in ['obs']: 
    #     da = obsD.get_obs_data(switch_var, switch, dataset, experiment)         
    # if source in ['nextgems'] and dataset == 'ICON-ESM_ngc2013': 
    #     da = iconD.get_icon_data(switch_var, switch, dataset)
    return da



# ------------------------
#         Test
# ------------------------
# ------------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':

    switch_var = {
        'pr':       False,                                                                                       # Precipitation
        'tas':      False, 'ta':            False,                                                              # Temperature
        'wap':      True,                                                                                      # Circulation
        'hur':      False, 'hus' :          False,                                                              # Humidity                   
        'rlds':     False, 'rlus':          False,  'rlut':     False,  'netlw':    False,                      # Longwave radiation
        'rsdt':     False, 'rsds':          False,  'rsus':     False,  'rsut':     False,  'netsw':    False,  # Shortwave radiation
        'cl':       False, 'cl_p_hybrid':   False,  'p_hybrid': False,  'ds_cl':    False,                      # Cloudfraction (ds_cl is for getting pressure levels)
        'zg':       False,                                                                                      # Height coordinates
        'hfss':     False, 'hfls':          False,                                                              # Surface fluxes
        'clwvi':    False,                                                                                      # Cloud ice and liquid water
        }

    switch = {
        'ocean_mask':    False, # mask
        'test_sample':   False  # save
        }
    
    da = load_variable(switch_var = {'pr': True}, switch = {'test_sample': False, 'ocean_mask': False}, dataset = cD.datasets[0], experiment = cD.experiments[0], 
                  resolution = cD.resolutions[0], timescale = cD.timescales[0])
    print(da)




