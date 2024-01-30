'''
# ------------------------
#  Getting base variable
# ------------------------
'''



# ------------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util-data')
import get_data.test_data   as gD_test
import get_data.cmip_data   as gD_cmip
import get_data.obs_data    as gD_obs
import get_data.icon_data   as gD_icon

sys.path.insert(0, f'{os.getcwd()}/util-core')
import myFuncs          as mF



def load_variable(switch_var = {'pr': True}, switch = {'constructed_fields': False}, dataset = 'TaiESM1', experiment = 'historical'):
    ''' Loading variable data.
        Sometimes sections of years of a dataset will be used instead of the full data ex: if dataset = GPCP_1998-2010 (for obsservations) 
        (There is a double trend in high percentile precipitation rate for the first 12 years of the data (that affects the area picked out by the time-mean percentile threshold)'''
    source = mF.find_source(dataset)
    if source in ['test']:
        da = gD_test.get_cF_var(switch_var, dataset)    
    if source in ['cmip5', 'cmip6']:
        da = gD_cmip.get_cmip_data(switch_var, switch, dataset, experiment)   
    if source in ['obs']: 
        da = gD_obs.get_obs_data(switch_var, switch, dataset, experiment)         
    if source in ['nextgems'] and dataset == 'ICON-ESM_ngc2013': 
        da = gD_icon.get_icon_data(switch_var, switch, dataset)
    # file_pattern = "/Users/cbla0002/Desktop/pr/ICON-ESM_ngc2013/ICON-ESM_ngc2013_pr_daily_*.nc" 
    # paths = sorted(glob.glob(file_pattern))
    # da = xr.open_mfdataset(paths, combine='by_coords', parallel=True) # chunks="auto"
    return da



# ------------------------
#         Test
# ------------------------
# ------------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import myVars          as mV
    import myFuncs_plots    as mFp     

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
    
    da = load_variable(switch_var = switch_var, switch = switch, dataset = mV.datasets[0], experiment = mV.experiments[0])
    print(da)
    # mFp.get_snapshot(da, plot = True, show_type = 'cycle') # show_type = [show, save_cwd, cycle] 




