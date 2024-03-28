'''
# ------------------------
#  Getting base variable
# ------------------------
Collects variables from different sources
Function:
    da = load_variable(switch_var, switch, dataset, experiment, resolution, timescale)

Input:
    da_cmip - cmip5 or cmip6 data   dim: (time, (plev), lat, lon)       get from: util-data/cmip/cmip_data.py
    da_obs  - observational data    dim: (time, (plev), lat, lon)       get from: util-data/observations/obs_data.py
    da_icon - icon model data       dim: (time, (plev), lat, lon)       get from: util-data/icon/icon_data.py

Output:
    da: - data array                dim: (time, (plev), lat, lon) 
'''


# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import numpy as np


# ------------------------------------------------------------------------------------ imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util-core')
import choose_datasets as cD  

sys.path.insert(0, f'{os.getcwd()}/util-data')
                                                # necessary var_process scripts are imported in load_variable()



# ------------------------
#       Get data
# ------------------------
# ----------------------------------------------------------------------------------- get model / obs data --------------------------------------------------------------------------------------------------- #
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
    if source in ['test']:
        import constructed.constructed_var as cF
        da = cF.get_cF_var(switch_var, dataset)

    if source in ['cmip5', 'cmip6']:
        import cmip.cmip_data as cmipD
        da = cmipD.get_cmip_data(switch_var, switch = switch, model = dataset, experiment = experiment, resolution = resolution, timescale = timescale)

    if source in ['obs']: 
        import observations.obs_data as obsD
        da = obsD.get_obs_data(switch_var, switch, dataset, experiment)      

    if source in ['nextgems'] and dataset == 'ICON-ESM_ngc2013': 
        import icon.icon_data as iconD
        da = iconD.get_icon_data(switch_var, switch, dataset)
    return da



# ------------------------
#         Test
# ------------------------
# ------------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    import xarray as xr

    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import missing_data             as mD

    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import get_plot.map_plot        as mP
    import get_plot.show_plots      as sP


    switch_var = {
        'pr':       False,                                                                                      # Precipitation
        'tas':      False, 'ta':            True,                                                              # Temperature
        'wap':      False,                                                                                      # Circulation
        'hur':      False, 'hus' :          False,                                                              # Humidity                   
        'rlds':     False, 'rlus':          False,  'rlut':     False,  'netlw':    False,                      # Longwave radiation
        'rsdt':     False, 'rsds':          False,  'rsus':     False,  'rsut':     False,  'netsw':    False,  # Shortwave radiation
        'cl':       False, 'cl_p_hybrid':   False,  'p_hybrid': False,  'ds_cl':    False,                      # Cloudfraction (ds_cl is for getting pressure levels)
        'zg':       False,                                                                                      # Height coordinates
        'hfss':     False, 'hfls':          False,                                                              # Surface fluxes
        'clwvi':    False,                                                                                      # Cloud ice and liquid water
        }

    switch = {
        'test_sample':      False,                                                                              # save
        'from_scratch':     True,   're_process':   False,  # if both are true the scratch file is replaced with the reprocessed version
        }
    
    switch_test = {
        'remove_test_plots':        False,
        'plot_scene':               False,
        }
    
    sP.remove_test_plots()  if switch_test['remove_test_plots']     else None


    # -------------------------------------------------------------------- Run to test / save variable temporarily --------------------------------------------------------------------------------------------------- #
    for var_name in [k for k, v in switch_var.items() if v]:
        # --------------------------------------------------------------------------------- Get data --------------------------------------------------------------------------------------------------- #
        print(f'variable: {var_name}')
        for experiment in cD.experiments:
            print(f'experiment: {experiment}')
            for dataset in mD.run_dataset_only(var_name, cD.datasets):
                print(f'dataset: {dataset}')
                da = load_variable(switch_var = {var_name: True}, switch = switch, dataset = cD.datasets[0], experiment = cD.experiments[0], resolution = cD.resolutions[0], timescale = cD.timescales[0])
                print(da)
                break       # comment out if saving all models in scratch (break for testing)
            break           # comment out if saving both experiments in scratch
        
        
        ds = xr.Dataset()
        # -------------------------------------------------------------------------------- Calculate --------------------------------------------------------------------------------------------------- #    
        if switch_test['plot_scene']:   # intended for running one dataset and several variables
            if 'plev' in da.dims:
                level = 500e2
                ds[f'{dataset}_{var_name}'] = da.isel(time = 0).sel(plev = level)
            else:
                ds[f'{dataset}_{var_name}'] = da.isel(time = 0)
    
    
    # -------------------------------------------------------------------------------------- Plot --------------------------------------------------------------------------------------------------- #
    if switch_test['plot_scene']:       # plots individual colormaps for different variables
        label = '[units]'
        vmin = None
        vmax = None
        cmap = 'Blues'
        filename = f'{dataset}_{experiment}_variables.png'
        fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()), cat_cmap = False)
        sP.show_plot(fig, show_type = 'save_cwd', filename = filename)



    # ----------------------------
    #  Call from different script
    # ----------------------------
    # sys.path.insert(0, f'{os.getcwd()}/util-data')
    # import get_data.variable_base as vB
    # da = vB.load_variable({'pr': True}, switch, dataset, experiment, resolution, timescale = 'daily')



