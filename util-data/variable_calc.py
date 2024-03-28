'''
# ------------------
#   Variable_calc
# ------------------
Calculates variables from base variables and picks horizontal / vertical regions     
get base_variables from: util-data/variable_base.py
'''


# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import numpy as np
import xarray as xr
import re
from pathlib import Path


# ------------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util-core')
import choose_datasets  as cD  

sys.path.insert(0, f'{os.getcwd()}/util-files')
import save_folders as sF

sys.path.insert(0, f'{os.getcwd()}/util-data')
import missing_data     as mD
import variable_base    as vB
                                        # necessary util-data/var_calc scripts are imported in calc_variable()
sys.path.insert(0, f'{os.getcwd()}/util-calc')
                                        # necessary util-calc scripts are imported in pick_vert_reg()



# ------------------------
#      Pick region
# ------------------------
# ------------------------------------------------------------------------------------- Vertical mask --------------------------------------------------------------------------------------------------- #
def pick_vert_reg(switch, da):
    import statistics.means_calc as mC
    da, region = mC.get_vert_reg(switch, da)
    return da, region


# ------------------------------------------------------------------------------------- Horizontal mask --------------------------------------------------------------------------------------------------- #
def pick_ocean_region(da):
    mask = xr.open_dataset('/home/565/cb4968/Documents/code/phd/util-data/cmip/ocean_mask.nc')['ocean_mask']
    da = da * mask
    return da

def pick_hor_reg(switch, dataset, experiment, da, resolution, timescale):
    ''' Ascent/descent region based on 500 hPa vertical pressure velocity (wap)
    # loading data deals with picking out ocean (as it can be done before or after interpolation) '''
    da, region = da, ''
    for met_type in [k for k, v in switch.items() if v]:
        if ('descent' in switch and met_type == 'descent') or \
            ('ascent' in switch and met_type == 'ascent'):
            wap500 = vB.load_variable(switch, 'wap', dataset, experiment, resolution, timescale).sel(plev = 500e2)         
            da, region = [da.where(wap500 > 0), '_d']  if met_type == 'descent' else [da, region]
            da, region = [da.where(wap500 < 0), '_a']  if met_type == 'ascent'  else [da, region]
        if ('descent_fixed' in switch and met_type == 'descent_fixed') or \
            ('ascent_fixed' in switch and met_type == 'ascent_fixed'):
            wap500 = vB.load_variable(switch, 'wap', dataset, experiment, resolution, timescale).sel(plev = 500e2).mean(dim='time')
            da, region = [da.where(wap500 > 0), '_fd']  if met_type == 'descent_fixed' else [da, region]
            da, region = [da.where(wap500 < 0), '_fa']  if met_type == 'ascent_fixed'  else [da, region]
        region = f'_o{region}'  if met_type == 'ocean_mask'         else region                    
    return da, region



# ----------------------------
#   Calculate / save variable
# ----------------------------
def calc_variable(switch, var_name, dataset, experiment, resolution, timescale):
    # print(f'getting {var_name}{[key for key, value in switch.items() if value]}')    
    if var_name == 'pe':
        import var_calc.pe_var          as pE
        da = pE.get_pe(switch, var_name, dataset, experiment, resolution, timescale)  
    elif var_name == 'distance_matrix':
        import var_calc.distance_matrix as dM
        da = dM.get_distance_matrix(switch, dataset, experiment, resolution)
    elif var_name == 'conv_obj':
        import var_calc.conv_obj_var    as cO
        da, _ = cO.get_conv_obj(switch, dataset, experiment, resolution, percentile = int(cD.conv_percentiles[0])*0.01)
    elif var_name == 'obj_id':
        import var_calc.conv_obj_var    as cO
        _, da = cO.get_conv_obj(switch, dataset, experiment, resolution, percentile = int(cD.conv_percentiles[0])*0.01)
    if var_name == 'h': # mse
        import var_calc.mse_var         as mV
        da = mV.get_mse(switch, var_name, dataset, experiment, resolution, timescale)
    # if var_name == 'h_anom2': # squared anomalies
    #     get_mse_anom2(switch, var_name, dataset, experiment, resolution, timescale)
    # if var_name in ['lcf', 'hcf']:
    #      da = get_clouds(switch, var_name, dataset, experiment, resolution, timescale)   
    # if var_name == 'stability':
    #      da = get_stability(switch, var_name, dataset, experiment, resolution, timescale)
    # if var_name == 'netlw':
    #      da = get_netlw(switch, var_name, dataset, experiment, resolution, timescale) 
    # if var_name == 'netsw':
    #     da = get_netsw(switch, var_name, dataset, experiment, resolution, timescale) 
    else:   # base_variable
        da = vB.load_variable({var_name: True}, switch, dataset, experiment, resolution, timescale)    # basic metrics
    return da

def mask_variable(switch, var_name, dataset, experiment, resolution, timescale, da):
    da, vert_reg = pick_vert_reg(switch, da) if 'plev' in da.dims else [da, '']
    da, hor_reg  = pick_hor_reg(switch, dataset, experiment, da, resolution, timescale)    # experiment needed as wap is loaded to pick region
    return da, f'{vert_reg}{hor_reg}'

def process_data(switch, var_name, dataset, experiment, resolution, timescale, source):
    print(f'Processing {dataset} {resolution} {timescale} {var_name} data from {experiment} experiment with variable_calc script')
    if resolution == 'regridded':
        print(f'Regridding to {cD.x_res}x{cD.y_res} degrees')
    da = calc_variable(switch, var_name, dataset, experiment, resolution, timescale)
    da, region = mask_variable(switch, var_name, dataset, experiment, resolution, timescale, da)
    ds = xr.Dataset({var_name: da}) if not var_name == 'ds_cl' else da
    return ds, region

def request_process(switch, var_name, dataset, experiment, resolution, timescale, source, path, region):
    print(f'no {dataset} {experiment} {resolution} {timescale} {var_name}{region} data at {cD.x_res}x{cD.y_res} deg in \n {path}')
    response = input(f"Do you want to process {var_name}{region} from {dataset}? (y/n) (check folder first): ").lower()
    response = 'y'
    if response == 'y':
        ds, region = process_data(switch, var_name, dataset, experiment, resolution, timescale, source)
    if response == 'n':
        print('exiting')
        exit()
    return ds, region



# -------------------------
#  Handle request / source
# -------------------------
def find_source(dataset):
    '''Determining source of dataset '''
    source = 'test'     if np.isin(cD.test_fields, dataset).any()      else None     
    source = 'cmip5'    if np.isin(cD.models_cmip5, dataset).any()     else source      
    source = 'cmip6'    if np.isin(cD.models_cmip6, dataset).any()     else source         
    source = 'dyamond'  if np.isin(cD.models_dyamond, dataset).any()   else source        
    source = 'nextgems' if np.isin(cD.models_nextgems, dataset).any()  else source   
    source = 'obs'      if np.isin(cD.observations, dataset).any()     else source
    return source
    
def region_name(switch):
    ''' Ascent/descent region based on 500 hPa vertical pressure velocity (wap)
    # loading data deals with picking out ocean (as it can be done before or after interpolation) '''
    h_region = ''
    v_region = ''
    if switch.get('descent', False):
        h_region = '_d'           if switch['descent']    else h_region
    if switch.get('ascent', False):
        h_region = '_a'           if switch['ascent']     else h_region
    if switch.get('descent_fixed', False):
        h_region = '_fd'          if switch['descent']    else h_region
    if switch.get('ascent_fixed', False):
        h_region = '_fa'          if switch['ascent']     else h_region
    for met_type in [k for k, v in switch.items() if v]:
        number = re.findall(r'\d+', met_type)
        if number and 'hpa' in met_type: # if there is a number and hpa in string in switch option
            v_region = f'_{number[0]}hpa'
        if met_type == 'vMean':
            v_region = '_vMean'
    return f'{v_region}{h_region}'

def folder_structure(var_name, region, source):
    folder = f'{sF.folder_scratch}/sample_data/{var_name}{region}/{source}'
    # print(f'folder:{folder}')
    return folder

def filename_structure(dataset, experiment, var_name, region, source, timescale = cD.timescales[0]):
    if source in ['cmip5', 'cmip6'] and experiment == 'historical':
        filename = f'{dataset}_{var_name}{region}_{timescale}_{experiment}_{cD.cmip_years[0][0]}_{cD.resolutions[0]}'
    if source in ['cmip5', 'cmip6'] and experiment in ['ssp585', 'rcp85']:
        filename = f'{dataset}_{var_name}{region}_{timescale}_{experiment}_{cD.cmip_years[0][1]}_{cD.resolutions[0]}'
    if source in ['obs']:  
        filename = f'{dataset}_{var_name}_{timescale}_{cD.obs_years[0]}_obs_{cD.resolutions[0]}'
    if source in ['icon']:  
        filename = f'{dataset}_{var_name}_{timescale}_{cD.obs_years[0]}_icon_{cD.resolutions[0]}'
    if cD.resolutions[0] == 'regridded':
        filename = f'{filename}_{int(360/cD.x_res)}x{int(180/cD.y_res)}'
    # print(f'filename: {filename}')
    return filename

def save_in_scratch(var_name, region, dataset, experiment, resolution, timescale, source, ds):
    folder = folder_structure(var_name, region, source)
    filename = filename_structure(dataset, experiment, var_name, region, source, timescale = cD.timescales[0])
    Path(folder).mkdir(parents=True, exist_ok=True)
    path = Path(f'{folder}/{filename}.nc')
    ds.to_netcdf(path, mode="w")
    print(f'{dataset} {var_name}{region} data saved at {path}')
    return path

def check_if_in_scratch(var_name, region, dataset, experiment, resolution, timescale, source):
    folder = folder_structure(var_name, region, source)
    filename = filename_structure(dataset, experiment, var_name, region, source, timescale = cD.timescales[0])
    path = f'{folder}/{filename}.nc'
    return path, os.path.exists(path)

def get_data_from_calc_folder(switch, var_name, region, dataset, experiment, resolution, timescale, source):
    if switch.get('re_process_calc', False):
        ds, region = process_data(switch, var_name, dataset, experiment, resolution, timescale, source)
        path = save_in_scratch(var_name, region, dataset, experiment, resolution, timescale, source, ds)
    path, in_scratch = check_if_in_scratch(var_name, region, dataset, experiment, resolution, timescale, source)
    if in_scratch:
        ds = xr.open_dataset(path, chunks= {'time': 'auto'}) 
    else:
        ds, region = request_process(switch, var_name, dataset, experiment, resolution, timescale, source, path, region)
        path = save_in_scratch(var_name, region, dataset, experiment, resolution, timescale, source, ds)
    return ds

def get_variable(switch_var = {'pr': True}, switch = {'test_sample': False, 'ocean_mask': False}, dataset = '', experiment = '', 
                  resolution = cD.resolutions[0], timescale = cD.timescales[0]):
    var_name = next((key for key, value in switch_var.items() if value), None)
    region = region_name(switch)
    source = find_source(dataset)
    if switch.get('from_scratch_calc', False):
        ds = get_data_from_calc_folder(switch, var_name, region, dataset, experiment, resolution, timescale, source)
    else:   
        ds, _ = process_data(switch, var_name, dataset, experiment, resolution, timescale, source) # if calculating from base variables and not saving
    if switch.get('test_sample', False):
        ds = ds.isel(time = slice(0, 365))
    if switch.get('ocean_mask', False):
        ds = pick_ocean_region(ds)
        region = f'{region}_o'
    return ds[var_name], region



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':

    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import get_plot.map_plot    as mP
    import get_plot.show_plots  as sP

    switch_var = {                                                      # Choose variable (only runs one per function call)
        # calculated (8)
        'conv_obj':         False,  'obj_id':       False,              # Convective object identifiers
        'lcf':              False,  'hcf':          False,              # Cloud fraction                
        'h':                True,                                      # Moist Static Energy
        'pe':               False,                                      # Precipitation efficiency      
        'netlw':            False,                                      # Longwave radiation
        'netsw':            False,                                      # Shortwave radiation
        # base variables (20)
        'pr':       False,                                                                                      # Precipitation
        'tas':      False, 'ta':            False,                                                              # Temperature
        'wap':      False,                                                                                      # Circulation
        'hur':      False, 'hus' :          False,                                                              # Humidity                   
        'rlds':     False, 'rlus':          False,  'rlut':     False,  'netlw':    False,                      # Longwave radiation
        'rsdt':     False, 'rsds':          False,  'rsus':     False,  'rsut':     False,  'netsw':    False,  # Shortwave radiation
        'cl':       False, 'cl_p_hybrid':   False,  'p_hybrid': False,  'ds_cl':    False,                      # Cloudfraction (ds_cl is for getting pressure levels)
        'zg':       False,                                                                                      # Height coordinates
        'hfss':     False, 'hfls':          False,                                                              # Surface fluxes
        'clwvi':    False,                                                                                      # Cloud ice and liquid water
        }

    switch = {                                                                                                  # choose data to use
        'ocean_mask':           False,                                                                          # horizontal mask
        'ascent_fixed':         False,  'descent_fixed':    False,  'ascent':  False,  'descent':  False,       # horizontal mask
        'vMean':                False,  '700hpa':           False,  '500hpa':  False,   '250hpa':  False,       # vertical mask
        'test_sample':          False,                             # data to use (test_sample uses first file (usually first year))
        'from_scratch_calc':    True,  're_process_calc':  False,  # if both are true the scratch file is replaced with the reprocessed version (only matters for calculated variables / masked variables)
        'from_scratch':         True,  're_process':       False   # same as above, but for base variables
        }
    
    switch_test = {
        'delete_previous_plots': True,
        'plot_scene':            True
        }
    sP.remove_test_plots() if switch_test['delete_previous_plots'] else None


    # -------------------------------------------------------------------- Run to test / save variable temporarily --------------------------------------------------------------------------------------------------- #
    for var_name in [k for k, v in switch_var.items() if v]:
        # -------------------------------------------------------------------------------- Get data --------------------------------------------------------------------------------------------------- #
        print(f'variable: {var_name}')
        for experiment in cD.experiments:
            print(f'experiment: {experiment}')
            for dataset in mD.run_dataset_only(var_name, cD.datasets):
                print(f'dataset: {dataset}')
                da, region = get_variable(switch_var, switch, dataset = dataset, experiment = experiment, 
                  resolution = cD.resolutions[0], timescale = cD.timescales[0])
                da.load()
                print(da)
                break       # comment out if saving all models in scratch (break for testing)
            break           # comment out if saving both experiments in scratch


        ds = xr.Dataset()
        # -------------------------------------------------------------------------------- Calculate --------------------------------------------------------------------------------------------------- #    
        if switch_test['plot_scene']:   # intended for running one dataset for several variables
            if 'plev' in da.dims:
                level = 500e2
                ds[f'{dataset}_{var_name}{region}'] = da.isel(time = 0).sel(plev = level)
            else:
                ds[f'{dataset}_{var_name}{region}'] = da.isel(time = 0)


        # ----------------------------------------------------------------------------------- Plot --------------------------------------------------------------------------------------------------- #
        if switch_test['plot_scene']:   # plots individual colormaps for different variables
            label = '[units]'
            vmin = None
            vmax = None 
            cmap = 'Blues'
            title = f'{dataset}_{experiment}_variables'
            filename = f'{title}.png'
            fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()), cat_cmap = False)
            sP.show_plot(fig, show_type = 'save_cwd', filename = filename)

