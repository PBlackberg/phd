'''
# ------------------
#   Variable_calc
# ------------------
These variables are calculated from base variables (the base variables are at times saved in scratch for quick access) (the calculated variables are rarely stored)
Some variables need to be calculated, and sometimes a horizontal or vertical region is picked out

To use:
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-core')
import choose_datasets as cD  

sys.path.insert(0, f'{os.getcwd()}/util-data')
import variable_calc as vC
da = vC.get_variable(switch_var = {'pr': True}, switch = {'test_sample': False, 'ocean_mask': False}, dataset = '', experiment = '', 
                  resolution = cD.resolutions[0], timescale = cD.timescales[0], from_folder = False, re_process = False)
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

sys.path.insert(0, f'{os.getcwd()}/util-data')
import variable_base            as vB
import dimensions_data          as dD
import missing_data             as mD
import var_calc.pe_var          as pE
import var_calc.conv_obj_var    as cO
import var_calc.distance_matrix as dM

sys.path.insert(0, f'{os.getcwd()}/util-calc')
import ls_state.means_calc  as mean_calc

sys.path.insert(0, f'{os.getcwd()}/util-files')
import save_folders as sF



# ------------------------
#      Pick region
# ------------------------
# ------------------------------------------------------------------------------------- Vertical mask --------------------------------------------------------------------------------------------------- #
def pick_vMean(da, plevs0 = 850e2, plevs1 = 0):         
    ''' # free troposphere (as most values at 1000 hPa and 925 hPa over land are NaN)
        # Where there are no values, exclude the associated pressure levels from the weights '''
    da = da.sel(plev = slice(plevs0, plevs1))
    w = ~np.isnan(da) * da['plev']                      
    da = (da * w).sum(dim='plev') / w.sum(dim='plev') 
    return da

def pick_vert_reg(switch, da):
    da, region = da, ''
    for met_type in [k for k, v in switch.items() if v]:
        number = re.findall(r'\d+', met_type)
        if number and 'hpa' in met_type: # if there is a number and hpa in string
            level = int(re.findall(r'\d+', met_type)[0]) * 10**2
            da, region = [da.sel(plev = level), f'_{number}hpa']
        if met_type == 'vMean':
            da, region = [pick_vMean(da), 'vMean']
    return da, region


# ------------------------------------------------------------------------------------ Horizontal mask --------------------------------------------------------------------------------------------------- #
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

def pick_ocean_region(da):
    mask = xr.open_dataset('/home/565/cb4968/Documents/code/phd/util-data/cmip/ocean_mask.nc')['ocean_mask']
    da = da * mask
    return da



# ------------------------
#  Calculated variables
# ------------------------
# ------------------------------------------------------------------------------------ calculate --------------------------------------------------------------------------------------------------- #
def get_clouds(switch, var_name, dataset, experiment, resolution, timescale):
    ''' # can also do 250 up (in schiro 'spread paper') '''
    da = vB.load_variable({'cl':True}, switch, dataset, experiment, resolution, timescale)
    da = da.sel(plev = slice(1000e2, 600e2)).max(dim = 'plev') if var_name == 'lcf' else da
    da = da.sel(plev = slice(400e2, 0)).max(dim = 'plev')      if var_name == 'hcf' else da  
    return da

def get_stability(switch, var_name, dataset, experiment, resolution, timescale):
    ''' # Differnece in potential temperature between two vertical sections 
    # Temperature at pressure levels (K) 
    # Where there are no temperature values, exclude the associated pressure levels from the weights'''
    da = vB.load_variable({'ta': True}, switch, dataset, experiment, resolution, timescale)                    
    theta =  da * (1000e2 / da['plev'])**(287/1005) 
    plevs1, plevs2 = [400e2, 250e2], [925e2, 700e2]
    da1, da2 = [theta.sel(plev=slice(plevs1[0], plevs1[1])), theta.sel(plev=slice(plevs2[0], plevs2[1]))]
    w1, w2 = ~np.isnan(da1) * da1['plev'], ~np.isnan(da2) * da2['plev']                 
    da = ((da1 * w1).sum(dim='plev') / w1.sum(dim='plev')) - ((da2 * w2).sum(dim='plev') / w2.sum(dim='plev'))
    return da

def get_netlw(switch, var_name, dataset, experiment, resolution, timescale):
    rlds, rlus, rlut = [vB.load_variable({var: True}, switch, dataset, experiment, resolution, timescale) for var in ['rlds', 'rlus', 'rlut']]
    da = -rlds + rlus - rlut
    return da

def get_netsw(switch, var_name, dataset, experiment, resolution, timescale):
    rsdt, rsds, rsus, rsut = [vB.load_variable({var: True}, switch, dataset, experiment, resolution, timescale) for var in ['rsdt', 'rsds', 'rsus', 'rsut']]
    da = rsdt - rsds + rsus - rsut
    return da

def get_mse(switch, var_name, dataset, experiment, resolution, timescale):
    '''  # h - Moist Static Energy (MSE) '''
    c_p, L_v = dD.dims_class.c_p, dD.dims_class.L_v
    ta, zg, hus = [vB.load_variable({var: True}, switch, dataset, experiment, resolution, timescale) for var in ['ta', 'zg', 'hus']]
    da = c_p * ta + zg + L_v * hus
    return da

def get_mse_anom2(switch, var_name, dataset, experiment, resolution, timescale):
    '''# MSE variance from the tropical mean  '''
    c_p, L_v = dD.dims_class.c_p, dD.dims_class.L_v
    ta, zg, hus = [vB.load_variable({var: True}, switch, dataset, experiment, resolution, timescale) for var in ['ta', 'zg', 'hus']]
    da = c_p * ta + zg + L_v * hus
    da, _ = pick_vert_reg(switch, dataset, da)
    da_sMean = mean_calc.get_sMean(da)
    da_anom = da - da_sMean
    da = da_anom**2
    return da


# ------------------------------------------------------------------------------------ summarize --------------------------------------------------------------------------------------------------- #
def calc_variable(switch, var_name, dataset, experiment, resolution, timescale):
    ''' Gets variable (some need to be calculated) '''
    # print(f'getting {var_name}{[key for key, value in switch.items() if value]}')    
    if var_name in ['lcf', 'hcf', 'stability', 'netlw', 'netsw', 'h', 'h_anom2', 'pe', 'conv_obj', 'obj_id', 'distance_matrix']:
        da = get_clouds(switch, var_name, dataset, experiment, resolution, timescale)       if var_name in ['lcf', 'hcf']       else None
        da = get_stability(switch, var_name, dataset, experiment, resolution, timescale)    if var_name == 'stability'          else da
        da = get_netlw(switch, var_name, dataset, experiment, resolution, timescale)        if var_name == 'netlw'              else da
        da = get_netsw(switch, var_name, dataset, experiment, resolution, timescale)        if var_name == 'netsw'              else da
        da = get_netsw(switch, var_name, dataset, experiment, resolution, timescale)        if var_name == 'netsw'              else da
        da = get_mse(switch, var_name, dataset, experiment, resolution, timescale)          if var_name == 'h'                  else da
        da = get_mse_anom2(switch, var_name, dataset, experiment, resolution, timescale)    if var_name == 'h_anom2'            else da
        da = pE.get_pe(switch, var_name, dataset, experiment, resolution, timescale)        if var_name == 'pe'                 else da
        da, _ = cO.get_conv_obj(switch, dataset, experiment, resolution)                    if var_name == 'conv_obj'           else [da, None]
        _, da = cO.get_conv_obj(switch, dataset, experiment, resolution)                    if var_name == 'obj_id'             else [None, da]
        da = dM.get_distance_matrix(switch, dataset, experiment, resolution)                if var_name == 'distance_matrix'    else da
    else:
        da = vB.load_variable({var_name: True}, switch, dataset, experiment, resolution, timescale)    # basic metrics
    return da

def get_variable_data(switch, var_name, dataset, experiment, resolution, timescale):
    ''' Picks region of variable '''
    da = calc_variable(switch, var_name, dataset, experiment, resolution, timescale)
    da, vert_reg = pick_vert_reg(switch, da) if 'plev' in da.dims else [da, '']
    da, hor_reg  = pick_hor_reg(switch, dataset, experiment, da, resolution, timescale)    # experiment needed as wap is loaded to pick region
    return da, f'{vert_reg}{hor_reg}'


# ----------------------------------------------------------------------------------- Check scratch ----------------------------------------------------------------------------------------------------- #
def save_in_scratch(var_name, dataset, experiment, resolution, timescale, source, ds, region):
    folder = f'{sF.folder_scratch}/sample_data/{var_name}{region}/{source}'
    filename = f'{dataset}_{var_name}{region}_{timescale}_*_{experiment}_{resolution}.nc'
    if resolution == 'regridded':
        filename = f'{dataset}_{var_name}{region}_{timescale}_{experiment}_{resolution}_{int(360/cD.x_res)}x{int(180/cD.y_res)}.nc'
    Path(folder).mkdir(parents=True, exist_ok=True)
    path = Path(f'{folder}/{filename}')
    ds.to_netcdf(path, mode="w")
    print(f'{dataset} {var_name}{region} data saved at {path}')
    return path

def process_data(switch, var_name, dataset, experiment, resolution, timescale, source):
    print(f'Processing {dataset} {resolution} {timescale} {var_name} data from {experiment} experiment')
    if resolution == 'regridded':
        print(f'Regridding to {cD.x_res}x{cD.y_res} degrees')
    da, region = get_variable_data(switch, var_name, dataset, experiment, resolution, timescale)
    ds = xr.Dataset({var_name: da}) if not var_name == 'ds_cl' else da
    path = save_in_scratch(var_name, dataset, experiment, resolution, timescale, source, ds, region)
    return xr.open_dataset(path, chunks= {'time': 'auto'})[var_name]

def request_process(switch, var_name, dataset, experiment, resolution, timescale, source, path, region):
    print(f'no {dataset} {experiment} {resolution} {timescale} {var_name}{region} data at {cD.x_res}x{cD.y_res} deg in \n {path}')
    response = input(f"Do you want to process {var_name}{region} from {dataset}? (y/n/y_all) (check folder first): ").lower()
    # response = 'y'
    if response == 'y':
        da = process_data(switch, var_name, dataset, experiment, resolution, timescale, source)
        print('requested dataset is processed and saved in scratch')
    if response == 'n':
        print('exiting')
        exit()
    if response == 'y_all':
        for dataset, experiment in mD.run_dataset(var_name, cD.datasets, cD.experiments):
            process_data(switch, var_name, dataset, experiment, resolution, timescale, source)
        print('all requested datasets processed and saved in scratch')
        exit()
    return da

def check_scratch(var_name, dataset, experiment, resolution, timescale, source, region):
    folder = f'{sF.folder_scratch}/sample_data/{var_name}{region}/{source}'
    filename = f'{dataset}_{var_name}{region}_{timescale}_{experiment}_{resolution}.nc'
    if resolution == 'regridded':
        filename = f'{dataset}_{var_name}{region}_{timescale}_{experiment}_{resolution}_{int(360/cD.x_res)}x{int(180/cD.y_res)}.nc'
    path = f'{folder}/{filename}'
    return path, os.path.exists(path)

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

def find_source(dataset):
    '''Determining source of dataset '''
    source = 'test'     if np.isin(cD.test_fields, dataset).any()      else None     
    source = 'cmip5'    if np.isin(cD.models_cmip5, dataset).any()     else source      
    source = 'cmip6'    if np.isin(cD.models_cmip6, dataset).any()     else source         
    source = 'dyamond'  if np.isin(cD.models_dyamond, dataset).any()   else source        
    source = 'nextgems' if np.isin(cD.models_nextgems, dataset).any()  else source   
    source = 'obs'      if np.isin(cD.observations, dataset).any()     else source
    return source

def get_variable(switch_var = {'pr': True}, switch = {'test_sample': False, 'ocean_mask': False}, dataset = '', experiment = '', 
                  resolution = cD.resolutions[0], timescale = cD.timescales[0], from_folder = False, re_process = False):
    var_name = next((key for key, value in switch_var.items() if value), None)
    region = region_name(switch)
    if from_folder:
        source = find_source(dataset)
        path, in_scratch = check_scratch(var_name, dataset, experiment, resolution, timescale, source, region)
        if re_process:
            da = request_process(switch, var_name, dataset, experiment, resolution, timescale, source, path, region)
        elif in_scratch:
            ds = xr.open_dataset(path, chunks= 'auto') 
            if 'time' in ds.dims:
                ds.chunk({'time': 'auto'})
            da = ds[var_name]
        else:
            da = request_process(switch, var_name, dataset, experiment, resolution, timescale, source, path, region)
    else:   # if calculating from base variables
        da, _ = get_variable_data(switch, var_name, dataset, experiment, resolution, timescale)
    if switch.get('test_sample', False):
        da = da.isel(time = slice(0, 365))  if switch['test_sample']   else da
    if switch.get('ocean_mask', False):
        da = pick_ocean_region(da)          if switch['ocean_mask']    else da
        region = f'{region}_o'              if switch['ocean_mask']    else region
    return da, region



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    switch_var = {                                                      # Choose variable (only runs one per function call)
        'pr':               False,  'pe':           False,              # Precipitation (pr not calculated, used for testing that it can call variable_base)               
        'wap':              False,                                      # Circulation
        'hur':              False,  'hus' :         False,              # Humidity               
        'stability':        False,                                      # Temperature
        'netlw':            False,                                      # Longwave radiation
        'netsw':            False,                                      # Shortwave radiation
        'lcf':              False,  'hcf':          False,              # Cloud fraction
        'zg':               False,                                      # Height coordinates
        'h':                False,  'h_anom2':      False,              # Moist Static Energy
        'conv_obj':         False,  'obj_id':       False,
        'distance_matrix':  True
        }

    switch = {                                                                                                 # choose data to use and mask
        'constructed_fields':   False,  'test_sample':      False,                                             # data to use (test_sample uses first file (usually first year))
        '700hpa':               False,  '500hpa':           False,   '250hpa':  False,  'vMean':    False,      # vertical mask (3D variables are: wap, hur, ta, zg, hus)
        'ocean_mask':           False,                                                                         # horizontal mask
        'ascent_fixed':         False,  'descent_fixed':    False,  'ascent':  False,  'descent':  False,      # horizontal mask
        }
    
    switch_test = {
        'delete_previous_plots': True,
        'plot_scene':            False
        }


    for var_name in [k for k, v in switch_var.items() if v]:
        # ----------------------------------------------------------------------------------- Get data --------------------------------------------------------------------------------------------------- #
        print(f'variable: {var_name}')
        for experiment in cD.experiments:
            print(f'experiment: {experiment}')
            for dataset in mD.run_dataset_only(var_name, cD.datasets):
                da, region = get_variable(switch_var, switch, dataset, experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0], from_folder = True, re_process = False)
                print(da)
                break
            break


    ds = xr.Dataset()
    # -------------------------------------------------------------------------------------- Plot --------------------------------------------------------------------------------------------------- #
    if switch_test['plot_scene']:
        sys.path.insert(0, f'{os.getcwd()}/util-plot')
        import get_plot.map_plot         as mP
        mP.remove_test_plots() if switch_test['delete_previous_plots'] else None
        ds[dataset] = da.isel(time = 0) #.sel(plev = 500e2)
        label = '[units]'
        vmin = None
        vmax = None 
        cmap = 'Blues'
        title = f'{var_name}{region}_{dataset}_{experiment}'
        filename = f'{title}.png'
        fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()), cat_cmap = False)
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

