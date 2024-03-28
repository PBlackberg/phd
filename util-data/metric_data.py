'''
# ------------------------
#     Metric data
# ------------------------
This script saves / loads metrics
It tries to open files from different potential folders (folder_save, folder_scratch and at different temporal scale if the requested is not available)
'''


# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np


# ------------------------------------------------------------------------------------ imported scripts --------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")

sys.path.insert(0, f'{os.getcwd()}/util-core')
import choose_datasets as cD

sys.path.insert(0, f'{os.getcwd()}/util-files')
import save_folders as sF



# ------------------------
#  General dir structure
# ------------------------
def find_source(dataset):
    ''' Determining source of dataset '''
    source = 'test'     if np.isin(cD.test_fields, dataset).any()      else None     
    source = 'cmip5'    if np.isin(cD.models_cmip5, dataset).any()     else source      
    source = 'cmip6'    if np.isin(cD.models_cmip6, dataset).any()     else source         
    source = 'dyamond'  if np.isin(cD.models_dyamond, dataset).any()   else source        
    source = 'nextgems' if np.isin(cD.models_nextgems, dataset).any()  else source   
    source = 'obs'      if np.isin(cD.observations, dataset).any()     else source
    return source

def folder_structure(user_folder, met_type, metric_name, source):
    folder = f'{user_folder}/metrics/{met_type}/{metric_name}/{source}'
    # print(folder)
    return folder

def filename_structure(dataset, experiment, metric_name, source, timescale = cD.timescales[0]):
    # print(metric_name)
    if source in ['cmip5', 'cmip6'] and experiment == 'historical':
        filename = f'{dataset}_{metric_name}_{timescale}_{experiment}_{cD.cmip_years[0][0]}_{cD.resolutions[0]}'
    if source in ['cmip5', 'cmip6'] and experiment in ['ssp585', 'rcp85']:
        filename = f'{dataset}_{metric_name}_{timescale}_{experiment}_{cD.cmip_years[0][1]}_{cD.resolutions[0]}'
    if source in ['obs']:  
        filename = f'{dataset}_{metric_name}_{timescale}_{cD.obs_years[0]}_obs_{cD.resolutions[0]}'
    if cD.resolutions[0] == 'regridded':
        filename = f'{filename}_{int(360/cD.x_res)}x{int(180/cD.y_res)}'
    # print(f'filename is: {filename}')
    return filename



# --------------------
#  save / load metric
# --------------------
# ----------------------------------------------------------------------------------------- save --------------------------------------------------------------------------------------------- #
def save_file(data, folder=f'{home}/Documents/code/phd', filename='test.nc', path = ''):
    ''' Overwrites existing metric with new metric '''
    if folder and filename:
        path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    data.to_netcdf(path, mode = 'w')
    
def save_metric(switch, met_type, dataset, experiment, metric, metric_name):
    ''' Saves in variable/metric specific folders '''
    source = find_source(dataset)
    filename = filename_structure(dataset, experiment, metric_name, source)
    for save_type in [k for k, v in switch.items() if v]:
        folder = folder_structure(f'{home}/Desktop/{metric_name}', met_type, metric_name, source) if save_type == 'save_folder_desktop'   else None
        folder = folder_structure(sF.folder_scratch, met_type, metric_name, source)               if save_type == 'save_scratch'          else folder
        folder = folder_structure(sF.folder_save, met_type, metric_name, source)                  if save_type == 'save'                  else folder
        if type(folder) == str: # if not saving folder will be None
            save_file(xr.Dataset({metric_name: metric}), folder, f'{filename}.nc')
            print(f'\t\t\t saved: {metric_name} \n \t\t\t in: {folder} \n \t\t\t as: {filename}.nc')


# ------------------------------------------------------------------------------------------ load --------------------------------------------------------------------------------------------- #
def try_opening(paths_file, metric_name):
    ''' Metrics can be saved at multiple spots, so trying to open files from different locations '''
    for path in paths_file:
        try:
            ds = xr.open_dataset(path)
            da = ds[metric_name]
            return da
        except:
            continue

def load_metric(met_type, metric_name, dataset, experiment, dataset_org = 'GPCP', timescale = cD.timescales[0]):
    source = find_source(dataset)
    if source in ['obs']:
        if met_type in ['pr', 'conv_org']: # GPCP observations always used for obs precipitation based metrics
            dataset = dataset_org 

    folders_check = [sF.folder_scratch] #, sF.folder_save, f'{home}/Desktop/']
    filenames = []
    paths_file = []
    for user_folder in folders_check:    # check different folders                       
        folder = folder_structure(user_folder, met_type, metric_name, source)                  
        filename = filename_structure(dataset, experiment, metric_name, source, timescale)    
        filenames.append(f'{filename}.nc')    
        paths_file.append(f'{folder}/{filename}.nc')
    da = try_opening(paths_file, metric_name)
    if da is None:
        print(f"Couldn't open {metric_name} from {source} {dataset} {experiment} {timescale} from the provided paths")
        print(f'check folders:')
        print('\n'.join(folder for folder in folders_check))
        print('and filenames:')
        print('\n'.join(filename for filename in filenames))
        print('exiting')
        exit()
    return da


# ------------------------------------------------------------------------------------------- remove --------------------------------------------------------------------------------------------- #
def remove_metric(switch, met_type, dataset, experiment, metric_name):
    ''' Removes metric in variable/metric specific folders '''
    source = find_source(dataset)
    filename = filename_structure(dataset, experiment, metric_name, source)
    for save_type in [k for k, v in switch.items() if v]:
        folder = folder_structure(f'{home}/Desktop/{metric_name}', met_type, metric_name, source) if save_type == 'save_folder_desktop'   else None
        folder = folder_structure(sF.folder_scratch, met_type, metric_name, source)               if save_type == 'save_scratch'          else folder
        folder = folder_structure(sF.folder_save, met_type, metric_name, source)                  if save_type == 'save'                  else folder
        if type(folder) == str: # if not saving folder will be None
            path = f'{folder}/{filename}.nc'
            os.remove(path)
            print(f'\t\t\t removed {metric_name} \n \t\t\t at: {path}')
            return 



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':

    switch_test = {
        'save':             True,
        'load':             True,
        'remove_metric':    True
        }

    switch = {
        'save': False,  'save_scratch': True, 'save_folder_desktop':   False,
        }

    met_type = 'test'
    dataset = cD.datasets[0]
    experiment = cD.experiments[0]
    timescale = cD.timescales[0]
    metric = xr.DataArray(np.array([1, 2, 3]), dims = ['time'], coords = {'time': np.array([1, 2, 3])})
    metric_name = 'test_metric'

    if switch_test['save']:
        save_metric(switch, met_type, dataset, experiment, metric, metric_name)

    if switch_test['load']:
        da = load_metric(met_type, metric_name, dataset, experiment, timescale = timescale)
        print(da)

    if switch_test['remove_metric']:
        remove_metric(switch, met_type, dataset, experiment, metric_name)



