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

sys.path.insert(0, f'{os.getcwd()}/util-data')
import organize_files.save_folders as sF


# -----------------------
#     Get metric
# -----------------------
def find_source(dataset):
    '''Determining source of dataset '''
    source = 'test'     if np.isin(cD.test_fields, dataset).any()      else None     
    source = 'cmip5'    if np.isin(cD.models_cmip5, dataset).any()     else source      
    source = 'cmip6'    if np.isin(cD.models_cmip6, dataset).any()     else source         
    source = 'dyamond'  if np.isin(cD.models_dyamond, dataset).any()   else source        
    source = 'nextgems' if np.isin(cD.models_nextgems, dataset).any()  else source   
    source = 'obs'      if np.isin(cD.observations, dataset).any()     else source
    return source


# ------------------------------------------------------------------------------------- file structure --------------------------------------------------------------------------------------------- #
def get_folder(folder_parent, met_type, metric_name, source):
    folder = f'{folder_parent}/metrics/{met_type}/{metric_name}/{source}'
    # print(folder)
    return folder

def get_filename(dataset, experiment, metric_name, source, timescale = cD.timescales[0]):
    # print(metric_name)
    filename = f'{dataset}_{metric_name}_{timescale}_{experiment}_{cD.resolutions[0]}'
    if source in ['obs']:  
        filename = f'{dataset}_{metric_name}_{timescale}_{cD.obs_years[0]}_obs_{cD.resolutions[0]}'
    if cD.resolutions[0] == 'regridded':
        filename = f'{filename}_{int(360/cD.x_res)}x{int(180/cD.y_res)}'
    # print(f'filename is: {filename}')
    return filename

def save_file(data, folder=f'{home}/Documents/phd', filename='test.nc', path = ''):
    ''' Basic saving function '''
    if folder and filename:
        path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    data.to_netcdf(path, mode = 'w')

# ----------------------------------------------------------------------------------------- save --------------------------------------------------------------------------------------------- #
def save_metric(switch, met_type, dataset, experiment, metric, metric_name):
    ''' Saves in variable/metric specific folders '''
    source = find_source(dataset)
    filename = get_filename(dataset, experiment, metric_name, source)
    for save_type in [k for k, v in switch.items() if v]:
        folder = get_folder(f'{home}/Desktop/{metric_name}', met_type, metric_name, source) if save_type == 'save_folder_desktop'   else None
        folder = get_folder(sF.folder_scratch, met_type, metric_name, source)               if save_type == 'save_scratch'          else folder
        folder = get_folder(sF.folder_save, met_type, metric_name, source)                  if save_type == 'save'                  else folder
        if type(folder) == str:
            save_file(xr.Dataset({metric_name: metric}), folder, f'{filename}.nc')
            print(f'\t\t\t saved {metric_name} ({save_type})')
            return f'{folder}/{filename}.nc'


# ----------------------------------------------------------------------------------------- load --------------------------------------------------------------------------------------------- #
def try_opening(file_paths, metric_name):
    for path in file_paths:
        try:
            ds = xr.open_dataset(path)
            da = ds[metric_name]
            return da
        except:
            continue

def load_metric(metric_type, metric_name, dataset, experiment, dataset_org = 'GPCP'):
    source = find_source(dataset)
    if metric_type in ['pr', 'conv_org'] and source in ['obs']: # GPCP observations used for precipitation based metrics
        dataset = dataset_org 
    filename = get_filename(dataset, experiment, metric_name, source)
    paths_file = []
    for folder_parent in [sF.folder_scratch, sF.folder_save, f'{home}/Desktop/']:    # check both folders                       
        folder = get_folder(folder_parent, met_type = metric_type, metric_name = metric_name, source = source)   
        for timescale in cD.timescales:                          # check other timescales                     
            filename = get_filename(dataset, experiment, metric_name, source, timescale)            
            paths_file.append(f'{folder}/{filename}.nc')
    da = try_opening(paths_file, metric_name)
    if da is None:
        print('\n'.join(path for path in paths_file))
        print(f"Couldn't open {metric_name} from {dataset} {experiment} from the provided paths")
        exit()
    return da



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    # ds = xr.open_dataset(f'/scratch/w40/cb4968/metrics/wap/wap_500hpa_itcz_width/cmip6/ACCESS-CM2_wap_500hpa_itcz_width_monthly_historical_regridded_144x72.nc')
    # print(ds)

    dataset = cD.datasets[0]
    experiment = cD.experiments[0]

    getit = True
    if getit:
        da = load_metric(metric_type = 'conv_org', metric_name = f'rome_{cD.conv_percentiles[0]}thprctile', 
                         dataset = dataset, experiment = experiment)
        print(da)




