'''
# ------------------------
#    Metric data
# ------------------------
This script saves / loads metrics
'''



# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import xarray as xr


# ------------------------------------------------------------------------------------ imported scripts --------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")

sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars as mV
import myFuncs as mF



# -----------------------
#     Get metric
# -----------------------
def get_folder(folder_parent, met_type, metric_name, source):
    folder = f'{folder_parent}/metrics/{met_type}/{metric_name}/{source}'
    # print(folder)
    return folder

def get_filename(dataset, experiment, metric_name, source, timescale = mV.timescales[0]):
    # print(metric_name)
    filename = f'{dataset}_{metric_name}_{timescale}_{experiment}_{mV.resolutions[0]}'
    if source in ['obs']:  
        filename = f'{dataset}_{metric_name}_{timescale}_{mV.obs_years[0]}_obs_{mV.resolutions[0]}'
    if mV.resolutions[0] == 'regridded':
        filename = f'{filename}_{int(360/mV.x_res)}x{int(180/mV.y_res)}'
    # print(f'filename is: {filename}')
    return filename

def save_metric(switch, met_type, dataset, experiment, metric, metric_name):
    ''' Saves in variable/metric specific folders '''
    source = mF.find_source(dataset)
    filename = get_filename(dataset, experiment, metric_name, source)
    for save_type in [k for k, v in switch.items() if v]:
        folder = get_folder(f'{home}/Desktop/{metric_name}', met_type, metric_name, source) if save_type == 'save_folder_desktop'   else None
        folder = get_folder(mV.folder_scratch, met_type, metric_name, source)               if save_type == 'save_scratch'          else folder
        folder = get_folder(mV.folder_save, met_type, metric_name, source)                  if save_type == 'save'                  else folder
        if type(folder) == str:
            mF.save_file(xr.Dataset({metric_name: metric}), folder, f'{filename}.nc')
            print(f'\t\t\t saved {metric_name} ({save_type})')
            return f'{folder}/{filename}.nc'

def try_opening(file_paths, metric_name):
    for path in file_paths:
        try:
            ds = xr.open_dataset(path)
            da = ds[metric_name]
            return da
        except:
            continue

def load_metric(metric_type, metric_name, dataset, experiment, dataset_org = 'GPCP'):
    source = mF.find_source(dataset)
    if metric_type in ['pr', 'conv_org'] and source in ['obs']: # GPCP observations used for precipitation based metrics
        dataset = dataset_org 
    filename = get_filename(dataset, experiment, metric_name, source)
    paths_file = []
    for folder_parent in [mV.folder_scratch, mV.folder_save, f'{home}/Desktop/']:    # check both folders                       
        folder = get_folder(folder_parent, met_type = metric_type, metric_name = metric_name, source = source)   
        for timescale in mV.timescales:                          # check other timescales                     
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

    dataset = mV.datasets[0]
    experiment = mV.experiments[0]

    getit = True
    if getit:
        da = load_metric(metric_type = 'conv_org', metric_name = f'rome_{mV.conv_percentiles[0]}thprctile', 
                         dataset = dataset, experiment = experiment)
        print(da)




