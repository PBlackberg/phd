import xarray as xr
import numpy as np

import os

# ---------------------------------------------------------------------------------------- Load/save data ----------------------------------------------------------------------------------------------------------#

def save_file(data, folder, filename):
    ''' Saves file to specified folder and filename '''
    os.makedirs(folder, exist_ok=True)
    path = folder + '/' + filename
    if os.path.exists(path):
        os.remove(path)    
    data.to_netcdf(path)
    return

def save_figure(figure, folder, filename):
    ''' Save figure to specified folder and filename '''
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)

    if os.path.exists(path):
        os.remove(path)    
    figure.savefig(path)
    return


def save_sample_data(folder_save, variable_name, data, source, dataset, experiment='historical', timescale='monthly', resolution='regridded'):
    ''' Save sample data (gadi) '''
    folder = f'{folder_save}/sample_data/{source}'
    os.makedirs(folder, exist_ok=True)
    filename = f'{dataset}_{variable_name}_{timescale}_{experiment}_{resolution}.nc'

    save_file(data, folder, filename)
    return


def load_sample_data(variable_name, dataset, experiment='historical', timescale='monthly', resolution='regridded', folder_load = f'{os.path.expanduser("~")}/Documents/data'):
    ''' Load saved sample data'''
    data_sources = ['cmip5', 'cmip6', 'obs']

    for source in data_sources:
        folder = f'{folder_load}/{variable_name}/sample_data/{source}'
        filename = f'{dataset}_{variable_name}_{timescale}_{experiment}_{resolution}.nc'
        file_path = os.path.join(folder, filename)
        try:
            ds = xr.open_dataset(file_path)
            return ds
        except FileNotFoundError:
            continue
    print(f'Error: no file at {file_path}')
    return None


def save_metric(metric, data, folder_save, source, dataset, experiment='historical', resolution='regridded'):
    ''' Save calculated metric to file '''
    folder = f'{folder_save}/metrics/{metric}/{source}'
    os.makedirs(folder, exist_ok=True)
    filename = f'{dataset}_{metric}_{experiment}_{resolution}.nc'
    
    save_file(data, folder, filename)
    return

def load_metric(variable_name, metric, dataset, experiment='historical', folder_load=os.path.expanduser("~") + '/Documents/data', resolution='regridded'):
    ''' Load metric data '''
    data_sources = ['cmip5', 'cmip6', 'obs']

    for source in data_sources:
        folder = f'{folder_load}/{variable_name}/metrics/{metric}/{source}'
        filename = f'{dataset}_{metric}_{experiment}_{resolution}.nc'
        file_path = os.path.join(folder, filename)

        try:
            ds = xr.open_dataset(file_path)
            return ds
        except FileNotFoundError:
            continue

    print(f"Error: no file found for {dataset} - {metric}, example: {file_path}")
    return None


def save_metric_figure(name, metric, figure, folder_save, source, resolution='regridded'):
    ''' Save calculated metric to file '''
    folder = f'{folder_save}/figures/{metric}/{source}'
    os.makedirs(folder, exist_ok=True)
    filename = f'{name}_{resolution}.pdf'
    
    save_figure(figure, folder, filename)
    return None



# ------------------------------------------------------------------------------------- common operations functions --------------------------------------------------------------------------------------------------- #

def snapshot(da):
    ''' Take a snapshot from a timestep of the dataset '''
    return da.isel(time=0)

def tMean(da):
    ''' Calculate time-mean '''
    return da.mean(dim='time', keep_attrs=True)

def sMean(da):
    ''' Calculate area-weighted spatial mean '''
    aWeights = np.cos(np.deg2rad(da.lat))
    return da.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)

def mean(da):
    ''' Calculate spatial mean, then time-mean '''
    return tMean(sMean(da))

def pick_region(da, dataset, experiment = 'historical', region = 'total', timeMean_option='monthly'):
    ''' Pick out data in regions of ascent/descent based on 500 hPa vertical pressure velocity (wap)'''
    wap500 = get_wap(dataset, experiment)['wap'].sel(plev = 500e2)
    if 'time' in da.dims:
        wap500 = resample_timeMean(wap500, timeMean_option)
        wap500 = wap500.assign_coords(time=da.time)
    else: 
        wap500 = wap500.mean(dim='time') # when looking at variables in regions of mean descent
    if region == 'total':
        pass
    elif region == 'descent':
        da = da.where(wap500>0)
    elif region == 'ascent':
        da = da.where(wap500<0)
    return da

def monthly_clim(da):
    ''' Creates a data array with the climatology of each month  '''
    year = da.time.dt.year
    month = da.time.dt.month
    da = da.assign_coords(year=("time", year.data), month=("time", month.data))
    return da.set_index(time=("year", "month")).unstack("time") # reshape the array to ("month", "year")

def resample_timeMean(da, timeMean_option=''):
    ''' Resample data to specified timescale [annual, seasonal, monthly, daily]'''
    if timeMean_option == 'annual' and len(da) >= 100:
        da = da.resample(time='Y').mean(dim='time', keep_attrs=True)

    elif timeMean_option == 'seasonal' and len(da) >= 100:
        da = da.resample(time='QS-DEC').mean(dim="time")
        da = to_monthly(da)
        da = da.rename({'month':'season'})
        da = da.assign_coords(season=["MAM", "JJA", "SON", "DJF"])
        da = da.isel(year=slice(1, None))

    elif timeMean_option == 'monthly' and len(da) > 360:
        da = da.resample(time='M').mean(dim='time', keep_attrs=True)

    elif timeMean_option == 'daily' or not timeMean_option:
        pass
    return da

def pick_region(data, dataset, experiment = 'historical', region = 'descent'):
    ''' Picks out total region, region of descent, or region of ascent based on vertical pressure velocity at 500 hPa'''
    wap = get_dsvariable('wap500', dataset, experiment)['wap500']
    if 'time' in data.dims:
        wap = wap.assign_coords(time=data.time)
    else:
        wap = wap.mean(dim='time')

    if region == 'total':
        pass
    elif region == 'descent':
        data = data.where(wap>0)
    elif region == 'ascent':
        data = data.where(wap<0)
    return data
    
def get_super(x):
    ''' For adding superscripts in strings '''
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)








# --------------------------------------------------------------------------   saved for the moment   ---------------------------------------------------------------------------------- #


# def get_dsvariable(variable, dataset, experiment = 'historical', home = os.path.expanduser("~") + '/Documents', resolution='regridded', timescale = 'monthly'):
#     ''' Load saved variable '''
#     folder = '{}/data/CMIP5/ds_cmip5_{}/{}'.format(home, resolution, dataset)
#     filename = dataset + '_' + variable + '_' + timescale +'_' + experiment + '_' + resolution + '.nc'
#     path_cmip5 = os.path.join(folder, filename)
#     try:
#         ds = xr.open_dataset(path_cmip5)
#     except FileNotFoundError:

#         try:
#             folder = '{}/data/CMIP6/ds_cmip6_{}/{}'.format(home, resolution, dataset)
#             filename = dataset + '_' + variable + '_' + timescale +'_' + experiment + '_' + resolution + '.nc'
#             path_cmip6 = os.path.join(folder, filename)
#             ds = xr.open_dataset(path_cmip6)
#         except FileNotFoundError:

#             try:
#                 folder = '{}/data/obs/ds_obs_{}/{}'.format(home, resolution, dataset)
#                 filename = dataset + '_' + variable + '_' + timescale +'_' + resolution + '.nc'
#                 path_obs = os.path.join(folder, filename)
#                 ds = xr.open_dataset(path_obs)
#             except FileNotFoundError:
#                 print(f"Error: no file at {path_cmip5}, {path_cmip6}, or {path_obs}")
#     return ds





# def get_metric(metric, dataset, experiment='historical', home=os.path.expanduser("~") + '/Documents', resolution='regridded'):
#     ''' load   '''

#     folder = '{}/data/CMIP5/metrics_cmip5_{}/{}'.format(home, resolution, dataset)
#     filename = dataset + '_' + metric + '_' + experiment + '_' + resolution + '.nc'
#     path_cmip5 = os.path.join(folder, filename)

#     folder = '{}/data/CMIP6/metrics_cmip6_{}/{}'.format(home, resolution, dataset)
#     filename = dataset + '_' + metric + '_' + experiment + '_' + resolution + '.nc'
#     path_cmip6 = os.path.join(folder, filename)

#     folder = '{}/data/obs/ds_obs_{}/{}'.format(home, resolution, dataset)
#     filename = dataset + '_' + metric + '_' + resolution + '.nc'
#     path_obs = os.path.join(folder, filename)

#     try:
#         ds = xr.open_dataset(path_cmip5)
#     except FileNotFoundError:
#         try:
#             ds = xr.open_dataset(path_cmip6)
#         except FileNotFoundError:
#             try:
#                 ds = xr.open_dataset(path_obs)
#             except FileNotFoundError:
#                 print(f"Error: no file at {path_cmip5}, {path_cmip6}, or {path_obs}")
#     return ds



