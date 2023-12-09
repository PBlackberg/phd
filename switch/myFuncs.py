'''
# ------------------------
#       myFuncs
# ------------------------
This script has functions that are communly used in other scripts, including

operations -    (ex: looping through datasets/experiments, load/save data, time functions)
Calculation -   (ex: connect lon boundary, calculate spherical distance)
Plots -         (ex: plot figure / subplots, move column and rows)
'''


# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cftime
import datetime
import time
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from functools import wraps 
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

# ------------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util')
import constructed_fields as cF                 # manually created scripts for testing
import get_data as gD                           # for loading data from supercomp    
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV                             # list of datasets to use    
import myClasses as mC                          # list of metrics to choose from



# ------------------------
#       Operations
# ------------------------
# -------------------------------------------------------------------------- Identify source of dataset ----------------------------------------------------------------------------------------------------- #
def find_source(dataset, models_cmip5 = mV.models_cmip5, models_cmip6 = mV.models_cmip6, observations = mV.observations):
    '''Determining source of dataset '''
    source = 'cmip5' if np.isin(models_cmip5, dataset).any() else None      
    source = 'cmip6' if np.isin(models_cmip6, dataset).any() else source         
    source = 'test'  if np.isin(mV.test_fields, dataset).any()        else source     
    source = 'obs'   if np.isin(observations, dataset).any() else source
    return source

def find_list_source(datasets, models_cmip5 = mV.models_cmip5, models_cmip6 = mV.models_cmip6, observations = mV.observations):
    ''' Determining source of dataset list (for plots) '''
    sources = set()
    for dataset in datasets:
        sources.add('cmip5') if dataset in models_cmip5 else None
        sources.add('cmip6') if dataset in models_cmip6 else None
        sources.add('obs')   if dataset in observations else None
    list_source = 'cmip5' if 'cmip5' in sources else 'test'
    list_source = 'cmip6' if 'cmip6' in sources else list_source
    list_source = 'obs'   if 'obs'   in sources else list_source
    list_source = 'mixed' if 'cmip5' in sources and 'cmip6' in sources else list_source
    return list_source

def find_ifWithObs(datasets, observations):
    ''' Indicate if there is observations in the dataset list (for plots) '''
    for dataset in datasets:
        if dataset in observations:
            return '_withObs'
    return ''


# --------------------------------------------------------------------------------------- Loading --------------------------------------------------------------------------------------------------- #
def data_available(source = '', dataset = '', experiment = '', var = '', switch = {'ocean_mask': False}, resolution = 'regridded'):
    ''' Check if dataset has variable. Returning False skips the loop in calc '''
    if [source, experiment] == ['cmip5', 'ssp585'] or [source, experiment] == ['cmip6', 'rcp85']: # only run fitting scenario for cmip version
        return  False
    if not experiment and not source in ['obs', 'test']:                                          # only run obs or test data for experiment == ''
        return False
    if experiment and source in ['obs']:                                                          # only run models when experiment ~= '' 
        return False
    
    if var in ['lcf', 'hcf', 'cl', 
               'ds_cl', 'cl_p_hybrid', 'p_hybrid'] \
        and dataset in ['INM-CM5-0', 'KIOST-ESM', 'EC-Earth3', 'INM-CM4-8', 
                        'CNRM-CM6-1-HR', 'GFDL-ESM4']:                                            # Some models do not have cloud variable
        print(f'No {var} data for this dataset')
        return False                                                                   
    return True

def load_variable(switch = {'constructed_fields': False, 'sample_data': True, 'gadi_data': False}, var = 'pr', 
                    dataset = 'random', experiment = mV.experiments[0]):
    ''' Loading variable data.

        Sometimes sections of years of a dataset will be used instead of the full data ex: if dataset = GPCP_1998-2010 (for obsservations) 
        (There is a double trend in high percentile precipitation rate for the first 12 years of the data (that affects the area picked out by the time-mean percentile threshold)'''
    dataset_alt = dataset.split('_')[0] if '_' in dataset else dataset    
    var = 'pr'  if var == 'var_2d' else var # for testing
    var = 'hur' if var == 'var_3d' else var # for testing

    source = find_source(dataset_alt)                                                            
    da = cF.get_cF_var(dataset_alt, var)                                                                                                                            if switch['constructed_fields']             else None
    da = xr.open_dataset(f'{mV.folder_save[0]}/sample_data/{var}/{source}/{dataset_alt}_{var}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc')[f'{var}']    if switch['sample_data']                    else da  
    da = gD.get_var_data(source, dataset_alt, experiment, var)                                                                                                      if switch['gadi_data']                      else da
    if '_' in dataset:                                                  
        start_year, end_year = dataset.split('_')[1].split('-')
        da = da.sel(time= slice(start_year, end_year))
    return da

def load_metric(metric_class, folder_load = mV.folder_save[0], source = find_source(''), dataset = 'random', timescale = mV.timescales[0], experiment = mV.experiments[0], resolution = mV.resolutions[0]):
    path = f'{folder_load}/metrics/{metric_class.var_type}/{metric_class.name}/{source}/{dataset}_{metric_class.name}_{timescale}_{experiment}_{resolution}.nc'
    # print(path)   # for debugging
    ds = xr.open_dataset(path)     
    da = ds[f'{metric_class.name}']
    return da

def convert_to_datetime(dates, data):
    ''' Some models have cftime datetime format which isn't very nice for plotting.

        Further, models have different calendars. When converting to datetime objects only standard calendar dates are valid.
        In some models the extra day in leap-years is missing. In some models all months have 30 days.
        Calling this function removes the extra days with invalid datetime days. '''
    if isinstance(dates[0], cftime.datetime):
        dates_new, data_new = [], []
        for i, date in enumerate(dates):
            year, month, day = date.year, date.month, date.day
            if month == 2 and day > 29:                                                     # remove 30th of feb                                                
                continue                                                                
            if not (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) and day > 28:   # remove 29th of feb in non-leap years
                continue
            dates_new.append(datetime.datetime(year, month, day))
            data_new.append(data[i])
        data = xr.DataArray(data_new, coords={'time': dates_new}, dims=['time'])
    else:
        dates_new = pd.to_datetime(dates)
    return dates_new, data

def run_experiment(dataset, var = 'pr'):
    for experiment in mV.experiments:
        if not data_available(find_source(dataset), dataset, experiment, var):
            continue
        print(f'\t\t {experiment}') if experiment else print(f'\t observational dataset')
        yield dataset, experiment
        
def run_dataset(var = 'pr'):
    for dataset in mV.datasets:
        print(f'\t{dataset} ({find_source(dataset)})')
        yield from run_experiment(dataset, var = '')


# --------------------------------------------------------------------------------------- Saving --------------------------------------------------------------------------------------------------- #
def save_file(data, folder=f'{home}/Documents/phd', filename='test.nc', path = ''):
    ''' Basic saving function '''
    if folder and filename:
        path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    data.to_netcdf(path)

def save_structured(dataset, experiment, da, metric_name, folder, var):
    ''' Saves in variable/metric specific folders '''
    ds = xr.Dataset(data_vars = {metric_name: da})
    folder = f'{mV.folder_save[0]}/{folder}/{var}/{metric_name}/{find_source(dataset)}'
    filename = f'{dataset}_{metric_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'
    save_file(ds, folder, filename)

def save_figure(figure, folder =f'{home}/Documents/phd', filename = 'test.pdf', path = ''):
    ''' Basic plot saving function '''
    if folder and filename:
        path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    figure.savefig(path)

def save_plot(switch= {'save_test_desktop': True, 'save_folder_desktop': False, 'save_folder_cwd': False}, 
              fig = '', home = home, filename = 'test'):
    ''' Saves figure to desktop or cwd '''
    for save_type in [k for k, v in switch.items() if v]:
        save_figure(fig, f'{home}/Desktop', 'test.pdf')                         if save_type == 'save_test_desktop'   else None
        save_figure(fig, f'{home}/Desktop/plots', f'{filename}.pdf')            if save_type == 'save_folder_desktop' else None
        save_figure(fig, f'{os.getcwd()}/test/plot_test', f'{filename}.png')    if save_type == 'save_folder_cwd'     else None


# --------------------------------------------------------------------------------------- Decorators --------------------------------------------------------------------------------------------------- #
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ''' prints start and end of function call, with time taken for the function to finish '''
        print(f'{func.__name__} started')
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_taken = end_time - start_time
        print(f'{func.__name__} took {time_taken/60:.2f} minutes')
        return result
    return wrapper



# ------------------------
#      Calculation
# ------------------------
# ----------------------------------------------------------------- For objects (contiguous convective regions) --------------------------------------------------------------------------------------------------- #
def connect_boundary(da):
    ''' Connect objects across boundary 
    Objects that touch across lon=0, lon=360 boundary are the same object.
    Takes array(lat, lon)) '''
    s = np.shape(da)
    for row in np.arange(0,s[0]):
        if da[row,0]>0 and da[row,-1]>0:
            da[da==da[row,0]] = min(da[row,0],da[row,-1])
            da[da==da[row,-1]] = min(da[row,0],da[row,-1])
    return da

def haversine_dist(lat1, lon1, lat2, lon2):
    '''Great circle distance (from Haversine formula) (used for distance between objects)
    h = sin^2(phi_1 - phi_2) + (cos(phi_1)cos(phi_2))sin^2(lambda_1 - lambda_2)
    (1) h = sin(theta/2)^2
    (2) theta = d_{great circle} / R    (central angle, theta)
    (1) in (2) and rearrange for d gives
    d = R * sin^-1(sqrt(h))*2 

    where 
    phi -latitutde
    lambda - longitude
    (Takes vectorized input) '''
    R = 6371 # radius of earth in km
    lat1 = np.deg2rad(lat1)                       
    lon1 = np.deg2rad(lon1-180) # function requires lon [-180 to 180]
    lat2 = np.deg2rad(lat2)                       
    lon2 = np.deg2rad(lon2-180)
    
    h = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2) * np.sin((lon2 - lon1)/2)**2 # Haversine formula
    return 2 * R * np.arcsin(np.sqrt(h))


# -------------------------------------------------------------------------------------- Other --------------------------------------------------------------------------------------------------- #
def monthly_clim(da):
    ''' Creates a data array with the climatology of each month  '''
    year = da.time.dt.year
    month = da.time.dt.month
    da = da.assign_coords(year=("time", year.data), month=("time", month.data))
    return da.set_index(time=("year", "month")).unstack("time") # reshape the array to ("month", "year")

def resample_timeMean(da, timeMean_option=''):
    ''' Resample data to specified timescale [annual, seasonal, monthly, daily] '''
    if timeMean_option == 'annual' and len(da) >= 100:
        da = da.resample(time='Y').mean(dim='time', keep_attrs=True)
    elif timeMean_option == 'seasonal' and len(da) >= 100:
        da = da.resample(time='QS-DEC').mean(dim="time")
        da = monthly_clim(da)
        da = da.rename({'month':'season'})
        da = da.assign_coords(season=["MAM", "JJA", "SON", "DJF"])
        da = da.isel(year=slice(1, None))
    elif timeMean_option == 'monthly' and len(da) > 360:
        da = da.resample(time='1MS').mean(dim='time')
    elif timeMean_option == 'daily' or not timeMean_option:
        pass
    else:
        pass
    return da



# ------------------------
#         Plots
# ------------------------
# ------------------------------------------------------------------------------------------ Limits --------------------------------------------------------------------------------------------------- #
def find_limits(switchM, datasets, metric_class, func = haversine_dist, # dummy function (use metric function when calling in plot script)
                quantileWithin_low = 0, quantileWithin_high = 1, 
                quantileBetween_low = 0, quantileBetween_high=1, 
                vmin = '', vmax = ''): # could pot use , *args, **kwargs):    
    ''' If vmin and vmax is not set, the specified quantile values are used as limits '''
    if vmin == '' and vmax == '':
        vmin_list, vmax_list = [], []
        for dataset in datasets:
            data, _, _ = func(switchM, dataset, metric_class) #, *args, **kwargs)
            vmin_list, vmax_list = np.append(vmin_list, np.nanquantile(data, quantileWithin_low)), np.append(vmax_list, np.nanquantile(data, quantileWithin_high))
        return np.nanquantile(vmin_list, quantileBetween_low), np.nanquantile(vmax_list, quantileBetween_high)
    else:
        return vmin, vmax
    

# ---------------------------------------------------------------------------------------- Format figure --------------------------------------------------------------------------------------------------- #
def create_figure(width = 12, height = 4, nrows = 1, ncols = 1):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width,height))
    return fig, axes

def move_col(ax, moveby):
    ax_position = ax.get_position()
    _, bottom, width, height = ax_position.bounds
    new_left = _ + moveby
    ax.set_position([new_left, bottom, width, height])

def move_row(ax, moveby):
    ax_position = ax.get_position()
    left, _, width, height = ax_position.bounds
    new_bottom = _ + moveby
    ax.set_position([left, new_bottom, width, height])

def scale_ax(ax, scaleby):
    ax_position = ax.get_position()
    left, bottom, _1, _2 = ax_position.bounds
    new_width = _1 * scaleby
    new_height = _2 * scaleby
    ax.set_position([left, bottom, new_width, new_height])

def scale_ax_x(ax, scaleby):
    ax_position = ax.get_position()
    left, bottom, _1, _2 = ax_position.bounds
    new_width = _1 * scaleby
    new_height = _2
    ax.set_position([left, bottom, new_width, new_height])

def scale_ax_y(ax, scaleby):
    ax_position = ax.get_position()
    left, bottom, _1, _2 = ax_position.bounds
    new_width = _1 
    new_height = _2 * scaleby
    ax.set_position([left, bottom, new_width, new_height])

def plot_xlabel(fig, ax, xlabel, pad, fontsize):
    ax_position = ax.get_position()
    lon_text_x =  ax_position.x0 + (ax_position.x1 - ax_position.x0) / 2
    lon_text_y =  ax_position.y0 - pad
    ax.text(lon_text_x, lon_text_y, xlabel, ha = 'center', fontsize = fontsize, transform=fig.transFigure)

def plot_ylabel(fig, ax, ylabel, pad, fontsize):
    ax_position = ax.get_position()
    lat_text_x = ax_position.x0 - pad
    lat_text_y = ax_position.y0 + (ax_position.y1 - ax_position.y0) / 2
    ax.text(lat_text_x, lat_text_y, ylabel, va = 'center', rotation='vertical', fontsize = fontsize, transform=fig.transFigure)

def plot_axtitle(fig, ax, title, xpad, ypad, fontsize):
    ax_position = ax.get_position()
    title_text_x = ax_position.x0 + xpad 
    title_text_y = ax_position.y1 + ypad
    ax.text(title_text_x, title_text_y, title, fontsize = fontsize, transform=fig.transFigure)

def delete_remaining_axes(fig, axes, num_subplots, nrows, ncols):
    for i in range(num_subplots, nrows * ncols):
        fig.delaxes(axes.flatten()[i])

def cbar_below_axis(fig, ax, pcm, cbar_height, pad, numbersize = 8, cbar_label = '', text_pad = 0.1):
    # colorbar position
    ax_position = ax.get_position()
    cbar_bottom = ax_position.y0 - cbar_height - pad
    cbar_left = ax_position.x0
    cbar_width = ax_position.width
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=numbersize)
    # colobar label
    cbar_text_x = ax_position.x0 + (ax_position.x1 - ax_position.x0) / 2
    cbar_text_y = cbar_bottom - text_pad
    ax.text(cbar_text_x, cbar_text_y, cbar_label, ha = 'center', fontsize = 12, transform=fig.transFigure)
    return cbar

def cbar_right_of_axis(fig, ax, pcm, width_frac, height_frac, pad, numbersize = 8, cbar_label = '', text_pad = 0.1, fontsize = 10):
    # colorbar position
    ax_position = ax.get_position()
    cbar_bottom = ax_position.y0
    cbar_left = ax_position.x1 + pad
    cbar_width = ax_position.width * width_frac
    cbar_height = ax_position.height * height_frac
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='vertical')
    cbar.ax.tick_params(labelsize=numbersize)
    # colobar label
    cbar_text_y = ax_position.y0 + (ax_position.y1 - ax_position.y0) / 2
    cbar_text_x = cbar_left + cbar_width + text_pad
    ax.text(cbar_text_x, cbar_text_y, cbar_label, rotation = 'vertical', va = 'center', fontsize = fontsize, transform=fig.transFigure)
    return cbar


# ------------------------------------------------------------------------------------------ With cartopy --------------------------------------------------------------------------------------------------- #
def create_map_figure(width = 12, height = 4, nrows = 1, ncols = 1, projection = ccrs.PlateCarree(central_longitude=180)):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width,height), subplot_kw=dict(projection=projection))
    return fig, axes

def format_ticks(ax, i = 0, num_subplots = 1, ncols = 1, col = 0, labelsize = 8, xticks = [30, 90, 150, 210, 270, 330], yticks = [-20, 0, 20]):
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_xticklabels('')
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_yticklabels('')
    if i >= num_subplots-ncols:
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.xaxis.set_tick_params(labelsize=labelsize)
    if col == 0:
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.yaxis.set_tick_params(labelsize=labelsize)
        ax.yaxis.set_ticks_position('both')

def plot_axMapScene(ax, scene, cmap, vmin = None, vmax = None, zorder = 0):
    lat = scene.lat
    lon = scene.lon
    lonm,latm = np.meshgrid(lon,lat)
    ax.add_feature(cfeat.COASTLINE)
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())    
    pcm = ax.pcolormesh(lonm,latm, scene, transform=ccrs.PlateCarree(),zorder=zorder, cmap=cmap, vmin=vmin, vmax=vmax)
    return pcm

def plot_scene(scene, cmap = 'Blues', label = '[units]', figure_title = 'test', ax_title= 'test', vmin = None, vmax = None):
    fig, ax = create_map_figure(width = 12, height = 4)
    pcm = plot_axMapScene(ax, scene, cmap, vmin = vmin, vmax = vmax)
    move_col(ax, moveby = -0.055)
    move_row(ax, moveby = 0.075)
    scale_ax(ax, scaleby = 1.15)
    cbar_below_axis(fig, ax, pcm, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = label, text_pad = 0.125)
    plot_xlabel(fig, ax, 'Lon', pad = 0.1, fontsize = 12)
    plot_ylabel(fig, ax, 'Lat', pad = 0.055, fontsize = 12)
    ax.text(0.5, 0.925, figure_title, ha = 'center', fontsize = 15, transform=fig.transFigure)
    plot_axtitle(fig, ax, ax_title, xpad = 0.005, ypad = 0.025, fontsize = 15)
    format_ticks(ax, labelsize = 11)
    return fig
    # scene = conv_regions.isel(time=1)
    # mF.plot_one_scene(scene)

def cycle_plot(fig):
    plt.ion()
    plt.show()
    plt.pause(0.5)
    plt.close(fig)
    plt.ioff()

# ----------------------------------------------------------------------------------------------- Trend plot --------------------------------------------------------------------------------------------------- #
def plot_scatter(ax, x, y, metric_class):
    h = ax.scatter(x, y, facecolors='none', edgecolor= metric_class.color)    
    return h

def plot_ax_datapointDensity(ax, x, y, metric_class):
    h = ax.hist2d(x,y,[20,20], range=[[np.nanmin(x), np.nanmax(x)], [np.nanmin(y), np.nanmax(y)]], cmap = metric_class.cmap)
    return h

def plot_ax_line(ax, x, y, color = 'b'):
    h = ax.plot(x, y, color)
    return h









