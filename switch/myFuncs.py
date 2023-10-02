import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import time
from functools import wraps



# ------------------------
#      Calculations
# ------------------------
def connect_boundary(da):
    ''' Connect objects across boundary 
    Objects that touch across lon=0, lon=360 boundary are the same object.
    Takes array(lat, lon)) 
    '''
    s = np.shape(da)
    for row in np.arange(0,s[0]):
        if da[row,0]>0 and da[row,-1]>0:
            da[da==da[row,0]] = min(da[row,0],da[row,-1])
            da[da==da[row,-1]] = min(da[row,0],da[row,-1])

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
    (Takes vectorized input)
    '''
    R = 6371 # radius of earth in km
    lat1 = np.deg2rad(lat1)                       
    lon1 = np.deg2rad(lon1-180) # function requires lon [-180 to 180]
    lat2 = np.deg2rad(lat2)                       
    lon2 = np.deg2rad(lon2-180)
    
    h = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2) * np.sin((lon2 - lon1)/2)**2 # Haversine formula
    return 2 * R * np.arcsin(np.sqrt(h))

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

def find_limits(switchM, datasets, metric_class, func = resample_timeMean, # dummy function (use function for getting metric when calling)
                quantileWithin_low = 0, quantileWithin_high = 1, 
                quantileBetween_low = 0, quantileBetween_high=1, 
                vmin = '', vmax = ''):    
    ''' If vmin and vmax is not set, the specified quantile values are used as limits '''
    if vmin == '' and vmax == '':
        vmin_list, vmax_list = [], []
        for dataset in datasets:
            data, _, _ = func(switchM, dataset, metric_class)
            vmin_list, vmax_list = np.append(vmin_list, np.nanquantile(data, quantileWithin_low)), np.append(vmax_list, np.nanquantile(data, quantileWithin_high))
        return np.nanquantile(vmin_list, quantileBetween_low), np.nanquantile(vmax_list, quantileBetween_high)
    else:
        return vmin, vmax



# ------------------------
#       Operations
# ------------------------
def save_file(data, folder='', filename='', path = ''):
    ''' Saves file to specified folder and filename, or path '''
    if folder and filename:
        path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    data.to_netcdf(path)

def save_in_structured_folders(da_calc, folder_save, var, metric_name, source, dataset, timescale, experiment, resolution):
    ds_calc = xr.Dataset(data_vars = {metric_name: da_calc})
    folder = f'{folder_save}/{var}/{metric_name}/{source}'
    filename = f'{dataset}_{metric_name}_{timescale}_{experiment}_{resolution}.nc'
    save_file(ds_calc, folder, filename)

def save_figure(figure, folder = '', filename = '', path = ''):
    ''' Save figure to specified folder and filename, or path '''
    if folder and filename:
        path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    figure.savefig(path)

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ''' wrapper '''
        print(f'{func.__name__} started')
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_taken = end_time - start_time
        print(f'{func.__name__} took {time_taken/60:.2f} minutes')
        return result
    return wrapper

def load_metric(metric_class, folder_save, source, dataset, timescale = 'daily', experiment = 'historical', resolution = 'regridded'):
    ds = xr.open_dataset(f'{folder_save}/metrics/{metric_class.var_type}/{metric_class.name}/{source}/{dataset}_{metric_class.name}_{timescale}_{experiment}_{resolution}.nc')     
    return ds

def save_plot(switch, fig, home, filename):
    save_figure(fig, f'{home}/Desktop',            'test.pdf')           if switch['save_test_desktop']   else None
    save_figure(fig, f'{home}/Desktop/plots',     f'{filename}.pdf')     if switch['save_folder_desktop'] else None
    save_figure(fig, f'{os.getcwd()}/plot_gadi_test', f'{filename}.png') if switch['save_folder_cwd']     else None



# ------------------------
#       Plotting
# ------------------------
# -------------------------------------------------------------------------------- General --------------------------------------------------------------------------------------------------- #
def create_figure(width, height, nrows = 1, ncols = 1):
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



# ----------------------------------------------------------------------------------- Cartopy --------------------------------------------------------------------------------------------------- #
def create_map_figure(width, height, nrows = 1, ncols = 1, projection = ccrs.PlateCarree(central_longitude=180)):
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

def plot_one_scene(scene, metric, figure_title = '', ax_title= '', vmin = None, vmax = None):
    fig, ax = create_map_figure(width = 12, height = 4)
    pcm = plot_axMapScene(ax, scene, metric.cmap, vmin = vmin, vmax = vmax)
    move_col(ax, moveby = -0.055)
    move_row(ax, moveby = 0.075)
    scale_ax(ax, scaleby = 1.15)
    cbar_below_axis(fig, ax, pcm, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = metric.label, text_pad = 0.125)
    plot_xlabel(fig, ax, 'Lon', pad = 0.1, fontsize = 12)
    plot_ylabel(fig, ax, 'Lat', pad = 0.055, fontsize = 12)
    plot_axtitle(fig, ax, figure_title, xpad = 0.005, ypad = 0.025, fontsize = 15)
    format_ticks(ax, labelsize = 11)
    return fig



# -------------------------------------------------------------------------------------- Trend --------------------------------------------------------------------------------------------------- #
def plot_scatter(ax, x, y, metric_class):
    h = ax.scatter(x, y, facecolors='none', edgecolor= metric_class.color)    
    return h

def plot_ax_datapointDensity(ax, x, y, metric_class):
    h = ax.hist2d(x,y,[20,20], cmap = metric_class.cmap)
    return h

def plot_ax_line(ax, x, y, metric_class):
    h = ax.plot(x, y, metric_class.color)
    return h



























