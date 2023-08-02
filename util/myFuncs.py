import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import time
from functools import wraps

# ------------------------------------------------------- functions for common operations --------------------------------------------------------------------------------------------------- #

def get_super(x):
    ''' For adding superscripts in strings (input is string) '''
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

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
        da = da.resample(time='M').mean(dim='time', keep_attrs=True)
    elif timeMean_option == 'daily' or not timeMean_option:
        pass
    else:
        pass
    return da

def find_limits(switch, datasets, options, metric, func, quantileWithin_low, quantileWithin_high, quantileBetween_low = 0, quantileBetween_high=1):    
    vmin_list, vmax_list = [], []
    for dataset in datasets:
        scene = func(switch, dataset, options, metric)
        vmin_list = np.append(vmin_list, np.nanquantile(scene, quantileWithin_low))
        vmax_list = np.append(vmax_list, np.nanquantile(scene, quantileWithin_high))
    vmin = np.nanquantile(vmin_list, quantileBetween_low)
    vmax = np.nanquantile(vmax_list, quantileBetween_high)
    return vmin, vmax

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ''' wrapper '''
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Function '{func.__name__}' took {time_taken/60:.2f} minutes.")
        return result
    return wrapper


# ---------------------------------------------------------------- functions for plotting --------------------------------------------------------------------------------------------------- #

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

def cbar_right_of_axis(fig, ax, pcm, width_frac, height_frac, pad, numbersize = 8, cbar_label = '', text_pad = 0.1):
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
    ax.text(cbar_text_x, cbar_text_y, cbar_label, rotation = 'vertical', va = 'center', fontsize = 10, transform=fig.transFigure)
    return cbar

# ------------------
#    For cartopy
# ------------------
def create_map_figure(width, height, nrows = 1, ncols = 1, projection = ccrs.PlateCarree(central_longitude=180)):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width,height), subplot_kw=dict(projection=projection))
    return fig, axes

def plot_axScene(ax, scene, cmap, vmin = None, vmax = None, zorder = 0):
    lat = scene.lat
    lon = scene.lon
    lonm,latm = np.meshgrid(lon,lat)
    ax.add_feature(cfeat.COASTLINE)
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())    
    pcm = ax.pcolormesh(lonm,latm, scene, transform=ccrs.PlateCarree(),zorder=zorder, cmap=cmap, vmin=vmin, vmax=vmax)
    return pcm

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




# ------------------------------------------------------------ functions for saving / loading data ----------------------------------------------------------------------------------------------------- #

# --------------------
# structure of folders 
# for metric: [folder_save]/[variable_type]/metrics/[metric]/[source]/[dataset]_[filename] ex: [folder_save]/pr/metrics/rxday/cmip6/[filename]
# for figure: [folder_save]/[variable_type]/figures/[plot_metric]/[source]/[source]_[filename] ex: [folder_save]/pr/figures/rxday_tMean/cmip6/[filename]

# structure of filename
# for metric: [dataset]_[metric]_[timescale]_[experiment]_[resolution] ex: FGOALS-g3_rxday_daily_historical_regridded.nc
# for figure_metric: [source]_[metric]_[timescale]_[resolution]  ex: cmip6_rx1day_daily_regridded.pdf 
# --------------------

def find_source(dataset, models_cmip5, models_cmip6, observations):
    '''Determining source of dataset '''
    if np.isin(models_cmip5, dataset).any():
        source = 'cmip5' 
    elif np.isin(models_cmip6, dataset).any():
        source = 'cmip6' 
    elif np.isin(observations, dataset).any():
        source = 'obs' 
    else:
        source = 'test' 
    return source

def find_list_source(datasets, models_cmip5, models_cmip6, observations):
    ''' Determining source of dataset list '''
    sources = set()
    for dataset in datasets:
        sources.add('cmip5') if dataset in models_cmip5 else None
        sources.add('cmip6') if dataset in models_cmip6 else None
        sources.add('obs') if dataset in observations else None
    if   'cmip5' in sources and 'cmip6' in sources:
         return 'mixed'
    elif 'cmip5' in sources:
         return 'cmip5'
    elif 'cmip6' in sources:
         return 'cmip6'
    else:
         return 'obs'

def find_ifWithObs(datasets, observations):
    ''' Indicate if there is observations in the dataset list (for filename of figures) '''
    for dataset in datasets:
        if dataset in observations:
            return '_withObs'
    return ''

def data_exist(model, experiment):
    ''' Check if model/project has data
    (for precipitation a model is not included if it does not have daily precipitation data)
    '''
    data_exist = 'True'
    return data_exist

def no_data(source, experiment, data_exists):
    if experiment and source in ['cmip5', 'cmip6']:
        pass
    elif not experiment and source == 'obs':
        pass
    else:
        return True

    if [source, experiment] == ['cmip5', 'ssp585'] or [source, experiment] == ['cmip6', 'rcp85']:
        return True

    if not data_exists:
        return True

def save_file(data, folder, filename):
    ''' Saves file to specified folder and filename '''
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    os.remove(path) if os.path.exists(path) else None
    data.to_netcdf(path)
    return

def save_figure(figure, folder, filename):
    ''' Save figure to specified folder and filename '''
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    os.remove(path) if os.path.exists(path) else None
    figure.savefig(path)
    return



# ------------------------------------------------------------ functions for getting available metrics ----------------------------------------------------------------------------------------------------- #

class dataset_class():
    ''' Creates object with dataset options'''
    def __init__(self, timescale, experiment, resolution):
        self.timescale = timescale
        self.experiment = experiment
        self.resolution = resolution

class metric_class():
    ''' Creates object with properties corresponding to folders where metric / figure is stored or loaded from'''
    def __init__(self, variable_type, metric, metric_option, cmap, label, color='k'):
        self.variable_type = variable_type
        self.name = metric
        self.option = metric_option
        self.color = color
        self.cmap = cmap
        self.label = label
    def get_figure_folder(self, folder_save, source):
        return f'{folder_save}/{self.variable_type}/figures/{self.name}/{source}'
    def get_metric_folder(self, folder_save, name, source):    
        return f'{folder_save}/{self.variable_type}/metrics/{name}/{source}'
    def get_filename(self, name, source, dataset, timescale, experiment, resolution):
        return f'{dataset}_{name}_{timescale}_{experiment}_{resolution}.nc' if not source == 'obs' else f'{dataset}_{name}_{timescale}_{resolution}.nc'

def pick_region(switch):
    region = ''
    region = '_d' if switch['descent'] else region
    region = '_a' if switch['ascent'] else region
    return region

def get_metric_object(switch):
    variable_type, metric, metric_option, cmap, label, color = [None, None, None, None, None, None]
    keys = [k for k, v in switch.items() if v]  # list of True keys
    for key in keys: # loop over true keys
        # precipitation metrics (pr)
        if key in ['pr', 'pr99', 'pr99_meanIn', 'rx1day_pr', 'rx5day_pr']:
            variable_type, cmap, label, color = ['pr', 'Blues', 'pr [mm day{}]'.format(get_super('-1')), 'b']
        metric, metric_option = ['pr', key] if key ==                     'pr' else [metric, metric_option]
        metric, metric_option = ['percentiles_pr', key]       if key == 'pr95' else [metric, metric_option]
        metric, metric_option = ['percentiles_pr', key]       if key == 'pr97' else [metric, metric_option]
        metric, metric_option = ['percentiles_pr', key]       if key == 'pr99' else [metric, metric_option]
        metric, metric_option = ['meanInPercentiles_pr', 'pr99'] if key == 'pr99_meanIn' else [metric, metric_option]
        metric, metric_option = ['rxday_pr', key]             if key == 'rx1day_pr' else [metric, metric_option]
        metric, metric_option = ['rxday_pr', key]             if key == 'rx5day_pr' else [metric, metric_option]

        # vertical pressure velocity (wap)
        if key in ['wap']:
            variable_type, cmap, label = ['wap', 'RdBu_r', 'wap [hPa day' + get_super('-1') +']']
            cmap = 'Reds' if switch['descent'] else cmap
            cmap = 'Blues' if switch['ascent'] else cmap
        metric, metric_option = [f'wap{pick_region(switch)}', f'wap{pick_region(switch)}'] if key == 'wap' else [metric, metric_option]

        # surface temperature (tas)
        if key in ['tas']:
            variable_type, cmap, label, color = ['tas', 'RdBu_r', 'Temperature [\u00B0C]', 'r']
        metric, metric_option = [f'tas{pick_region(switch)}', f'tas{pick_region(switch)}'] if key == 'tas' else [metric, metric_option]

        # Relative humididty (hur)
        if key in ['hur']:
            variable_type, cmap, label, color = ['hur', 'Greens', 'Relative humidity [%]', 'g']
        metric, metric_option = [f'hur{pick_region(switch)}', f'hur{pick_region(switch)}'] if key == 'hur' else [metric, metric_option]

        # longwave radiation
        if key in ['rlut']:
            variable_type, cmap, label, color = ['lw', 'Purples', 'OLR [W m' + get_super('-2') +']', 'purple']
            metric, metric_option = [f'rlut{pick_region(switch)}', f'rlut{pick_region(switch)}'] if key == 'rlut' else [metric, metric_option]

        #  cloud fraction (cl)
        if key in ['lcf', 'hcf']:
            variable_type, cmap, label, color = ['cl', 'Blues', 'Cloud fraction [%]', 'b']
            metric, metric_option = [f'lcf{pick_region(switch)}', f'lcf{pick_region(switch)}'] if key == 'lcf' else [metric, metric_option]
            metric, metric_option = [f'hcf{pick_region(switch)}', f'hcf{pick_region(switch)}'] if key == 'hcf' else [metric, metric_option]

        # Specific humidity (hus)
        if key in ['hus']:
            variable_type, cmap, label = ['hus', 'Greens', 'Specific humidity [kg/kg]']
        metric, metric_option = [f'hus{pick_region(switch)}', f'hus{pick_region(switch)}'] if key == 'hus' else [metric, metric_option]

        # organization
        if key in ['obj']:
            variable_type, cmap, label,  = 'org', 'Greys', 'binary'
            metric, metric_option = 'obj', 'o_scene' if key == 'obj' else [metric, metric_option]

        if key in ['rome']:
            variable_type, cmap, label,  = 'org', 'Greys', 'ROME [km' + get_super('2') + ']' 
            metric, metric_option = 'rome', 'rome' if key == 'rome' else [metric, metric_option]

        if key in ['rome_equal_area']:
            variable_type, cmap, label,  = 'org', 'Greys', 'ROME [km' + get_super('2') + ']' 
            metric, metric_option = 'rome_equal_area', 'rome' if key == 'rome_equal_area' else [metric, metric_option]

        if key in ['ecs']:
            metric_option, color, label = 'ecs', 'red', 'ECS [K]'

        if key in ['change with warming']:
            cmap = 'RdBu_r'
            label = '{}{} K{}'.format(label[:-1], get_super('-1'), label[-1:]) if switch['per_kelvin'] else label
    return metric_class(variable_type, metric, metric_option, cmap, label, color)







