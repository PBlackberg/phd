import numpy as np
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import time
from functools import wraps

# ------------------------------------------------------- functions for common calculation --------------------------------------------------------------------------------------------------- #

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

def find_limits(switch, datasets, metric, func = resample_timeMean, 
                quantileWithin_low = 0, quantileWithin_high = 1, 
                quantileBetween_low = 0, quantileBetween_high=1, 
                vmin = '', vmax = ''):    
    if vmin == '' and vmax == '':
        vmin_list, vmax_list = [], []
        for dataset in datasets:
            data, _, _ = func(switch, dataset, metric)
            vmin_list, vmax_list = np.append(vmin_list, np.nanquantile(data, quantileWithin_low)), np.append(vmax_list, np.nanquantile(data, quantileWithin_high))
        return np.nanquantile(vmin_list, quantileBetween_low), np.nanquantile(vmax_list, quantileBetween_high)
    else:
        return vmin, vmax


# ------------------------------------------------------- functions for common operations --------------------------------------------------------------------------------------------------- #

def get_super(x):
    ''' For adding superscripts in strings (input is string) '''
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

def save_file(data, folder='', filename='', path = ''):
    ''' Saves file to specified folder and filename, or path '''
    if folder and filename:
        path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    data.to_netcdf(path)

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

# ------------------------------------------------------------ functions for checking available data ----------------------------------------------------------------------------------------------------- #

def find_list_source(datasets, models_cmip5, models_cmip6, observations):
    ''' Determining source of dataset list (for figures) '''
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
    ''' Indicate if there is observations in the dataset list (for figures) '''
    for dataset in datasets:
        if dataset in observations:
            return '_withObs'
    return ''

def find_source(dataset, models_cmip5, models_cmip6, observations):
    '''Determining source of dataset '''
    source = 'cmip5' if np.isin(models_cmip5, dataset).any() else 'test'      
    source = 'cmip6' if np.isin(models_cmip6, dataset).any() else source         
    source = 'obs' if np.isin(observations, dataset).any() else source
    return source

def data_available(source, dataset, experiment):
    ''' Check if dataset has variable '''
    if [source, experiment] == ['cmip5', 'ssp585'] or [source, experiment] == ['cmip6', 'rcp85']: # different warm scenario names for cmip5, cmip6
        return False
    if not experiment and not source in ['obs']: # When looping 'no experiment', only run obs
        return False
    if experiment and source in ['obs']: # when looping experiment, only run models 
        return False
    return True

# ------------------------------------------------------------ functions for getting available metric specs ----------------------------------------------------------------------------------------------------- #

class metric_class():
    ''' Gives metric: name (of saved dataset), option (data array in dataset), label, cmap, color
        (used for plots of calculated metrics)
    '''
    def __init__(self, variable_type, name, option, cmap, label, color='k'):
        self.variable_type = variable_type
        self.name   = name
        self.option = option
        self.label  = label
        self.cmap   = cmap
        self.color  = color

def reg(switch):
    region = ''
    region = '_d' if switch['descent'] else region
    region = '_a' if switch['ascent']  else region
    return region

def thres(switch):
    threshold = ''
    threshold = '_fixed_area' if switch['fixed area'] else threshold
    return threshold

def get_metric_object(switch):
    ''' list of metric: name (of saved dataset), option (data array in dataset), label, cmap, color
        Used for plots of metrics
    '''
    variable_type, name, option, label, cmap, color = [None, None, None, 'Greys', None, 'k']
    keys = [k for k, v in switch.items() if v]  # list of True keys
    for key in keys: # loop over true keys
        # -------------
        # precipitation
        # -------------
        variable_type, name, option, label, cmap, color = ['pr', 'pr',                   key, 'pr [mm day{}]'.format(get_super('-1')), 'Blues', 'b'] if key in ['pr']                                   else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = ['pr', 'rxday_pr',             key, 'pr [mm day{}]'.format(get_super('-1')), 'Blues', 'b'] if key in ['rx1day_pr','rx5day_pr']                else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = ['pr', 'rxday_pr_sMean',       key, 'pr [mm day{}]'.format(get_super('-1')), 'Blues', 'b'] if key in ['rx1day_pr_sMean','rx5day_pr_sMean']    else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = ['pr', 'percentiles_pr',       key, 'pr [mm day{}]'.format(get_super('-1')), 'Blues', 'b'] if key in ['pr95','pr97','pr99']                   else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = ['pr', 'percentiles_pr_sMean', key, 'pr [mm day{}]'.format(get_super('-1')), 'Blues', 'b'] if key in ['pr95_sMean','pr97_sMean','pr99_sMean'] else [variable_type, name, option, label, cmap, color]

        # ------------
        # organization
        # ------------
        variable_type, name, option, label, cmap, color = ['org', f'{key}{thres(switch)}', key, 'obj [binary]',                       cmap, color] if key in ['obj']            else [variable_type, name, option, label, cmap, color]          
        variable_type, name, option, label, cmap, color = ['org', f'{key}{thres(switch)}', key, 'ROME [km{}]'.format(get_super('2')), cmap, color] if key in ['rome', 'rome_n'] else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = ['org', f'{key}{thres(switch)}', key, 'number index [Nb]',                  cmap, color] if key in ['ni']             else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = ['org', f'ni{thres(switch)}',    key, 'areafraction [%]',                   cmap, color] if key in ['areafraction']   else [variable_type, name, option, label, cmap, color]

        # -----------------------------
        # large-scale environment state
        # -----------------------------
        variable_type, name, option, label, cmap, color = [None, key,                   key,                   'ECS [K]',                                 'Reds',     'r']   if key in ['ecs'] else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = [key,  f'{key}{reg(switch)}', f'{key}{reg(switch)}', 'temp. [\u00B0C]',                         'coolwarm', 'r'] if key in ['tas'] else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = [key,  f'{key}{reg(switch)}', f'{key}{reg(switch)}', 'rel. humiid. [%]',                        'Greens',   'g'] if key in ['hur'] else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = [key,  f'{key}{reg(switch)}', f'{key}{reg(switch)}', 'wap [hPa day{}]'.format(get_super('-1')), 'RdBu_r',   color] if key in ['wap'] else [variable_type, name, option, label, cmap, color]

        # ----------
        # Radiation
        # ----------
        variable_type, name, option, label, cmap, color = ['lw', f'{key}{reg(switch)}', f'{key}{reg(switch)}', 'OLR [W m{}]'.format(get_super('-2')), 'Purples', 'purple'] if key in ['rlut'] else [variable_type, name, option, label, cmap, color]
                    
        # ---------
        #  clouds
        # ---------
        variable_type, name, option, label, cmap, color = ['cl', f'cl{key}{reg(switch)}', f'cl{key}{reg(switch)}', 'cloud fraction [%]', 'Blues', 'b'] if key == 'lcf' else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = ['cl', f'cl{key}{reg(switch)}', f'cl{key}{reg(switch)}', 'cloud fraction [%]', 'Blues', 'b'] if key == 'hcf' else [variable_type, name, option, label, cmap, color]

        # -------------------
        # Moist static energy
        # -------------------
        variable_type, name, option, label, cmap, color = [key,  key, key, 'spec. humiid. [%]', 'Greens', 'g'] if key in ['hus'] else [variable_type, name, option, label, cmap, color]
   
    # ---------
    # Settings
    # ---------
    cmap = 'Reds'                                                      if switch['descent'] and switch['wap'] else cmap
    cmap = 'Blues'                                                     if switch['ascent']  and switch['wap'] else cmap
    # cmap = 'Reds'
    for key in keys: # loop over true keys
        cmap = 'RdBu_r'                                                    if key == 'change with warming' else cmap
        label = '{} K{}{}'.format(label[:-1], get_super('-1'), label[-1:]) if key == 'per kelvin' else label
    
    # cmap, color = 'Reds', 'r'
    return metric_class(variable_type, name, option, cmap, label, color)


class variable_class():
    ''' Gives variable details (name, option, label, cmap)
        (Used for animation of fields)
    '''
    def __init__(self, variable_type, name, cmap, label):
        self.variable_type = variable_type
        self.name   = name
        self.label  = label
        self.cmap   = cmap

def get_variable_object(switch):
    ''' list of variable: name (of saved dataset), option (data array in dataset), label, cmap, color
        Used for animation of fields
    '''
    variable_type, name, label, cmap = [None, None, 'Greys']
    keys = [k for k, v in switch.items() if v]  # list of True keys
    for key in keys: # loop over true keys
        variable_type, name, label, cmap = [key,  key, 'pr [mm day{}]'.format(get_super('-1')), 'Blues']     if key in ['pr']   else [variable_type, name, label, cmap] 
        variable_type, name, label, cmap = [key,  key, 'pr [mm day{}]'.format(get_super('-1')), 'Reds']      if key in ['pr']   else [variable_type, name, label, cmap] 
        variable_type, name, label, cmap = [key,  key, 'rel. humiid. [%]',                      'Greens']    if key in ['hur']  else [variable_type, name, label, cmap] 
        variable_type, name, label, cmap = ['lw', key, 'OLR [W m{}]'.format(get_super('-2')),   'Purples',]  if key in ['rlut'] else [variable_type, name, label, cmap] 


































