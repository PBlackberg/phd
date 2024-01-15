'''
# ------------------------
#     myFuncs - plots
# ------------------------
This script include general functions used for plotting

'''

# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import os
home = os.path.expanduser("~")



# ------------------------
#    Show/Save  Plots
# ------------------------
def save_figure(figure, folder =f'{home}/Documents/phd', filename = 'test.pdf', path = ''):
    ''' Basic plot saving function '''
    if folder and filename:
        path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    figure.savefig(path)


def save_plot(switch= {'save_folder_desktop': False, 'save_test_desktop': False,  'save_test_cwd': True}, fig = '', home = home, filename = 'test'):
    ''' Saves figure to desktop or cwd '''
    for save_type in [k for k, v in switch.items() if v]:
        save_figure(fig, folder = f'{home}/Desktop/plots', filename = f'{filename}.pdf')            if save_type == 'save_folder_desktop' else None
        save_figure(fig, folder = f'{home}/Desktop', filename = 'test.pdf')                         if save_type == 'save_test_desktop'   else None
        save_figure(fig, folder = f'{os.getcwd()}', filename = 'test.png')                          if save_type == 'save_test_cwd'       else None


def show_plot(fig, show_type = 'cycle', cycle_time = 0.5):
    ''' If using this on supercomputer, x11 forwarding is required with XQuartz installed on your computer '''
    if show_type == 'cycle':
        plt.ion()
        plt.show()
        plt.pause(cycle_time)
        plt.close(fig)
        plt.ioff()
    elif show_type == 'save_cwd':
        save_plot(fig = fig) 
        return True
    elif show_type == 'show':
        plt.show()
        return True



# ------------------------
#         Plots
# ------------------------
# ------------------------------------------------------------------------------------------ Limits --------------------------------------------------------------------------------------------------- #
def find_limits(switchM, metric_class, func = save_plot,   # dummy function (use metric function when calling in plot script)
                quantileWithin_low = 0, quantileWithin_high = 1, 
                quantileBetween_low = 0, quantileBetween_high=1, 
                vmin = '', vmax = ''): # could pot use , *args, **kwargs):    
    ''' If vmin and vmax is not set, the specified quantile values are used as limits '''
    if vmin == '' and vmax == '':
        vmin_list, vmax_list = [], []
        for dataset in mV.datasets:
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



# ----------------------------------------------------------------------------------------------- Scatter plot --------------------------------------------------------------------------------------------------- #
def calc_meanInBins(x, y, binwidth_frac=100):
    bin_width = (x.max() - x.min())/binwidth_frac # Each bin is one percent of the range of x values
    bins = np.arange(x.min(), x.max() + bin_width, bin_width)
    y_bins = []
    for i in np.arange(0,len(bins)-1):
        y_bins.append(y.where((x>=bins[i]) & (x<bins[i+1])).mean())
    return bins, y_bins

def plot_slope(x, y, ax, color):
    ''' Line of best fit (linear) '''
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    ax.plot(x, y_fit, color=color)

def plot_axScatter(switch, ax, x, y, metric_classY, binwidth_frac=100):
    h = None
    for plot_type in [k for k, v in switch.items() if v]:
        ax.scatter(x, y, facecolors='none', edgecolor= metric_classY.color)                                                         if plot_type == 'scatter'       else None
        h = ax.hist2d(x,y,[20,20], range=[[np.nanmin(x), np.nanmax(x)], [np.nanmin(y), np.nanmax(y)]], cmap = metric_classY.cmap)   if plot_type == 'density_map'   else h
        plot_slope(x, y, ax, metric_classY.color)                                                                                                        if plot_type == 'slope'       else None
        if plot_type == 'bin_trend':
            bins, y_bins = calc_meanInBins(x, y, binwidth_frac)            
            ax.plot(bins[:-1], y_bins, color = metric_classY.color)                                              
    return h

def plot_scatter(switch, metric_classX, metric_classY, x, y, metric_title, axtitle, xmin, xmax, ymin, ymax):
    fig, ax = create_figure(width = 9, height = 6)
    fig_title = f'{metric_classX.name} and {metric_classY.name} ({metric_title})'
    h = plot_axScatter(switch, ax, x, y, metric_classY)
    res= stats.pearsonr(x,y)
    placement = (0.825, 0.05) if res[0]>0 else (0.825, 0.9) 
    ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', xytext=placement, textcoords='axes fraction', fontsize = 12, color = 'r') if res[1]<=0.05 else None      
    move_col(ax, -0.03)
    move_row(ax, 0.01)
    scale_ax_x(ax, 0.95)
    scale_ax_y(ax, 1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12) 
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    ax.xaxis.set_major_formatter(formatter)
    plot_xlabel(fig, ax, metric_classX.label, pad = 0.09,   fontsize = 12)
    plot_ylabel(fig, ax, metric_classY.label, pad = 0.075, fontsize = 12)
    plot_axtitle(fig, ax, axtitle, xpad = 0.005, ypad = 0.01, fontsize = 12)
    fig.text(0.5, 0.95, fig_title, ha = 'center', fontsize = 15.5, transform=fig.transFigure)
    cbar_right_of_axis(fig, ax, h[3], width_frac= 0.05, height_frac=1, pad=0.035, numbersize = 12, cbar_label = 'months [Nb]', text_pad = 0.05, fontsize = 12) if switch['density_map'] else None
    return fig

