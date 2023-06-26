import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
import timeit

import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/functions')
import myFuncs as mF # imports common operators
import myVars as mV # imports common variables
import constructed_fields as cF # imports fields for testing


# -------------------------------------------------------------------------------------- Formatting axes ----------------------------------------------------------------------------------------------------- #

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

def format_ticks(ax, i = 0, num_subplots = 1, nrows = 1, col = 0, labelsize = 8, xticks = [30, 90, 150, 210, 270, 330], yticks = [-20, 0, 20]):
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_xticklabels('')
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_yticklabels('')
    if i >= num_subplots-nrows:
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.xaxis.set_tick_params(labelsize=labelsize)
    if col == 0:
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.yaxis.set_tick_params(labelsize=labelsize)
        ax.yaxis.set_ticks_position('both')

def delete_remaining_axes(fig, axes, num_subplots, nrows, ncols):
    for i in range(num_subplots, nrows * ncols):
        fig.delaxes(axes.flatten()[i])

def plot_axScene(ax, scene, cmap, vmin = None, vmax = None, zorder = 0):
    lat = scene.lat
    lon = scene.lon
    lonm,latm = np.meshgrid(lon,lat)
    ax.add_feature(cfeat.COASTLINE)
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())    
    pcm = ax.pcolormesh(lonm,latm, scene, transform=ccrs.PlateCarree(),zorder=zorder, cmap=cmap, vmin=vmin, vmax=vmax)
    return pcm

def create_map_figure(width, height, nrows = 1, ncols = 1, projection = ccrs.PlateCarree(central_longitude=180)):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width,height), subplot_kw=dict(projection=projection))
    return fig, axes


# -------------------------------------------------------------------------------------- Calculation ----------------------------------------------------------------------------------------------------- #

def get_scene(switch, variable_type, metric, metric_option, dataset, resolution, folder_save):
    if switch['climatology']:
        scene = mV.load_metric(folder_save, variable_type, metric, dataset, resolution=resolution)[metric_option]

    if switch['change with warming']:
        scene_historical = mV.load_metric(folder_save, variable_type, metric, dataset, experiment=mV.experiments[0], resolution=resolution)[metric_option]
        scene_warm = mV.load_metric(folder_save, variable_type, metric, dataset, experiment=mV.experiments[1], resolution=resolution)[metric_option]
        scene = scene_warm - scene_historical 
    return scene


def find_limits(switch, variable_type, metric, metric_option, quantileWithin_low = 0, quantileWithin_high = 1, quantileBetween_low = 0, quantileBetween_high = 1, datasets = mV.datasets, resolution = '', folder_save=''):    
    vmin_list, vmax_list = [], []
    for dataset in datasets:
        scene = get_scene(switch, variable_type, metric, metric_option, dataset, resolution, folder_save)
        vmin_list = np.append(vmin_list, np.quantile(scene, quantileWithin_low))
        vmax_list = np.append(vmax_list, np.quantile(scene, quantileWithin_high))

    vmin = np.quantile(vmin_list, quantileBetween_low)
    vmax = np.quantile(vmax_list, quantileBetween_high)

    return (vmin, vmax) if switch['climatology'] else (-vmax, vmax)


# -------------------------------------------------------------------------------------- different plots ----------------------------------------------------------------------------------------------------- #

def plot_one_scene(switch, variable_type, metric, metric_option, cmap, title, cbar_label, dataset, resolution, folder_save):
    # create figure
    fig, ax = create_map_figure(width = 12, height = 4)

    # find limits
    vmin, vmax = find_limits(switch, variable_type, metric, metric_option, datasets = [dataset], 
        quantileWithin_low = 0, 
        quantileWithin_high = 0.95, # remove extreme values from colorbar range
        resolution = resolution,
        folder_save = folder_save
        )
    
    # get scene
    scene = get_scene(switch, variable_type, metric, metric_option, dataset, resolution, folder_save)

    # plot
    pcm = plot_axScene(ax, scene, cmap, vmin = vmin, vmax = vmax)

    # adjust axis position
    move_col(ax, moveby = -0.055)
    move_row(ax, moveby = 0.075)
    scale_ax(ax, scaleby = 1.15)

    # create colorbar
    cbar_below_axis(fig, ax, pcm, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = cbar_label, text_pad = 0.125)

    # plot text
    plot_xlabel(fig, ax, xlabel='Lon', pad = 0.1, fontsize = 12)
    plot_ylabel(fig, ax, ylabel='Lat', pad = 0.055, fontsize = 12)
    plot_axtitle(fig, ax, f'{dataset}: {title}', xpad = 0.005, ypad = 0.025, fontsize=15)

    # format ticks
    format_ticks(ax, labelsize = 10)
    return fig


def plot_multiple_scenes(switch, variable_type, metric, metric_option, cmap, title, cbar_label, datasets, resolution, folder_save):
    # create figure
    nrows = 4
    ncols = 4
    fig, axes = create_map_figure(width = 14, height = 5, nrows=nrows, ncols=ncols)
    
    num_subplots = len(datasets)
    for i, dataset in enumerate(datasets):

        # find limits
        vmin, vmax = find_limits(switch, variable_type, metric, metric_option, 
            quantileWithin_low = 0, 
            quantileWithin_high = 0.90, # remove extreme values from colorbar range
            quantileBetween_low = 0, 
            quantileBetween_high = 1, # remove extreme models' range for colorbar range
            resolution = resolution,
            folder_save = folder_save
            )
        
        # get scene
        scene = get_scene(switch, variable_type, metric, metric_option, dataset, resolution, folder_save)

        # determine the row and column indices
        row = i // ncols
        col = i % ncols

        # plot
        ax = axes.flatten()[i]
        pcm = plot_axScene(ax, scene, cmap, vmin = vmin, vmax = vmax)

        # adjust axis position and size (columns and rows at a time, then adjust scale to fill figure)
        scale_ax(ax, scaleby=1.3)

        move_col(ax, moveby = -0.0825 + 0.0025) if col == 0 else None
        move_col(ax, moveby = -0.0435 + 0.0025) if col == 1 else None
        move_col(ax, moveby = - 0.005 + 0.0025) if col == 2 else None
        move_col(ax, moveby = 0.0325 + 0.0025)  if col == 3 else None
            
        move_row(ax, moveby = 0.025+0.005) if row == 0 else None
        move_row(ax, moveby = 0.04+0.005)  if row == 1 else None
        move_row(ax, moveby = 0.045+0.01)  if row == 2 else None
        move_row(ax, moveby = 0.05+0.01)   if row == 3 else None

        # Plot text
        plot_axtitle(fig, ax, dataset, xpad = 0.002, ypad = 0.0095, fontsize = 9)
        plot_xlabel(fig, ax, xlabel='Lon', pad = 0.0725, fontsize = 8) if i >= num_subplots-nrows else None
        plot_ylabel(fig, ax, ylabel='Lat', pad = 0.0375, fontsize = 8) if col == 0 else None

        # format ticks
        format_ticks(ax, i, num_subplots, nrows, col, labelsize=9)


    # fig title
    title_text_x = 0.5
    title_text_y = 0.95
    ax.text(title_text_x, title_text_y, title, ha = 'center', fontsize = 15, transform=fig.transFigure)

    # create colorbar
    cbar_position = [0.225, 0.0875, 0.60, 0.02] # [left, bottom, width, height]
    cbar_ax = fig.add_axes(cbar_position)
    cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
    # colobar label
    cbar_text_x = cbar_position[0] + cbar_position[2] / 2
    cbar_text_y = cbar_position[1]-0.07
    ax.text(cbar_text_x, cbar_text_y, cbar_label, ha = 'center', fontsize = 9, transform=fig.transFigure)

    delete_remaining_axes(fig, axes, num_subplots, nrows, ncols)
    return fig




# ------------------
#    Run script
# ------------------

def run_map_plot(switch, datasets = mV.datasets, folder_save = mV.folder_save, resolution = 'regridded'):

    if  switch['rx1day'] or switch['rx5day']:
        variable_type = 'pr'
        cmap = 'Blues'
        cbar_label = 'pr [mm day{}]'.format(mF.get_super('-1'))

        metric = 'rxday_pr_tMean' if switch['rx1day'] or switch['rx5day'] else None
        metric_option = 'rx1day' if switch['rx1day'] else 'rx5day'


    if switch['climatology']:
        title = f'{metric_option} time mean'
    else:
        title = f'{metric_option}, change with warming'
        cbar_label = cbar_label[:-1] + ' K' + mF.get_super('-1') + cbar_label[-1:] 
        cmap = 'RdBu_r'


    if switch['one scene']:
        dataset = datasets[0]
        fig = plot_one_scene(switch, variable_type, metric, metric_option, cmap, title, cbar_label, dataset, resolution, folder_save)
        source = mV.find_list_source(datasets, mV.models_cmip5, mV.models_cmip6, mV.observations)
        filename = f'{dataset}_{metric_option}' if switch['climatology'] else f'{dataset}_{metric_option}_difference' 
    else:
        fig = plot_multiple_scenes(switch, variable_type, metric, metric_option, cmap, title, cbar_label, datasets, resolution, folder_save)
        source = mV.find_list_source(datasets, mV.models_cmip5, mV.models_cmip6, mV.observations)
        with_obs = mV.find_ifWithObs(datasets, mV.observations)
        filename = f'{source}_{metric_option}{with_obs}' if switch['climatology'] else f'{source}_{metric_option}_difference' 
        
    mF.save_metric_figure(name=filename, metric=metric, figure = fig, folder_save=folder_save, source = source) if switch['save'] else None
    plt.show() if switch['show'] else None
    plt.close()



if __name__ == '__main__':

    start = timeit.default_timer()

    # choose which metrics to plot
    switch = {
        'rx1day': True, 
        'rx5day': False,

        'snapshot':False,
        'climatology': True,
        'change with warming': False,

        'one scene': False,
        'show': True,
        'save': False,
        }
    

    # plot and save figure
    run_map_plot(switch,
                 folder_save = mV.folder_save)

    stop = timeit.default_timer()
    print(f'Finshed, script finished in {round((stop-start)/60, 2)} minutes.')












































