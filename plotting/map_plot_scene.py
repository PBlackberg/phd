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


# ----------------------------------------------------------------------------------- Formatting axes for map plot ----------------------------------------------------------------------------------------------------- #

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

def get_scene(switch, variable_type, metric, metric_option, dataset, timescale, resolution, folder_load):
    source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
    if switch['climatology'] or switch['snapshot']:
        scene = mV.load_metric(folder_load, variable_type, metric, source, dataset, timescale, experiment = mV.experiments[0], resolution=resolution)[metric_option]

    if switch['change with warming']:
        scene_historical = mV.load_metric(folder_load, variable_type, metric, dataset, experiment=mV.experiments[0], resolution=resolution)[metric_option]
        scene_warm = mV.load_metric(folder_load, variable_type, metric, dataset, experiment=mV.experiments[1], resolution=resolution)[metric_option]
        scene = scene_warm - scene_historical 
    return scene


def find_limits(switch, variable_type, metric, metric_option, datasets, timescale, resolution, folder_load, quantileWithin_low, quantileWithin_high, quantileBetween_low = 0, quantileBetween_high=1):    
    vmin_list, vmax_list = [], []
    for dataset in datasets:
        scene = get_scene(switch, variable_type, metric, metric_option, dataset, timescale, resolution, folder_load)
        vmin_list = np.append(vmin_list, np.nanquantile(scene, quantileWithin_low))
        vmax_list = np.append(vmax_list, np.nanquantile(scene, quantileWithin_high))

    vmin = np.nanquantile(vmin_list, quantileBetween_low)
    vmax = np.nanquantile(vmax_list, quantileBetween_high)

    return (vmin, vmax) if switch['climatology'] or switch['snapshot'] else (-vmax, vmax)


# -------------------------------------------------------------------------------------- different plots ----------------------------------------------------------------------------------------------------- #

def plot_one_scene(switch, variable_type, metric, metric_option, cmap, title, cbar_label, dataset, timescale, resolution, folder_save):
    # Adjust position and scale of axes
    move_col_by = -0.055
    move_row_by = 0.075
    scale_ax_by = 1.15     

    # Adjust position of colorbar
    cbar_height = 0.05
    pad = 0.15
    numbersize = 12
    text_pad = 0.125 

    # Set/adjust position and size of text
    xlabel = 'Lon'
    xlabel_pad = 0.1
    ylabel='Lat'
    ylabel_pad = 0.055
    xylabel_fontsize = 12

    title = f'{dataset}: {title}'
    title_xpad = 0.005
    title_ypad = 0.025
    title_fontsize = 15

    ticklabel_size = 11

    # adjust colorbar limits
    vmin, vmax = find_limits(switch, variable_type, metric, metric_option, datasets = [dataset], timescale = timescale, resolution = resolution, folder_load = folder_save,
        quantileWithin_low = 0,    # remove extreme low values from colorbar range 
        quantileWithin_high = 1,   # remove extreme high values from colorbar range 
        )


    fig, ax = create_map_figure(width = 12, height = 4)
    scene = get_scene(switch, variable_type, metric, metric_option, dataset, timescale, resolution, folder_save)
    pcm = plot_axScene(ax, scene, cmap, vmin = vmin, vmax = vmax)
    mF.move_row(ax, moveby = move_row_by)
    mF.move_col(ax, moveby = move_col_by)
    mF.scale_ax(ax, scaleby = scale_ax_by)
    cbar_below_axis(fig, ax, pcm, cbar_height, pad, numbersize, cbar_label, text_pad)
    mF.plot_xlabel(fig, ax, xlabel, xlabel_pad, xylabel_fontsize)
    mF.plot_ylabel(fig, ax, ylabel, ylabel_pad, xylabel_fontsize)
    mF.plot_axtitle(fig, ax, title, title_xpad, title_ypad, title_fontsize)
    format_ticks(ax, labelsize = ticklabel_size)
    return fig


def plot_multiple_scenes(switch, variable_type, metric, metric_option, cmap, title, cbar_label, datasets, timescale, resolution, folder_save):
    nrows = 4
    ncols = 4
    
    # Adjust position of cols
    move_col0_by, move_col1_by, move_col2_by, move_col3_by = -0.0825 + 0.0025, -0.0435 + 0.0025, -0.005 + 0.0025, 0.0325 + 0.0025

    # Adjust position of rows
    move_row0_by = 0.025+0.005
    move_row1_by = 0.04+0.005
    move_row2_by = 0.045+0.01
    move_row3_by = 0.05+0.01
    move_row4_by = 0.0325 + 0.0025

    # Adjust scale of ax
    scale_ax_by = 1.3                                                             

    # Set position of colorbar [left, bottom, width, height]
    cbar_position = [0.225, 0.0875, 0.60, 0.02]

    # Set/adjust position and size of text
    xlabel = 'Lon'
    xlabel_pad = 0.0725
    ylabel='Lat'
    ylabel_pad = 0.0375
    xylabel_fontsize = 8

    axtitle_xpad = 0.002
    axtitle_ypad = 0.0095
    axtitle_fontsize = 9

    title_x = 0.5
    title_y = 0.95
    title_fontsize = 15

    cbar_text_x = cbar_position[0] + cbar_position[2] / 2   # In th middle of colorbar
    cbar_text_y = cbar_position[1]-0.07                     # essentially pad
    cbar_text_fontsize = 9
    ticklabel_size = 9

    # Find common limits
    vmin, vmax = find_limits(switch, variable_type, metric, metric_option, datasets, timescale, resolution, folder_load = folder_save,
        quantileWithin_low = 0,    # remove extreme low values from colorbar range 
        quantileWithin_high = 1,   # remove extreme high values from colorbar range 
        quantileBetween_low = 0,   # remove extreme low models' from colorbar range
        quantileBetween_high = 1   # remove extreme high models' from colorbar range
        )
    
    fig, axes = create_map_figure(width = 14, height = 5, nrows=nrows, ncols=ncols)
    num_subplots = len(datasets)
    for i, dataset in enumerate(datasets):
        row = i // ncols  # determine row index
        col = i % ncols   # determine col index
        ax = axes.flatten()[i]
        scene = get_scene(switch, variable_type, metric, metric_option, dataset, timescale, resolution, folder_save)
        pcm = plot_axScene(ax, scene, cmap, vmin = vmin, vmax = vmax)

        mF.move_col(ax, move_col0_by) if col == 0 else None
        mF.move_col(ax, move_col1_by) if col == 1 else None
        mF.move_col(ax, move_col2_by) if col == 2 else None
        mF.move_col(ax, move_col3_by) if col == 3 else None

        mF.move_row(ax, move_row0_by) if row == 0 else None
        mF.move_row(ax, move_row1_by) if row == 1 else None
        mF.move_row(ax, move_row2_by) if row == 2 else None
        mF.move_row(ax, move_row3_by) if row == 3 else None
        mF.move_row(ax, move_row4_by) if col == 4 else None

        mF.scale_ax(ax, scale_ax_by)

        mF.plot_xlabel(fig, ax, xlabel, xlabel_pad, xylabel_fontsize) if i >= num_subplots-nrows else None
        mF.plot_ylabel(fig, ax, ylabel, ylabel_pad, xylabel_fontsize) if col == 0 else None
        mF.plot_axtitle(fig, ax, dataset, axtitle_xpad, axtitle_ypad, axtitle_fontsize)
        format_ticks(ax, i, num_subplots, nrows, col, ticklabel_size )

    ax.text(title_x, title_y, title, ha = 'center', fontsize = title_fontsize, transform=fig.transFigure)
    cbar_ax = fig.add_axes(cbar_position)
    cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
    ax.text(cbar_text_x, cbar_text_y, cbar_label, ha = 'center', fontsize = cbar_text_fontsize , transform=fig.transFigure)
    mF.delete_remaining_axes(fig, axes, num_subplots, nrows, ncols)
    return fig



# ---------------------------------------------------------------------------------- Find the metric and units ----------------------------------------------------------------------------------------------------- #

def name_region(switch):
    if switch['descent']:
        region = '_d' 
    elif switch['ascent']:
        region = '_a' 
    else:
        region = ''
    return region

def find_general_metric_and_specify_cbar(switch):
    if switch['pr'] or switch['percentiles_pr'] or switch['rx1day_pr'] or switch['rx5day_pr']:
        variable_type = 'pr'
        cmap = 'Blues'
        cbar_label = 'pr [mm day{}]'.format(mF.get_super('-1'))
    if switch['pr']:
        metric = 'pr' 
        metric_option = metric
    if switch['percentiles_pr']:
        metric = 'percentiles_pr' 
        metric_option = 'pr97' # there is also pr95, pr99
    if  switch['rx1day_pr'] or switch['rx5day_pr']:
        metric = 'rxday_pr'
        metric_option = 'rx1day_pr' if switch['rx1day_pr'] else 'rx5day_pr'

    if switch['wap']:
        variable_type = 'wap'
        cmap = 'RdBu_r' if not switch['ascent'] and not switch['descent'] else 'Reds'
        cbar_label = 'wap [hPa day' + mF.get_super('-1') + ']'
        region = name_region(switch)
        metric = f'wap{region}'
        metric_option = metric

    if switch['tas']:
        variable_type = 'tas'
        cmap = 'RdBu_r'
        cbar_label = 'Temperature [\u00B0C]'
        region = name_region(switch)
        metric = f'tas{region}'
        metric_option = metric

    if switch['hur']:
        variable_type = 'hur'
        cmap = 'Greens'
        cbar_label = 'Relative humidity [%]'
        region = name_region(switch)
        metric = f'hur{region}'
        metric_option = metric

    if switch['rlut']:
        variable_type = 'lw'
        cmap = 'Purples'
        cbar_label = 'OLR [W m' + mF.get_super('-2') +']'
        region = name_region(switch)
        metric = f'rlut{region}'
        metric_option = metric

    if switch['lcf'] or switch['hcf']:
        variable_type = 'cl'
        cmap = 'Blues'
        cbar_label = 'cloud fraction [%]'
        region = name_region(switch)
        metric = f'lcf{region}' if switch['lcf'] else f'hcf{region}'
        metric_option = metric

    if switch['hus']:
        variable_type = 'hus'
        cmap = 'Greens'
        cbar_label = 'Specific humidity [mm]'
        region = name_region(switch)
        metric = f'hus{region}'
        metric_option = metric

    if switch['change with warming']:
        cmap = 'RdBu_r'
        cbar_label = '{}{} K{}'.format(cbar_label[:-1], mF.get_super('-1'), cbar_label[-1:]) if switch['per_kelvin'] else cbar_label
    return variable_type, metric, metric_option, cmap, cbar_label

def specify_metric_and_title(switch, metric, metric_option):
    if switch['snapshot']:
        title = f'{metric_option} snapshot'
        metric = f'{metric}_snapshot'
        metric_option = f'{metric_option}_snapshot'

    if switch['climatology'] or switch['change with warming']:
        metric = f'{metric}_tMean'
        metric_option = f'{metric_option}_tMean'
        title = f'{metric_option} time mean' if switch['climatology'] else f'{metric_option}, change with warming'
    return metric, metric_option, title


# ------------------
#    Run script
# ------------------

def run_map_plot(switch, datasets, timescale, resolution, folder_save = mV.folder_save):
    print(f'Plotting map_plot with {resolution} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')

    variable_type, metric, metric_option, cmap, cbar_label = find_general_metric_and_specify_cbar(switch)
    metric, metric_option, title  = specify_metric_and_title(switch, metric, metric_option)

    if switch['one scene']:
        dataset = datasets[0]
        fig = plot_one_scene(switch, variable_type, metric, metric_option, cmap, title, cbar_label, dataset, timescale, resolution, folder_save)
        source = mV.find_list_source(datasets, mV.models_cmip5, mV.models_cmip6, mV.observations)
        filename = f'{dataset}_{metric_option}' if switch['climatology'] else f'{dataset}_{metric_option}_difference' 
    else:
        fig = plot_multiple_scenes(switch, variable_type, metric, metric_option, cmap, title, cbar_label, datasets, timescale, resolution, folder_save)
        source = mV.find_list_source(datasets, mV.models_cmip5, mV.models_cmip6, mV.observations)
        with_obs = mV.find_ifWithObs(datasets, mV.observations)
        filename = f'{source}_{metric_option}{with_obs}' if switch['climatology'] else f'{source}_{metric_option}_difference' 
        
    mF.save_metric_figure(name = filename, metric = metric, figure = fig, folder_save = folder_save, source = source) if switch['save'] else None
    plt.show() if switch['show'] else None


if __name__ == '__main__':

    start = timeit.default_timer()

    # choose which metrics to plot
    switch = {
        'pr':                  False,       # 1
        'percentiles_pr':      False,       # 2
        'rx1day_pr':           False,       # 3
        'rx5day_pr':           False,       # 4

        'obj_scene':           False,       # 5

        'wap':                 False,       # 6
        'tas':                 False,       # 7

        'hur':                 True,       # 8
        'rlut':                False,       # 9

        'lcf':                 False,       # 10
        'hcf':                 False,       # 11

        'hus':                 False,       # 12

        'descent':             False,
        'ascent':              False,
        'per_kelvin':          False,

        'snapshot':            True,
        'climatology':         False,
        'change with warming': False,
        
        'one scene':           True,   # plots all chosen datasets if False (20 datasets max)
        'show':                True,
        'save':                False,
        }
    

    # plot and save figure
    run_map_plot(switch, 
                 datasets =    mV.datasets, 
                 timescale =  mV.timescales[0],
                 resolution =  mV.resolutions[0],
                #  folder_save = f'{mV.folder_save_gadi}'
                 )

    stop = timeit.default_timer()
    print(f'Finshed, script finished in {round((stop-start)/60, 2)} minutes.')












































