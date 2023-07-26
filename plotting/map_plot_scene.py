import numpy as np
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
import pick_metric as pM

# could create a file for selecting datasets only.
# Then one file for the list of metrics (myMetrics)
# The saving functions could either go into myFuncs or myMetrics

# --------------------------------------------------------------------------------- Calculate plot metric and find limits ----------------------------------------------------------------------------------------------------- #

def get_scene(switch, dataset, metric):
    source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
    if switch['snapshot']:
        title = f'{metric.option} snapshot'
        scene = mV.load_metric(metric.variable_type, f'{metric.name}_snapshot', source, dataset, timescale = mV.timescales[0], experiment = mV.experiments[0], resolution= mV.resolutions[0], folder_load = mV.folder_save)[metric_option]
    if switch['climatology']:
        title = f'{metric.option} time mean'
        scene = mV.load_metric(folder_load, metric.variable_type, f'{metric.name}_tMean', source, dataset, timescale = mV.timescales[0], experiment = mV.experiments[0], resolution= mV.resolutions[0])[metric_option]
    if switch['change with warming']:
        title = f'{metric.option} change with warming'
        scene_historical = mV.load_metric(folder_load, metric.variable_type, f'{metric.name}_tMean', dataset, timescale = mV.timescales[0], experiment = mV.experiments[0], resolution= mV.resolutions[0])[metric_option]
        scene_warm = mV.load_metric(folder_load, metric.variable_type, f'{metric.name}_tMean', dataset, experiment=mV.experiments[1], resolution=resolution)[metric_option]
        scene = scene_warm - scene_historical 
    return scene, title

# -------------------------------------------------------------------------------------- different plots ----------------------------------------------------------------------------------------------------- #

def plot_one_scene(switch, dataset, metric):
    # Set limits
    # vmin, vmax = [ , ]
    vmin, vmax = mF.find_limits(switch, [dataset], metric, get_scene(),
        quantileWithin_low = 0,    # remove extreme low values from colorbar range 
        quantileWithin_high = 1,   # remove extreme high values from colorbar range 
        )

    fig, ax = mF.create_map_figure(width = 12, height = 4)
    scene, title = get_scene(switch, dataset, metric)
    pcm = mF.plot_axScene(ax, scene, metric.cmap, vmin = vmin, vmax = vmax)

    mF.move_col(ax, moveby = -0.055)
    mF.move_row(ax, moveby = 0.075)
    mF.scale_ax(ax, scaleby = 1.15)
    mF.cbar_below_axis(fig, ax, pcm, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = metric.cbar_label, text_pad = 0.125)
    mF.plot_xlabel(fig, ax, 'Lon', xlabel_pad = 0.1, xylabel_fontsize = 12)
    mF.plot_ylabel(fig, ax, 'Lat', ylabel_pad = 0.055, xylabel_fontsize = 12)
    mF.plot_axtitle(fig, ax, f'{dataset}: {title}', title_xpad = 0.005, title_ypad = 0.025, title_fontsize = 15)
    mF.format_ticks(ax, labelsize = 11)
    return fig


def plot_multiple_scenes(switch, variable_type, metric, metric_option, cmap, title, cbar_label, datasets, timescale, resolution, folder_save):
    # Set limits
    # vmin, vmax = [ , ]
    vmin, vmax = mF.find_limits(switch, variable_type, metric, metric_option, datasets, timescale, resolution, folder_load = folder_save,
        quantileWithin_low = 0,    # remove extreme low values from colorbar range 
        quantileWithin_high = 1,   # remove extreme high values from colorbar range 
        quantileBetween_low = 0,   # remove extreme low models' from colorbar range
        quantileBetween_high = 1   # remove extreme high models' from colorbar range
        )
    nrows = 4
    ncols = 4                                                     
    fig, axes = mF.create_map_figure(width = 14, height = 5, nrows=nrows, ncols=ncols)
    num_subplots = len(datasets)
    for i, dataset in enumerate(datasets):
        row = i // ncols  # determine row index
        col = i % ncols   # determine col index
        ax = axes.flatten()[i]
        scene, title = get_scene(switch, dataset, metric)
        pcm = mF.plot_axScene(ax, scene, cmap, vmin = vmin, vmax = vmax)

        mF.move_col(ax, -0.0825 + 0.0025) if col == 0 else None
        mF.move_col(ax, -0.0435 + 0.0025) if col == 1 else None
        mF.move_col(ax, -0.005 + 0.0025) if col == 2 else None
        mF.move_col(ax, 0.0325 + 0.0025) if col == 3 else None

        mF.move_row(ax, 0.025+0.005) if row == 0 else None
        mF.move_row(ax, 0.04+0.005) if row == 1 else None
        mF.move_row(ax, 0.045+0.01) if row == 2 else None
        mF.move_row(ax, 0.05+0.01) if row == 3 else None
        mF.move_row(ax, 0.0325 + 0.0025) if col == 4 else None

        mF.scale_ax(ax, 1.3)

        mF.plot_xlabel(fig, ax, 'Lon', xlabel_pad = 0.0725, fontsize = 8) if i >= num_subplots-nrows else None
        mF.plot_ylabel(fig, ax, 'Lat', ylabel_pad = 0.0375, fontsize = 8) if col == 0 else None
        mF.plot_axtitle(fig, ax, dataset, axtitle_xpad = 0.002, axtitle_ypad = 0.0095, axtitle_fontsize = 9)
        mF.format_ticks(ax, i, num_subplots, nrows, col, ticklabel_size = 9)

    ax.text(0.5, 0.95, title, ha = 'center', fontsize = 15, transform=fig.transFigure)

    cbar_position = [0.225, 0.0875, 0.60, 0.02]
    cbar_ax = fig.add_axes(cbar_position)
    cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
    ax.text(cbar_position[0] + cbar_position[2] / 2 , cbar_position[1]-0.07, cbar_label, ha = 'center', fontsize = 9, transform=fig.transFigure)
    mF.delete_remaining_axes(fig, axes, num_subplots, nrows, ncols)
    return fig


# ---------------------------------------------------------------------------------- Find the metric and units ----------------------------------------------------------------------------------------------------- #

# ------------------
#    Run script
# ------------------

def save_plot(switch, fig, metric):
    if not switch['save']:
        return
    
    source = mV.find_list_source(mV.datasets, mV.models_cmip5, mV.models_cmip6, mV.observations)
    with_obs = mV.find_ifWithObs(mV.datasets, mV.observations)
    if switch['snapshot']:
        filename = f'{mV.datasets[0]}_{metric.option}_snapshot' if switch['one scene'] else f'{mV.datasets[0]}_{metric.option}_difference' 
    else:
        filename = f'{source}_{metric.option}{with_obs}' if switch['climatology'] else f'{source}_{metric.option}_difference' 

        mF.save_figure_metric(name = filename, metric = metric, figure = fig, folder_save = mV.folder_save[0], source = source) if switch['save'] else None

    # manual save
    # folder = f'{home}/Desktop'
    # filename = 'snapshot'    
    # mV.save_figure(fig, folder, f'{filename}.pdf') if switch['save'] else None




def run_map_plot(switch, datasets = mV.datasets, timescale = mV.timescales[0], resolution = mV.resolutions[0]):
    print(f'Plotting map_plot with {timescale} {resolution} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')
    metric = pM.define_metric_object(switch)
    fig = plot_one_scene(switch, datasets[0], metric) if switch['one scene'] else plot_multiple_scenes(switch, datasets, metric)
    plt.show() if switch['show'] else None


    



if __name__ == '__main__':

    start = timeit.default_timer()
    # choose which metrics to plot
    switch = {
        'pr':                  False,
        'pr99':                False,
        'rx1day_pr':           False,
        'rx5day_pr':           False,

        'obj_scene':           False,

        'wap':                 False,
        'tas':                 True,

        'hur':                 False,
        'rlut':                False,

        'lcf':                 False,
        'hcf':                 False,

        'hus':                 True,

        'descent':             False,
        'ascent':              True,
        'per_kelvin':          False,

        'snapshot':            False,
        'climatology':         False,
        'change with warming': False,
        
        'one scene':           False,
        'multiple scenes':     False,
        'show':                False,
        'save':                False,
        }


    # plot and save figure
    run_map_plot(switch)

    stop = timeit.default_timer()
    print(f'Finshed, script finished in {round((stop-start)/60, 2)} minutes.')












































