import matplotlib.pyplot as plt
import xarray as xr

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/util')
import myFuncs as mF # imports common operators / functions
import myVars as mV # imports common variables
import constructed_fields as cF # imports fields for testing


# ---------------------------------------------------------------------------------- Calculate plot metric ----------------------------------------------------------------------------------------------------- #

def get_data(switch, source, dataset, metric, experiment):
    if switch['snapshot']:
        metric_name, metric_option = f'{metric.name}_snapshot', f'{metric.option}_snapshot'
    if switch['climatology'] or switch['change with warming']:
        metric_name, metric_option = f'{metric.name}_tMean', f'{metric.option}_tMean'
    path = f'{mV.folder_save[0]}/{metric.variable_type}/metrics/{metric_name}/{source}/{dataset}_{metric_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc' 
    return xr.open_dataset(path)[metric_option]
    
def calc_scene(switch, dataset, metric):
    source = mF.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
    scene = get_data(switch, source, dataset, metric, mV.experiments[0]) if switch['snapshot'] or switch['climatology'] else None
    if switch['change with warming']:
        scene_historical = get_data(switch, source, dataset, metric, mV.experiment[0])
        scene_warm = get_data(switch, source, dataset, metric, mV.experiment[1])
        scene = scene_warm - scene_historical 
    return scene

# -------------------------------------------------------------------------------------- plot / format plot ----------------------------------------------------------------------------------------------------- #

def find_title(switch, metric):
    title = f'{metric.option} snapshot' if switch['snapshot'] else None
    title = f'{metric.option} clim' if switch['climatology'] else title
    title = f'{metric.option} change with warming' if switch['change with warming'] else title
    return title

def plot_one_scene(switch, dataset, metric):
    calc_cbar_limits = False
    if calc_cbar_limits:
        vmin, vmax = mF.find_limits(switch, [dataset], metric, calc_scene,
            quantileWithin_low = 0,    # remove extreme low values from colorbar range 
            quantileWithin_high = 1,   # remove extreme high values from colorbar range 
            )
    else:
        vmin, vmax = [None , None]

    fig, ax = mF.create_map_figure(width = 12, height = 4)
    scene = calc_scene(switch, dataset, metric)
    pcm = mF.plot_axScene(ax, scene, metric.cmap, vmin = vmin, vmax = vmax)

    mF.move_col(ax, moveby = -0.055)
    mF.move_row(ax, moveby = 0.075)
    mF.scale_ax(ax, scaleby = 1.15)
    mF.cbar_below_axis(fig, ax, pcm, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = metric.label, text_pad = 0.125)
    mF.plot_xlabel(fig, ax, 'Lon', pad = 0.1, fontsize = 12)
    mF.plot_ylabel(fig, ax, 'Lat', pad = 0.055, fontsize = 12)
    mF.plot_axtitle(fig, ax, f'{dataset}: {find_title(switch, metric)}', xpad = 0.005, ypad = 0.025, fontsize = 15)
    mF.format_ticks(ax, labelsize = 11)
    return fig


def plot_multiple_scenes(switch, datasets, metric):
    calc_cbar_limits = True
    if calc_cbar_limits:
        vmin, vmax = mF.find_limits(switch, datasets, metric, calc_scene,
        quantileWithin_low = 0,    # remove extreme low values from colorbar range 
        quantileWithin_high = 1,   # remove extreme high values from colorbar range 
        quantileBetween_low = 0,   # remove extreme low models' from colorbar range
        quantileBetween_high = 1   # remove extreme high models' from colorbar range
        )
    else:
        vmin, vmax = [None , None] # needs to be fixed values if used, otherwise the limits will vary from subplot to subplot (not reflecting common colorbar)
    
    nrows = 5
    ncols = 4                                                     
    fig, axes = mF.create_map_figure(width = 14, height = 6, nrows=nrows, ncols=ncols)
    num_subplots = len(datasets)
    for i, dataset in enumerate(datasets):
        row = i // ncols  # determine row index
        col = i % ncols   # determine col index
        ax = axes.flatten()[i]
        scene = calc_scene(switch, dataset, metric)
        pcm = mF.plot_axScene(ax, scene, metric.cmap, vmin = vmin, vmax = vmax)

        mF.move_col(ax, -0.0825 + 0.0025) if col == 0 else None
        mF.move_col(ax, -0.0435 + 0.0025) if col == 1 else None
        mF.move_col(ax, -0.005 + 0.0025) if col == 2 else None
        mF.move_col(ax, 0.0325 + 0.0025) if col == 3 else None

        mF.move_row(ax, 0.025+0.005) if row == 0 else None
        mF.move_row(ax, 0.04+0.005) if row == 1 else None
        mF.move_row(ax, 0.045+0.01) if row == 2 else None
        mF.move_row(ax, 0.05+0.01) if row == 3 else None
        mF.move_row(ax, 0.05+0.01) if row == 4 else None

        mF.scale_ax(ax, 1.3)

        mF.plot_xlabel(fig, ax, 'Lon', pad = 0.064, fontsize = 8) if i >= num_subplots-ncols else None
        mF.plot_ylabel(fig, ax, 'Lat', pad = 0.0375, fontsize = 8) if col == 0 else None
        mF.plot_axtitle(fig, ax, dataset, xpad = 0.002, ypad = 0.0095, fontsize = 9)
        mF.format_ticks(ax, i, num_subplots, ncols, col, labelsize = 9)

    ax.text(0.5, 0.95, find_title(switch, metric), ha = 'center', fontsize = 15, transform=fig.transFigure)

    cbar_position = [0.225, 0.095, 0.60, 0.02] # [left, bottom, width, height]
    cbar_ax = fig.add_axes(cbar_position)
    fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
    ax.text(cbar_position[0] + cbar_position[2] / 2 , cbar_position[1]-0.075, metric.label, ha = 'center', fontsize = 10, transform=fig.transFigure)
    mF.delete_remaining_axes(fig, axes, num_subplots, nrows, ncols)
    return fig

# ----------------------------------------------------------------------- Find the metric and labels / run ----------------------------------------------------------------------------------------------------- #

def save_the_plot(switch, fig, metric):
    source = mF.find_list_source(mV.datasets, mV.models_cmip5, mV.models_cmip6, mV.observations)
    with_obs = mF.find_ifWithObs(mV.datasets, mV.observations)

    folder = metric.get_figure_folder(mV.folder_save[0], source)
    filename = f'{metric.option}_snapshot'              if switch['snapshot'] else None
    filename = f'{metric.option}_clim'                  if switch['climatology'] else filename
    filename = f'{metric.option}_difference'            if switch['change with warming'] else filename
    filename = f'{mV.datasets[0]}_{filename}' if switch['one dataset'] else f'{source}_{filename}{with_obs}'

    mF.save_figure(fig, folder, f'{filename}.pdf') if switch['save'] else None
    mF.save_figure(fig, f'{home}/Desktop', f'{filename}.pdf') if switch['save to desktop'] else None


@mF.timing_decorator
def run_map_plot(switch):
    metric = mF.get_metric_object(switch)
    print(f'Plotting map_plot from {mV.timescales[0]} {mV.resolutions[0]} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')

    fig = plot_one_scene(switch, mV.datasets[0], metric) if switch['one dataset'] else plot_multiple_scenes(switch, mV.datasets, metric)
    save_the_plot(switch, fig, metric) if switch['save'] or switch['save to desktop'] else None
    plt.show() if switch['show'] else None



if __name__ == '__main__':
    run_map_plot(switch = {
        # metrics
        'pr':                  False,
        'pr99':                False,
        'rx1day_pr':           False,
        'rx5day_pr':           False,

        'obj':                 False,

        'wap':                 False,
        'tas':                 False,

        'hus':                 False,
        'hur':                 False,
        'rlut':                False,

        'lcf':                 True,
        'hcf':                 False,


        # metric calculation
        'snapshot':            False,
        'climatology':         True,
        'change with warming': False,


        # masked by
        'descent':             False,
        'ascent':              False,
        'per_kelvin':          False,
        

        # show/save
        'one dataset':         True,
        'show':                True,
        'save':                False,
        'save to desktop':     False
        }
    )












































