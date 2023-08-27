import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util')
import myFuncs as mF                                # imports common operators
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV                                 # imports common variables

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


# ----------------------------------------------------------------------------- plot / format plot ----------------------------------------------------------------------------------------------------- #
    
def plot_one_scene(scene, metric, title = '', vmin = None, vmax = None):
    fig, ax = mF.create_map_figure(width = 12, height = 4)
    pcm = mF.plot_axScene(ax, scene, metric.cmap, vmin = vmin, vmax = vmax)
    mF.move_col(ax, moveby = -0.055)
    mF.move_row(ax, moveby = 0.075)
    mF.scale_ax(ax, scaleby = 1.15)
    mF.cbar_below_axis(fig, ax, pcm, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = metric.label, text_pad = 0.125)
    mF.plot_xlabel(fig, ax, 'Lon', pad = 0.1, fontsize = 12)
    mF.plot_ylabel(fig, ax, 'Lat', pad = 0.055, fontsize = 12)
    mF.plot_axtitle(fig, ax, title, xpad = 0.005, ypad = 0.025, fontsize = 15)
    mF.format_ticks(ax, labelsize = 11)
    return fig

def plot_multiple_scenes(datasets, metric, func, title = '',  vmin = None, vmax = None, switch = {}):
    nrows, ncols = 5, 4                                                     
    fig, axes = mF.create_map_figure(width = 14, height = 6, nrows=nrows, ncols=ncols)
    num_subplots = len(datasets)
    for i, dataset in enumerate(datasets):
        row = i // ncols  # determine row index
        col = i % ncols   # determine col index
        ax = axes.flatten()[i]
        scene, _, axtitle = func(switch, dataset, metric)
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
        mF.format_ticks(ax, i, num_subplots, ncols, col, labelsize = 9)
        mF.plot_axtitle(fig, ax, axtitle, xpad = 0.002, ypad = 0.0095, fontsize = 9)

    ax.text(0.5, 0.95, title, ha = 'center', fontsize = 15, transform=fig.transFigure)
    cbar_position = [0.225, 0.095, 0.60, 0.02] # [left, bottom, width, height]
    cbar_ax = fig.add_axes(cbar_position)
    fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
    ax.text(cbar_position[0] + cbar_position[2] / 2 , cbar_position[1]-0.075, metric.label, ha = 'center', fontsize = 10, transform=fig.transFigure)
    mF.delete_remaining_axes(fig, axes, num_subplots, nrows, ncols)
    return fig

# ---------------------------------------------------------------------------------- Get scene and run ----------------------------------------------------------------------------------------------------- #

def get_scene(switch, dataset, metric):
    source = mF.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
    if switch['snapshot']:
        title = f'{metric.option}_snapshot'
        axtitle = dataset
        metric_name, metric_option = f'{metric.name}_snapshot', f'{metric.option}_snapshot'
        ds = xr.open_dataset(f'{mV.folder_save[0]}/{metric.variable_type}/metrics/{metric_name}/{source}/{dataset}_{metric_name}_{mV.conv_percentiles[0]}thPrctile_{mV.timescales[0]}_{mV.experiments[0]}_{mV.resolutions[0]}.nc') if metric.variable_type == 'org' else None      
        ds = xr.open_dataset(f'{mV.folder_save[0]}/{metric.variable_type}/metrics/{metric_name}/{source}/{dataset}_{metric_name}_{mV.timescales[0]}_{mV.experiments[0]}_{mV.resolutions[0]}.nc') if not metric.variable_type == 'org' else ds     
        scene = ds[metric_option]
        
    if switch['climatology']:
        title = f'{metric.option}_clim'
        metric_name, metric_option = f'{metric.name}_tMean', f'{metric.option}_tMean'
        ds = xr.open_dataset(f'{mV.folder_save[0]}/{metric.variable_type}/metrics/{metric_name}/{source}/{dataset}_{metric_name}_{mV.timescales[0]}_{mV.experiments[0]}_{mV.resolutions[0]}.nc') 
        axtitle = dataset
        scene = ds[metric_option]

    if switch['change with warming']:
        title = f'{metric.option}_change_with_warming'
        axtitle = dataset
        metric_name, metric_option = f'{metric.name}_tMean', f'{metric.option}_tMean'
        scene_historical = xr.open_dataset(f'{mV.folder_save[0]}/{metric.variable_type}/metrics/{metric_name}/{source}/{dataset}_{metric_name}_{mV.timescales[0]}_{mV.experiments[0]}_{mV.resolutions[0]}.nc')[metric_option]
        scene_warm = xr.open_dataset(f'{mV.folder_save[0]}/{metric.variable_type}/metrics/{metric_name}/{source}/{dataset}_{metric_name}_{mV.timescales[0]}_{mV.experiments[1]}_{mV.resolutions[0]}.nc')[metric_option]
        scene_change = scene_warm - scene_historical
        if switch['per kelvin']:
                tas_historical = xr.open_dataset(f'{mV.folder_save[0]}/tas/metrics/tas_sMean/{source}/{dataset}_tas_sMean_{mV.timescales[0]}_{mV.experiments[0]}_{mV.resolutions[0]}.nc')['tas_sMean'].mean(dim='time')
                tas_warm = xr.open_dataset(f'{mV.folder_save[0]}/tas/metrics/tas_sMean/{source}/{dataset}_tas_sMean_{mV.timescales[0]}_{mV.experiments[1]}_{mV.resolutions[0]}.nc')['tas_sMean'].mean(dim='time')
                tas_change = tas_warm - tas_historical
                axtitle = f'{dataset:20} dT = {np.round(tas_change.data,2)} K'
                scene_change = scene_change/tas_change
        if switch['per kelvin (ecs)']:
            tas_change = mV.ecs_list[dataset] 
            axtitle = f'{dataset:20} ECS = {np.round(tas_change,2)} K'
            scene = scene_change/tas_change
    return scene, title, axtitle


def run_map_plot(switch):
    print(f'Plotting map_plot from {mV.timescales[0]} {mV.resolutions[0]} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')

    metric = mF.get_metric_object(switch) # gets labels associated with metric (full name, colorbar, units etc.)
    source_list = mF.find_list_source(mV.datasets, mV.models_cmip5, mV.models_cmip6, mV.observations)
    ifWith_obs = mF.find_ifWithObs(mV.datasets, mV.observations)

    if switch['one scene']:
        scene, title, _ = get_scene(switch, mV.datasets[0], metric)
        title = f'{mV.datasets[0]}_{title}'
        vmin, vmax = mF.find_limits(switch, [mV.datasets[0]], metric, get_scene, 
                                 quantileWithin_low = 0, quantileWithin_high = 1, quantileBetween_low = 0, quantileBetween_high=1, # if calculating limits (set lims to '', or comment out)
                                 vmin = '', vmax = ''                                                                              # if manually setting limits
                                 )      
        vmin = -vmax if switch['change with warming'] or switch['wap'] else vmin                                                           
        fig = plot_one_scene(scene, metric, title = title, vmin = vmin, vmax = vmax)

    elif switch['multiple_scenes']:
        _, title, _ = get_scene(switch, mV.datasets[0], metric)
        title = f'{source_list}_{title}{ifWith_obs}'
        vmin, vmax = mF.find_limits(switch, mV.datasets, metric, get_scene, 
                                 quantileWithin_low = 0, quantileWithin_high = 1, quantileBetween_low = 0, quantileBetween_high=1,
                                #  vmin = -200, vmax = 200                                                                        # Need fixed limits for common colorbar
                                 )  
        vmin = -vmax if switch['change with warming'] or switch['wap'] else vmin
        fig = plot_multiple_scenes(mV.datasets, metric, get_scene, title, vmin, vmax, switch)

    folder = f'{mV.folder_save[0]}/{metric.variable_type}/figures/{metric.name}'
    filename = title
    mF.save_figure(fig, os.getcwd(), 'test.png')       if switch['save to cwd'] else None
    mF.save_figure(fig, f'{home}/Desktop', 'test.pdf') if switch['save to desktop'] else None
    mF.save_figure(fig, folder, f'{filename}.pdf')            if switch['save'] else None
    plt.show() if switch['show'] else None


if __name__ == '__main__':
    run_map_plot(switch = {
        # ------
        # metric
        # ------
            # organization
            'obj':                 False,

            # precipitation
            'pr':                  False,
            'pr99':                False,
            'rx1day_pr':           False,
            'rx5day_pr':           False,

            # Large scale state
            'tas':                 False,
            'hur':                 False,
            'rlut':                True,
            'wap':                 False,

            # clouds
            'lcf':                 False,
            'hcf':                 False,

            # moist static energy
            'hus':                 False,

        # --------
        # settings
        # --------
        # scene type
        'snapshot':            True,
        'climatology':         False,
        'change with warming': False,

        # masked by
        'fixed area':          False,
        'descent':             True,
        'ascent':              False,
        'per kelvin':          False,
        'per kelvin (ecs)':    False,
        
        # show/save
        'one scene':           False,
        'multiple_scenes':     True,
        'show':                True,
        'save':                False,
        'save to cwd':         False,
        'save to desktop':     True
        }
    )





























# '/Users/cbla0002/Documents/data/rad/metrics/rlut_d_snapshot/cmip6/BCC-CSM2-MR_rlut_d_snapshot_monthly_historical_regridded.nc'










