import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV
import myClasses as mC
import myFuncs as mF
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)



# ------------------------
#       Get scene
# ------------------------
# ---------------------------------------------------------------------------------------- get scene ----------------------------------------------------------------------------------------------------- #
def get_scene(switchM, dataset, metric_class):
    ''' Gets the metric and calculates the scene from options in the switch '''
    source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
    axtitle = dataset
    title = ''
    if any(switchM[key] for key in ['snapshot', 'tMean']):
        scene = mF.load_metric(metric_class, mV.folder_save[0], source, dataset, mV.timescales[0], mV.experiments[0], mV.resolutions[0])[metric_class.name]
        
    if switchM['change with warming']:
        title = '_change_with_warming'
        scene_historical = mF.load_metric(metric_class, mV.folder_save[0], source, dataset, mV.timescales[0], mV.experiments[0], mV.resolutions[0])[metric_class.name]
        scene_warm       = mF.load_metric(metric_class, mV.folder_save[0], source, dataset, mV.timescales[0], mV.experiments[1], mV.resolutions[0])[metric_class.name]
        scene = scene_warm - scene_historical
        if switchM['per kelvin']:
                title = f'_per_K (dTas)'
                tas_class      = mF.get_metric_class('tas', {'sMean': True, 'ascent': False, 'descent': False})
                tas_historical = mF.load_metric(tas_class, mV.folder_save[0], source, dataset, mV.timescales[0], mV.experiments[0], mV.resolutions[0])[metric_class.name].mean(dim='time')
                tas_warm       = mF.load_metric(tas_class, mV.folder_save[0], source, dataset, mV.timescales[0], mV.experiments[1], mV.resolutions[0])[metric_class.name].mean(dim='time')
                tas_change = tas_warm - tas_historical
                scene = scene/tas_change
                axtitle = f'{dataset:20} dT = {np.round(tas_change.data,2)} K'
        if switchM['per kelvin (ecs)']:
            title = f'_per_K (ECS)'
            tas_change = mV.ecs_list[dataset] 
            scene = scene/tas_change
            axtitle = f'{dataset:20} ECS = {np.round(tas_change,2)} K'
    return scene, title, axtitle



# ----------------------------------------------------------------------------------------- get limits ----------------------------------------------------------------------------------------------------- #
def get_limits(switchM, metric_class, datasets):
    ''' Some models and some geographical locations stick out, such that it rest of the distribution is hard to see. 
    This function gives pre-set quantile limits within models (limiting extreme geographical locations) of between models (limiting extreme models) 
    The limits can be set manually here too '''
    qWithin_low, qWithin_high, qBetween_low, qBetween_high = 0, 1, 0, 1 # quantiles
    if switchM['snapshot']:
        qWithin_low, qWithin_high, qBetween_low, qBetween_high = [0, 0.9, 0, 1]     if metric_class.name in ['pr_snapshot']        else [qWithin_low, qWithin_high, qBetween_low, qBetween_high]
        qWithin_low, qWithin_high, qBetween_low, qBetween_high = [0, 0.8, 0, 1]     if metric_class.name in ['pr_rx1day_snapshot'] else [qWithin_low, qWithin_high, qBetween_low, qBetween_high] 

    if switchM['tMean']:
        qWithin_low, qWithin_high, qBetween_low, qBetween_high = [0, 0.98, 0, 1]    if metric_class.name in ['pr_tMean']           else [qWithin_low, qWithin_high, qBetween_low, qBetween_high]
        qWithin_low, qWithin_high, qBetween_low, qBetween_high = [0, 0.8, 0, 1]     if metric_class.name in ['pr_rx1day_tMean']    else [qWithin_low, qWithin_high, qBetween_low, qBetween_high] 

    if switchM['change with warming']:
        qWithin_low, qWithin_high, qBetween_low, qBetween_high = [0.05, 0.95, 0, 1] if metric_class.name in ['pr_tMean']           else [qWithin_low, qWithin_high, qBetween_low, qBetween_high]
        qWithin_low, qWithin_high, qBetween_low, qBetween_high = [0, 0.9, 0, 1]     if metric_class.name in ['pr_rx1day_tMean']    else [qWithin_low, qWithin_high, qBetween_low, qBetween_high] 



    vmin, vmax = mF.find_limits(switchM, datasets, metric_class, get_scene, 
                                qWithin_low, qWithin_high, qBetween_low, qBetween_high, # if calculating limits (set lims to '', or comment out)
                                vmin = '', vmax = ''                                    # if manually setting limits
                                )      
    if metric_class.var_type in ['wap', 'tas'] or switchM['change with warming']:
        vabs_max = np.maximum(np.abs(vmin), np.abs(vmax))
        vmin, vmax = -vabs_max, vabs_max 
    return vmin, vmax



# ------------------------
#     Plot map plot
# ------------------------
# --------------------------------------------------------------------------------------- plot / format plot ----------------------------------------------------------------------------------------------------- #
def plot_one_scene(switchM, metric_class, vmin = None, vmax = None):
    ''' Plotting singular scene, mainly for testing '''
    source = mV.find_source(mV.datasets[0], mV.models_cmip5, mV.models_cmip6, mV.observations)
    if not mV.data_available(source, mV.datasets[0], mV.experiments[0], var = metric_class.var_type):
        print('No data for this')
    fig, ax = mF.create_map_figure(width = 12, height = 4)
    scene, title, _ = get_scene(switchM, mV.datasets[0], metric_class)
    fig_title = f'{metric_class.name}{title}'
    pcm = mF.plot_axMapScene(ax, scene, metric_class.cmap, vmin = vmin, vmax = vmax)
    mF.move_col(ax, moveby = -0.055)
    mF.move_row(ax, moveby = 0.075)
    mF.scale_ax(ax, scaleby = 1.15)
    mF.cbar_below_axis(fig, ax, pcm, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = f'{metric_class.label}', text_pad = 0.125)
    mF.plot_xlabel(fig, ax, 'Lon', pad = 0.1, fontsize = 12)
    mF.plot_ylabel(fig, ax, 'Lat', pad = 0.055, fontsize = 12)
    mF.plot_axtitle(fig, ax, fig_title, xpad = 0.005, ypad = 0.025, fontsize = 15)
    mF.format_ticks(ax, labelsize = 11)
    return fig, fig_title

def plot_multiple_scenes(switchM, metric_class, vmin = None, vmax = None):
    ''' Plotting multiple scenes. Can plot up to 20 here '''
    nrows, ncols = 5, 4                                                     
    fig, axes = mF.create_map_figure(width = 14, height = 6, nrows=nrows, ncols=ncols)
    num_subplots = len(mV.datasets)
    for i, dataset in enumerate(mV.datasets):
        source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        if not mV.data_available(source, dataset, mV.experiments[0], var = metric_class.var_type):
            continue

        row = i // ncols  # determine row index
        col = i % ncols   # determine col index
        ax = axes.flatten()[i]
        scene, title, axtitle = get_scene(switchM, dataset, metric_class)
        fig_title = f'{metric_class.name}{title}'
        pcm = mF.plot_axMapScene(ax, scene, metric_class.cmap, vmin = vmin, vmax = vmax)

        mF.move_col(ax, -0.0825 + 0.0025) if col == 0 else None
        mF.move_col(ax, -0.0435 + 0.0025) if col == 1 else None
        mF.move_col(ax, -0.005 + 0.0025)  if col == 2 else None
        mF.move_col(ax, 0.0325 + 0.0025)  if col == 3 else None

        mF.move_row(ax, 0.025+0.005)      if row == 0 else None
        mF.move_row(ax, 0.04+0.005)       if row == 1 else None
        mF.move_row(ax, 0.045+0.01)       if row == 2 else None
        mF.move_row(ax, 0.05+0.01)        if row == 3 else None
        mF.move_row(ax, 0.05+0.01)        if row == 4 else None

        mF.scale_ax(ax, 1.3)

        mF.plot_xlabel(fig, ax, 'Lon', pad = 0.064, fontsize = 8) if i >= num_subplots-ncols else None
        mF.plot_ylabel(fig, ax, 'Lat', pad = 0.0375, fontsize = 8) if col == 0 else None
        mF.format_ticks(ax, i, num_subplots, ncols, col, labelsize = 9)
        mF.plot_axtitle(fig, ax, axtitle, xpad = 0.002, ypad = 0.0095, fontsize = 9)

    ax.text(0.5, 0.95, fig_title, ha = 'center', fontsize = 15, transform=fig.transFigure)
    cbar_position = [0.225, 0.095, 0.60, 0.02] # [left, bottom, width, height]
    cbar_ax = fig.add_axes(cbar_position)
    fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
    ax.text(cbar_position[0] + cbar_position[2] / 2 , cbar_position[1]-0.075, metric_class.label, ha = 'center', fontsize = 10, transform=fig.transFigure)
    mF.delete_remaining_axes(fig, axes, num_subplots, nrows, ncols)
    return fig, fig_title



# ------------------------
#     Run / save plot
# ------------------------
# ----------------------------------------------------------------------------------------- get scene and run ----------------------------------------------------------------------------------------------------- #
def plot_metric(switchM, switch, metric_class):
    if switch['one_scene']:
        vmin, vmax = get_limits(switchM, metric_class, [mV.datasets[0]])      
        fig, fig_title = plot_one_scene(switchM, metric_class, vmin, vmax)
        filename = f'{mV.datasets[0]}_{fig_title}'

    if switch['multiple_scenes']:
        vmin, vmax = get_limits(switchM, metric_class, mV.datasets)
        fig, fig_title = plot_multiple_scenes(switchM, metric_class, vmin, vmax)
        source_list = mV.find_list_source(mV.datasets, mV.models_cmip5, mV.models_cmip6, mV.observations)
        ifWith_obs = mV.find_ifWithObs(mV.datasets, mV.observations)
        filename = f'{source_list}_{fig_title}{ifWith_obs}'
    
    mF.save_plot(switch, fig, home, filename)
    plt.show() if switch['show'] else None

def run_map_plot(switch_metric, switchM, switch):
    print(f'Plotting map_plot from {mV.timescales[0]} {mV.resolutions[0]} data')
    print(f'metric: {[key for key, value in switch_metric.items() if value]} {[key for key, value in switchM.items() if value]}')
    print(f'metric_type: {[key for key, value in switchM.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')
    for metric in [k for k, v in switch_metric.items() if v] :
        metric_class = mC.get_metric_class(metric, switchM, prctile = mV.conv_percentiles[0])
        plot_metric(switchM, switch, metric_class)



# ------------------------
#   Choose what to run
# ------------------------
if __name__ == '__main__':
# ---------------------------------------------------------------------------------- metric ----------------------------------------------------------------------------------------------------- #
 
    switch_metric = {
        # organization
        'obj':                 False,
        # precipitation
        'pr':                  False,
        'pr99':                False,
        'pr_rx1day':           False,
        'pr_rx5day':           False,
        # Large scale state
        'tas':                 False,
        'hur':                 False,
        'rlut':                False,
        'wap':                 False,
        'stability':           True,
        # clouds
        'lcf':                 False,
        'hcf':                 False,
        'ws_lc':               False,
        'ws_hc':               False,
        # moist static energy
        'hus':                 False,
        }

    switchM = {    
        # masked by
        'fixed area':          False, # only applies to org_metrics
        '250hpa':              False,
        '500hpa':              False,
        '700hpa':              False,
        'descent':             False,
        'ascent':              False,
        'per kelvin':          False,
        'per kelvin (ecs)':    False,
        # scene type
        'snapshot':            True,
        'tMean':               False,
        'change with warming': False,
        }



# ---------------------------------------------------------------------------------- settings ----------------------------------------------------------------------------------------------------- #
    switch = {
        # type of figure
        'one_scene':           True,
        'multiple_scenes':     False,
        # show/save
        'show':                False,
        'save_test_desktop':   False,
        'save_folder_desktop': False,
        'save_folder_cwd':     True,
        }
    
    run_map_plot(switch_metric, switchM, switch)



















