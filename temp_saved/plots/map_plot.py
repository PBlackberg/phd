''' 
# ------------------------
#       Map plot
# ------------------------
This script plot scenes with coastlines in the background
The commonly plotted metrics include
Snapshots
Gridbox time-means

'''


# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------------------------ imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars as mV
import myFuncs as mF
import myFuncs_plots as mFp



# ------------------------
#       Get metric
# ------------------------
# ---------------------------------------------------------------------------------------- get scene ----------------------------------------------------------------------------------------------------- #
def tas_weighted(metric, title, axtitle, dataset):        
    title = f'_per_K (dTas)'
    tas_class      = mF.get_metric_class('tas', {'sMean': True, 'ascent': False, 'descent': False})
    tas_historical = mF.load_metric(tas_class, dataset, mV.experiments[0]).mean(dim='time')
    tas_warm       = mF.load_metric(tas_class, dataset, mV.experiments[1]).mean(dim='time')
    tas_change = tas_warm - tas_historical
    metric = metric / tas_change
    axtitle = f'{dataset:20} dT = {np.round(tas_change.data,2)} K'
    return metric, title, axtitle

def ecs_weighted(metric, title, axtitle, dataset):     
    title = f'_per_K (ECS)'
    ecs = mV.ecs_list[dataset] 
    metric = metric / ecs
    axtitle = f'{dataset:20} ECS = {np.round(ecs,2)} K'
    return metric, title, axtitle

def get_change_with_warming(metric_class, title, axtitle, dataset, switchM):
    title = '_change_with_warming'
    metric_historical, metric_warm  = mF.load_metric(metric_class, dataset, mV.experiments[0]), mF.load_metric(metric_class, dataset, mV.experiments[1])
    metric = metric_warm - metric_historical        
    metric, title, axtitle = tas_weighted(metric, title, axtitle, dataset) if switchM['per kelvin']   else [metric, title, axtitle]
    metric, title, axtitle = tas_weighted(metric, title, axtitle, dataset) if switchM['per ecs']      else [metric, title, axtitle]
    return metric, title, axtitle

def get_metric(switchM, dataset, metric_class):
    ''' Gets the metric and performs calculation according to switchM '''
    metric, title, axtitle = mF.load_metric(metric_class, dataset), '', dataset
    metric, title, axtitle = get_change_with_warming(metric_class, title, axtitle, dataset, switchM) if switchM['change with warming'] else [metric, title, axtitle]
    return metric, title, axtitle



# ----------------------------------------------------------------------------------------- get limits ----------------------------------------------------------------------------------------------------- #
def get_limits(switchM, metric_name):
    ''' Pre-set quantile limits 
    Within models  - limiting extreme geographical locations
    Between models - limiting extreme models '''
    metric_class = mC.get_metric_class(metric_name, switchM)   
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
    vmin, vmax = mF.find_limits(switchM, metric_class, get_metric, qWithin_low, qWithin_high, qBetween_low, qBetween_high,
                                
                                # vmin = 0, vmax = 0.4    # if manually setting limits (comment out for normal limits) o_heatmap
                                # vmin = 0, vmax = 0.2    # if manually setting limits (comment out for normal limits) o_heatmap (with_warming)
                                )      
    if metric_class.var in ['wap'] or switchM['change with warming']:
        vabs_max = np.maximum(np.abs(vmin), np.abs(vmax))
        vmin, vmax = -vabs_max, vabs_max 
    return vmin, vmax



# ------------------------
#     Plot map plot
# ------------------------
# --------------------------------------------------------------------------------------- plot / format plot ----------------------------------------------------------------------------------------------------- #
def format_fig(num_subplots):
    ncols = 4 if num_subplots > 4 else num_subplots # max 4 subplots per row
    nrows = int(np.ceil(num_subplots / ncols))
    width, height = [14, 3]     if nrows == 1   else [14, 6] 
    width, height = [14, 3.5]   if nrows == 2   else [width, height]
    width, height = [14, 4.5]   if nrows == 3   else [width, height]
    width, height = [14, 5]     if nrows == 4   else [width, height]
    width, height = [14, 6]     if nrows == 5   else [width, height] 
    width, height = [14, 7.5]   if nrows == 6   else [width, height]
    width, height = [14, 8.25]  if nrows == 7   else [width, height]
    width, height = [14, 8.25]  if nrows == 8   else [width, height]
    fig, axes = mFp.create_map_figure(width = width, height = height, nrows=nrows, ncols=ncols)
    mFp.delete_remaining_axes(fig, axes, num_subplots, nrows, ncols)
    return fig, axes, nrows, ncols

def format_axes(fig, ax, subplot, num_subplots, nrows, ncols, axtitle):
    row, col = subplot // ncols, subplot % ncols    # determine row and col index
    mFp.move_col(ax, -0.0825 + 0.0025) if col == 0 else None
    mFp.move_col(ax, -0.0435 + 0.0025) if col == 1 else None
    mFp.move_col(ax, -0.005 + 0.0025)  if col == 2 else None
    mFp.move_col(ax, 0.0325 + 0.0025)  if col == 3 else None
    mFp.move_row(ax, 0.025+0.005)      if row == 0 else None
    mFp.move_row(ax, 0.04+0.005)       if row == 1 else None
    mFp.move_row(ax, 0.045+0.01)       if row == 2 else None
    mFp.move_row(ax, 0.05+0.01)        if row == 3 else None
    mFp.move_row(ax, 0.05+0.01)        if row == 4 else None
    mFp.move_row(ax, 0.05+0.01)        if row == 5 else None
    mFp.move_row(ax, 0.05+0.01)        if row == 6 else None
    mFp.move_row(ax, 0.05+0.01)        if row == 7 else None
    mFp.scale_ax(ax, 1.3)
    mFp.plot_xlabel(fig, ax, 'Lon', pad = 0.064, fontsize = 8) if subplot >= num_subplots-ncols else None
    mFp.plot_ylabel(fig, ax, 'Lat', pad = 0.0375, fontsize = 8) if col == 0 else None
    mFp.format_ticks(ax, subplot, num_subplots, ncols, col, labelsize = 9)
    mFp.plot_axtitle(fig, ax, axtitle, xpad = 0.002, ypad = 0.0095, fontsize = 9) if nrows < 6 else mFp.plot_axtitle(fig, ax, axtitle, xpad = 0.002, ypad = 0.0095/2, fontsize = 9) 

def plot_colobar(fig, ax, pcm, label, variable_list = mV.datasets):
    if len(variable_list) > 4:
        cbar_position = [0.225, 0.095, 0.60, 0.02]      # [left, bottom, width, height]
    else:
        cbar_position = [0.225, 0.15, 0.60, 0.05] 
    cbar_ax = fig.add_axes(cbar_position)
    fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
    ax.text(cbar_position[0] + cbar_position[2] / 2 , cbar_position[1]-0.075, label, ha = 'center', fontsize = 10, transform=fig.transFigure)

def plot_scenes(switchM, metric_name, vmin = None, vmax = None):
    ''' Plotting multiple scenes. Can plot up to 20 here '''
    metric_class = mF.get_metric_class(metric_name, switchM)                                                        # has details about name, cmap, label and so on
    vmin, vmax = get_limits(switchM, metric_name)
    if len(mV.datasets) == 1:
        metric, title, axtitle = get_metric(switchM, mV.datasets[0], metric_class)
        fig = mF.plot_scene(metric, metric_class.cmap, metric_class.label, title, axtitle, vmin, vmax)
        return fig, f'{metric_class.name}{title}'
    fig, axes, nrows, ncols = format_fig(len(mV.datasets))
    for subplot, dataset in enumerate(mV.datasets):
        metric, title, axtitle = get_metric(switchM, dataset, metric_class)
        fig.text(0.5, 0.95, f'{metric_class.name}{title}', ha = 'center', fontsize = 15, transform=fig.transFigure)  # fig title
        ax = axes.flatten()[subplot]
        pcm = mF.plot_axMapScene(ax, metric, metric_class.cmap, vmin = vmin, vmax = vmax)
        format_axes(fig, ax, subplot, len(mV.datasets), nrows, ncols, axtitle)
    plot_colobar(fig, ax, pcm, metric_class.label)
    return fig, f'{metric_class.name}{title}'


def plot_dsScenes(ds, label = 'units []', title = 'test', vmin = None, vmax = None, cmap = 'Blues', variable_list = mV.datasets):
    ''' Plotting multiple scenes based on daatset with scenes '''
    if len(variable_list) == 1:
        for dataset in variable_list:
            return mFp.plot_scene(ds[dataset], cmap = cmap, label = label, figure_title = title, ax_title= dataset, vmin = vmin, vmax = vmax)
    fig, axes, nrows, ncols = format_fig(len(variable_list))
    for subplot, dataset in enumerate(variable_list):
        da = ds[dataset]
        top_text = 0.95 if len(variable_list) > 4 else 0.85
        fig.text(0.5, top_text, title, ha = 'center', fontsize = 15, transform=fig.transFigure)  # fig title
        ax = axes.flatten()[subplot]
        pcm = mFp.plot_axMapScene(ax, da, cmap, vmin = vmin, vmax = vmax)
        format_axes(fig, ax, subplot, len(mV.datasets), nrows, ncols, axtitle = dataset)
    plot_colobar(fig, ax, pcm, label)
    return fig, axes
















# ------------------------
#     Run / save plot
# ------------------------
# ----------------------------------------------------------------------------------------- get scene and run ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator(show_time = True)
def run_map_plot(switch_metric, switchM, switch):
    print(f'{len(mV.datasets)} datasets with {mV.timescales[0]} {mV.resolutions[0]} data')
    print(f'metric: {[key for key, value in switch_metric.items() if value]}')
    print(f'metric_type: {[key for key, value in switchM.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')
    for metric_name in [k for k, v in switch_metric.items() if v]:
        fig, fig_title = plot_scenes(switchM, metric_name, vmin = None, vmax = None)
        mF.save_plot(switch, fig, home, filename = f'{mF.find_list_source(mV.datasets)}_{fig_title}{mF.find_ifWithObs(mV.datasets)}')
        plt.show() if switch['show'] else None



# ------------------------
#   Choose what to plot
# ------------------------
if __name__ == '__main__':
# ---------------------------------------------------------------------------------- metric ----------------------------------------------------------------------------------------------------- #
    switch_metric = {                                                                                                       # pick metric (can pick multiple)
        'pr':                   False,  'pr_99':         False, 'pr_97':        False,  'pr_95':    False,  'pr_90': False, # precipitation percentiles
        'pr_rx1day':            False,  'pr_rx5day':     False,                                                             # precipitaiton extremes
        'conv':                 False,  'obj':           False, 'o_heatmap':    False,                                      # organization
        'wap':                  False,                                                                                      # circulation
        'tas':                  True,  'stability':     False,                                                             # temperature
        'hur':                  False,  'hus':           False,                                                             # humidity
        'netlw':                False,  'rlds':          False, 'rlus':         False,  'rlut':     False,                  # LW
        'netsw':                False,  'rsdt':          False, 'rsds':         False,  'rsus':     False,  'rsut': False,  # SW
        'lcf':                  False,  'hcf':           False, 'ws_lc':        False,  'ws_hc':    False,                  # clouds
        'h':                    False,  'h_anom2':      False,                                                              # moist static energy
        }

    switchM = {                                                                         # choose seetings for metrics
        'fixed area':           False,                                                  # org threshold
        '250hpa':               False,  '500hpa':       False,  '700hpa':    False,     # mask: vertical
        'descent':              False,  'ascent':       False,  'ocean':     False,     # mask: horizontal
        'descent_fixed':        False,  'ascent_fixed': False,                           # mask: horizontal
        
        'snapshot':             False,  'tMean':        True,                           # scene type
        'change with warming':  False,  'per kelvin':   False, 'per ecs':   False,      # scenario type
        }


# ---------------------------------------------------------------------------------- settings ----------------------------------------------------------------------------------------------------- #
    switch = {
        'show':                 False,                                                          # show
        'save_folder_desktop':  False, 'save_test_desktop': False,   'save_test_cwd':   True,   # save
        }
    

# ------------------------------------------------------------------------------------ run ----------------------------------------------------------------------------------------------------- #
    run_map_plot(switch_metric, switchM, switch)




