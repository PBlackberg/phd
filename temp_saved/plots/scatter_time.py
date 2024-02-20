''' 
# ------------------------
#     Scatter plot
# ------------------------
This script plots scatter plots with trend line and correlation coefficient, including
Correlating
    values
    anomalies

Plotting data as
    scatter
    datapoint density map

Plotting trend as
    linear slope
    bin-mean trend line

'''



# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats


# ------------------------------------------------------------------------------------ imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
import pandas as pd
home = os.path.expanduser("~")                           
sys.path.insert(0, f'{os.getcwd()}/switch')
import temp_saved.myVars_saved as mV                                 
import myFuncs as mF     
import myClasses as mC



# ------------------------
#       Get metric
# ------------------------
# ---------------------------------------------------------------------------------------- load list ----------------------------------------------------------------------------------------------------- #
def calc_anomalies(metric, title, axtitle):
    title = f'{title} anomalies'
    if mV.timescales[0] == 'daily' and len(metric) > 30: 
        rolling_mean = metric.rolling(time=12, center=True).mean()
        metric = metric - rolling_mean
        metric = metric.dropna(dim='time')
    if mV.timescales[0] == 'monthly': 
        climatology = metric.groupby('time.month').mean('time')
        metric = metric.groupby('time.month') - climatology 
    return metric, title, axtitle

def get_metric(switchM, dataset, metric_class):
    ''' Gets the metric and performs calculation according to switchM '''
    metric, title, axtitle = mF.load_metric(metric_class, dataset), '', dataset
    metric = mF.resample_timeMean(metric, mV.timescales[0])
    metric, title, axtitle = calc_anomalies(metric, title, axtitle) if switchM['anomalies'] else [metric, title, axtitle]
    return metric, title, axtitle


# ---------------------------------------------------------------------------------------- get limits ----------------------------------------------------------------------------------------------------- #
def get_limits(switchM, metric_class):
    qWithin_low, qWithin_high, qBetween_low, qBetween_high = 0, 1, 0, 1 # quantiles
    vmin, vmax = mF.find_limits(switchM, metric_class, get_metric, qWithin_low, qWithin_high, qBetween_low, qBetween_high,
                                vmin = '', vmax = ''    # if manually setting limits (comment out for normal limits)
                                )   
    return vmin, vmax



# --------------------------
#   Plot correlation plot
# --------------------------
# --------------------------------------------------------------------------------------- plot figure ----------------------------------------------------------------------------------------------------- #
def format_fig(num_subplots):
    ncols = 4 if num_subplots > 4 else num_subplots # max 4 subplots per row
    nrows = int(np.ceil(len(mV.datasets)/ ncols))
    width, height = [12, 8.5]   if nrows == 5 else [12, 8.5] 
    width, height = [12, 10]    if nrows == 6 else [width, height]
    width, height = [12, 11.5]  if nrows == 7 else [width, height]
    width, height = [12, 11.5]  if nrows == 8 else [width, height]
    ncols = 4 if num_subplots > 4 else num_subplots # max 4 subplots per row
    fig, axes = mF.create_figure(width = 12, height = 8.5, nrows=nrows, ncols=ncols)
    mF.delete_remaining_axes(fig, axes, num_subplots, nrows, ncols)
    return fig, axes, nrows, ncols

def highlight_subplot_frame(ax, color):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)  # Adjust line width

def format_axes(switch, switch_highlight, metric_classX, metric_classY, dataset, fig, ax, subplot, num_subplots, nrows, ncols, axtitle, xmin, xmax, ymin, ymax, h):
    row, col = subplot // ncols, subplot % ncols    # determine row and col index
    mF.move_col(ax, -0.0715+0.0025)        if col == 0 else None
    mF.move_col(ax, -0.035)                if col == 1 else None
    mF.move_col(ax, 0.0)                   if col == 2 else None
    mF.move_col(ax, 0.035)                 if col == 3 else None

    mF.move_row(ax, 0.0875 - 0.025 +0.025) if row == 0 else None
    mF.move_row(ax, 0.0495 - 0.0135+0.025) if row == 1 else None
    mF.move_row(ax, 0.01   - 0.005+0.025)  if row == 2 else None
    mF.move_row(ax, -0.0195+0.025)         if row == 3 else None
    mF.move_row(ax, -0.05+0.025)           if row == 4 else None
    mF.move_row(ax, -0.05+0.01)            if row == 5 else None
    mF.move_row(ax, -0.05+0.01)            if row == 6 else None
    mF.move_row(ax, -0.05+0.01)            if row == 7 else None

    mF.scale_ax_x(ax, 0.9) # 0.95
    mF.scale_ax_y(ax, 0.85)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    ax.xaxis.set_major_formatter(formatter)
    mF.plot_xlabel(fig, ax, metric_classX.label, pad=0.055, fontsize = 10)    if subplot >= num_subplots-ncols else None
    mF.plot_ylabel(fig, ax, metric_classY.label, pad = 0.0475, fontsize = 10) if col == 0 else None
    mF.plot_axtitle(fig, ax, axtitle, xpad = 0.035, ypad = 0.0075, fontsize = 10)
    ax.set_xticklabels([]) if not subplot >= num_subplots-ncols else None
    highlight_subplot_frame(ax, color = 'b') if dataset in mV.highlight_models(switch_highlight, mV.datasets, switch_subset= mV.switch_subset) else None
    highlight_subplot_frame(ax, color = 'r') if mF.find_source(dataset) in ['obs'] else None
    if switch['density_map']:
        if col == 3:
            mF.cbar_right_of_axis(fig, ax, h[3], width_frac= 0.05, height_frac=1, pad=0.015, numbersize = 9, cbar_label = 'months [Nb]', text_pad = 0.035)
        else:
            mF.cbar_right_of_axis(fig, ax, h[3], width_frac= 0.05, height_frac=1, pad=0.015, numbersize = 9, cbar_label = '', text_pad = 0.1)

def add_correlation_coeff(x, y, ax):
    res= stats.pearsonr(x,y)
    placement = (0.675, 0.05) if res[0]>0 else (0.675, 0.85) 
    ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', xytext = placement, textcoords='axes fraction', fontsize = 8, color = 'r') if res[1]<=0.05 else None

def split_conv_threshold(switchM):
    prctileM = '90' if switchM['90'] else mV.conv_percentiles[0]
    prctileM = '95' if switchM['95'] else prctileM
    prctileM = '97' if switchM['90'] else prctileM
    return prctileM
        
def plot_scatters(switchX, switchY, switch_highlight, switch, metric_nameX, metric_nameY):
    metric_classX = mC.get_metric_class(metric_nameX, switchX, prctile = split_conv_threshold(switchX))
    metric_classY = mC.get_metric_class(metric_nameY, switchY, prctile = split_conv_threshold(switchY))
    xmin, xmax = get_limits(switchX, metric_classX)
    ymin, ymax = get_limits(switchY, metric_classY)
    if len(mV.datasets) == 1:
        x, metric_title, axtitle = get_metric(switchX, mV.datasets[0], metric_classX)
        y, metric_title, axtitle = get_metric(switchY, mV.datasets[0], metric_classY)
        # x = x.assign_coords(time=y.time)
        x, y = x.dropna(dim='time', how='any'), y.dropna(dim='time', how='any')
        x, y = xr.align(x, y, join='inner')
        fig = mF.plot_scatter(switch, metric_classX, metric_classY, x, y, metric_title, axtitle, xmin, xmax, ymin, ymax)
        return fig, f'{metric_classX.name} and {metric_classY.name} ({metric_title})'

    fig, axes, nrows, ncols = format_fig(len(mV.datasets))
    for subplot, dataset in enumerate(mV.datasets):
        x, metric_title, axtitle = get_metric(switchX, dataset, metric_classX)  # must be this order to use in mF.find_limits
        y, metric_title, axtitle = get_metric(switchY, dataset, metric_classY)
        # x = x.assign_coords(time=y.time)
        x, y = x.dropna(dim='time', how='any'), y.dropna(dim='time', how='any')
        x, y = xr.align(x, y, join='inner')
        fig.text(0.5, 0.985, f'{metric_classX.name} and {metric_classY.name} ({metric_title})', ha = 'center', fontsize = 9, transform=fig.transFigure)
        ax = axes.flatten()[subplot]
        h = mF.plot_axScatter(switch, ax, x, y, metric_classY)
        format_axes(switch, switch_highlight, metric_classX, metric_classY, dataset, fig, ax, subplot, len(mV.datasets), nrows, ncols, axtitle, xmin, xmax, ymin, ymax, h)
        add_correlation_coeff(x, y, ax)
    return fig, f'{metric_classX.name} and {metric_classY.name} ({metric_title})'



# ------------------------
#     Run / save plot
# ------------------------
# ---------------------------------------------------------------------------------- Find metric / labels and run ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator()
def run_scatter_plot(switch_metricX, switch_metricY, switchX, switchY, switch_highlight, switch):
    print(f'{mV.timescales[0]} correlation in {len(mV.datasets)} datasets')
    print(f'metricX: {[key for key, value in switch_metricX.items() if value]} {[key for key, value in switchX.items() if value]}')
    print(f'metricY: {[key for key, value in switch_metricY.items() if value]} {[key for key, value in switchY.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')
    metric_nameX = next((key for key, value in switch_metricX.items() if value), None)
    for metric_nameY in [k for k, v in switch_metricY.items() if v]:
        fig, fig_title = plot_scatters(switchX, switchY, switch_highlight, switch, metric_nameX, metric_nameY)
        mF.save_plot(switch, fig, home, filename = f'{mF.find_list_source(mV.datasets)}_{fig_title}{mF.find_ifWithObs(mV.datasets)}')
        plt.show() if switch['show'] else None


# ------------------------
#  Choose what to plot
# ------------------------
if __name__ == '__main__':
# ---------------------------------------------------------------------------------- x-metric ----------------------------------------------------------------------------------------------------- #
    switch_metricX = {                                                                                                  # pick x-metrics (pick one)
        'pr':           False,  'pr_99':        False,  'pr_97':        False,  'pr_95':    False,  'pr_90':    False,  # precipitation percentiles
        'pr_rx1day':    False,  'pr_rx5day':    False,                                                                  # precipitation extremes
        'rome':         True,  'ni':           False,  'areafraction': False,                                           # organization
        'wap':          False,                                                                                          # circulation
        'tas':          False,  'stability':    False,  'oni':          False,                                          # temperature
        'hur':          False,  'hus':          False,                                                                  # humidity
        'netlw':        False,  'rlut':         False,  'rlds':         False,  'rlus':     False,  'rlut':     False,  # LW
        'netsw':        False,  'rsdt':         False,  'rsds':         False,  'rsus':     False,  'rsut':     False,  # SW
        'lcf':          False,  'hcf':          False,  'ws_lc':        False,   'ws_hc':   False,                      # clouds
        'h':            False,   'h_anom2':      False,                                                                 # moist static energy
        }
    
    switchX = {                                                                         # choose seetings for y-metrics
        '250hpa':       False,  '500hpa':   False,  '700hpa':   False,                  # mask: vertical
        'descent':      False,  'ascent':   False,  'ocean':    False,                  # mask: horizontal
        'fixed area':   False,  '90':       False,  '95':       False,  '97':   False,  # conv threshold (95th default)
        'sMean':        False,   'area':     False,                                      # metric type
        'anomalies':    False,                                                           # calc type
        }
    

# ---------------------------------------------------------------------------------- y-metric ----------------------------------------------------------------------------------------------------- #
    switch_metricY = {                                                                                                  # pick y-metrics (Can pick multiple)
        'pr':           False,  'pr_99':        False,  'pr_97':        False,  'pr_95':    False,  'pr_90':    False,  # precipitation percentiles
        'pr_rx1day':    False,  'pr_rx5day':    False,                                                                  # precipitation extremes
        'rome':         False,  'ni':           False,  'areafraction': False,  'o_heatmap': False,                     # organization
        'wap':          False,                                                                                          # circulation
        'tas':          True,  'stability':    False,  'oni':          False,                                          # temperature
        'hur':          False,  'hus':          False,                                                                  # humidity
        'netlw':        False,  'rlut':         False,  'rlds':         False,  'rlus':     False,  'rlut':     False,  # LW
        'netsw':        False,  'rsdt':         False,  'rsds':         False,  'rsus':     False,  'rsut':     False,  # SW
        'lcf':          False,   'hcf':         False,  'ws_lc':        False,   'ws_hc':   False,                      # clouds
        'h':            False,   'h_anom2':     False,                                                                  # moist static energy
        }
    
    switchY = {                                                                                 # choose seetings for y-metrics
        '250hpa':           False,  '500hpa':       False,  '700hpa':   False,                  # mask: vertical
        'descent':          False,  'ascent':       False,  'ocean':    False,                  # mask: horizontal
        'descent_fixed':    False,  'ascent_fixed': False,                                      # mask: horizontal
        'fixed area':       False,  '90':           False,  '95':       False,  '97':   False,  # conv threshold (95th default)
        'sMean':            True,   'area':         False,                                      # metric type
        'anomalies':        False,                                                               # calc type
        }

# ---------------------------------------------------------------------------------- settings ----------------------------------------------------------------------------------------------------- #
    switch_highlight = {                                            # models to highlight
        'dTas': False, 'subset_switch': False, 'custom':    False,  # higlight models
        }
    
    switch = {                                                                                  # overall settings
        'scatter':              False, 'density_map':         True,                            # plot type
        'slope':                True, 'bin_trend':           False,                            # plot type
        'show':                 False,                                                          # show
        'save_test_desktop':    True,  'save_folder_desktop': False, 'save_cwd': False,  # save
        }
    
# ----------------------------------------------------------------------------------- run ----------------------------------------------------------------------------------------------------- #
    run_scatter_plot(switch_metricX, switch_metricY, switchX, switchY, switch_highlight, switch)






