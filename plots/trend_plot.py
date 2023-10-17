import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
import os
import sys
import pandas as pd
home = os.path.expanduser("~")                           
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV                                 
import myFuncs as mF     
import myClasses as mC



# ------------------------
#       Get list
# ------------------------
# ---------------------------------------------------------------------------------------- load list ----------------------------------------------------------------------------------------------------- #
def get_list(switchM, dataset, metric_class):
    source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
    timescale = 'daily' if metric_class.var_type in ['pr', 'org', 'hus', 'ws'] else 'monthly'
    experiment  = '' if source == 'obs' else mV.experiments[0]
    # dataset = 'GPCP_1998-2009' if source == 'obs' and metric_class.var_type in ['pr', 'org'] else dataset
    dataset = 'GPCP_2010-2022' if source == 'obs' and metric_class.var_type in ['pr', 'org'] else dataset # pick a time range
    # dataset = 'GPCP' if source == 'obs' and metric_class.var_type in ['pr', 'org'] else dataset # complete record

    alist = mF.load_metric(metric_class, mV.folder_save[0], source, dataset, timescale, experiment, mV.resolutions[0])[metric_class.name]
    alist = mF.resample_timeMean(alist, mV.timescales[0])
    axtitle = dataset
    
    if dataset == 'CERES': # this observational dataset have monthly data with day specified as the middle of the month instead of the first
        alist['time'] = alist['time'] - pd.Timedelta(days=14)

    metric_title = ''
    if switchM['anomalies']:
        metric_title = 'anomalies'
        if mV.timescales[0] == 'daily': 
            rolling_mean = alist.rolling(time=12, center=True).mean()
            alist = alist - rolling_mean
            alist = alist.dropna(dim='time')
        if mV.timescales[0] == 'monthly': 
            climatology = alist.groupby('time.month').mean('time')
            alist = alist.groupby('time.month') - climatology 
        if mV.timescales[0] == 'annual':
            '' 
    return alist, metric_title, axtitle


# ---------------------------------------------------------------------------------------- get limits ----------------------------------------------------------------------------------------------------- #
def get_limits(switchM, metric_class, datasets):
    qWithin_low, qWithin_high, qBetween_low, qBetween_high = 0, 1, 0, 1 # quantiles
    vmin, vmax = mF.find_limits(switchM, datasets, metric_class, get_list, 
                                qWithin_low, qWithin_high, qBetween_low, qBetween_high, # if calculating limits (set lims to '', or comment out)
                                vmin = '', vmax = ''                                    # if manually setting limits
                                )   
    return vmin, vmax



# --------------------------
#   Plot correlation plot
# --------------------------
# -------------------------------------------------------------------------------------- plot axis plot ----------------------------------------------------------------------------------------------------- #
def calc_meanInBins(x, y, binwidth_frac=100):
    bin_width = (x.max() - x.min())/binwidth_frac # Each bin is one percent of the range of x values
    bins = np.arange(x.min(), x.max() + bin_width, bin_width)
    y_bins = []
    for i in np.arange(0,len(bins)-1):
        y_bins = np.append(y_bins, y.where((x>=bins[i]) & (x<bins[i+1])).mean())
    return bins, y_bins

def plot_axTrend(switch, fig, ax, x, y, metric_classY):
    h_points, h_pcm, h_line, cbar = None, None, None, None
    if switch['scatter']:
        h_points = mF.plot_scatter(ax, x, y, metric_classY)
    
    if switch['bins']:
        bins, y_bins = calc_meanInBins(x, y, binwidth_frac=100)
        h_pcm = mF.plot_ax_datapointDensity(ax, x, y, metric_classY)
        h_line = mF.plot_ax_line(ax, bins[:-1], y_bins, metric_classY)    
    return h_points, h_pcm, h_line, cbar



# --------------------------------------------------------------------------------------- plot figure ----------------------------------------------------------------------------------------------------- #
def plot_one_dataset(switchX, switchY, switch, metric_classX, metric_classY, xmin = None, xmax = None, ymin = None, ymax = None):
    fig, ax = mF.create_figure(width = 9, height = 6)

    x, _, _ =                  get_list(switchX, mV.datasets[0], metric_classX)
    y, metric_title, axtitle = get_list(switchY, mV.datasets[0], metric_classY)
    x, y = xr.align(x, y, join='inner')

    fig_title = f'{metric_classX.name} and {metric_classY.name} ({metric_title})'
    h_points, h_pcm, h_line, cbar = plot_axTrend(switch, fig, ax, x, y, metric_classY)
    res= stats.pearsonr(x,y)
    placement = (0.825, 0.05) if res[0]>0 else (0.825, 0.9) 
    ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', xytext=placement, textcoords='axes fraction', fontsize = 12, color = 'r') if res[1]<=0.05 else None      
    
    mF.move_col(ax, -0.03)
    mF.move_row(ax, 0.01)
    mF.scale_ax_x(ax, 0.95)
    mF.scale_ax_y(ax, 1)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12) 
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    ax.xaxis.set_major_formatter(formatter)
    mF.plot_xlabel(fig, ax, metric_classX.label, pad = 0.09,   fontsize = 12)
    mF.plot_ylabel(fig, ax, metric_classY.label, pad = 0.075, fontsize = 12)
    mF.plot_axtitle(fig, ax, axtitle, xpad = 0.005, ypad = 0.01, fontsize = 12)
    ax.text(0.5, 0.95, fig_title, ha = 'center', fontsize = 15.5, transform=fig.transFigure)

    mF.cbar_right_of_axis(fig, ax, h_pcm[3], width_frac= 0.05, height_frac=1, pad=0.035, numbersize = 12, cbar_label = 'months [Nb]', text_pad = 0.05, fontsize = 12) if switch['bins'] else None
    return fig, fig_title

def highlight_subplot_frame(ax, color):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)  # Adjust line width

def plot_multiple_datasets(switchX, switchY, switch_highlight, switch, metric_classX, metric_classY, xmin = None, xmax = None, ymin = None, ymax = None):
    nrows, ncols = 5, 4
    fig, axes = mF.create_figure(width = 12, height = 8.5, nrows=nrows, ncols=ncols)
    num_subplots = len(mV.datasets)
    for i, dataset in enumerate(mV.datasets):
        row = i // ncols  # determine row index
        col = i % ncols   # determine col index
        ax = axes.flatten()[i]

        dataset_highlight = mV.get_ds_highlight(switch_highlight, mV.datasets)
        highlight_subplot_frame(ax, color = 'b') if dataset in dataset_highlight else None
        highlight_subplot_frame(ax, color = 'r') if mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations) in ['obs'] else None

        x, _, _ =                  get_list(switchX, dataset, metric_classX)
        y, metric_title, axtitle = get_list(switchY, dataset, metric_classY)
        fig_title = f'{metric_classX.name} and {metric_classY.name} ({metric_title})'
        x = x.assign_coords(time=y.time)
        h_points, h_pcm, h_line, cbar = plot_axTrend(switch, fig, ax, x, y, metric_classY)
        res= stats.pearsonr(x,y)
        placement = (0.675, 0.05) if res[0]>0 else (0.675, 0.85) 
        ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', xytext = placement, textcoords='axes fraction', fontsize = 8, color = 'r') if res[1]<=0.05 else None
        
        mF.move_col(ax, -0.0715+0.0025)        if col == 0 else None
        mF.move_col(ax, -0.035)                if col == 1 else None
        mF.move_col(ax, 0.0)                   if col == 2 else None
        mF.move_col(ax, 0.035)                 if col == 3 else None

        mF.move_row(ax, 0.0875 - 0.025 +0.025) if row == 0 else None
        mF.move_row(ax, 0.0495 - 0.0135+0.025) if row == 1 else None
        mF.move_row(ax, 0.01   - 0.005+0.025)  if row == 2 else None
        mF.move_row(ax, -0.0195+0.025)         if row == 3 else None
        mF.move_row(ax, -0.05+0.025)           if row == 4 else None

        mF.scale_ax_x(ax, 0.9)
        mF.scale_ax_y(ax, 0.95)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1,1))
        ax.xaxis.set_major_formatter(formatter)
        mF.plot_xlabel(fig, ax, metric_classX.label, pad=0.055, fontsize = 10)    if i >= num_subplots-ncols else None
        mF.plot_ylabel(fig, ax, metric_classY.label, pad = 0.0475, fontsize = 10) if col == 0 else None
        mF.plot_axtitle(fig, ax, axtitle, xpad = 0, ypad = 0.0075, fontsize = 10)
        ax.text(0.5, 0.985, fig_title, ha = 'center', fontsize = 9, transform=fig.transFigure)

        if switch['bins']:
            if col == 3:
                mF.cbar_right_of_axis(fig, ax, h_pcm[3], width_frac= 0.05, height_frac=1, pad=0.015, numbersize = 9, cbar_label = 'months [Nb]', text_pad = 0.035)
            else:
                mF.cbar_right_of_axis(fig, ax, h_pcm[3], width_frac= 0.05, height_frac=1, pad=0.015, numbersize = 9, cbar_label = '', text_pad = 0.1)
    mF.delete_remaining_axes(fig, axes, num_subplots, nrows, ncols)
    return fig, fig_title



# ------------------------
#     Run / save plot
# ------------------------
# ---------------------------------------------------------------------------------- Find metric / labels and run ----------------------------------------------------------------------------------------------------- #
def plot_trend(switchX, switchY, switch_highlight, switch, metric_classX, metric_classY):
    if len(mV.datasets) == 1:
        xmin, xmax = get_limits(switchX, metric_classX, [mV.datasets[0]])
        ymin, ymax = get_limits(switchY, metric_classY, [mV.datasets[0]])
        fig, title = plot_one_dataset(switchX, switchY, switch, metric_classX, metric_classY, xmin, xmax, ymin, ymax)
        filename = f'{mV.datasets[0]}{title}'
    else:
        xmin, xmax = get_limits(switchX, metric_classX, mV.datasets)
        ymin, ymax = get_limits(switchY, metric_classY, mV.datasets)
        fig, fig_title = plot_multiple_datasets(switchX, switchY, switch_highlight, switch, metric_classX, metric_classY, xmin, xmax, ymin, ymax)
        source_list = mV.find_list_source(mV.datasets, mV.models_cmip5, mV.models_cmip6, mV.observations)
        ifWith_obs = mV.find_ifWithObs(mV.datasets, mV.observations)
        filename = f'{source_list}_{fig_title}{ifWith_obs}'

    mF.save_plot(switch, fig, home, filename)
    plt.show() if switch['show'] else None

@mF.timing_decorator
def run_trend_plot(switch_metricX, switch_metricY, switchX, switchY, switch_highlight, switch):
    print(f'Plotting {mV.timescales[0]} correlation between')
    print(f'metricX: {[key for key, value in switch_metricX.items() if value]} {[key for key, value in switchX.items() if value]}')
    print(f'metricY: {[key for key, value in switch_metricY.items() if value]} {[key for key, value in switchY.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')

    metricX = next((key for key, value in switch_metricX.items() if value), None)
    for metricY in [k for k, v in switch_metricY.items() if v]:
        prctileX = '90' if switchX['90'] else mV.conv_percentiles[0]
        prctileX = '95' if switchX['95'] else prctileX
        prctileX = '97' if switchX['90'] else prctileX

        prctileY = '90' if switchX['90'] else mV.conv_percentiles[0]
        prctileY = '95' if switchX['95'] else prctileY
        prctileY = '97' if switchX['97'] else prctileY

        metric_classX = mC.get_metric_class(metricX, switchX, prctile = prctileX)
        metric_classY = mC.get_metric_class(metricY, switchY, prctile = prctileY)
        plot_trend(switchX, switchY, switch_highlight, switch, metric_classX, metric_classY)



# ------------------------
#   Choose what to run
# ------------------------
if __name__ == '__main__':
# ---------------------------------------------------------------------------------- x-metric ----------------------------------------------------------------------------------------------------- #
    switch_metricX = {          # pick x-metric (pick one)
        # organization
        'rome':                True,
        'ni':                  False,
        'areafraction':        False,
        # precipitation
        'pr':                  False,
        'pr99':                False,
        'pr_rx1day':           False,
        'pr_rx5day':           False,
        # ascent/descent
        'wap':                 False,
        # humidity
        'hur':                 False,
        # temperature
        'tas':                 False,
        # radiation
            # longwave
            'rlut':               False,
            'rlds':               False,
            'rlus':               False, 
            'rlut':               False, 
            'netlw':              False,
            # shortwave
            'rsdt':               False,
            'rsds':               False,
            'rsus':               False,
            'rsut':               False,
            'netsw':              False,
        # clouds
        'stability':           False,
        'lcf':                 False,
        'hcf':                 False,
        'ws_lc':               False,
        'ws_hc':               False,
        # moist static energy
        'hus':                 False,
        }
    

    switchX = {               # choose seetings for x-metric
        # masked by
        'descent':             False,
        'ascent':              False,
        '250hpa':              False,
        '500hpa':              False,
        '700hpa':              False,
        # conv threshold
        'fixed area':          False,
        '90':                  False,
        '95':                  False,
        '97':                  False,
        # metric type
        'sMean':               False,
        # calc type
        'anomalies':           True,
        }

# ---------------------------------------------------------------------------------- y-metric ----------------------------------------------------------------------------------------------------- #
    switch_metricY = {         # pick y-metrics (Can pick multiple, mV.timescales[0] must be available for both metrics)
        # organization
        'rome':                False,
        'ni':                  False,
        'areafraction':        False,
        # precipitation
        'pr':                  False,
        'pr99':                False,
        'pr_rx1day':           False,
        'pr_rx5day':           False,
        # ascent/descent
        'wap':                 False,
        # humidity
        'hur':                 False,
        # temperature
        'tas':                 False,
        # radiation
            # longwave
            'rlds':               False,
            'rlus':               False, 
            'rlut':               False, 
            'netlw':              False,
            # shortwave
            'rsdt':               False,
            'rsds':               False,
            'rsus':               False,
            'rsut':               False,
            'netsw':              True,
        # clouds
        'stability':           False,
        'lcf':                 False,
        'hcf':                 False,
        'ws_lc':               False,
        'ws_hc':               False,
        # moist static energy
        'hus':                 False,
        }
    
    switchY = {                # pick settings for y-metric (must align between y-metrics)
        # masked by
        'descent':             False,
        'ascent':              False,
        '250hpa':              False,
        '500hpa':              False,
        '700hpa':              False,
        # conv threshold
        'fixed area':          False,
        '90':                  False,
        '95':                  False,
        '97':                  False,
        # metric type
        'sMean':               True,
        # calc type
        'anomalies':           True,
        }

# ---------------------------------------------------------------------------------- settings ----------------------------------------------------------------------------------------------------- #
    switch_highlight = {       # models to highlight
        'by_dTas':             False,
        'by_org_hur_corr':     False,
        'by_obs_sim':          False,
        }
    
    switch = {                 # overall settings
        # plot type
        'scatter':             False,
        'bins':                True,
        # show/save
        'show':                False,
        'save_test_desktop':   True,
        'save_folder_desktop': False,
        'save_folder_cwd':     False,
        }
    
# ----------------------------------------------------------------------------------- run ----------------------------------------------------------------------------------------------------- #
    run_trend_plot(switch_metricX, switch_metricY, switchX, switchY, switch_highlight, switch)













