''' 
# -----------------------------
#   Scatter plot (time-mean)
# -----------------------------
This script plots a scatter plot to compare time-mean tropical mean properties between models
(could color models that are included in correlation calc, leaving the remaning models as black dots)

'''



# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
from scipy import stats


# ------------------------------------------------------------------------------------ imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV
import myFuncs as mF    
import myClasses as mC



# ------------------------
#       Get metric
# ------------------------
# ---------------------------------------------------------------------------------------- get datapoint ----------------------------------------------------------------------------------------------------- #
def tas_weighted(metric, title, axtitle, dataset):        
    title = f'_per_K (dTas)'
    tas_class      = mC.get_metric_class('tas', {'sMean': True, 'ascent': False, 'descent': False})
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
    metric_historical, metric_warm  = mF.load_metric(metric_class, dataset, mV.experiments[0]).mean(dim='time'), mF.load_metric(metric_class, dataset, mV.experiments[1]).mean(dim='time')
    metric = metric_warm - metric_historical   
    metric, title, axtitle = tas_weighted(metric, title, axtitle, dataset) if switchM['per_kelvin']   else [metric, title, axtitle]
    metric, title, axtitle = tas_weighted(metric, title, axtitle, dataset) if switchM['per_ecs']      else [metric, title, axtitle]
    return metric, title, axtitle

def get_orig_res(dataset, metric_class):
    da = xr.open_dataset(f'/Users/cbla0002/Documents/data/sample_data/pr/{mF.find_source(dataset)}/{dataset}_pr_daily_historical_orig.nc')['pr']
    dlat, dlon = da['lat'].diff(dim='lat').data[0], da['lon'].diff(dim='lon').data[0]
    resolution = dlat * dlon
    datapoint = dlat       if metric_class.name == 'dlat'       else None
    datapoint = dlon       if metric_class.name == 'dlon'       else datapoint
    datapoint = resolution if metric_class.name == 'res'        else datapoint
    return datapoint, '', dataset

def get_ecs(dataset):
    return mV.ecs_list[dataset], '', dataset

def get_metric(switchM, dataset, metric_class):
    ''' Gets the metric and performs calculation according to switchM '''
    if mF.find_source(dataset) in ['obs']:
        metric, title, axtitle = mF.load_metric(metric_class, dataset), 'clim', dataset
        aSlice = slice('2010', '2022') if not dataset in ['ISCCP'] else slice('2010', '2018') # organization metrics are from 2010-2022, so try to stick to that
        metric = metric.sel(time = aSlice).mean(dim = 'time')
    else:
        metric, title, axtitle = mF.load_metric(metric_class, dataset).mean(dim = 'time'), 'clim', dataset
        metric, title, axtitle = get_change_with_warming(metric_class, title, axtitle, dataset, switchM)    if switchM['change_with_warming']   else [metric, title, axtitle]
    return metric, title, axtitle

def get_datapoint(switchX, switchY, dataset, metric_classX, metric_classY):
    if metric_classX.name not in ['dlat', 'dlon', 'res', 'ecs'] and metric_classY.name not in ['dlat', 'dlon', 'res', 'ecs']:
        x, metric_titleX, axtitle = get_metric(switchX, dataset, metric_classX)  # must be this order to use in mF.find_limits
        y, metric_titleY, axtitle = get_metric(switchY, dataset, metric_classY)

    if metric_classX.name not in ['dlat', 'dlon', 'res', 'ecs']:
        x, metric_titleX, axtitle = get_metric(switchX, dataset, metric_classX)  # must be this order to use in mF.find_limits

        # x, y = x.dropna(dim='time', how='any'), y.dropna(dim='time', how='any')
        # x, y = xr.align(x, y, join='inner')
        # x, y = x.mean(dim = 'time'), y.mean(dim = 'time')

    if metric_classX.name in ['dlat', 'dlon', 'res']:    
        x, metric_titleX, axtitle = get_orig_res(dataset, metric_classX)
    if metric_classY.name in ['dlat', 'dlon', 'res']:    
        y, metric_titleY, axtitle = get_orig_res(dataset, metric_classY)
    if metric_classX.name == 'ecs':
        x, metric_titleX, axtitle = get_ecs(dataset)
    if metric_classY.name == 'ecs':
        y, metric_titleY, axtitle = get_ecs(dataset)

    return x, y, metric_titleX, metric_titleY



# ---------------------------------------------------------------------------------------- get limits ----------------------------------------------------------------------------------------------------- #
def get_limits(switchM, metric_class): 
    vmin, vmax = [None, None]
    return vmin, vmax



# --------------------------
#     Correlation plot
# --------------------------
# ---------------------------------------------------------------------------------------- plot figure ----------------------------------------------------------------------------------------------------- #
def split_conv_threshold(switchM):
    prctileM = '90' if switchM['90'] else mV.conv_percentiles[0]
    prctileM = '95' if switchM['95'] else prctileM
    prctileM = '97' if switchM['90'] else prctileM
    return prctileM

def add_correlation_coeff(x, y, ax):
    res= stats.pearsonr(x,y) if not mF.find_ifWithObs(mV.datasets) else stats.pearsonr(x[0:-len(mV.observations)],y[0:-len(mV.observations)])
    print('r: ', res[0])
    print('p-value:', res[1])
    if res[1]<=0.05:
        ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', 
                    xytext=(0.8, 0.9), textcoords='axes fraction', fontsize = 12, color = 'r')

def format_ax(metric_classX, metric_classY, fig, ax, xmin, xmax, ymin, ymax):
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    ax.xaxis.set_major_formatter(formatter)
    mF.move_col(ax, 0)
    mF.move_row(ax, 0.03)
    mF.scale_ax_x(ax, 1)
    mF.scale_ax_y(ax, 1)
    mF.plot_xlabel(fig, ax, metric_classX.label, pad=0.1, fontsize = 12)
    mF.plot_ylabel(fig, ax, metric_classY.label, pad = 0.075, fontsize = 12)

def plot_letters(switch_highlight, ax, x, y):
    model_highlight = mV.highlight_models(switch_highlight, mV.datasets, mV.switch_subset)
    legend_handles, legend_labels = [], []
    for dataset in mV.datasets: # put 2 letters as label for each dataset (with associated legend)
        dataset_idx = mV.datasets.index(dataset)
        label_text = dataset[0:2]
        label_text = f'{label_text}-L'  if dataset[-2] == 'L'    else label_text 
        label_text = f'{label_text}-H'  if dataset[-2] == 'H'    else label_text
        text_color = 'b'                if dataset in model_highlight          else 'k'        # highlighted model
        text_color = 'r'                if mF.find_source(dataset) in ['obs']  else text_color # highlighted obs
        ax.text(x[dataset_idx], y[dataset_idx], label_text, color=text_color, ha='center', va='center')
        legend_handles.append(Patch(facecolor=text_color, edgecolor='none'))
        legend_labels.append(f"{label_text} - {dataset}")
    ax.legend(handles=legend_handles, labels=legend_labels, bbox_to_anchor=(1.1, 0.475), loc='center left')


def plot_scatter_timeMean(switchX, switchY, switch_highlight, metric_nameX, metric_nameY):
    metric_classX = mC.get_metric_class(metric_nameX, switchX, prctile = split_conv_threshold(switchX))
    metric_classY = mC.get_metric_class(metric_nameY, switchY, prctile = split_conv_threshold(switchY))
    xmin, xmax = get_limits(switchX, metric_classX)
    ymin, ymax = get_limits(switchY, metric_classY)
    fig, ax = mF.create_figure(width = 11, height = 6.25)
    mF.scale_ax_x(ax, scaleby=0.75)
    x, y = [], []
    for dataset in mV.datasets:
        x_datapoint, y_datapoint, metric_titleX, metric_titleY = get_datapoint(switchX, switchY, dataset, metric_classX, metric_classY)
        x.append(x_datapoint)
        y.append(y_datapoint)    
    fig_title = f'{metric_classX.name} {metric_titleX} and \n {metric_classY.name} {metric_titleY}'
    ax.scatter(x, y, alpha=0)                                                               # Completely transparent points 
    add_correlation_coeff(x, y, ax)
    plot_letters(switch_highlight, ax, x, y)                                                                  # dataset latters at points
    plt.axhline(y=0, color='k', linestyle='--') if np.min(y)<0 and np.max(y)>0 else None    # add dashed line if datapoints cross zero 
    plt.axvline(x=0, color='k', linestyle='--') if np.min(x)<0 and np.max(x)>0 else None    # add dashed line if datapoints cross zero 
    format_ax(metric_classX, metric_classY, fig, ax, xmin, xmax, ymin, ymax)
    mF.plot_axtitle(fig, ax, fig_title, xpad = 0.15, ypad = 0.0075, fontsize = 12)
    return fig, fig_title



# ------------------------
#     Run / save plot
# ------------------------
# ----------------------------------------------------------------------------------- Find metric / labels and run ----------------------------------------------------------------------------------------------------- #
def run_scatter_timeMean(switch_metricX, switch_metricY, switchX, switchY, switch_highlight, switch):
    print(f'{mV.timescales[0]} correlation in {len(mV.datasets)} datasets')
    print(f'metricX: {[key for key, value in switch_metricX.items() if value]} {[key for key, value in switchX.items() if value]}')
    print(f'metricY: {[key for key, value in switch_metricY.items() if value]} {[key for key, value in switchY.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')
    metric_nameX = next((key for key, value in switch_metricX.items() if value), None)
    for metric_nameY in [k for k, v in switch_metricY.items() if v]:
        fig, fig_title = plot_scatter_timeMean(switchX, switchY, switch_highlight, metric_nameX, metric_nameY)
        mF.save_plot(switch, fig, home, filename = f'{mF.find_list_source(mV.datasets)}_{fig_title}{mF.find_ifWithObs(mV.datasets)}')
        plt.show() if switch['show'] else None



# ------------------------
#  Choose what to plot
# ------------------------
if __name__ == '__main__':
# ---------------------------------------------------------------------------------- x-metric ----------------------------------------------------------------------------------------------------- #
    switch_metricX = {                                                                                                  # pick y-metric (pick one)
        'pr':           False,  'pr_99':        False,  'pr_97':        False,  'pr_95':    False,  'pr_90':    False,  # precipitation percentiles
        'pr_rx1day':    False,  'pr_rx5day':    False,                                                                  # precipitation extremes
        'rome':         True,  'ni':           False,  'areafraction': False,  'F_pr10':   False,                      # organization
        'wap':          False,                                                                                          # circulation
        'hur':          False,  'hus':          False,                                                                  # humidity
        'tas':          False,  'ecs':          False,  'stability':    False,                                          # temperature
        'netlw':        False,  'rlds':         False,  'rlus':         False,  'rlut':     False,                      # LW
        'netsw':        False,  'rsdt':         False,  'rsds':         False,  'rsus':     False,  'rsut':     False,  # SW
        'lcf':          False,   'hcf':         False, 'ws_lc':        False,  'ws_hc':    False,                       # clouds
        'res':          False,  'dlat':         False,  'dlon':         False,                                          # orig resolution dims
        'h':            False,  'h_anom2':      False,                                                                  # moist static energy
        }
    
    switchX = {                                                                                     # choose seetings for x-metric
        '250hpa':       False,  '500hpa':               False,  '700hpa':   False,                  # mask: vertical
        'descent':      False,  'ascent':               False,  'ocean':    False,                  # mask: horizontal
        'fixed area':   False,  '90':                   False,  '95':       False,  '97':   False,  # conv threshold
        'sMean':        False,  'area':                 False,                                      # metric type
        'clim':         True,   'change_with_warming': False,                                      # Scenario type   
        'per_kelvin':  False, 'per_ecs':              False,                                      # by warming
        }

    

# ---------------------------------------------------------------------------------- y-metric ----------------------------------------------------------------------------------------------------- #
    switch_metricY = {                                                                                                  # pick y-metric (can pick multiple)
        'pr':           False,  'pr_99':        False,  'pr_97':        False,  'pr_95':    False,  'pr_90':    False,  # precipitation percentiles
        'pr_rx1day':    False,  'pr_rx5day':    False,                                                                  # precipitation extremes
        'rome':         False,  'ni':           False,  'areafraction': False,  'F_pr10':   False,                      # organization
        'wap':          False,                                                                                          # circulation
        'hur':          False,  'hus':          False,                                                                  # humidity
        'tas':          False,  'ecs':          True,  'stability':    False,                                          # temperature
        'netlw':        False,  'rlds':         False,  'rlus':         False,  'rlut':     False,                      # LW
        'netsw':        False,  'rsdt':         False,  'rsds':         False,  'rsus':     False,  'rsut':     False,  # SW
        'lcf':          False,   'hcf':         False, 'ws_lc':        False,  'ws_hc':    False,                       # clouds
        'res':          False,  'dlat':         False,  'dlon':         False,                                          # orig resolution dims
        'h':            False,  'h_anom2':      False,                                                                  # moist static energy
        }
    
    switchY = {                                                                                         # choose seetings for x-metric
        '250hpa':           False,  '500hpa':               False,  '700hpa':   False,                  # mask: vertical
        'descent':          False,  'ascent':               False,  'ocean':    False,                  # mask: horizontal
        'descent_fixed':    False,  'ascent_fixed':         False,                                      # mask: horizontal
        'fixed area':       False,  '90':                   False,  '95':       False,  '97':   False,  # conv threshold
        'sMean':            True,   'area':                 False,                                      # metric type
        'clim':             False,  'change_with_warming':  False,                                       # Scenario type   
        'per_kelvin':       False,  'per_ecs':              False,                                      # by warming
        }


# ----------------------------------------------------------------------------------- settings ----------------------------------------------------------------------------------------------------- #
    switch_highlight = {                                                                                  
        'by_dTas':             False, 'by_org_hur_corr':    False, 'by_obs_sim':      False, 'by_excluded': False,      # models to highlight
        }
    
    switch = {                                                                                                          # overall settings
        'show':                False,                                                                                   # show
        'save_test_desktop':   True, 'save_folder_desktop': False, 'save_folder_cwd': False,                            # save
        }
    
# -------------------------------------------------------------------------------------- run ----------------------------------------------------------------------------------------------------- #
    run_scatter_timeMean(switch_metricX, switch_metricY, switchX, switchY, switch_highlight, switch)









