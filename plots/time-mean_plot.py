import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
from scipy import stats
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV
import myFuncs as mF    
import myClasses as mC



# ------------------------
#     Get datapoint
# ------------------------
# ---------------------------------------------------------------------------------------- get datapoint ----------------------------------------------------------------------------------------------------- #
def get_orig_res(dataset, source):
    da = xr.open_dataset(f'/Users/cbla0002/Documents/data/sample_data/pr/{source}/{dataset}_pr_daily_historical_orig.nc')['pr']
    dlat, dlon = da['lat'].diff(dim='lat').data[0], da['lon'].diff(dim='lon').data[0]
    resolution = dlat * dlon
    return dlat, dlon, resolution


def get_datapoint(switchM, dataset, metric_class):
    source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
    timescale = 'daily' if metric_class.var_type in ['pr', 'org', 'hus', 'ws'] else 'monthly'

    axtitle = ''
    metric_title = ''
    if switchM['clim']:
            metric_title = '_clim'
            if source == 'obs':
                experiment  = ''
                aslice = slice('2010', '2018') if dataset in ['ISCCP']         else slice('2010', '2022')        
                aslice = slice('2010', '2022') if dataset in ['CERES', 'ERA5'] else aslice    
                dataset = 'GPCP_2010-2022' if metric_class.var_type in ['pr', 'org'] else dataset # pick a time range
                alist = mF.load_metric(metric_class, mV.folder_save[0], source, dataset, timescale, experiment, mV.resolutions[0])[metric_class.name]
                alist = alist.sel(time = aslice)
                datapoint = alist.mean(dim='time')
            else:
                datapoint = mF.load_metric(metric_class, mV.folder_save[0], source, dataset, timescale, mV.experiments[0], mV.resolutions[0])[metric_class.name].mean(dim='time')

    elif switchM['change_with_warming']:
        title = '_change_with_warming'
        metric_title = '_change_with_warming'
        datapoint_historical = mF.load_metric(metric_class, mV.folder_save[0], source, dataset, timescale, mV.experiments[0], mV.resolutions[0])[metric_class.name].mean(dim='time')
        datapoint_warm       = mF.load_metric(metric_class, mV.folder_save[0], source, dataset, timescale, mV.experiments[1], mV.resolutions[0])[metric_class.name].mean(dim='time')
        datapoint = datapoint_warm - datapoint_historical
        if switchM['per kelvin']:
            metric_title = f'{title}_per_K (dTas)'
            tas_class      = mC.get_metric_class('tas', {'sMean': True, 'ascent': False, 'descent': False})
            tas_historical = mF.load_metric(tas_class, mV.folder_save[0], source, dataset, mV.timescales[0], mV.experiments[0], mV.resolutions[0])[tas_class.name].mean(dim='time')
            tas_warm       = mF.load_metric(tas_class, mV.folder_save[0], source, dataset, mV.timescales[0], mV.experiments[1], mV.resolutions[0])[tas_class.name].mean(dim='time')
            tas_change = tas_warm - tas_historical
            datapoint = datapoint/tas_change
        if switchM['per ecs']:
            metric_title = f'{title}_per_K (ECS)'
            ecs = mV.ecs_list[dataset] 
            datapoint = datapoint/ecs
    else:
        dlat, dlon, resolution = get_orig_res(dataset, source)
        datapoint = dlat       if metric_class.name == 'dlat'       else None
        datapoint = dlon       if metric_class.name == 'dlon'       else datapoint
        datapoint = resolution if metric_class.name == 'res'        else datapoint

    datapoint = mV.ecs_list[dataset] if metric_class.name == 'ecs' else datapoint
    return datapoint, metric_title, axtitle


# ---------------------------------------------------------------------------------------- get limits ----------------------------------------------------------------------------------------------------- #
def get_limits(switchM, metric_class, datasets):
    qWithin_low, qWithin_high, qBetween_low, qBetween_high = 0, 1, 0, 1 # quantiles
    vmin, vmax = mF.find_limits(switchM, datasets, metric_class, get_datapoint, 
                                qWithin_low, qWithin_high, qBetween_low, qBetween_high, # if calculating limits (set lims to '', or comment out)
                                vmin = '', vmax = ''                                    # if manually setting limits
                                )   
    vmin, vmax = [None, None]
    return vmin, vmax



# --------------------------
#     Correlation plot
# --------------------------
# ---------------------------------------------------------------------------------------- plot figure ----------------------------------------------------------------------------------------------------- #
def plot_scatter(switchX, switchY, switch_highlight, metric_classX, metric_classY, xmin = None, xmax = None, ymin = None, ymax = None):
    fig, ax = mF.create_figure(width = 11, height = 6.25)
    mF.scale_ax_x(ax, scaleby=0.75)
    x, y = [], []
    for dataset in mV.datasets:
        x_datapoint, x_metric_title, _ = get_datapoint(switchX, dataset, metric_classX)
        x = np.append(x, x_datapoint)

        y_datapoint, y_metric_title, _ = get_datapoint(switchY, dataset, metric_classY)
        y = np.append(y, y_datapoint)

    fig_title = f'{metric_classX.name} {x_metric_title} and \n {metric_classY.name}{y_metric_title}'

    ax.scatter(x, y, alpha=0)  # Completely transparent points (putting text over)    
    res= stats.pearsonr(x,y) if not mV.find_ifWithObs(mV.datasets, mV.observations) else stats.pearsonr(x[0:-1],y[0:-1])
    print('r: ', res[0])
    print('p-value:', res[1])
    if res[1]<=0.05:
        ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', 
                    xytext=(0.8, 0.9), textcoords='axes fraction', fontsize = 12, color = 'r')
    model_highlight = mV.get_ds_highlight(switch_highlight, mV.datasets, mV.switch_subset, mV.exclude_models)
    legend_handles, legend_labels = [], []
    for dataset in mV.datasets: # put 2 letters as label for each dataset (with associated legend)
        dataset_idx = mV.datasets.index(dataset)
        label_text = dataset[0:2]
        label_text = f'{label_text}-L' if dataset[-2] == 'L' else label_text 
        label_text = f'{label_text}-H' if dataset[-2] == 'H' else label_text
        text_color = 'b' if dataset in model_highlight                                                            else 'k'        # highlighted model
        text_color = 'r' if mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations) in ['obs'] else text_color # highlighted obs
        ax.text(x[dataset_idx], y[dataset_idx], label_text, color=text_color, ha='center', va='center')
        legend_handles.append(Patch(facecolor=text_color, edgecolor='none'))
        legend_labels.append(f"{label_text} - {dataset}")
    ax.legend(handles=legend_handles, labels=legend_labels, bbox_to_anchor=(1.1, 0.475), loc='center left')

    plt.axhline(y=0, color='k', linestyle='--') if np.min(y)<0 and np.max(y)>0 else None
    plt.axvline(x=0, color='k', linestyle='--') if np.min(x)<0 and np.max(x)>0 else None

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
    mF.plot_axtitle(fig, ax, fig_title, xpad = 0, ypad = 0.0075, fontsize = 12)
    return fig, fig_title



# ------------------------
#     Run / save plot
# ------------------------
# ----------------------------------------------------------------------------------- Find metric / labels and run ----------------------------------------------------------------------------------------------------- #
def run_corr(switchX, switchY, switch_highlight, switch, metric_classX, metric_classY):
    xmin, xmax = get_limits(switchX, metric_classX, mV.datasets)
    ymin, ymax = get_limits(switchY, metric_classY, mV.datasets)
    fig, fig_title = plot_scatter(switchX, switchY, switch_highlight, metric_classX, metric_classY, xmin, xmax, ymin, ymax)

    source_list = mV.find_list_source(mV.datasets, mV.models_cmip5, mV.models_cmip6, mV.observations)
    ifWith_obs = mV.find_ifWithObs(mV.datasets, mV.observations)
    filename = f'{source_list}_{fig_title}{ifWith_obs}'

    mF.save_plot(switch, fig, home, filename)
    plt.show() if switch['show'] else None

def run_timeMean_corr(switch_metricX, switch_metricY, switchX, switchY, switch_highlight, switch):
    print(f'Plotting {mV.timescales[0]} correlation between')
    print(f'metricX: {[key for key, value in switch_metricX.items() if value]} {[key for key, value in switchX.items() if value]}')
    print(f'metricY: {[key for key, value in switch_metricY.items() if value]} {[key for key, value in switchY.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')

    metricX = next((key for key, value in switch_metricX.items() if value), None)
    for metricY in [k for k, v in switch_metricY.items() if v]:
        metric_classX = mC.get_metric_class(metricX, switchX, prctile = mV.conv_percentiles[0])
        metric_classY = mC.get_metric_class(metricY, switchY, prctile = mV.conv_percentiles[0])
        run_corr(switchX, switchY, switch_highlight, switch, metric_classX, metric_classY)



# ------------------------
#  Choose what to plot
# ------------------------
if __name__ == '__main__':
# ---------------------------------------------------------------------------------- x-metric ----------------------------------------------------------------------------------------------------- #
    switch_metricX = {                                                                                                  # pick y-metric (can pick multiple)
        'rome':         True,  'ni':           False,  'areafraction': False,  'F_pr10':   False,                      # organization
        'pr':           False,  'pr_99':        False,  'pr_97':        False,  'pr_95':    False,  'pr_90':    False,  # precipitation percentiles
        'pr_rx1day':    False,  'pr_rx5day':    False,                                                                  # precipitation extremes
        'wap':          False,                                                                                          # descent/ascent
        'hur':          False,  'hus':          False,                                                                  # humidity
        'tas':          False,  'ecs':          False,  'stability':    False,                                          # temperature
        'netlw':        False,  'rlds':         False,  'rlus':         False,  'rlut':     False,                      # LW
        'netsw':        False,  'rsdt':         False,  'rsds':         False,  'rsus':     False,  'rsut':     False,  # SW
        'lcf':          False,   'hcf':         False,                                                                  # cloudfraction
        'ws_lc':        False,  'ws_hc':        False,                                                                  # weather states
        'res':          False,  'dlat':         False,  'dlon':         False,                                          # orig resolution
        'h':            False,  'h_anom2':      False,                                                                  # moist static energy
        }
    
    switchX = {                                                                                     # choose seetings for x-metric
        '250hpa':       False,  '500hpa':               False,  '700hpa':   False,                  # mask: vertical
        'descent':      False,  'ascent':               False,  'ocean':    False,                  # mask: horizontal
        'fixed area':   False,  '90':                   False,  '95':       False,  '97':   False,  # conv threshold
        'sMean':        False,  'area':                 False,                                      # metric type
        'clim':         False,   'change_with_warming':  True,                                      # Scenario type   
        'per kelvin':   True,  'per ecs':              False,                                      # by warming
        }

    

# ---------------------------------------------------------------------------------- y-metric ----------------------------------------------------------------------------------------------------- #
    switch_metricY = {                                                                                                  # pick y-metric (can pick multiple)
        'rome':         False,  'ni':           False,  'areafraction': False,  'F_pr10':   False,                      # organization
        'pr':           False,  'pr_99':        False,  'pr_97':        False,  'pr_95':    False,  'pr_90':    False,  # precipitation percentiles
        'pr_rx1day':    False,  'pr_rx5day':    False,                                                                  # precipitation extremes
        'wap':          False,                                                                                          # descent/ascent
        'hur':          False,  'hus':          False,                                                                  # humidity
        'tas':          False,  'ecs':          False,  'stability':    False,                                          # temperature
        'netlw':        False,  'rlds':         False,  'rlus':         False,  'rlut':     False,                      # LW
        'netsw':        False,  'rsdt':         False,  'rsds':         False,  'rsus':     False,  'rsut':     False,  # SW
        'lcf':          False,   'hcf':          False,                                                                  # cloudfraction
        'ws_lc':        False,  'ws_hc':        False,                                                                  # weather states
        'res':          False,  'dlat':         False,  'dlon':         False,                                          # orig resolution
        'h':            False,  'h_anom2':      True,                                                                  # moist static energy
        }
    
    switchY = {                                                                                     # choose seetings for x-metric
        '250hpa':       False,  '500hpa':               False,  '700hpa':   False,                  # mask: vertical
        'descent':      False,  'ascent':               False,  'ocean':    False,                  # mask: horizontal
        'fixed area':   False,  '90':                   False,  '95':       False,  '97':   False,  # conv threshold
        'sMean':        True,   'area':                 False,                                      # metric type
        'clim':         False,   'change_with_warming':  True,                                      # Scenario type   
        'per kelvin':   True,  'per ecs':              False,                                      # by warming
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
    run_timeMean_corr(switch_metricX, switch_metricY, switchX, switchY, switch_highlight, switch)









