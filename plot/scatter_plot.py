import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats

import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util')
import myFuncs as mF # imports common operators
import myVars as mV # imports common variables

# ---------------------------------------------------------------------------------------- plot / format plot ----------------------------------------------------------------------------------------------------- #

def plot_scatter(x, y, metric_x, metric_y, title = '', xmin = None, xmax = None, ymin = None, ymax = None):
    fig, ax = mF.create_figure(width = 8, height = 5.5)
    ax.scatter(x, y, facecolors='none', edgecolor= metric_y.color)    
    res= stats.pearsonr(x,y)
    print('r: ', res[0])
    print('p-value:', res[1])
    if res[1]<=0.05:
        ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', 
                    xytext=(0.8, 0.9), textcoords='axes fraction', fontsize = 12, color = 'r')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    mF.move_col(ax, 0)
    mF.move_row(ax, 0.03)
    mF.scale_ax_x(ax, 1)
    mF.scale_ax_y(ax, 1)
    mF.plot_xlabel(fig, ax, metric_x.label, pad=0.1, fontsize = 12)
    mF.plot_ylabel(fig, ax, metric_y.label, pad = 0.075, fontsize = 12)
    mF.plot_axtitle(fig, ax, title, xpad = 0, ypad = 0.0075, fontsize = 12)
    return fig

# ---------------------------------------------------------------------------------- Find metric / units and run ----------------------------------------------------------------------------------------------------- #

def get_mean_value(source, dataset, metric, experiment):
    if metric.option == 'ecs':
        mean_value = mV.ecs_list[dataset]  
    elif  metric.variable_type in ['pr', 'org']: # metrics from these variables are always on daily timescale and then resampled
        timescale = 'daily'
        metric_name = metric.name
        metric_option = metric.option
        dataset_alt = 'GPCP' if dataset in ['ERA5', 'CERES'] else dataset # GPCP is used as corresponding precipitation for these obs
        path = f'{mV.folder_save[0]}/{metric.variable_type}/metrics/{metric_name}/{source}/{dataset_alt}_{metric_name}_{mV.conv_percentiles[0]}thPrctile_{timescale}_{experiment}_{mV.resolutions[0]}.nc' if metric.variable_type == 'org' else None     
        path = f'{mV.folder_save[0]}/{metric.variable_type}/metrics/{metric_name}/{source}/{dataset_alt}_{metric_name}_{timescale}_{experiment}_{mV.resolutions[0]}.nc'                                   if metric.variable_type == 'pr' else path     
        mean_value = xr.open_dataset(path)[metric_option].mean(dim='time').data
        mean_value = xr.open_dataset(path)[metric_option].sel(time = slice('2000-03', '2021')).mean(dim='time').data if dataset == 'CERES' else mean_value # full data between
    else:    
        metric_name = f'{metric.name}_sMean'
        metric_option = f'{metric.option}_sMean'
        timescale = mV.timescales[0] 
        path = f'{mV.folder_save[0]}/{metric.variable_type}/metrics/{metric_name}/{source}/{dataset}_{metric_name}_{timescale}_{experiment}_{mV.resolutions[0]}.nc'
        mean_value = xr.open_dataset(path)[metric_option].mean(dim='time').data
    return mean_value

def get_array(switch, dataset, metric):
    array_mean = []
    axtitle = ''
    for dataset in mV.datasets:
        source = mF.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        if switch['climatology']:
            title = '_clim'
            mean_value = get_mean_value(source, dataset, metric, mV.experiments[0])

        if switch['change with warming']:
            title = '_change_with_warming'
            value_historical = get_mean_value(source, dataset, metric, mV.experiments[0])
            value_warm = get_mean_value(source, dataset, metric, mV.experiments[1])
            mean_value = value_warm - value_historical 
            if switch['per_kelvin']:
                tas_historical = xr.open_dataset(f'{mV.folder_save[0]}/tas/metrics/tas_sMean/{source}/{dataset}_tas_sMean_{mV.timescales[0]}_{mV.experiments[0]}_{mV.resolutions[0]}.nc')['tas_sMean'].mean(dim='time')
                tas_warm = xr.open_dataset(f'{mV.folder_save[0]}/tas/metrics/tas_sMean/{source}/{dataset}_tas_sMean_{mV.timescales[0]}_{mV.experiments[1]}_{mV.resolutions[0]}.nc')['tas_sMean'].mean(dim='time')
                tas_change = tas_warm - tas_historical
                mean_value = mean_value/tas_change
            mean_value = value_historical if metric.option == 'ecs' else mean_value
            mean_value = value_warm - value_historical if metric.option == 'tas' else mean_value

        array_mean = np.append(array_mean, mean_value)
    return array_mean, title, axtitle


def run_scatter_plot(switch):
    keys = [k for k, v in switch.items() if v]  # list of True keys
    switch_x, switch_y = switch.copy(), switch.copy() 
    switch_x[keys[1]] = False # sets second variable to False
    switch_y[keys[0]] = False # sets first variable to False
    metric_x, metric_y = mF.get_metric_object(switch_x), mF.get_metric_object(switch_y)
    if not switch['xy']:
        metric_x, metric_y = metric_y, metric_x # plotting the reverse relationship
    source_list = mF.find_list_source(mV.datasets, mV.models_cmip5, mV.models_cmip6, mV.observations)
    ifWith_obs = mF.find_ifWithObs(mV.datasets, mV.observations)
    print(f'switch: {[key for key, value in switch.items() if value]}')
    print(f'Plotting {metric_x.option} and {metric_y.option} correlation with {mV.resolutions[0]} data')
    print(f'Plotting {keys[0]} and {keys[1]}, {mV.timescales[0]} correlation {ifWith_obs} \n with conv threshold at {mV.conv_percentiles[0]}th percentile of {mV.resolutions[0]} daily precipitation data')
    
    x, title, _ = get_array(switch, mV.datasets, metric_x)        
    y, _, _ = get_array(switch, mV.datasets, metric_y)     
    title = f'{metric_y.option}_and_{metric_x.option}{title}_{source_list}{ifWith_obs}' if not switch['fixed area'] else f'{metric_y.option}_and_{metric_x.option}_fixed_area{title}{source_list}{ifWith_obs}'

    xmin, xmax = mF.find_limits(switch, [mV.datasets[0]], metric_x, get_array, 
                                quantileWithin_low = 0, quantileWithin_high = 1, quantileBetween_low = 0, quantileBetween_high=1, # if calculating limits (set lims to '', or comment out)
                                vmin = None, vmax = None                                                                          # if manually setting limits
                                )    
    ymin, ymax = mF.find_limits(switch, [mV.datasets[0]], metric_y, get_array, 
                                quantileWithin_low = 0, quantileWithin_high = 1, quantileBetween_low = 0, quantileBetween_high=1, # if calculating limits (set lims to '', or comment out)
                                vmin = None, vmax = None                                                                          # if manually setting limits
                                )    
    fig = plot_scatter(x, y, metric_x, metric_y, title, xmin, xmax, ymin, ymax)

    folder = f'{mV.folder_save[0]}/corr/{metric_x.name}_and_{metric_y.name}'
    filename = title
    mF.save_figure(fig, folder, f'{filename}.pdf')     if switch['save'] else None
    mF.save_figure(fig, f'{home}/Desktop', 'test.pdf') if switch['save to desktop'] else None
    mF.save_figure(fig, os.getcwd(), 'test.png')       if switch['save to cwd'] else None
    plt.show() if switch['show'] else None


if __name__ == '__main__':
    run_scatter_plot(switch = {
        # -------
        # metrics
        # -------
            # organization
            'rome':                True,
            'ni':                  False,
            'areafraction':        False,

            # precipitation
            'pr':                  False,
            'pr95':                False,
            'pr97':                False,
            'pr99':                False,
            'pr99_sMean':          False,
            'rx1day_pr_sMean':     False,
            'rx5day_pr_sMean':     False,

            # Large scale state
            'ecs':                 False,
            'tas':                 False,
            'hur':                 False,
            'wap':                 True,

            # radiation
            'rlut':                False,

            # clouds
            'lcf':                 False,
            'hcf':                 False,

            # moist static energy
            'hus':                 False,

        # --------
        # settings
        # --------
        # plot type
        'xy':                  True,
        'climatology':         True,
        'change with warming': False,

        # masked by
        'descent':             False,
        'ascent':              False,
        'fixed area':          False,
        'per_kelvin':          True,

        # show/save
        'show':                False,
        'save':                False,
        'save to desktop':     True,
        'save to cwd':         False
        }
    )




