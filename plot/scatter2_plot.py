import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats

import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/util')
import myFuncs as mF # imports common operators
import myVars as mV # imports common variables


# ---------------------------------------------------------------------------------------- Calculation ----------------------------------------------------------------------------------------------------- #

def get_data(source, dataset, options, metric, experiment):
    if metric.option in ['rome']:
        folder = metric.get_metric_folder(mV.folder_save[0], f'{metric.name}', source)
        filename = metric.get_filename(f'{metric.name}', source, dataset, 'daily', experiment, options.resolution)
        array = xr.open_dataset(f'{folder}/{filename}')[f'{metric.option}']
        array = mF.resample_timeMean(array, options.timescale)
        mean_value = array.mean(dim=('time'))
    elif metric.option in ['pr99']:
        folder = metric.get_metric_folder(mV.folder_save[0], metric.name, source)
        filename = metric.get_filename(metric.name, source, dataset, 'daily', experiment, options.resolution)
        array = xr.open_dataset(f'{folder}/{filename}')[f'{metric.option}']
        mean_value = array.mean(dim=('time'))
    elif metric.option in ['rx1day_pr', 'rx5day_pr']:
        folder = metric.get_metric_folder(mV.folder_save[0], f'{metric.name}_sMean', source)
        filename = metric.get_filename(f'{metric.name}_sMean', source, dataset, 'daily', options.experiment[0], options.resolution)
        array = xr.open_dataset(f'{folder}/{filename}')[f'{metric.option}']    
        mean_value = array.mean(dim=('time'))
    elif metric.option in ['ecs']:
        mean_value = mV.ecs_list[dataset]
    else:
        folder = metric.get_metric_folder(mV.folder_save[0], f'{metric.name}_sMean', source)
        filename = metric.get_filename(f'{metric.name}_sMean', source, dataset, options.timescale, experiment, options.resolution)
        array = xr.open_dataset(f'{folder}/{filename}')[f'{metric.option}_sMean']
        mean_value = array.mean(dim=('time'))
    return mean_value

def calc_metric(switch, dataset, options, metric):
    source = mF.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
    if switch['climatology']:
        mean_value = get_data(source, dataset, options, metric, options.experiment[0])
    if switch['change with warming']:
        array_historical = get_data(switch, source, dataset, options, metric, options.experiment[0])[metric.option]
        array_warm = get_data(switch, source, dataset, options, metric, options.experiment[1])[metric.option]
        mean_value = array_warm - array_historical 
    return mean_value
    
def create_list(switch, dataset, options, metric_0, metric_1):
    x,y = [],[]
    for dataset in mV.datasets:
        x = np.append(x, calc_metric(switch, dataset, options, metric_0))
        y = np.append(y, calc_metric(switch, dataset, options, metric_1))
    return x,y

# ---------------------------------------------------------------------------------------- formatting plot ----------------------------------------------------------------------------------------------------- #

def find_title(switch, metric_0, metric_1):
    title = f'{metric_0.option}_and_{metric_1.option}_clim'                if switch['climatology'] else title
    title = f'{metric_0.option}_and_{metric_1.option}_change with warming' if switch['change with warming'] else title
    return title

def plot_ax_scatter(ax, x, y, metric_1):
    pcm = ax.scatter(x, y, facecolors='none', edgecolor= metric_1.color)    
    return pcm

def plot_correlation(ax, x,y, position, fontsize):
    res= stats.pearsonr(x,y)
    print('r: ', res[0])
    print('p-value:', res[1])
    if res[1]<=0.05:
        ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', xytext=position, textcoords='axes fraction', fontsize = fontsize, color = 'r')

def plot_one_scatter(switch, datasets, options, metric_0, metric_1):
    fig, ax = mF.create_figure(width = 8, height = 5.5)
    x,y = create_list(switch, datasets, options, metric_0, metric_1)        
    pcm = plot_ax_scatter(ax, x, y, metric_1)
    plot_correlation(ax, x,y, position = (0.8, 0.9), fontsize = 12)

    mF.move_col(ax, 0)
    mF.move_row(ax, 0.03)
    mF.scale_ax_x(ax, 1)
    mF.scale_ax_y(ax, 1)
    mF.plot_xlabel(fig, ax, metric_0.label, pad=0.1, fontsize = 12)
    mF.plot_ylabel(fig, ax, metric_1.label, pad = 0.075, fontsize = 12)
    mF.plot_axtitle(fig, ax, find_title(switch, metric_0, metric_1), xpad = 0, ypad = 0.0075, fontsize = 12)

# ---------------------------------------------------------------------------------- Find metric / units and run ----------------------------------------------------------------------------------------------------- #

def run_scatter_plot(switch):
    if not switch['run']:
        return
    options = mF.dataset_class(mV.timescales[0], mV.experiments, mV.resolutions[0])
    keys = [k for k, v in switch.items() if v]  # list of True keys
    switch_0, switch_1 = switch.copy(), switch.copy() 
    switch_0[keys[1]] = False # sets second variable to False
    switch_1[keys[0]] = False # sets first variable to False
    metric_0, metric_1 = mF.get_metric_object(switch_0), mF.get_metric_object(switch_1)
    if not switch['xy']:
        metric_0, metric_1 = metric_1, metric_0 # plotting the reverse relationship

    print(f'switch: {[key for key, value in switch.items() if value]}')
    print(f'Plotting {metric_0.option} and {metric_1.option} correlation with {options.resolution} data')

    fig = plot_one_scatter(switch, mV.datasets, options, metric_0, metric_1)

    if switch['save'] or switch['save to desktop']:
        source = mF.find_list_source(mV.datasets, mV.models_cmip5, mV.models_cmip6, mV.observations)
        with_obs = mF.find_ifWithObs(mV.datasets, mV.observations)

        folder = f'{mV.folder_save[0]}/corr/{metric_0.option}_and_{metric_1.option}_mean'
        filename = f'{metric_0.option}_and_{metric_1.option}_mean'
        filename = f'{source}_{filename}{with_obs}'

        mF.save_figure(fig, folder, f'{filename}.pdf') if switch['save'] else None
        mF.save_figure(fig, f'{home}/Desktop', f'{filename}.pdf') if switch['save to desktop'] else None

    plt.show() if switch['show'] else None


if __name__ == '__main__':
    run_scatter_plot(switch = {
        # metrics
            # organization
            'rome':                False,

            # other
            'ecs':                 True,
            'pr':                  False,
            'pr99':                True,
            'pr99_meanIn':         False,
            'rx1day_pr':           False,
            'rx5day_pr':           False,

            'wap':                 False,
            'tas':                 False,

            'hus':                 False,
            'hur':                 False,
            'rlut':                False,

            'lcf':                 False,
            'hcf':                 False,

        # masked by
        'descent':             False,
        'ascent':              False,

        # metric calculation
        'climatology':         True,
        'change with warming': False,

        # plot modifications
        'xy':                  True,

        # show/save
        'run':                 True,
        'show':                True,
        'save':                False,
        'save to desktop':     False
        }
    )




