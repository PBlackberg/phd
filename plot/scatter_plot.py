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

def get_data(source, dataset, metric):
    timescale = 'daily' if  metric.option in ['rome','rome_fixed_area', 'pr99', 'rx1day_pr', 'rx5day_pr'] else mV.timescales[0]
    dataset_alt = 'GPCP' if dataset in ['ERA5', 'CERES'] and metric.option in ['rome', 'pr99'] else dataset
    metric_name = metric.name if metric.option in ['rome','rome_fixed_area', 'pr99'] else f'{metric.option}_sMean'
    metric_option = metric.option if metric.option in ['rome','rome_fixed_area', 'pr99'] else f'{metric.option}_sMean'
    path = f'{mV.folder_save[0]}/{metric.variable_type}/metrics/{metric_name}/{source}/{dataset_alt}_{metric_name}_{timescale}_{mV.experiments[0]}_{mV.resolutions[0]}.nc' 
    return xr.open_dataset(path)[metric_option] if not dataset == 'CERES' else xr.open_dataset(path)[metric_option].sel(time = slice('2000-03', '2021'))

def calc_anomalies(array, timescale):
    if timescale == 'monthly': 
        climatology = array.groupby('time.month').mean('time')
        return array.groupby('time.month') - climatology 
    if timescale == 'daily': 
        rolling_mean = array.rolling(time=12, center=True).mean()
        array = array - rolling_mean
        return array.dropna(dim='time')

def calc_metric(switch, dataset, metric):
    source = mF.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
    array = get_data(source, dataset, metric)
    array = mF.resample_timeMean(array, mV.timescales[0])
    if switch['anomalies']:
        array = calc_anomalies(array, mV.timescales[0])
    return array

# ---------------------------------------------------------------------------------------- Plot / format plot ----------------------------------------------------------------------------------------------------- #

def plot_ax_scatter(switch, ax, x, y, metric_1):
    if switch['bins']:
        pcm = ax.hist2d(x,y,[20,20], cmap = metric_1.cmap)
        bin_width = (x.max() - x.min())/100 # Each bin is one percent of the range of x values
        bins = np.arange(x.min(), x.max() + bin_width, bin_width)
        y_bins = []
        for i in np.arange(0,len(bins)-1):
            y_bins = np.append(y_bins, y.where((x>=bins[i]) & (x<bins[i+1])).mean())
        ax.plot(bins[:-1], y_bins, metric_1.color)
    else:
        pcm = ax.scatter(x, y, facecolors='none', edgecolor= metric_1.color)    
    return pcm

def plot_correlation(ax, x,y, position, fontsize):
    res= stats.pearsonr(x,y)
    if res[1]<=0.05:
        ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', xytext=position, textcoords='axes fraction', fontsize = fontsize, color = 'r')

def plot_one_scatter(switch, metric_0, metric_1):
    fig, ax = mF.create_figure(width = 8, height = 5.5)
    x = calc_metric(switch, mV.datasets[0], metric_0)
    y = calc_metric(switch, mV.datasets[0], metric_1)
    x = x.assign_coords(time=y.time) if mV.timescales[0] == 'monthly' and switch['rome'] else x

    pcm = plot_ax_scatter(switch, ax, x, y, metric_1)
    plot_correlation(ax, x,y, position = (0.8, 0.9), fontsize = 12)

    mF.move_col(ax, -0.035)
    mF.move_row(ax, 0.03)
    mF.scale_ax_x(ax, 1)
    mF.scale_ax_y(ax, 1)
    mF.plot_xlabel(fig, ax, metric_0.label, pad=0.1, fontsize = 12)
    mF.plot_ylabel(fig, ax, metric_1.label, pad = 0.075, fontsize = 12)
    mF.plot_axtitle(fig, ax, mV.datasets[0], xpad = 0, ypad = 0.0075, fontsize = 12)

    if switch['bins']:
        mF.cbar_right_of_axis(fig, ax, pcm[3], width_frac= 0.05, height_frac=1, pad=0.015, numbersize = 9, cbar_label = 'months [Nb]', text_pad = 0.05)


def plot_multiple_scatter(switch, metric_0, metric_1):
    nrows, ncols = 5, 4
    fig, axes = mF.create_figure(width = 12, height = 8, nrows=nrows, ncols=ncols)
    num_subplots = len(mV.datasets)
    for i, dataset in enumerate(mV.datasets):
        row = i // ncols  # determine row index
        col = i % ncols   # determine col index
        ax = axes.flatten()[i]

        y = calc_metric(switch, dataset, metric_1)
        x = calc_metric(switch, dataset, metric_0)
        x = x.assign_coords(time=y.time) #if mV.timescales[0] == 'monthly' and metric_0.option in ['rome', 'rome_equal_area'] else x
            
        pcm = plot_ax_scatter(switch, ax, x, y, metric_1)
        plot_correlation(ax, x,y, position = (0.675, 0.85), fontsize = 8)

        mF.move_col(ax,  -0.0715+0.0025)   if col == 0 else None
        mF.move_col(ax, -0.035)            if col == 1 else None
        mF.move_col(ax, 0.0)               if col == 2 else None
        mF.move_col(ax, 0.035)             if col == 3 else None

        mF.move_row(ax, 0.0875)            if row == 0 else None
        mF.move_row(ax, 0.0495)            if row == 1 else None
        mF.move_row(ax, 0.01)              if row == 2 else None
        mF.move_row(ax, -0.0195)           if row == 3 else None
        mF.move_row(ax, -0.05)             if row == 4 else None

        mF.scale_ax_x(ax, 0.9)
        mF.scale_ax_y(ax, 1)

        if i >= num_subplots-ncols:
            mF.plot_xlabel(fig, ax, metric_0.label, pad=0.055, fontsize = 10)

        if col == 0:
            mF.plot_ylabel(fig, ax, metric_1.label, pad = 0.0475, fontsize = 10)
        mF.plot_axtitle(fig, ax, dataset, xpad = 0, ypad = 0.0075, fontsize = 10)


        if switch['bins']:
            if col == 3:
                mF.cbar_right_of_axis(fig, ax, pcm[3], width_frac= 0.05, height_frac=1, pad=0.015, numbersize = 9, cbar_label = 'months [Nb]', text_pad = 0.035)
            else:
                mF.cbar_right_of_axis(fig, ax, pcm[3], width_frac= 0.05, height_frac=1, pad=0.015, numbersize = 9, cbar_label = '', text_pad = 0.1)

    # ax.text(0.5, 0.9625, title, ha = 'center', fontsize = 15, transform=fig.transFigure) # title
    mF.delete_remaining_axes(fig, axes, num_subplots, nrows, ncols)
    return fig

# ---------------------------------------------------------------------------------- Find metric / labels and run ----------------------------------------------------------------------------------------------------- #

def save_the_plot(switch, fig, metric_0, metric_1):
    source = mF.find_list_source(mV.datasets, mV.models_cmip5, mV.models_cmip6, mV.observations)
    with_obs = mF.find_ifWithObs(mV.datasets, mV.observations)

    folder = f'{mV.folder_save[0]}/corr/{metric_0.option}_and_{metric_1.option}'
    filename = f'{metric_0.option}_and_{metric_1.option}'
    filename = f'{mV.datasets[0]}_{filename}' if switch['one dataset'] else f'{source}_{filename}{with_obs}'

    mF.save_figure(fig, folder, f'{filename}.pdf') if switch['save'] else None
    mF.save_figure(fig, f'{home}/Desktop', f'{filename}.pdf') if switch['save to desktop'] else None

@mF.timing_decorator
def run_scatter_plot(switch):
    keys = [k for k, v in switch.items() if v]  # list of True keys
    switch_0, switch_1 = switch.copy(), switch.copy() 
    switch_0[keys[1]] = False # sets second variable to False
    switch_1[keys[0]] = False # sets first variable to False
    metric_0, metric_1 = mF.get_metric_object(switch_0), mF.get_metric_object(switch_1)
    if not switch['xy']:
        metric_0, metric_1 = metric_1, metric_0 # if plotting the reverse relationship

    print(f'switch: {[key for key, value in switch.items() if value]}')
    print(f'Plotting {keys[0]} and {keys[1]} correlation with {mV.resolutions[0]} data')

    fig = plot_multiple_scatter(switch, metric_0, metric_1) if not switch['one dataset'] else plot_one_scatter(switch, metric_0, metric_1)

    save_the_plot(switch, fig, metric_0, metric_1) if switch['save'] or switch['save to desktop'] else None
    plt.show() if switch['show'] else None


if __name__ == '__main__':
    run_scatter_plot(switch = {
        # metrics
            # organization
            'rome':                True,
            'rome_fixed_area':     False,

            # other
            'pr':                  False,
            'pr99':                True,
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
        'anomalies':           True,

        # plot modifications
        'bins':                True,
        'xy':                  True,

        # run/show/save
        'one dataset':         False,
        'show':                True,
        'save':                False,
        'save to desktop':     True
        }
    )




