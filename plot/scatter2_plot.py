import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
import timeit

import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/util')
import myFuncs as mF # imports common operators
import myVars as mV # imports common variables



# ---------------------------------------------------------------------------------------- Calculation ----------------------------------------------------------------------------------------------------- #















# ---------------------------------------------------------------------------------------- formatting plot ----------------------------------------------------------------------------------------------------- #

def plot_correlation(ax, x,y, position, fontsize):
    res= stats.pearsonr(x,y)
    if res[1]<=0.05:
        ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', xytext=position, textcoords='axes fraction', fontsize = fontsize, color = 'r')

def plot_ax_scatter(ax, x, y, metric_1):
    pcm = ax.scatter(x, y, facecolors='none', edgecolor= metric_1.color)    
    return pcm

def plot_one_scatter(switch, dataset, options, metric_0, metric_1):
    fig, ax = mF.create_figure(width = 8, height = 5.5)
    x = calc_metric(switch, dataset, options, metric_0)
    y = calc_metric(switch, dataset, options, metric_1)
    x = x.assign_coords(time=y.time) if options.timescale == 'monthly' and switch['rome'] else x
        
    pcm = plot_ax_scatter(ax, x, y, metric_1)
    plot_correlation(ax, x,y, position = (0.8, 0.9), fontsize = 12)

    mF.move_col(ax, -0.035)
    mF.move_row(ax, 0.03)
    mF.scale_ax_x(ax, 1)
    mF.scale_ax_y(ax, 1)
    mF.plot_xlabel(fig, ax, metric_0.label, pad=0.1, fontsize = 12)
    mF.plot_ylabel(fig, ax, metric_1.label, pad = 0.075, fontsize = 12)
    mF.plot_axtitle(fig, ax, dataset, xpad = 0, ypad = 0.0075, fontsize = 12)

    if switch['bins']:
        mF.cbar_right_of_axis(fig, ax, pcm[3], width_frac= 0.05, height_frac=1, pad=0.015, numbersize = 9, cbar_label = 'months [Nb]', text_pad = 0.05)




# ---------------------------------------------------------------------------------- Find metric / units and run ----------------------------------------------------------------------------------------------------- #

def run_scatter_plot(switch):
    options = mF.dataset_class(mV.timescales[0], mV.experiments[0], mV.resolutions[0])
    keys = [k for k, v in switch.items() if v]  # list of True keys
    switch_0, switch_1 = switch.copy(), switch.copy() 
    switch_0[keys[1]] = False # sets second variable to False
    switch_1[keys[0]] = False # sets first variable to False
    metric_0, metric_1 = mF.get_metric_object(switch_0), mF.get_metric_object(switch_1)
    if not switch['xy']:
        metric_0, metric_1 = metric_1, metric_0 # plotting the reverse relationship

    print(f'switch: {[key for key, value in switch.items() if value]}')
    print(f'Plotting {keys[0]} and {keys[1]} correlation with {options.resolution} data')

    fig = plot_one_scatter(switch, mV.datasets[0], options, metric_0, metric_1)

    if switch['save'] or switch['save to desktop']:
        source = mF.find_list_source(mV.datasets, mV.models_cmip5, mV.models_cmip6, mV.observations)
        with_obs = mF.find_ifWithObs(mV.datasets, mV.observations)

        folder = f'{mV.folder_save[0]}/corr/{metric_0.option}_and_{metric_1.option}'
        filename = f'{metric_0.option}_and_{metric_1.option}'
        filename = f'{mV.datasets[0]}_{filename}' if switch['one dataset'] else f'{source}_{filename}{with_obs}'

        mF.save_figure(fig, folder, f'{filename}.pdf') if switch['save'] else None
        mF.save_figure(fig, f'{home}/Desktop', f'{filename}.pdf') if switch['save to desktop'] else None

    plt.show() if switch['show'] else None




if __name__ == '__main__':
    start = timeit.default_timer()
    # choose which metrics to plot
    switch = {
        # metrics
            # organization
            'rome':                True,

            # other
            'pr':                  False,
            'pr99':                False,
            'rx1day_pr':           False,
            'rx5day_pr':           False,

            'wap':                 False,
            'tas':                 False,

            'hus':                 False,
            'hur':                 False,
            'rlut':                True,

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

        # show/save
        'one dataset':         False,
        'show':                True,
        'save':                False,
        'save to desktop':     False
        }

    # plot and save figure
    run_scatter_plot(switch)

    stop = timeit.default_timer()
    print(f'Finshed, script finished in {round((stop-start)/60, 2)} minutes.')

















