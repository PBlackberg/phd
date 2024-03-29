import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
import timeit

import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/functions')
import myFuncs as mF # imports common operators
import myVars as mV # imports common variables

# ------------------------------------------------------------------------------ Formatting axes for scatter plot ----------------------------------------------------------------------------------------------------- #

def plot_correlation(ax, x,y):
    x_text = 0.725
    y_text = 0.075
    res= stats.pearsonr(x,y)
    if res[1]<=0.05:
        ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', xytext=(x_text, y_text), textcoords='axes fraction', fontsize = 8, color = 'r')

def plot_ax_bins(ax, x, y, color='k'):    
    bin_width = (x.max() - x.min())/100 # Each bin is one percent of the range of x values
    bins = np.arange(x.min(), x.max() + bin_width, bin_width)
    y_bins = []
    for i in np.arange(0,len(bins)-1):
        y_bins = np.append(y_bins, y.where((x>=bins[i]) & (x<bins[i+1])).mean())
    ax.plot(bins[:-1], y_bins, color)
    plot_correlation(ax, x,y)

def create_figure(width, height, nrows = 1, ncols = 1):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width,height))
    return fig, axes


# --------------------------------------------------------------------------------------- Calculation ----------------------------------------------------------------------------------------------------- #

def calc_anomalies(array, timescale):
    if timescale == 'daily': 
        rolling_mean = array.rolling(time=12, center=True).mean()
        array = array - rolling_mean
        array = array.dropna(dim='time')
    return array

def calc_plot_var(switch, variable_type, metric, metric_option, dataset, timescale, resolution, folder_load):
    source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
    array = mV.load_metric(folder_load, variable_type, metric, source, dataset, timescale, experiment = mV.experiments[0], resolution=resolution)[metric_option]
    array = calc_anomalies(array, timescale) if switch['anomalies'] else array
    return array


# -------------------------------------------------------------------------------------- different plots ----------------------------------------------------------------------------------------------------- #

def plot_multiple_scatter(switch, var0, var1, title, datasets, timescale, resolution, folder_save):    
    nrows = 4
    ncols = 4
    fig, axes = create_figure(width = 11, height = 7.75, nrows=nrows, ncols=ncols)
    num_subplots = len(datasets)
    for i, dataset in enumerate(datasets):
        row = i // ncols  # determine row index
        col = i % ncols   # determine col index
        ax = axes.flatten()[i]

        variable_type, metric, metric_option, xlabel = find_metric_and_units(var0)
        x = calc_plot_var(switch, variable_type, metric, metric_option, dataset, timescale, resolution, folder_save)
        variable_type, metric, metric_option, ylabel = find_metric_and_units(var1)
        y = calc_plot_var(switch, variable_type, metric, metric_option, dataset, timescale, resolution, folder_save)

        if timescale == 'monthly' and switch['rome']:
            x = x.assign_coords(time=y.time)

        ax.hist2d(x,y,[20,20], cmap ='Greys')
        color = 'k'
        sp = plot_ax_bins(ax, x, y, color)

        mF.move_col(ax,  -0.055) if col == 0 else None
        mF.move_col(ax, -0.02)   if col == 1 else None
        mF.move_col(ax, 0.015)   if col == 2 else None
        mF.move_col(ax, 0.05)    if col == 3 else None

        mF.move_row(ax, 0.05 -0.002)   if row == 0 else None
        mF.move_row(ax, 0.035 -0.006)  if row == 1 else None
        mF.move_row(ax, 0.015 -0.008)  if row == 2 else None
        mF.move_row(ax, -0.0025 -0.01) if row == 3 else None
        mF.move_row(ax, 0) if col == 4 else None

        mF.scale_ax_x(ax, 1.1)
        mF.scale_ax_y(ax, 0.9)

        if i >= num_subplots-nrows:
            mF.plot_xlabel(fig, ax, xlabel, pad=0.0725, fontsize = 10)

        if col == 0:
            mF.plot_ylabel(fig, ax, ylabel, pad = 0.055, fontsize = 10)
        mF.plot_axtitle(fig, ax, dataset, xpad = 0, ypad = 0.0075, fontsize = 10)

    ax.text(0.5, 0.9625, title, ha = 'center', fontsize = 15, transform=fig.transFigure)
    mF.delete_remaining_axes(fig, axes, num_subplots, nrows, ncols)
    return fig


# ---------------------------------------------------------------------------------- Find metric/units and run ----------------------------------------------------------------------------------------------------- #

def find_metric_and_units(plot_var):
    if plot_var['pr99']:
        variable_type = 'pr'
        axis_label = 'pr [mm day{}]'.format(mF.get_super('-1'))
        metric = 'meanInPercentiles_pr'
        metric_option = 'pr99'
    if plot_var['rome']:
        variable_type = 'org'
        axis_label = 'ROME [km' + mF.get_super('2') + ']' 
        metric = f'rome'
        metric_option = metric
    return variable_type, metric, metric_option, axis_label

def run_scatter_plot(switch, datasets, timescale, resolution, folder_save):
    print(f'Plotting rome and pr99 correlation with {timescale} {resolution} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')

    keys = [k for k, v in switch.items() if v]  # list of True keys
    var0, var1 = switch.copy(), switch.copy() 
    var0[keys[1]] = False # sets second variable to False
    var1[keys[0]] = False # sets first variable to False

    title  = f'ROME and pr99'
    fig = plot_multiple_scatter(switch, var0, var1, title, datasets, timescale, resolution, folder_save)

    source = mV.find_list_source(datasets, mV.models_cmip5, mV.models_cmip6, mV.observations)
    with_obs = mV.find_ifWithObs(datasets, mV.observations)

    folder = f'{home}/Desktop/GASS-CFMIP_poster'
    filename = f'{source}_rome_and_pr99{with_obs}'    
    mV.save_figure(fig, folder, f'{filename}.pdf') if switch['save'] else None
    plt.show() if switch['show'] else None


if __name__ == '__main__':
    start = timeit.default_timer()

    # choose two metrics (1-12) to plot, and other settings
    switch = {
        'rome':                True,       # First metric
        'pr99':                True,       

        'anomalies':           True,
        
        'show':                True,
        'save':                True,
        }
    

    # plot and save figure
    run_scatter_plot(switch, 
                     datasets =    mV.datasets, 
                     timescale =   mV.timescales[0],
                     resolution =  mV.resolutions[0],
                     folder_save = mV.folder_save[0]
                     )

    stop = timeit.default_timer()
    print(f'Finshed, script finished in {round((stop-start)/60, 2)} minutes.')







