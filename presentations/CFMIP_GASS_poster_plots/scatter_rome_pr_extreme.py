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
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV                                 # imports common variables

# ------------------------------------------------------------------------------ Formatting axes for scatter plot ----------------------------------------------------------------------------------------------------- #

def plot_correlation(ax, x,y):
    x_text = 0.65
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

def cbar_right_of_axis(fig, ax, pcm, width_frac, height_frac, pad, numbersize = 8, cbar_label = '', text_pad = 0.1):
    # colorbar position
    ax_position = ax.get_position()
    cbar_bottom = ax_position.y0
    cbar_left = ax_position.x1 + pad
    cbar_width = ax_position.width * width_frac
    cbar_height = ax_position.height * height_frac
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='vertical')
    cbar.ax.tick_params(labelsize=numbersize)
    # colobar label
    cbar_text_y = ax_position.y0 + (ax_position.y1 - ax_position.y0) / 2
    cbar_text_x = cbar_left + cbar_width + text_pad
    ax.text(cbar_text_x, cbar_text_y, cbar_label, rotation = 'vertical', va = 'center', fontsize = 10, transform=fig.transFigure)
    return cbar


# --------------------------------------------------------------------------------------- Calculation ----------------------------------------------------------------------------------------------------- #

def calc_anomalies(array, timescale):
    if timescale == 'daily': 
        rolling_mean = array.rolling(time=12, center=True).mean()
        array = array - rolling_mean
        array = array.dropna(dim='time')
    return array

def calc_plot_var(switch, variable_type, metric, metric_option, dataset, timescale, resolution, folder_load):
    source = mF.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
    array = mV.load_metric(folder_load, variable_type, metric, source, dataset, timescale, experiment = mV.experiments[0], resolution=resolution)[metric_option]
    array = calc_anomalies(array, timescale) if switch['anomalies'] else array
    return array


# -------------------------------------------------------------------------------------- different plots ----------------------------------------------------------------------------------------------------- #

xlims = {
    'TaiESM1':          [-500000, 500000],              # 1
    'BCC-CSM2-MR':      [-250000, 500000],       # 2
    'FGOALS-g3':        [-200000, 250000],              # 3
    'CNRM-CM6-1':       [-250000, 400000],     # 4
    'MIROC6':           [-500000, 500000],     # 5
    'MPI-ESM1-2-LR':    [-500000, 1000000],     # 6
    'NorESM2-MM':       [-500000, 500000],     # 7
    'GFDL-CM4':         [-250000, 300000],     # 8
    'CanESM5':          [-200000, 250000],       # 9
    'CMCC-ESM2':        [-500000, 500000],     # 10
    'UKESM1-0-LL':      [-500000, 500000],     # 11
    'MRI-ESM2-0':       [-500000, 500000],     # 12
    'CESM2':            [-500000, 500000],     # 19
    'NESM3':            [-1000000,1000000],     # 14
    'IITM-ESM':         [-500000, 500000],     # 15
    'EC-Earth3':        [-500000, 500000],     # 16
    'INM-CM5-0':        [-250000, 250000],     # 17
    'IPSL-CM6A-LR':     [-250000, 250000],     # 18
    'KIOST-ESM':        [-1000000,1000000],     # 19
    'GPCP':             [-500000, 500000],     # 20
}

xticks = {
    'TaiESM1':          [-500000, 0, 500000],              # 1
    'BCC-CSM2-MR':      [-250000, 0, 500000],       # 2
    'FGOALS-g3':        [-200000, 0, 250000],              # 3
    'CNRM-CM6-1':       [-250000, 0, 400000],     # 4
    'MIROC6':           [-500000, 0, 500000],     # 5
    'MPI-ESM1-2-LR':    [-500000, 0, 1000000],     # 6
    'NorESM2-MM':       [-500000, 0, 500000],     # 7
    'GFDL-CM4':         [-250000, 0, 300000],     # 8
    'CanESM5':          [-200000, 0, 250000],       # 9
    'CMCC-ESM2':        [-500000, 0, 500000],     # 10
    'UKESM1-0-LL':      [-500000, 0, 500000],     # 11
    'MRI-ESM2-0':       [-500000, 0, 500000],     # 12
    'CESM2':            [-500000, 0, 500000],     # 19
    'NESM3':            [-1000000,0, 1000000],     # 14
    'IITM-ESM':         [-500000, 0, 500000],     # 15
    'EC-Earth3':        [-500000, 0, 500000],     # 16
    'INM-CM5-0':        [-250000, 0, 250000],     # 17
    'IPSL-CM6A-LR':     [-250000, 0, 250000],     # 18
    'KIOST-ESM':        [-1000000,0, 1000000],     # 19
    'GPCP':             [-500000, 0, 500000],     # 20
}

def plot_multiple_scatter(switch, var0, var1, title, datasets, timescale, resolution, folder_save):    
    nrows = 5
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

        pcm = ax.hist2d(x,y,[20,20], cmap ='Blues')
        color = 'b'
        plot_ax_bins(ax, x, y, color)

        mF.move_col(ax,  -0.0715+0.0025) if col == 0 else None
        mF.move_col(ax, -0.035)   if col == 1 else None
        mF.move_col(ax, 0.0)   if col == 2 else None
        mF.move_col(ax, 0.035)    if col == 3 else None

        mF.move_row(ax, 0.0875)   if row == 0 else None
        mF.move_row(ax, 0.0495)  if row == 1 else None
        mF.move_row(ax, 0.01)  if row == 2 else None
        mF.move_row(ax, -0.0195) if row == 3 else None
        mF.move_row(ax, -0.05) if row == 4 else None

        mF.scale_ax_x(ax, 0.9)
        mF.scale_ax_y(ax, 1)

        ax.set_ylim([-15, 15])
        ax.set_xlim(xlims[dataset])
        ax.set_xticks(xticks[dataset])
        ax.set_xticklabels(xticks[dataset])
        ax.tick_params(axis='x', labelsize=10)

        if i >= num_subplots-ncols:
            mF.plot_xlabel(fig, ax, xlabel, pad=0.055, fontsize = 10)

        if col == 0:
            mF.plot_ylabel(fig, ax, ylabel, pad = 0.0475, fontsize = 10)
        mF.plot_axtitle(fig, ax, dataset, xpad = 0, ypad = 0.0075, fontsize = 10)

        if col == 3:
            cbar_right_of_axis(fig, ax, pcm[3], width_frac= 0.05, height_frac=1, pad=0.015, numbersize = 9, cbar_label = 'days [Nb]', text_pad = 0.0375)
        else:
            cbar_right_of_axis(fig, ax, pcm[3], width_frac= 0.05, height_frac=1, pad=0.015, numbersize = 9, cbar_label = '', text_pad = 0.1)

    # ax.text(0.5, 0.9625, title, ha = 'center', fontsize = 15, transform=fig.transFigure)
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

    source = mF.find_list_source(datasets, mV.models_cmip5, mV.models_cmip6, mV.observations)
    with_obs = mF.find_ifWithObs(datasets, mV.observations)

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
        'save':                False,
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







