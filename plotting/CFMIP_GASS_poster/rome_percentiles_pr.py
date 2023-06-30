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


def plot_ax_scatter(ax,x,y, color='k', xmin=None, ymin=None):
    ax.scatter(x,y,facecolors='none', edgecolor=color)
    plot_correlation(ax, x,y)

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

# -------------------------------------------------------------------------------------- Calculation ----------------------------------------------------------------------------------------------------- #

def name_region(switch):
    if switch['descent']:
        region = '_d' 
    elif switch['ascent']:
        region = '_a' 
    else:
        region = ''
    return region

def find_metric_and_units(plot_var):
    if plot_var['pr'] or plot_var['percentiles_pr'] or plot_var['rx1day_pr'] or plot_var['rx5day_pr']:
        variable_type = 'pr'
        axis_label = 'pr [mm day{}]'.format(mF.get_super('-1'))
    if plot_var['pr']:
        metric = 'pr' 
        metric_option = metric
    if plot_var['percentiles_pr']:
        metric = 'percentiles_pr' 
        metric_option = 'pr97' # there is also pr95, pr99
    if  plot_var['rx1day_pr'] or plot_var['rx5day_pr']:
        metric = 'rxday_pr'
        metric_option = 'rx1day_pr' if plot_var['rx1day_pr'] else 'rx5day_pr'

    if plot_var['wap']:
        variable_type = 'wap'
        axis_label = 'wap [hPa day' + mF.get_super('-1') + ']'
        region = name_region(plot_var)
        metric = f'wap{region}'
        metric_option = metric

    if plot_var['tas']:
        variable_type = 'tas'
        axis_label = 'Temperature [\u00B0C]'
        region = name_region(plot_var)
        metric = f'tas{region}'
        metric_option = metric

    if plot_var['hur']:
        variable_type = 'hur'
        axis_label = 'Relative humidity [%]'
        region = name_region(plot_var)
        metric = f'hur_sMean{region}'
        metric_option = metric

    if plot_var['lcf'] or plot_var['hcf']:
        variable_type = 'cl'
        axis_label = 'cloud fraction [%]'
        region = name_region(plot_var)
        metric = f'lcf{region}' if plot_var['lcf'] else f'hcf{region}'
        metric_option = metric

    if plot_var['hus']:
        variable_type = 'hus'
        axis_label = 'Specific humidity [mm]'
        region = name_region(plot_var)
        metric = f'hus{region}'
        metric_option = metric

    if plot_var['rome']:
        variable_type = 'org'
        axis_label = 'ROME [km' + mF.get_super('2') + ']' 
        metric = f'rome'
        metric_option = metric

        axis_label = '{}{} K{}'.format(axis_label[:-1], mF.get_super('-1'), axis_label[-1:]) if plot_var['per_kelvin'] else axis_label
    return variable_type, metric, metric_option, axis_label


def calc_plot_var(switch, variable_type, metric, metric_option, dataset, timescale, resolution, folder_load):
    source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
    if timescale == 'monthly' and (metric == 'percentiles_pr' or metric_option == 'rome'):
        var = mV.load_metric(folder_load, variable_type, metric, source, dataset, 'daily', experiment = mV.experiments[0], resolution=resolution)[metric_option]
        var = mF.resample_timeMean(var, timescale)
    else:
        var = mV.load_metric(folder_load, variable_type, metric, source, dataset, timescale, experiment = mV.experiments[0], resolution=resolution)[metric_option]
    return var


def find_limits(switch, plot_var, datasets, timescale, resolution, folder_load, quantileWithin_low, quantileWithin_high, quantileBetween_low = 0, quantileBetween_high=1):    
    vmin_list, vmax_list = [], []
    for dataset in datasets:
        variable_type, metric, metric_option, xlabel = find_metric_and_units(plot_var)
        var = calc_plot_var(switch, variable_type, metric, metric_option, dataset, timescale, resolution, folder_load)
        if timescale == 'monthly' and (switch['percentiles_pr'] or switch['rome']):
            var = mF.resample_timeMean(var, timescale)
        vmin_list = np.append(vmin_list, np.nanquantile(var, quantileWithin_low))
        vmax_list = np.append(vmax_list, np.nanquantile(var, quantileWithin_high))

    vmin = np.nanquantile(vmin_list, quantileBetween_low)
    vmax = np.nanquantile(vmax_list, quantileBetween_high)

    return (vmin, vmax)


# -------------------------------------------------------------------------------------- different plots ----------------------------------------------------------------------------------------------------- #

def plot_one_scatter(switch, var0, var1, title, dataset, timescale, resolution, folder_save):
    # adjust figure size
    width = 9
    height = 6

    # Adjust position and scale of axes
    move_col_by = 0
    move_row_by = 0
    scale_ax_by = 1     

    # Set position of text
    title = f'{dataset}: {title}'
    title_fontsize = 15
    xylabel_fontsize = 12

    title_xpad = 0.05
    title_ypad = 0.02
    xlabel_pad = 0.085
    ylabel_pad = 0.085

    # find variables and variable limits
    variable_type, metric, metric_option, xlabel = find_metric_and_units(var0)
    x = calc_plot_var(switch, variable_type, metric, metric_option, dataset, timescale, resolution, folder_save)
    xmin, xmax = find_limits(switch, var0, datasets = [dataset], timescale = timescale, resolution = resolution, folder_load = folder_save,
        quantileWithin_low = 0,    # remove extreme low values from colorbar range 
        quantileWithin_high = 1,   # remove extreme high values from colorbar range 
        )
    variable_type, metric, metric_option, ylabel = find_metric_and_units(var1)
    y = calc_plot_var(switch, variable_type, metric, metric_option, dataset, timescale, resolution, folder_save)
    ymin, ymax = find_limits(switch, var1, datasets = [dataset], timescale = timescale, resolution = resolution, folder_load = folder_save,
        quantileWithin_low = 0,    # remove extreme low values from colorbar range 
        quantileWithin_high = 1,   # remove extreme high values from colorbar range 
        )
    
    if timescale == 'monthly' and (switch['percentiles_pr'] or switch['rome']):
        x = x.assign_coords(time=y.time)

    fig, ax = create_figure(width = width, height = height)
    sp = plot_ax_scatter(ax, x, y) if switch['xy'] else plot_ax_scatter(ax, y, x)

    mF.move_col(ax, moveby = move_col_by)
    mF.move_row(ax, moveby = move_row_by)
    mF.scale_ax(ax, scaleby = scale_ax_by)
    mF.plot_xlabel(fig, ax, xlabel, xlabel_pad, xylabel_fontsize) if switch['xy'] else mF.plot_ylabel(fig, ax, xlabel, xlabel_pad, xylabel_fontsize) 
    mF.plot_ylabel(fig, ax, ylabel, ylabel_pad, xylabel_fontsize) if switch['xy'] else mF.plot_xlabel(fig, ax, ylabel, ylabel_pad, xylabel_fontsize)
    mF.plot_axtitle(fig, ax, title, title_xpad, title_ypad, title_fontsize)
    return fig


def plot_multiple_scatter(switch, var0, var1, title, datasets, timescale, resolution, folder_save):
    # adjust figure size
    width = 11
    height = 7.75
    
    nrows = 4
    ncols = 4
    
    # Adjust position of cols
    move_col0_by, move_col1_by, move_col2_by, move_col3_by = -0.055, -0.02, 0.015, 0.05

    # Adjust position of rows
    move_row0_by = 0.05 -0.002 
    move_row1_by = 0.035 -0.006
    move_row2_by = 0.015 -0.008
    move_row3_by = -0.0025 -0.01
    move_row4_by = 0

    # Adjust scale of ax
    scale_ax_x_by = 1.1          
    scale_ax_y_by = 0.9                                                  

    # Set position of text
    xylabel_fontsize = 10
    title_fontsize = 15
    axtitle_fontsize = 10

    title_x = 0.5
    title_y = 0.9625

    xlabel_pad = 0.0725
    ylabel_pad = 0.055

    axtitle_xpad = 0.002
    axtitle_ypad = 0.0095

    # Find common limits
    xmin, xmax = find_limits(switch, var1, datasets, timescale, resolution, folder_load = folder_save,
        quantileWithin_low = 0,    # remove extreme low values from colorbar range 
        quantileWithin_high = 1,   # remove extreme high values from colorbar range 
        quantileBetween_low = 0,   # remove extreme low models' from colorbar range
        quantileBetween_high = 1   # remove extreme high models' from colorbar range
        )
    ymin, ymax = find_limits(switch, var1, datasets, timescale, resolution, folder_load = folder_save,
        quantileWithin_low = 0,    # remove extreme low values from colorbar range 
        quantileWithin_high = 1,   # remove extreme high values from colorbar range 
        quantileBetween_low = 0,   # remove extreme low models' from colorbar range
        quantileBetween_high = 1   # remove extreme high models' from colorbar range
        )

    fig, axes = create_figure(width = width, height = height, nrows=nrows, ncols=ncols)
    num_subplots = len(datasets)
    for i, dataset in enumerate(datasets):
        row = i // ncols  # determine row index
        col = i % ncols   # determine col index
        ax = axes.flatten()[i]

        variable_type, metric, metric_option, xlabel = find_metric_and_units(var0)
        x = calc_plot_var(switch, variable_type, metric, metric_option, dataset, timescale, resolution, folder_save)
        variable_type, metric, metric_option, ylabel = find_metric_and_units(var1)
        y = calc_plot_var(switch, variable_type, metric, metric_option, dataset, timescale, resolution, folder_save)
        # sp = plot_ax_scatter(ax, x, y) if switch['xy'] else plot_ax_scatter(ax, y, x)
        color = 'Blues'
        ax.hist2d(x,y,[20,20], cmap = color) if switch['xy'] else ax.hist2d(y,x,[20,20], cmap = color)
        color = 'Blue'
        sp = plot_ax_bins(ax, x, y, color) if switch['xy'] else plot_ax_bins(ax, y, x, color)

        mF.move_col(ax, move_col0_by) if col == 0 else None
        mF.move_col(ax, move_col1_by) if col == 1 else None
        mF.move_col(ax, move_col2_by) if col == 2 else None
        mF.move_col(ax, move_col3_by) if col == 3 else None

        mF.move_row(ax, move_row0_by) if row == 0 else None
        mF.move_row(ax, move_row1_by) if row == 1 else None
        mF.move_row(ax, move_row2_by) if row == 2 else None
        mF.move_row(ax, move_row3_by) if row == 3 else None
        mF.move_row(ax, move_row4_by) if col == 4 else None

        mF.scale_ax_x(ax, scale_ax_x_by)
        mF.scale_ax_y(ax, scale_ax_y_by)

        if i >= num_subplots-nrows:
            mF.plot_xlabel(fig, ax, xlabel, xlabel_pad, xylabel_fontsize) if switch['xy'] else mF.plot_xlabel(fig, ax, ylabel, ylabel_pad, xylabel_fontsize)

        if col == 0:
            mF.plot_ylabel(fig, ax, ylabel, ylabel_pad, xylabel_fontsize) if switch['xy'] else mF.plot_ylabel(fig, ax, xlabel, xlabel_pad, xylabel_fontsize) 
        
        mF.plot_axtitle(fig, ax, dataset, axtitle_xpad, axtitle_ypad, axtitle_fontsize)

    ax.text(title_x, title_y, title, ha = 'center', fontsize = title_fontsize, transform=fig.transFigure)
    # mF.delete_remaining_axes(fig, axes, num_subplots, nrows, ncols)
    return fig



# ---------------------------------------------------------------------------------- Find the metric and units ----------------------------------------------------------------------------------------------------- #

# ------------------
#    Run script
# ------------------

def run_scatter_plot(switch, datasets, timescale, resolution, folder_save = mV.folder_save):
    print(f'Plotting scatter_plot with {resolution} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')

    keys = [k for k, v in switch.items() if v]  # list of True keys
    var0, var1 = switch.copy(), switch.copy() 
    var0[keys[1]] = False # sets second variable to False
    var1[keys[0]] = False # sets first variable to False
    title  = f'{keys[0]}{name_region(var0)}_and_{keys[1]}{name_region(var1)}'

    if switch['one dataset']:
        dataset = datasets[0]
        fig = plot_one_scatter(switch, var0, var1, title, dataset, timescale, resolution, folder_save)
        source = mV.find_list_source(datasets, mV.models_cmip5, mV.models_cmip6, mV.observations)
        filename = f'{dataset}_{title}'
    else:
        fig = plot_multiple_scatter(switch, var0, var1, title, datasets, timescale, resolution, folder_save)
        source = mV.find_list_source(datasets, mV.models_cmip5, mV.models_cmip6, mV.observations)
        with_obs = mV.find_ifWithObs(datasets, mV.observations)
        filename = f'{source}_{title}{with_obs}'
        
    mV.save_figure(fig, f'{folder_save}/corr/{source}', f'{filename}.pdf') if switch['save'] else None
    plt.show() if switch['show'] else None



if __name__ == '__main__':

    start = timeit.default_timer()

    # choose two metrics (1-12) to plot, and other settings
    switch = {
        'rome':                True,       # First metric

        'pr':                  False,      
        'percentiles_pr':      True,       
        'rx1day_pr':           False,       
        'rx5day_pr':           False,       

        'wap':                 False,       
        'tas':                 False,       
        'hur':                 False,       

        'lcf':                 False,       
        'hcf':                 False,       

        'hus':                 False,       # Last metric

        'descent':             False,
        'ascent':              False,
        'anomalies':           False,
        'per_kelvin':          False,
        
        'xy':                  True,   # if False, reverse explanatory and response variable
        'one dataset':         False,    # if False, plots all chosen datasets (20 datasets max)
        'show':                True,
        'save':                False,
        }
    

    # plot and save figure
    run_scatter_plot(switch, 
                 datasets =    mV.datasets, 
                 timescale =   mV.timescales[0],
                 resolution =  mV.resolutions[0],
                #  folder_save = f'{mV.folder_save_gadi}'
                 )

    stop = timeit.default_timer()
    print(f'Finshed, script finished in {round((stop-start)/60, 2)} minutes.')













































































































































