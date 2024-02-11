import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars as mV
import myFuncs as mF    
import myFuncs_plots as mFp    


# ------------------------
#       Get list
# ------------------------
# --------------------------------------------------------------------------------------- get metric ----------------------------------------------------------------------------------------------------- #
def get_metric(switchM, dataset, metric_class):
    source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
    _ = '' # dummy variable, needed for finding limits
    if switchM['clim']:
        experiment = mV.experiments[0] if not source == 'obs' else ''
        y = mF.load_metric(metric_class, mV.folder_save[0], source, dataset, timescale = mV.timescales[0], experiment = experiment, resolution = mV.resolutions[0])[metric_class.name]

    if switchM['change_with_warming']:
        y_historical = mF.load_metric(metric_class, mV.folder_save[0], source, dataset, timescale = mV.timescales[0], experiment = mV.experiments[0], resolution = mV.resolutions[0])[metric_class.name]
        y_warm       = mF.load_metric(metric_class, mV.folder_save[0], source, dataset, timescale = mV.timescales[0], experiment = mV.experiments[1], resolution = mV.resolutions[0])[metric_class.name]
        y = y_warm if np.max(y_warm) >= np.max(y_historical) else y_historical
    return y, _, _


# --------------------------------------------------------------------------------------- find limits ----------------------------------------------------------------------------------------------------- #
def get_limits(switchM, metric_class, datasets):
    qWithin_low, qWithin_high, qBetween_low, qBetween_high = 0, 1, 0, 1 # quantiles
    vmin, vmax = mF.find_limits(switchM, datasets, metric_class, get_metric, 
                                qWithin_low, qWithin_high, qBetween_low, qBetween_high, # if calculating limits (set lims to '', or comment out)
                                vmin = '', vmax = ''                                    # if manually setting limits
                                )   
    return vmin, vmax

# ------------------------------------------------------------------------------------ Calculate plot-metric ----------------------------------------------------------------------------------------------------- #
def r_eff(area):
    return np.sqrt(area/np.pi)

def get_r_gridbox(dataset, source):
    dim_class = mC.get_metric_class('pr', {'snapshot': True})
    experiment = mV.experiments[0] if not source == 'obs' else ''
    pr_snapshot = mF.load_metric(dim_class, mV.folder_save[0], source, dataset, timescale = 'daily', experiment = experiment, resolution = mV.resolutions[0])[dim_class.name]
    dims = mC.dims_class(pr_snapshot) # to get the dimensions of the model grid
    r_gridbox = r_eff(dims.aream.mean())
    return r_gridbox

def calc_pwad(o_area, pr_o, bin_width):
    bins = np.arange(0, r_eff(o_area.max()) + bin_width, bin_width)
    y_bins = []
    for i in np.arange(0,len(bins)-1):
        y_value = (o_area.where((r_eff(o_area)>=bins[i]) & (r_eff(o_area)<bins[i+1])) * pr_o).sum()/(o_area * pr_o).sum()
        y_bins = np.append(y_bins, y_value)
    bins = bins + 0.5*bin_width # place the points at the centre of each histogram
    return bins, y_bins

def calc_relFreq(y, bin_width):
    bins = np.arange(y.min(), y.max() + bin_width, bin_width)
    y_bins = []
    for i in np.arange(0,len(bins)-1):
        y_value = xr.where((y>=bins[i]) & (y<bins[i+1]), 1, 0).sum() / len(y)
        y_bins = np.append(y_bins, y_value)
    return bins, y_bins

# --------------------------------------------------------------------------------------- get plot-metric ----------------------------------------------------------------------------------------------------- #
def get_plot_metric(switchM, metric_class, dataset, source, bin_width):
    if switchM['clim']:
        experiment = mV.experiments[0] if not source == 'obs' else ''
        if metric_class.name == 'pwad':
            title = 'Precipitation Weighted Area Distribution (PWAD)'
            o_area_class = mC.get_metric_class('o_area', switchM)
            pr_o_class = mC.get_metric_class('pr_o', switchM)
            o_area = mF.load_metric(o_area_class, mV.folder_save[0], source, dataset, timescale = 'daily', experiment = experiment, resolution = mV.resolutions[0])[o_area_class.name]
            pr_o = mF.load_metric(pr_o_class, mV.folder_save[0], source, dataset, timescale = 'daily', experiment = experiment, resolution = mV.resolutions[0])[pr_o_class.name]
            bins, y_bins = calc_pwad(o_area, pr_o, bin_width)
        else:
            title = 'Frequency of occurence in bins'
            y = mF.load_metric(metric_class, mV.folder_save[0], source, dataset, timescale = mV.timescales[0], experiment = experiment, resolution = mV.resolutions[0])[metric_class.name]
            bins, y_bins = calc_relFreq(y, bin_width)

    if switchM['change_with_warming']:
        if metric_class.name == 'pwad':
            title = 'Precipitation Weighted Area Distribution (PWAD)'
            o_area_class = mC.get_metric_class('o_area', switchM)
            pr_o_class = mC.get_metric_class('pr_o', switchM)
            
            o_area = mF.load_metric(o_area_class, mV.folder_save[0], source, dataset, timescale = 'daily', experiment = mV.experiments[0], resolution = mV.resolutions[0])[o_area_class.name]
            pr_o = mF.load_metric(pr_o_class, mV.folder_save[0], source, dataset, timescale = 'daily', experiment = mV.experiments[0], resolution = mV.resolutions[0])[pr_o_class.name]
            bins_historical, y_bins_historical = calc_pwad(o_area, pr_o, bin_width)
            
            o_area = mF.load_metric(o_area_class, mV.folder_save[0], source, dataset, timescale = 'daily', experiment = mV.experiments[1], resolution = mV.resolutions[0])[o_area_class.name]
            pr_o = mF.load_metric(pr_o_class, mV.folder_save[0], source, dataset, timescale = 'daily', experiment = mV.experiments[1], resolution = mV.resolutions[0])[pr_o_class.name]
            bins_warm, y_bins_warm = calc_pwad(o_area, pr_o, bin_width)
        else:
            title = 'Frequency of occurence in bins'
            y = mF.load_metric(metric_class, mV.folder_save[0], source, dataset, timescale = mV.timescales[0], experiment = mV.experiments[0], resolution = mV.resolutions[0])[metric_class.name]
            bins_historical, y_bins_historical = calc_relFreq(y, bin_width)

            y = mF.load_metric(metric_class, mV.folder_save[0], source, dataset, timescale = mV.timescales[0], experiment = mV.experiments[1], resolution = mV.resolutions[0])[metric_class.name]
            bins_warm, y_bins_warm = calc_relFreq(y, bin_width)

        bins   = bins_historical if len(bins_historical) > len(bins_warm) else bins_warm      # take the longest list of bins (they have the same binwidth)  

        max_length = max(len(y_bins_historical), len(y_bins_warm))
        y_bins_historical = np.pad(y_bins_historical, (0, max_length - len(y_bins_historical))) # fill the shorter list with zeros to match the longer list
        y_bins_warm       = np.pad(y_bins_warm, (0, max_length - len(y_bins_warm)))

        y_bins = y_bins_warm - y_bins_historical                                              # take the difference in respective distribution

    return bins, y_bins, title



# ----------------
#      Plot
# ----------------
# ---------------------------------------------------------------------------------------- plot axis plot ----------------------------------------------------------------------------------------------------- #
def plot_ax_line(ax, x, y, color, linewidth = None):
    h = ax.plot(x, y, color, linewidth = linewidth)
    return h


# ----------------------------------------------------------------------------------------- plot figure ----------------------------------------------------------------------------------------------------- #
def plot_bins(switchM, metric_class, xmin, xmax):
    fig, ax = mF.create_figure(width = 9, height = 6)
    
    # i = 1 # 12 is Nor-MM

    for dataset in mV.datasets:
        source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        bin_width = xmax/100 if not metric_class.name == 'pwad' else get_r_gridbox(dataset, source) # mean gridbox area as binwidth, or 1% of maximum range of values for all models
        bins, y_bins, title = get_plot_metric(switchM, metric_class, dataset, source, bin_width)
        if not dataset == 'GPCP_2010-2022': # plot observations as histogram bars instead
            # if dataset == mV.datasets[i]:
            #     metric_class.color = 'b'
            

            plot_ax_line(ax, bins[0:-1], y_bins, metric_class)
            metric_class.color = 'k'
            
            # if dataset == mV.datasets[i]:
            #     max_index = np.argmax(y_bins) # plot name of datasetat highest point
            #     max_x = bins[max_index]
            #     ax.text(max_x, y_bins[max_index], dataset, ha='center', va='bottom', fontsize=12, color='b')
        
        else:
            for i, y in enumerate(y_bins):
                x0 = bins[i] - 0.5 * bin_width
                x1 = bins[i] + 0.5 * bin_width
                color = 'r'
                plt.plot([x0, x0], [0, y_bins[i]], color=color, linestyle='--')
                plt.plot([x1, x1], [0, y_bins[i]], color=color, linestyle='--')
                plt.plot([x0, x1], [y_bins[i], y_bins[i]], color=color, linestyle='--')
                metric_class.color = 'r'
    
    plt.xlim([None, None])
    plt.ylim([None, None])
    plt.title(title)

    plt.axhline(y=0, color='k', linestyle='--') if np.min(y_bins)<0 and np.max(y_bins)>0 else None

    xlabel = r'Effective radius [km$^2$]'           if metric_class.name == 'pwad' else metric_class.label
    ylabel = r'Fraction of total precipitation [%]' if metric_class.name == 'pwad' else r'Frequency of occurence [%]'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig_title = f'{metric_class.name}' if metric_class.name == 'pwad' else f'{metric_class.name}_foo_dist'
    return fig, fig_title



def plot_dsBins(ds, variable_list = mV.datasets, title = '', x_label = '', y_label = '', bins_middle = '', bins_given = False, shade = False, fig = '', ax = '', fig_ax_given = False):
    if not fig_ax_given:
        fig, ax = mFp.create_figure(width = 9, height = 6)
    
    bin_width = ds.bins[1] - ds.bins[0]
    if not bins_given:
        bins_middle = ds['bins_middle']

    if shade:
        data_arrays = [ds[var] for var in ds.data_vars if var != 'GPCP' and var != 'bins_middle']
        combined = xr.concat(data_arrays, dim='model')
        min_vals = combined.min(dim='model')
        max_vals = combined.max(dim='model')
        plt.fill_between(bins_middle, min_vals, max_vals, color='blue', alpha=0.5)
        mean_vals = combined.mean(dim='model')
        # plot_ax_line(ax, bins_middle, mean_vals, 'k', linewidth = 2)

        for i, y in enumerate(ds['GPCP']):
            x0 = bins_middle[i] - 0.5 * bin_width
            x1 = bins_middle[i] + 0.5 * bin_width
            color = 'r'
            plt.plot([x0, x0], [0, y], color=color, linestyle='--')
            plt.plot([x1, x1], [0, y], color=color, linestyle='--')
            plt.plot([x0, x1], [y, y], color=color, linestyle='--')
            color= 'r'
    else:
        for dataset in variable_list:
            if dataset == 'bins_middle':
                continue
            y_bins = ds[dataset]
            if dataset in mV.observations:   # plot observations as histogram bars
                for i, y in enumerate(y_bins):
                    x0 = bins_middle[i] - 0.5 * bin_width
                    x1 = bins_middle[i] + 0.5 * bin_width
                    color = 'r'
                    plt.plot([x0, x0], [0, y], color=color, linestyle='--')
                    plt.plot([x1, x1], [0, y], color=color, linestyle='--')
                    plt.plot([x0, x1], [y, y], color=color, linestyle='--')
                    color= 'r'
            else:
                plot_ax_line(ax, bins_middle, y_bins, 'k')
    plt.xlim([None, None])
    plt.ylim([None, None])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.axhline(y=0, color='k', linestyle='--') if np.min(y_bins)<0 and np.max(y_bins)>0 else None
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax



# ------------------------
#     Run / save plot
# ------------------------
def run_pdf_distribution(switch_metric, switchM, switch):
    print(f'Plotting {mV.timescales[0]} distribution')
    print(f'metric: {[key for key, value in switch_metric.items() if value]} {[key for key, value in switchM.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')

    for metric in [k for k, v in switch_metric.items() if v]:
        metric_class = mC.get_metric_class(metric, switchM)
        xmin, xmax = get_limits(switchM, metric_class, mV.datasets) if not metric_class.name == 'pwad' else [None, None]
        fig, fig_title = plot_bins(switchM, metric_class, xmin, xmax)
        if len(mV.datasets) == 1:
            filename = f'{mV.datasets[0]}{fig_title}'
        else:
            source_list = mV.find_list_source(mV.datasets, mV.models_cmip5, mV.models_cmip6, mV.observations)
            ifWith_obs = mV.find_ifWithObs(mV.datasets, mV.observations)
            filename = f'{source_list}_{fig_title}{ifWith_obs}'

    mF.save_plot(switch, fig, home, filename)
    plt.show() if switch['show'] else None



# ------------------------
#  Choose what to plot
# ------------------------
if __name__ == '__main__':
# ---------------------------------------------------------------------------------- metric to plot ----------------------------------------------------------------------------------------------------- #
    switch_metric = {                                                                                         # pick one
        'rome':       False, 'ni':        False, 'areafraction': False, 'F_pr10':    False, 'pwad':  True,   # organization
        'pr':         False, 'pr_99':     False, 'pr_97':        False, 'pr_95':     False, 'pr_90': False,   # precipitation
        'wap':        False,                                                                                  # circulation
        'hur':        False,                                                                                  # humidity
        'tas':        False, 'stability': False,                                                              # temperature
        'netlw':      False, 'rlut':      False, 'rlds':         False, 'rlus':      False, 'rlut': False,    # LW
        'netsw':      False, 'rsdt':      False, 'rsds':         False, 'rsus':      False, 'rsut': False,    # SW
        'lcf':        False, 'hcf':       False,                                                              # cloudfraction
        'ws_lc':      False, 'ws_hc':     False,                                                              # weather states
        'hus':        False,                                                                                  # moist static energy
        }

    switchM = {                                                                                               # can pick multiple
        '250hpa':     False, '500hpa':    False, '700hpa':       False,                                       # mask: vertical
        'descent':    False, 'ascent':    False, 'ocean':        False,                                       # mask: horizontal
        'fixed area': False, '90':        False, '95':           False, '97':        False,                   # conv threshold (95th default)
        'sMean':      True,                                                                                  # metric type
        'anomalies':  False,                                                                                  # calc type
        'clim':       True,  'change_with_warming': False,                                                    # Scenario type   
        }


# ------------------------------------------------------------------------------------- settings ----------------------------------------------------------------------------------------------------- #
    switch = {
        'show':                False,                                                                         # show
        'save_test_desktop':   True, 'save_folder_desktop': False, 'save_folder_cwd':     False,              # save
        }

    run_pdf_distribution(switch_metric, switchM, switch)







