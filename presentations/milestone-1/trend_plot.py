import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats

import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util')
import myFuncs as mF                                # imports common operators
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV                                 # imports common variables



# ---------------------------------------------------------------------------------------- Plot / format plot ----------------------------------------------------------------------------------------------------- #

def ax_plot(switch, ax, x, y, metric_y):
    if switch['bins']:
        pcm = ax.hist2d(x,y,[20,20], cmap = metric_y.cmap)
        bin_width = (x.max() - x.min())/100 # Each bin is one percent of the range of x values
        bins = np.arange(x.min(), x.max() + bin_width, bin_width)
        y_bins = []
        for i in np.arange(0,len(bins)-1):
            y_bins = np.append(y_bins, y.where((x>=bins[i]) & (x<bins[i+1])).mean())
        ax.plot(bins[:-1], y_bins, metric_y.color)

    if switch['scatter']:
        pcm = ax.scatter(x, y, facecolors='none', edgecolor= metric_y.color)    
    return pcm

def plot_one_dataset(switch, x, y, metric_x, metric_y, title = '', xmin = None, xmax = None, ymin = None, ymax = None):
    fig, ax = mF.create_figure(width = 8, height = 5.5)
    pcm = ax_plot(switch, ax, x, y, metric_y)
    res= stats.pearsonr(x,y)
    ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', 
                xytext=(0.8, 0.85), textcoords='axes fraction', fontsize = 12, color = 'r') if res[1]<=0.05 else None
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    mF.move_col(ax, -0.035)
    mF.move_row(ax, 0.03)
    mF.scale_ax_x(ax, 1)
    mF.scale_ax_y(ax, 1)
    mF.plot_xlabel(fig, ax, metric_x.label, pad=0.1, fontsize = 12)
    mF.plot_ylabel(fig, ax, metric_y.label, pad = 0.075, fontsize = 12)
    mF.plot_axtitle(fig, ax, title, xpad = 0, ypad = 0.0075, fontsize = 12)
    mF.cbar_right_of_axis(fig, ax, pcm[3], width_frac= 0.05, height_frac=1, 
                          pad=0.015, numbersize = 9, cbar_label = 'months [Nb]', text_pad = 0.05) if switch['bins'] else None
    return fig



xlims = {
    'TaiESM1':          [-200000, 200000],     # 1
    'BCC-CSM2-MR':      [-100000, 100000],     # 2
    'FGOALS-g3':        [-100000, 100000],     # 3
    'CNRM-CM6-1':       [-100000, 100000],     # 4
    'MIROC6':           [-100000, 100000],     # 5
    'MPI-ESM1-2-LR':    [-200000, 200000],     # 6
    'NorESM2-MM':       [-200000, 200000],     # 7
    'GFDL-CM4':         [-100000, 100000],     # 8
    'CanESM5':          [-50000, 50000],       # 9
    'CMCC-ESM2':        [-200000, 200000],     # 10
    'UKESM1-0-LL':      [-100000, 100000],     # 11
    'MRI-ESM2-0':       [-200000, 200000],     # 12
    'CESM2-WACCM':      [-200000, 200000],     # 19
    'NESM3':            [-200000, 200000],     # 14
    'IITM-ESM':         [-200000, 200000],     # 15
    'EC-Earth3':        [-100000, 100000],     # 16
    'INM-CM5-0':        [-100000, 100000],     # 17
    'IPSL-CM6A-LR':     [-100000, 100000],     # 18
    'KIOST-ESM':        [-250000, 250000],     # 19
    'ERA5':             [-200000, 200000],     # 20
    'CERES':            [-200000, 200000],     # 20
    'GPCP':            [-200000, 200000],      # 20
}

xticks = {
    'TaiESM1':          [-200000, 0, 200000],     # 1
    'BCC-CSM2-MR':      [-50000, 0, 50000],       # 2
    'FGOALS-g3':        [-100000, 0, 100000],     # 3
    'CNRM-CM6-1':       [-100000, 0, 100000],     # 4
    'MIROC6':           [-100000, 0, 100000],     # 5
    'MPI-ESM1-2-LR':    [-200000, 0, 200000],     # 6
    'NorESM2-MM':       [-200000, 0, 200000],     # 7
    'GFDL-CM4':         [-100000, 0, 100000],     # 8
    'CanESM5':          [-50000, 0, 50000],       # 9
    'CMCC-ESM2':        [-200000, 0, 200000],     # 10
    'UKESM1-0-LL':      [-100000, 0, 100000],     # 11
    'MRI-ESM2-0':       [-200000, 0, 200000],     # 12
    'CESM2-WACCM':      [-200000, 0, 200000],     # 19
    'NESM3':            [-200000, 0, 200000],     # 14
    'IITM-ESM':         [-200000, 0, 200000],     # 15
    'EC-Earth3':        [-100000, 0, 100000],     # 16
    'INM-CM5-0':        [-100000, 0, 100000],     # 17
    'IPSL-CM6A-LR':     [-100000, 0, 100000],     # 18
    'KIOST-ESM':        [-250000, 0, 250000],     # 19
    'ERA5':             [-200000, 0, 200000],     # 20
    'CERES':            [-200000, 0, 200000],     # 20
    'GPCP':            [-200000, 0, 200000],      # 20
}

def plot_multiple_datasets(switch, metric_x, metric_y, title = '', axtitle = '', xmin = None, xmax = None, ymin = None, ymax = None):
    nrows, ncols = 5, 4
    fig, axes = mF.create_figure(width = 12, height = 8.5, nrows=nrows, ncols=ncols)
    num_subplots = len(mV.datasets)
    for i, dataset in enumerate(mV.datasets):
        row = i // ncols  # determine row index
        col = i % ncols   # determine col index
        ax = axes.flatten()[i]
        x, _, _ = get_list(switch, dataset, metric_x)
        y, _, axtitle = get_list(switch, dataset, metric_y)
        x = x.assign_coords(time=y.time) #if mV.timescales[0] == 'monthly' and metric_0.option in ['rome', 'rome_equal_area'] else x
            
        pcm = ax_plot(switch, ax, x, y, metric_y)
        res= stats.pearsonr(x,y)
        ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', 
                    xytext=(0.675, 0.85), textcoords='axes fraction', fontsize = 8, color = 'r') if res[1]<=0.05 else None
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        mF.move_col(ax,  -0.0715+0.0025)   if col == 0 else None
        mF.move_col(ax, -0.035)            if col == 1 else None
        mF.move_col(ax, 0.0)               if col == 2 else None
        mF.move_col(ax, 0.035)             if col == 3 else None

        mF.move_row(ax, 0.0875 - 0.025 +0.025)            if row == 0 else None
        mF.move_row(ax, 0.0495 - 0.0135+0.025)            if row == 1 else None
        mF.move_row(ax, 0.01   - 0.005+0.025)              if row == 2 else None
        mF.move_row(ax, -0.0195+0.025)           if row == 3 else None
        mF.move_row(ax, -0.05+0.025)             if row == 4 else None

        mF.scale_ax_x(ax, 0.9)
        mF.scale_ax_y(ax, 0.95)

        mF.plot_xlabel(fig, ax, metric_x.label, pad=0.055, fontsize = 10) if i >= num_subplots-ncols else None
        mF.plot_ylabel(fig, ax, metric_y.label, pad = 0.0475, fontsize = 10) if col == 0 else None
        mF.plot_axtitle(fig, ax, axtitle, xpad = 0, ypad = 0.0075, fontsize = 10)
        # ax.text(0.5, 0.975, title, ha = 'center', fontsize = 15, transform=fig.transFigure)

        # ax.set_ylim([-2.5, 2.5])
        # ax.set_ylim([-1, 1])
        ax.set_xlim(xlims[dataset])
        ax.set_xticks(xticks[dataset])
        ax.set_xticklabels(xticks[dataset])
        ax.tick_params(axis='x', labelsize=10)

        if switch['bins']:
            if col == 3:
                mF.cbar_right_of_axis(fig, ax, pcm[3], width_frac= 0.05, height_frac=1, pad=0.015, numbersize = 9, cbar_label = 'months [Nb]', text_pad = 0.035)
            else:
                mF.cbar_right_of_axis(fig, ax, pcm[3], width_frac= 0.05, height_frac=1, pad=0.015, numbersize = 9, cbar_label = '', text_pad = 0.1)
    mF.delete_remaining_axes(fig, axes, num_subplots, nrows, ncols)
    return fig

# ---------------------------------------------------------------------------------- Find metric / labels and run ----------------------------------------------------------------------------------------------------- #

def get_list(switch, dataset, metric):
    source = mF.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
    metric_name = f'{metric.name}_sMean'
    metric_option = f'{metric.option}_sMean'
    timescale = mV.timescales[0] 
    experiment_alt = '' if dataset in ['GPCP','ERA5', 'CERES'] else mV.experiments[0]
    path = f'{mV.folder_save[0]}/{metric.variable_type}/metrics/{metric_name}/{source}/{dataset}_{metric_name}_{timescale}_{experiment_alt}_{mV.resolutions[0]}.nc'

    if  metric.variable_type in ['pr', 'org']: # metrics from these variables are always on daily timescale and then resampled
        timescale = 'daily'
        metric_name = metric.name
        metric_option = metric.option
        dataset_alt = 'GPCP' if dataset in ['ERA5', 'CERES'] else dataset # GPCP is used as corresponding precipitation for these obs
        path = f'{mV.folder_save[0]}/{metric.variable_type}/metrics/{metric_name}/{source}/{dataset_alt}_{metric_name}_{mV.conv_percentiles[0]}thPrctile_{timescale}_{experiment_alt}_{mV.resolutions[0]}.nc' if metric.variable_type == 'org' else path     
        path = f'{mV.folder_save[0]}/{metric.variable_type}/metrics/{metric_name}/{source}/{dataset_alt}_{metric_name}_{timescale}_{experiment_alt}_{mV.resolutions[0]}.nc'                                   if metric.variable_type == 'pr' else path     

    array = xr.open_dataset(path)[metric_option]
    array = xr.open_dataset(path)[metric_option].sel(time = slice('2000-03', '2021')) if dataset == 'CERES' else array # full data between
    array = mF.resample_timeMean(array, mV.timescales[0])

    title = ''
    axtitle = dataset
    if switch['anomalies']:
        title = '_anomalies'
        if mV.timescales[0] == 'monthly': 
            climatology = array.groupby('time.month').mean('time')
            array = array.groupby('time.month') - climatology 
        if mV.timescales[0] == 'daily': 
            rolling_mean = array.rolling(time=12, center=True).mean()
            array = array - rolling_mean
            array = array.dropna(dim='time')
    return array, title, axtitle
    
def run_scatter_trend_plot(switch):
    keys = [k for k, v in switch.items() if v]  # list of True keys
    switch_x, switch_y = switch.copy(), switch.copy() 
    switch_x[keys[1]] = False # sets second metric switch to False
    switch_y[keys[0]] = False # sets first metric switch to False
    metric_x, metric_y = mF.get_metric_object(switch_x), mF.get_metric_object(switch_y)
    print(metric_x.label)
    if not switch['xy']:
        metric_x, metric_y = metric_y, metric_x # if plotting the reverse relationship
    source_list = mF.find_list_source(mV.datasets, mV.models_cmip5, mV.models_cmip6, mV.observations)
    ifWith_obs = mF.find_ifWithObs(mV.datasets, mV.observations)
    print(f'switch: {[key for key, value in switch.items() if value]}')
    print(f'Plotting {keys[0]} and {keys[1]}, {mV.timescales[0]} correlation {ifWith_obs} \n with conv threshold at {mV.conv_percentiles[0]}th percentile of {mV.resolutions[0]} daily precipitation data')

    if switch['one dataset']:
        x, title, _ = get_list(switch, mV.datasets[0], metric_x)
        y, _, _ = get_list(switch, mV.datasets[0], metric_y)
        x = x.assign_coords(time=y.time) if mV.timescales[0] == 'monthly' else x

        title = f'{mV.datasets[0]}_{metric_y.option}_and_{metric_x.option}{title}' if not switch['fixed area'] else f'{metric_y.option}_and_{metric_x.option}_fixed_area{title}'

        xmin, xmax = mF.find_limits(switch, [mV.datasets[0]], metric_x, get_list, 
                                 quantileWithin_low = 0, quantileWithin_high = 1, quantileBetween_low = 0, quantileBetween_high=1, # if calculating limits (set lims to '', or comment out)
                                 vmin = None, vmax = None                                                                              # if manually setting limits
                                 )    
        ymin, ymax = mF.find_limits(switch, [mV.datasets[0]], metric_y, get_list, 
                                 quantileWithin_low = 0, quantileWithin_high = 1, quantileBetween_low = 0, quantileBetween_high=1, # if calculating limits (set lims to '', or comment out)
                                 vmin = None, vmax = None                                                                              # if manually setting limits
                                 )    
        fig = plot_one_dataset(switch, x, y, metric_x, metric_y, title, xmin, xmax, ymin, ymax)


    elif switch['multiple datasets']:
        _, title, _ = get_list(switch, mV.datasets[0], metric_x)
        title = f'{metric_y.option}_and_{metric_x.option}{title}_{source_list}{ifWith_obs}' if not switch['fixed area'] else f'{metric_y.option}_and_{metric_x.option}_fixed_area{title}{source_list}{ifWith_obs}'

        xmin, xmax = mF.find_limits(switch, [mV.datasets[0]], metric_x, get_list, 
                                 quantileWithin_low = 0, quantileWithin_high = 1, quantileBetween_low = 0, quantileBetween_high=1, # if calculating limits (set lims to '', or comment out)
                                 vmin = None, vmax = None                                                                              # if manually setting limits
                                 )    
        ymin, ymax = mF.find_limits(switch, [mV.datasets[0]], metric_y, get_list, 
                                 quantileWithin_low = 0, quantileWithin_high = 1, quantileBetween_low = 0, quantileBetween_high=1, # if calculating limits (set lims to '', or comment out)
                                 vmin = None, vmax = None                                                                              # if manually setting limits
                                 )    
        fig = plot_multiple_datasets(switch, metric_x, metric_y, title, xmin, xmax, ymin, ymax)

    folder = f'{mV.folder_save[0]}/corr/{metric_x.name}_and_{metric_y.name}'
    filename = title
    mF.save_figure(fig, folder, f'{filename}.pdf')     if switch['save'] else None
    mF.save_figure(fig, f'{home}/Desktop', 'test.pdf') if switch['save to desktop'] else None
    mF.save_figure(fig, os.getcwd(), 'test.png')       if switch['save to cwd'] else None
    plt.show() if switch['show'] else None


if __name__ == '__main__':
    run_scatter_trend_plot(switch = {
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
            'tas':                 False,
            'hur':                 False,
            'rlut':                False,
            'wap':                 False,
            'wap_area':            False,
            'stability':           False,

            # clouds
            'lcf':                 True,
            'hcf':                 False,

            # moist static energy
            'hus':                 False,

        # --------
        # settings
        # --------
        # plot type
        'xy':                  True,
        'scatter':             False,
        'bins':                True,

        # masked by
        'descent':             False,
        'ascent':              False,
        'fixed area':          False,
        'anomalies':           True,

        # show/save
        'one dataset':         False,
        'multiple datasets':   True,
        'show':                False,
        'save':                False,
        'save to desktop':     True,
        'save to cwd':         False
        }
    )









