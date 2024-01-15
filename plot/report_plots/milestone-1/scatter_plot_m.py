import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Patch

import os
home = os.path.expanduser("~")
import myFuncs_m as mF
import myVars_m as mV

# ---------------------------------------------------------------------------------------- get data ----------------------------------------------------------------------------------------------------- #

def get_mean_value(source, dataset, metric, experiment):
    if metric.option == 'ecs':
        mean_value = mV.ecs_list[dataset]  
    elif  metric.variable_type in ['pr', 'org']: # metrics from these variables are always on daily timescale and then resampled
        timescale = 'daily'
        metric_name = metric.name
        metric_option = metric.option
        experiment_alt = '' if dataset in ['GPCP','ERA5', 'CERES'] else experiment
        dataset_alt = 'GPCP' if dataset in ['ERA5', 'CERES'] else dataset # GPCP is used as corresponding precipitation for these obs
        if dataset_alt == 'GPCP':
            path = f'{mV.folder_save[0]}/{metric.variable_type}/metrics/{metric_name}/{source}/{dataset_alt}_{metric_name}_{mV.conv_percentiles[0]}thPrctile_{timescale}_{experiment_alt}_{mV.resolutions[0]}_short.nc' if metric.variable_type == 'org' else None     
            path = f'{mV.folder_save[0]}/{metric.variable_type}/metrics/{metric_name}/{source}/{dataset_alt}_{metric_name}_{timescale}_{experiment_alt}_{mV.resolutions[0]}_short.nc'                                   if metric.variable_type == 'pr' else path     
        else:
            path = f'{mV.folder_save[0]}/{metric.variable_type}/metrics/{metric_name}/{source}/{dataset_alt}_{metric_name}_{mV.conv_percentiles[0]}thPrctile_{timescale}_{experiment_alt}_{mV.resolutions[0]}.nc' if metric.variable_type == 'org' else None    
            path = f'{mV.folder_save[0]}/{metric.variable_type}/metrics/{metric_name}/{source}/{dataset_alt}_{metric_name}_{timescale}_{experiment_alt}_{mV.resolutions[0]}.nc'                                   if metric.variable_type == 'pr' else path     

        mean_value = xr.open_dataset(path)[metric_option].mean(dim='time').data
        # mean_value = xr.open_dataset(path)[metric_option].sel(time = slice('2000-03', '2021')).mean(dim='time').data if dataset == 'CERES' else mean_value # full data between
        mean_value = xr.open_dataset(path)[metric_option].sel(time = slice('2010', '2021')).mean(dim='time').data if dataset in ['CERES', 'ERA5'] else mean_value # full data between
    else:    
        metric_name = f'{metric.name}_sMean'
        metric_option = f'{metric.option}_sMean'
        timescale = mV.timescales[0] 
        experiment_alt = '' if dataset in ['GPCP','ERA5', 'CERES'] else experiment
        path = f'{mV.folder_save[0]}/{metric.variable_type}/metrics/{metric_name}/{source}/{dataset}_{metric_name}_{timescale}_{experiment_alt}_{mV.resolutions[0]}.nc'
        mean_value = xr.open_dataset(path)[metric_option].mean(dim='time').data
    return mean_value

def get_list(switch, dataset, metric):
    alist_mean = []
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
            if switch['per kelvin']:
                tas_historical = xr.open_dataset(f'{mV.folder_save[0]}/tas/metrics/tas_sMean/{source}/{dataset}_tas_sMean_{mV.timescales[0]}_{mV.experiments[0]}_{mV.resolutions[0]}.nc')['tas_sMean'].mean(dim='time')
                tas_warm = xr.open_dataset(f'{mV.folder_save[0]}/tas/metrics/tas_sMean/{source}/{dataset}_tas_sMean_{mV.timescales[0]}_{mV.experiments[1]}_{mV.resolutions[0]}.nc')['tas_sMean'].mean(dim='time')
                tas_change = tas_warm - tas_historical
                mean_value = mean_value/tas_change
            mean_value = value_historical if metric.option == 'ecs' else mean_value
            mean_value = value_warm - value_historical if metric.option == 'tas' else mean_value
            # if metric.variable_type == 'stability':
            #     mean_value = value_historical

        alist_mean = np.append(alist_mean, mean_value)
    return alist_mean, title, axtitle


# ---------------------------------------------------------------------------------------- plot / format plot ----------------------------------------------------------------------------------------------------- #

def plot_scatter(x, y, metric_x, metric_y, title = '', xmin = None, xmax = None, ymin = None, ymax = None):
    fig, ax = mF.create_figure(width = 10, height = 5.5)
    mF.scale_ax_x(ax, scaleby=0.75)

    # ax.scatter(x, y, facecolors='none', edgecolor= metric_y.color)    
    ax.scatter(x, y, alpha=0)  # Completely transparent points (putting text over)

    # res= stats.pearsonr(x[0:-1],y[0:-1]) # for with obs
    res= stats.pearsonr(x,y)

    print('r: ', res[0])
    print('p-value:', res[1])
    if res[1]<=0.05:
        ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', 
                    xytext=(0.8, 0.9), textcoords='axes fraction', fontsize = 12, color = 'r')
        
    legend_handles = []
    legend_labels = []
    # put 2 letters as label for each dataset (with associated legend)
    for dataset in mV.datasets:
        dataset_idx = mV.datasets.index(dataset)
        label_text = dataset[0:2]

        if dataset in ['TaiESM1', 'CMCC-ESM2', 'CESM2-WACCM', 'NorESM2-MM', 'CNRM-CM6-1', 'MIROC6']:
            scatter_color = 'b'
            text_color = 'b'
        elif dataset in ['GPCP', 'ISCCP', 'CERES', 'ERA5']:
            scatter_color = 'g'
            text_color = 'g'
        else:
            scatter_color = 'k'
            text_color = 'k'
        # ax.scatter(x[dataset_idx], y[dataset_idx], c=scatter_color)
        # ax.text(x[dataset_idx], y[dataset_idx], label_text, color=text_color)
        ax.text(x[dataset_idx], y[dataset_idx], label_text, color=text_color, ha='center', va='center')

        # # Add text to the plot
        # ax.text(x[dataset_idx], y[dataset_idx], label_text, color=text_color, ha='center', va='center', fontsize=12)

        # Create custom legend handle and label
        legend_handles.append(Patch(facecolor=text_color, edgecolor='none'))
        legend_labels.append(f"{label_text} - {dataset}")

    # Add the custom legend
    ax.legend(handles=legend_handles, labels=legend_labels, bbox_to_anchor=(1.05, 0.5), loc='center left')



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

# ---------------------------------------------------------------------------------- interpret switch and run ----------------------------------------------------------------------------------------------------- #

def run_scatter_plot(switch):
    keys = [k for k, v in switch.items() if v]  # list of True keys
    switch_x, switch_y = switch.copy(), switch.copy() 
    switch_x[keys[1]] = False # sets second variable to False
    switch_y[keys[0]] = False # sets first variable to False
    metric_x, metric_y = mF.get_metric_object(switch_x), mF.get_metric_object(switch_y)
    source_list = mF.find_list_source(mV.datasets, mV.models_cmip5, mV.models_cmip6, mV.observations)
    ifWith_obs = mF.find_ifWithObs(mV.datasets, mV.observations)
    print(f'switch: {[key for key, value in switch.items() if value]}')
    print(f'Plotting {metric_x.option} and {metric_y.option} correlation with {mV.resolutions[0]} data')
    # print(f'Plotting {keys[0]} and {keys[1]}, {mV.timescales[0]} correlation {ifWith_obs} \n with conv threshold at {mV.conv_percentiles[0]}th percentile of {mV.resolutions[0]} daily precipitation data')
    
    x, title, _ = get_list(switch, mV.datasets, metric_x)        
    y, _, _ = get_list(switch, mV.datasets, metric_y)     
    title = f'{metric_y.option}_and_{metric_x.option}{title}_{source_list}{ifWith_obs}' if not switch['fixed area'] else f'{metric_y.option}_and_{metric_x.option}_fixed_area{title}{source_list}{ifWith_obs}'

    xmin, xmax = mF.find_limits(switch, [mV.datasets[0]], metric_x, get_list, 
                                quantileWithin_low = 0, quantileWithin_high = 1, quantileBetween_low = 0, quantileBetween_high=1, # if calculating limits (set lims to '', or comment out)
                                vmin = None, vmax = None                                                                          # if manually setting limits
                                )    
    ymin, ymax = mF.find_limits(switch, [mV.datasets[0]], metric_y, get_list, 
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
            'F_pr10':              False,
            
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
            'hur_250hpa':          False,
            'hur_700hpa':          False,
            'wap_area':            False,
            'ecs':                 False,
            'rlut':                False,
            'stability':           False,
            'wap':                 False,

            # radiation

            # clouds
            'lcf':                 True,
            'hcf':                 False,


            # moist static energy
            'hus':                 False,

        # --------
        # settings
        # --------
        # plot type
        'climatology':         False,
        'change with warming': True,

        # masked by
        'descent':             True,
        'ascent':              False,
        'fixed area':          False,
        'per kelvin':          True,

        # show/save
        'show':                True,
        'save to cwd':         False,
        'save to desktop':     True,
        'save':                False
        }
    )




