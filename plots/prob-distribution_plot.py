import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV
import myFuncs as mF    
import myClasses as mC



dataset = mV.datasets[0]

def r_eff(area):
    return np.sqrt(area/np.pi)

da = xr.open_dataset(f'/Users/cbla0002/Documents/data/sample_data/pr/cmip6/{dataset}_pr_daily_historical_regridded.nc')['pr']
dims = mC.dims_class(da)
r_gridbox = r_eff(dims.aream.mean())

o_area = xr.open_dataset(f'/Users/cbla0002/Documents/data/metrics/org/o_area_95thprctile/cmip6/{dataset}_o_area_95thprctile_daily_historical_regridded.nc')['o_area_95thprctile']
pr_o = xr.open_dataset(f'/Users/cbla0002/Documents/data/metrics/pr/pr_o_sMean_95thprctile/cmip6/{dataset}_pr_o_sMean_95thprctile_daily_historical_regridded.nc')['pr_o_sMean_95thprctile']
metric_class = mC.get_metric_class('o_area')

def calc_pwad(o_area, pr_o, bin_width):
    bins = np.arange(0, r_eff(o_area.max()) + bin_width, bin_width)
    y_bins = []
    for i in np.arange(0,len(bins)-1):
        y_value = (o_area.where((r_eff(o_area)>=bins[i]) & (r_eff(o_area)<bins[i+1])) * pr_o).sum()/(o_area * pr_o).sum()
        y_bins = np.append(y_bins, y_value)
    return bins, y_bins
bins, y_bins = calc_pwad(o_area, pr_o, bin_width = r_gridbox)


def plot_ax_line(ax, x, y, metric_class):
    h = ax.plot(x, y, metric_class.color)
    return h

def plot_one(ax, bins, y_bins, dataset, metric_class):
    plot_ax_line(ax, bins[0:-1], y_bins, metric_class)
    max_index = np.argmax(y_bins)
    max_x = bins[max_index]
    ax.text(max_x, y_bins[max_index], dataset, ha='center', va='bottom', fontsize=12, color='k')

plot = False
if plot:
    fig, ax = mF.create_figure(width = 9, height = 6)
    plot_one(ax, bins, y_bins, dataset, metric_class)

plot = True
if plot:
    fig, ax = mF.create_figure(width = 9, height = 6)
    for dataset in mV.datasets:

        if dataset == 'GPCP_2010-2022':
            o_area = xr.open_dataset(f'/Users/cbla0002/Documents/data/metrics/org/o_area_95thprctile/obs/{dataset}_o_area_95thprctile_daily__regridded.nc')['o_area_95thprctile']
            pr_o = xr.open_dataset(f'/Users/cbla0002/Documents/data/metrics/pr/pr_o_sMean_95thprctile/obs/{dataset}_pr_o_sMean_95thprctile_daily__regridded.nc')['pr_o_sMean_95thprctile']
            metric_class.color = 'r'
        else:
            metric_class.color = 'k'
            if dataset in ['INM-CM5-0', 'CanESM5']:
                metric_class.color = 'y'

            o_area = xr.open_dataset(f'/Users/cbla0002/Documents/data/metrics/org/o_area_95thprctile/cmip6/{dataset}_o_area_95thprctile_daily_historical_regridded.nc')['o_area_95thprctile']
            pr_o = xr.open_dataset(f'/Users/cbla0002/Documents/data/metrics/pr/pr_o_sMean_95thprctile/cmip6/{dataset}_pr_o_sMean_95thprctile_daily_historical_regridded.nc')['pr_o_sMean_95thprctile']
        
        bins, y_bins = calc_pwad(o_area, pr_o, bin_width = r_gridbox)
        bins = bins + 0.5*r_gridbox
        plot_one(ax, bins, y_bins, dataset, metric_class)

        
        if dataset == 'GPCP_2010-2022':
            for i, y in enumerate(y_bins):
                x0 = bins[i] - 0.5 * r_gridbox
                x1 = bins[i] + 0.5 * r_gridbox
                color = 'r'
                plt.plot([x0, x0], [0, y_bins[i]], color=color, linestyle='--')
                plt.plot([x1, x1], [0, y_bins[i]], color=color, linestyle='--')
                plt.plot([x0, x1], [y_bins[i], y_bins[i]], color=color, linestyle='--')

    
    plt.xlim([0, 2000])
    plt.ylim([0, None])
    plt.title('Precipitation Weighted Area Distribution (PWAD)')
    plt.xlabel(r'Effective radius [km$^2$]')
    plt.ylabel(r'Fraction of precipitation in object size bin')


    switch = {                 # overall settings
        # show/save
        'show':                True,
        'save_test_desktop':   False,
        'save_folder_desktop': False,
        'save_folder_cwd':     False,
        }
    

    mF.save_plot(switch, fig, home, filename='')
    plt.show() if switch['show'] else None


















































