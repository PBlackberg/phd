import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats

import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util')
import myFuncs as mF                                # imports common operators

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


class metric_class():
    def __init__(self, label, cmap, color):
        self.label = label
        self.cmap  = cmap
        self.color = color

x = xr.open_dataset('/Users/cbla0002/Documents/data/metrics/org/rome_95thprctile/obs/GPCP_1998-2009_rome_95thprctile_daily__regridded.nc')['rome_95thprctile'].sel(time=slice('2000', '2009')).data
# print(x.time[0:3])
metric_x = metric_class(r'ROME [km$^2$]', 'Blues', 'b')

y = xr.open_dataset('/Users/cbla0002/Documents/data/metrics/ws/ws_low_clouds_sMean/ISCCP_ws_low_clouds_sMean_daily__regridded.nc')['ws_low_clouds_sMean'][0:3653]
metric_y = metric_class(r'ws freq [Nb]', 'Blues', 'b')
switch = {'bins': False, 'scatter': True}

plot_one_dataset(switch, x, y, metric_x, metric_y, title = 'Weather states (low clouds) and organization')
plt.show()











