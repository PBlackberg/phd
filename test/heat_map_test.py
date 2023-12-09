''' 
This script plots a heatmap of objects (contiguous convective regions)
Per definition, the objects will fall where the 95th percentile precipitation falls.  This has the double ITCZ structure pattern
'''



import numpy as np
import xarray as xr
import skimage.measure as skm
import matplotlib.pyplot as plt
import warnings                               
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util')
import constructed_fields as cF                     
import get_data as gD                               
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV                                 
import myClasses as mC
import myFuncs as mF 



switch = {
    'daily_scene': True, 
    'heatmap': True
    }

title, cmap, label = 'heat_map_plot', 'Blues', 'freq [Nb]'

def plot_one_scene(scene, cmap, title, label, vmin = None, vmax = None):
    ''' Plotting singular scene, mainly for testing '''
    fig, ax = mF.create_map_figure(width = 12, height = 4)
    pcm = mF.plot_axMapScene(ax, scene, cmap, vmin = vmin, vmax = vmax)
    mF.move_col(ax, moveby = -0.055)
    mF.move_row(ax, moveby = 0.075)
    mF.scale_ax(ax, scaleby = 1.15)
    mF.cbar_below_axis(fig, ax, pcm, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = f'{label}', text_pad = 0.125)
    mF.plot_xlabel(fig, ax, 'Lon', pad = 0.1, fontsize = 12)
    mF.plot_ylabel(fig, ax, 'Lat', pad = 0.055, fontsize = 12)
    mF.plot_axtitle(fig, ax, title, xpad = 0.005, ypad = 0.025, fontsize = 15)
    mF.format_ticks(ax, labelsize = 11)
    return fig

da = xr.open_dataset('/Users/cbla0002/Documents/data/sample_data/pr/cmip6/ACCESS-CM2_pr_daily_historical_orig.nc')['pr']
dim = mC.dims_class(da)
conv_threshold = da.quantile(int(mV.conv_percentiles[0])*0.01, dim=('lat', 'lon'), keep_attrs=True)
conv_threshold = xr.DataArray(data = conv_threshold.mean(dim='time').data * np.ones(shape = len(da.time)), dims = 'time', coords = {'time': da.time.data})
da = da.where(da >= conv_threshold.isel(time= 0))
da = (da > 0).astype(int)
da_heatmap = da.sum(dim = 'time')


plot_one_scene(da.isel(time=0), cmap, title, label, vmin = None, vmax = None) # one scene of objects
plt.show() if switch['daily_scene'] else None

plot_one_scene(da_heatmap, cmap, title, label, vmin = None, vmax = None) # heat map, frequency of objects
plt.show() if switch['heatmap'] else None









































