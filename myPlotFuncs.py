import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import cartopy

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)













def plot_timeseries(y, color= '#1f77b4', title ='', xlabel='', ylabel='', ax='', ymin=None, ymax=None, fig_width=20, fig_height=5):

    if not ax:
        f, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.plot(y, color=color)
    ax.axhline(y=y.mean(dim='time'), color='k')
    ax.set_title(title)
    ax.set_ylim([ymin,ymax])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
































