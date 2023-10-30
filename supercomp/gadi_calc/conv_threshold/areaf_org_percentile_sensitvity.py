import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util')
import myVars as mF
import myFuncs as mF
sys.path.insert(0, f'{os.getcwd()}/plot')
import scatter_plot as sP
import map_plot as mP

import xarray as xr
import matplotlib.pyplot as plt

'''
Organization index may vary with the arefraction used to calculate the metric
ROME takes a lot longer to compute for larger areafraction (probably because a larger matrix need to be created when replicating each gridbox contained in the object)
Here are plots showing how the areafraction may impact the organization index 
'''

prctiles = [
    '97',
    '95',
    '90'
    ]

#  ---------------------------------------------------------------------- Visual representation of area used ----------------------------------------------------------------------------- #
class metric_class():
    def __init__(self):
        self.cmap = 'Blues'
        self.label = 'mm/day'
metric = metric_class()
switch = {'save to desktop': True}
plot = False
if plot:
    da = xr.open_dataset('/Users/cbla0002/Documents/data/pr/sample_data/cmip6/TaiESM1_pr_daily_historical_regridded.nc')['pr']
    fig = mP.plot_one_scene(da.isel(time=0), metric, title = 'TaiESM1: precipitation',  vmin = 0, vmax = 10)
    filename = f'precipitation field'
    mF.save_figure(fig, f'{home}/Desktop', f'{filename}.pdf') if switch['save to desktop'] else None
    plt.show()


class metric_class():
    def __init__(self):
        self.cmap = 'Greys'
        self.label = 'mm/day'
metric = metric_class()
plot = False
if plot:
    da = xr.open_dataset('/Users/cbla0002/Documents/data/pr/sample_data/cmip6/TaiESM1_pr_daily_historical_regridded.nc')['pr']
    conv_thresholds = [ da.quantile(0.95, dim=('lat', 'lon'), keep_attrs=True), da.quantile(0.97, dim=('lat', 'lon'), keep_attrs=True)]
    for prctile in prctiles:
        conv_threshold = da.quantile(int(prctile)*0.01, dim=('lat', 'lon'), keep_attrs=True).mean(dim='time')
        scene = (da.isel(time=0).where(da.isel(time=0) >= conv_threshold)>0)*1
        fig = mP.plot_one_scene(scene, metric)
        filename = f'{prctile} threshold on precipitation'
        mF.save_figure(fig, f'{home}/Desktop', f'{filename}.pdf') if switch['save to desktop'] else None
        plt.show()


# ---------------------------------------------------------------------- plot area fraction variation ----------------------------------------------------------------------------- #
plot = False
if plot:
    for prctile in prctiles:
        da = xr.open_dataset(f'/Users/cbla0002/Documents/data/org/metrics/ni/cmip6/TaiESM1_ni_{prctile}thPrctile_daily_historical_regridded.nc')
        plt.figure()
        plt.title(f'{prctile}th percentile threshold (fixed precipitation rate)')
        plt.plot(da['areafraction'])
        plt.ylabel('areafraction')
        plt.xlabel('time (days)')
        plt.show()


# ----------------------------------------------------------------- plot correlation with organization index ----------------------------------------------------------------------------- #

class metric_class():
    def __init__(self):
        self.cmap = 'Greys'
        self.color = 'k'
metric = metric_class()
switch = {'bins':True, 
          'save to desktop': True}

plot= False
if plot:
    for prctile in prctiles:
        da = xr.open_dataset(f'/Users/cbla0002/Documents/data/org/metrics/ni/cmip6/TaiESM1_ni_{prctile}thPrctile_daily_historical_regridded.nc')
        fig, ax = plt.subplots()
        x = da['areafraction']
        y = da['ni']
        sP.plot_ax_scatter(switch, ax, x, y, metric)
        plt.title(f'{prctile}th percentile threshold (fixed precipitation rate)')
        plt.xlabel('areafraction')
        plt.ylabel('number of objects')

        filename = f'areaf_and_ni_{prctile}_threshold'
        mF.save_figure(fig, f'{home}/Desktop', f'{filename}.pdf') if switch['save to desktop'] else None
        plt.show()

plot= True
if plot:
    for prctile in prctiles:
        da = xr.open_dataset(f'/Users/cbla0002/Documents/data/org/metrics/ni/cmip6/TaiESM1_ni_{prctile}thPrctile_daily_historical_regridded.nc')
        da2 = xr.open_dataset(f'/Users/cbla0002/Documents/data/org/metrics/rome/cmip6/TaiESM1_rome_{prctile}thPrctile_daily_historical_regridded.nc')
        fig, ax = plt.subplots()
        x = da['areafraction']
        y = da2['rome']
        sP.plot_ax_scatter(switch, ax, x, y, metric)
        plt.title(f'{prctile}th percentile threshold (fixed precipitation rate)')
        plt.xlabel('areafraction')
        plt.ylabel('rome')

        filename = f'areaf_and_rome_{prctile}_threshold'
        mF.save_figure(fig, f'{home}/Desktop', f'{filename}.pdf') if switch['save to desktop'] else None
        plt.show()











