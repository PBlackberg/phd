import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import cartopy

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import os
home = os.path.expanduser("~")



# --------------------------------------------------------------------------------- basic plot functions ----------------------------------------------------------------------------------------- #

def plot_scene(scene, cmap='Reds', zorder= 0, title='', ax='', vmin=None, vmax=None, fig_width=17.5, fig_height=8):
    projection = cartopy.crs.PlateCarree(central_longitude=180)
    lat = scene.lat
    lon = scene.lon

    # if the scene is plotted as a figure by itself
    if not ax:
        f, ax = plt.subplots(subplot_kw=dict(projection=projection), figsize=(fig_width, fig_height))
        pcm = scene.plot(transform=cartopy.crs.PlateCarree(), cbar_kwargs={'orientation': 'horizontal','pad':0.125, 'aspect':50,'fraction':0.055}, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.add_feature(cartopy.feature.COASTLINE)
        ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=cartopy.crs.PlateCarree())
        
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-20, 0, 20])
        ax.set_xticklabels([0, 90, 180, 270, 360])
        plt.tight_layout()

    # if the scene is plotted as subplots in a larger figure
    else:
        lonm,latm = np.meshgrid(lon,lat)
        ax.add_feature(cartopy.feature.COASTLINE)
        ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=cartopy.crs.PlateCarree())

        pcm = ax.pcolormesh(lonm,latm, scene, transform=cartopy.crs.PlateCarree(),zorder=zorder, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(title)

    return pcm



def plot_timeseries(y, timeMean_option=[''], title='', ax='', ymin=None, ymax=None, fig_width=20, fig_height=5):

    if not ax:
        f, ax = plt.subplots(figsize=(fig_width, fig_height))

    if timeMean_option[0] == 'annual':
        y = y.resample(time='Y').mean(dim='time', keep_attrs=True)

        ax.plot(y)
        ax.axhline(y=y.mean(dim='time'), color='k')

    if timeMean_option[0] == 'seasonal':
        y = y.resample(time='QS-DEC').mean(dim="time")
        y = to_monthly(y)
        y = y.rename({'month':'season'})
        y = y.assign_coords(season = ["MAM", "JJA", "SON", "DJF"])
        y = y.isel(year=slice(1, None))

        ax.plot(y, label = y.season.values)

    if timeMean_option[0] == 'monthly':
        y = y.resample(time='M').mean(dim='time', keep_attrs=True)

        ax.plot(y)
        ax.axhline(y=y.mean(dim='time'), color='k')

    if timeMean_option[0] == 'daily' or not timeMean_option[0]:
        ax.plot(y)
        ax.axhline(y=y.mean(dim='time'), color='k')
    
    ax.set_title(title)
    ax.set_ylim([ymin, ymax])



def plot_bar(y, timeMean_option=[''], title='', ax='', ymin=None, ymax=None, fig_width=20, fig_height=5):

    if not ax:
        f, ax = plt.subplots(figsize=(fig_width, fig_height))

    if not timeMean_option[0]:
        y.to_series().plot.bar(ax=ax)

    if timeMean_option[0] == 'seasonal':
        y = y.resample(time='QS-DEC').mean(dim="time")
        y = to_monthly(y)
        y = y.rename({'month':'season'})
        y = y.assign_coords(season = ["MAM", "JJA", "SON", "DJF"])
        y = y.isel(year=slice(1, None))
        y= (y.mean(dim='year') - y.mean(dim='year').mean(dim='season'))

        y.to_series().plot.bar(ax=ax)
        ax.axhline(y=0, color='k',linestyle='--')
        ax.set_xticklabels(y.season.values, rotation=30, ha='right')

    if timeMean_option[0] == 'monthly':
        y = to_monthly(y)
        y = y.assign_coords(month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul','aug', 'sep', 'oct', 'nov', 'dec'])
        y= (y.mean(dim='year') - y.mean(dim='year').mean(dim='month'))

        # ax.plot(y)
        y.to_series().plot.bar(ax=ax)
        ax.axhline(y=0, color='k',linestyle='--')
        ax.set_xticks(np.arange(0,12))
        ax.set_xticklabels(y.month.values,rotation=30, ha='right')
    
    ax.set_ylim([ymin, ymax])
    ax.set_title(title)




def plot_boxplot(y, title='', ylabel='', ax=''):

    if not ax:
        plt.figure(figsize=(4,6))

    plt.xlim(0,1)
    plt.boxplot(y,vert=True, positions= [0.3], patch_artist=True, medianprops = dict(color="b",linewidth=1),boxprops = dict(color="b",facecolor='w',zorder=0)
                ,sym='+',flierprops = dict(color="r"))

    x = np.linspace(0.3-0.025, 0.3+0.025, len(y))
    plt.scatter(x, y, c='k', alpha=0.4)

    plt.xticks([])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(0.6,0.5,0.4,0.4))
    sns.despine(top=True, right=True, left=False, bottom=True)









# ------------------------------------------------------------------------------------- common operations functions ------------------------------------------------------------------------------------- #


def to_monthly(da):
    year = da.time.dt.year
    month = da.time.dt.month

    # assign new coords
    da = da.assign_coords(year=("time", year.data), month=("time", month.data))

    # reshape the array to (..., "month", "year")
    return da.set_index(time=("year", "month")).unstack("time")



def get_dsvariable(variable, dataset, experiment, home=home, resolution='regridded'):

    if resolution == 'regridded':
        folder_model = '{}/Documents/data/CMIP5/ds_cmip5/{}'.format(home,dataset)
        fileName_model = dataset + '_' + variable + '_' + experiment + '.nc'
        path1 = os.path.join(folder_model, fileName_model)

        folder_obs = home + '/Documents/data/obs/ds_obs/' + dataset
        fileName_obs = dataset + '_' + variable + '.nc'
        path2 = os.path.join(folder_obs, fileName_obs)

        folder_model = '{}/Documents/data/CMIP5/ds_cmip5_raw/{}'.format(home,dataset)
        fileName_model = dataset + '_' + variable + '_' + experiment + '.nc'
        path3 = os.path.join(folder_model, fileName_model)

        try:
            ds = xr.open_dataset(path1)
        except FileNotFoundError:
            try:
                ds = xr.open_dataset(path2)
            except FileNotFoundError:
                print(f"Error: no file at {path1} or {path2}")
                try:
                    ds = xr.open_dataset(path3)
                except FileNotFoundError:
                    print(f"Error: no file at {path1} or {path2} or {path3}")

    if resolution == 'original':
        folder_model = '{}/Documents/data/CMIP5/ds_cmip5_orig/{}'.format(home,dataset)
        fileName_model = dataset + '_' + variable + '_'+ experiment+ '_orig.nc'
        path1 = os.path.join(folder_model, fileName_model)

        folder_obs = '{}/Documents/data/obs/ds_obs_orig/{}'.format(home,dataset)
        fileName_obs = dataset + '_' + variable + '_orig.nc'
        path2 = os.path.join(folder_obs, fileName_obs)

        folder_model = '{}/Documents/data/CMIP5/ds_cmip5_raw/{}'.format(home,dataset)
        fileName_model = dataset + '_' + variable + '_' + experiment + '_orig.nc'
        path3 = os.path.join(folder_model, fileName_model)

        try:
            ds = xr.open_dataset(path1)
        except FileNotFoundError:
            try:
                ds = xr.open_dataset(path2)
            except FileNotFoundError:
                print(f"Error: no file at {path1} or {path2}")
                try:
                    ds = xr.open_dataset(path3)
                except FileNotFoundError:
                    print(f"Error: no file at {path1} or {path2} or {path3}")
    return ds

                        

def get_metric(metric, dataset, experiment='historical', home=home, resolution='regridded'):

    if resolution == 'regridded':
        folder_model = '{}/Documents/data/CMIP5/metrics_cmip5/{}'.format(home,dataset)
        fileName_model = dataset + '_' + metric + '_' + experiment + '.nc'
        path1 = os.path.join(folder_model, fileName_model)

        folder_obs = home + '/Documents/data/obs/metrics_obs/' + dataset
        fileName_obs = dataset + '_' + metric + '.nc'
        path2 = os.path.join(folder_obs, fileName_obs)

        try:
            ds = xr.open_dataset(path1)
        except FileNotFoundError:
            try:
                ds = xr.open_dataset(path2)
            except FileNotFoundError:
                print(f"Error: no file at {path1} or {path2}")

    if resolution == 'original':
        folder_model = '{}/Documents/data/CMIP5/metrics_cmip5_orig/{}'.format(home,dataset)
        fileName_model = dataset + '_' + metric + '_'+ experiment+ '_orig.nc'
        path1 = os.path.join(folder_model, fileName_model)

        folder_obs = '{}/Documents/data/obs/metrics_obs_orig/{}'.format(home,dataset)
        fileName_obs = dataset + '_' + metric + '_orig.nc'
        path2 = os.path.join(folder_obs, fileName_obs)

        try:
            ds = xr.open_dataset(path1)
        except FileNotFoundError:
            try:
                ds = xr.open_dataset(path2)
            except FileNotFoundError:
                print(f"Error: no file at {path1} or {path2}")

    return ds
















# def find_limits(variable, datasets, resolutions, experiments, home=home, timeMean_option=[''], quantile_low=0, quantile_high=1, scene_type=''):

#     vmin, vmax = [], []
#     for dataset in datasets:
#         for resolution in resolutions:
#             for experiment in experiments:
                
#                 data = get_dsvariable(variable, dataset, resolution, experiment, home)[variable]

#                 if scene_type == 'example':
#                     data = get_dsvariable(variable, dataset, resolution, experiment, home)[variable].isel(time=0)
                
#                 elif scene_type == 'experiment':
#                     data = get_dsvariable(variable, dataset, resolution, experiment, home)[variable].mean(dim=('time'),keep_attrs=True)

#                 elif scene_type == 'difference':
#                     data_historical = get_dsvariable(variable, dataset, resolution, experiment='historical')[variable].mean(dim=('time'))
#                     data_rcp85 = get_dsvariable(variable, dataset, resolution, experiment='rcp85')[variable].mean(dim=('time'))

#                     if variable != 'tas':
#                         tas_historical = get_dsvariable(variable='tas', dataset=dataset, resolution=resolution, experiment='historical')[variable].mean(dim=('time'))
#                         tas_rcp85 = get_dsvariable(variable='tas', dataset=dataset, resolution=resolution, experiment='rcp85')[variable].mean(dim=('time'))
#                         tas_difference = tas_rcp85 - tas_historical
#                         data = (data_rcp85 - data_historical)/(data_historical*tas_difference)
#                     else:
#                         data = (data_rcp85 - data_historical)/data_historical


#                     if timeMean_option[0] == 'difference':
#                         aWeights = np.cos(np.deg2rad(data_historical.lat))
#                         data = (data_rcp85.weighted(aWeights).mean(dim=('lat','lon')) - data_historical.weighted(aWeights).mean(dim=('lat','lon')))/data_historical.weighted(aWeights).mean(dim=('lat','lon'))


#                 if variable != 'tas' and len(np.shape(data))>2:
#                     aWeights = np.cos(np.deg2rad(data.lat))
#                     y= data.weighted(aWeights).mean(dim=('lat','lon'))
#                 else:
#                     y= data

#                 if timeMean_option[0] == 'experiment':
#                     y = y.mean(dim='time', keep_attrs=True)

#                 if timeMean_option[0] == 'annual':
#                     y = y.resample(time='Y').mean(dim='time', keep_attrs=True)

#                 if timeMean_option[0] == 'seasonal':
#                     y = y.resample(time='QS-DEC').mean(dim="time")
#                     y = to_monthly(y)
#                     y = y.rename({'month':'season'})
#                     y = y.assign_coords(season = ["MAM", "JJA", "SON", "DJF"])
#                     y = y.isel(year=slice(1, None))

#                 if timeMean_option[0] == 'monthly':
#                     if variable == 'tas' and datasets[0] == 'FGOALS-g2':
#                         pass
#                     else:
#                         y = y.resample(time='M').mean(dim='time', keep_attrs=True)

#                 if timeMean_option[0] == 'daily':
#                     y = y
                
#                 vmin = np.append(vmin, np.quantile(y, quantile_low))
#                 vmax = np.append(vmax, np.quantile(y, quantile_high))

#     vmin = np.min(vmin)
#     vmax = np.max(vmax)

#     return vmin, vmax




















