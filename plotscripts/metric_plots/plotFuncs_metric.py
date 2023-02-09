import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import cartopy

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)




# ---------------------------------------------------------------- single axes plots ----------------------------------------------------------------

def plot_scene(scene, cmap='Reds', title='', vmin=None, vmax=None,fig_width=17.5 ,fig_height=8):
    projection = cartopy.crs.PlateCarree(central_longitude=180)
    lat = scene.lat
    lon = scene.lon

    f, ax = plt.subplots(subplot_kw=dict(projection=projection), figsize=(fig_width, fig_height))
    scene.plot(transform=cartopy.crs.PlateCarree(), cbar_kwargs={'orientation': 'horizontal','pad':0.125, 'aspect':50,'fraction':0.055}, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=cartopy.crs.PlateCarree())
    ax.set_title(title)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels([0, 90, 180, 270, 360])
    ax.set_yticks([-20, 0, 20])
    plt.tight_layout()



def plot_timeseries(y, title ='', ylabel='', xlabel ='', ymin = None, ymax = None, fig_width=20 ,fig_height=10):
    f, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(y)
    ax.axhline(y=y.mean(dim='time'), color='k')
    ax.set_title(title)
    ax.set_ylabel(ylabel + ' ['+y.units+']')
    ax.set_xlabel(xlabel + list(y.dims)[0])
    ax.set_ylim([ymin,ymax])



def plot_timeMean_option(y, ax, timeMean_option):
    if timeMean_option == 'annual' or timeMean_option == 'month' or timeMean_option == 'day':
        ax.plot(y)
        ax.axhline(y=y.mean(dim='time'), color='k')

    if timeMean_option == 'season':
        ax.plot(y, label = y.season.values)

    if timeMean_option == 'season_mean':
        y.to_series().plot.bar(ax=ax)
        ax.axhline(y=0, color='k',linestyle='--')

        ax.set_xlabel('')
        ax.set_xticklabels(y.season.values, rotation=30, ha='right')

    if timeMean_option == 'month_mean':
        ax.plot(y.mean(dim='year'))
        ax.set_xticks(np.arange(0,12))
        ax.set_xticklabels(y.month.values)


def plot_boxplot(y, title='', ylabel=''):
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


def plot_sceneThreshold(scene_background, scene, cmap_background, cmap, title, fig_width=20 ,fig_height=7.5):
    fig= plt.figure(figsize=(fig_width, fig_height))
    lat = scene_background.lat
    lon = scene_background.lon
    lonm,latm = np.meshgrid(lon,lat)
    ax = fig.add_subplot(projection=cartopy.crs.PlateCarree(central_longitude=180))
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=cartopy.crs.PlateCarree())

    pcm= ax.pcolormesh(lonm,latm, scene_background, transform=cartopy.crs.PlateCarree(),zorder=0, cmap=cmap_background)
    ax.pcolormesh(lonm,latm, scene, transform=cartopy.crs.PlateCarree(), cmap=cmap)

    ax.set_title(title)
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')

    ax.set_yticks([-20, 0, 20])
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels([0, 90, 180, 270, 360])

    plt.colorbar(pcm, ax=ax, orientation='horizontal',pad=0.10, aspect=50, fraction=0.055, label = scene_background.units)




# ---------------------------------------------------------------- plot with subplots ----------------------------------------------------------------


def plot_scenes(ds, cmap='Reds', title='', vmin = None, vmax=None):

    n_da = len(ds.data_vars)

    fig_width=17.5
    fig_height=8
    if n_da == 1:
        dataset = list(ds.data_vars.keys())[0]
        scene = ds[dataset]

        plot_scene(scene, cmap, dataset, vmin, vmax,fig_width ,fig_height)
        plt.suptitle(title, fontsize=12, x = 0.62, y=0.65)
        

    elif n_da == 2:
        dataset = list(ds.data_vars.keys())[0]
        scene = ds[dataset]
        plot_scene(scene, cmap, dataset, vmin, vmax)
        plt.suptitle(title, fontsize=12, x = 0.62, y=0.65)

        dataset = list(ds.data_vars.keys())[1]
        scene = ds[dataset]
        plot_scene(scene, cmap, dataset, vmin, vmax)
        

    else:
        n_cols = 4
        n_rows = (n_da + n_cols - 1) // n_cols
        figsize=(22, (15/5)*n_rows)

        fig= plt.figure(figsize = figsize)
        fig.suptitle(title, fontsize=18, y=0.90) #y=0.89

        lat = ds.lat
        lon = ds.lon
        lonm,latm = np.meshgrid(lon,lat)

        for i, dataset in enumerate(list(ds.data_vars.keys())):
            ax= fig.add_subplot(n_rows, n_cols, i + 1, projection=cartopy.crs.PlateCarree(central_longitude=180))
            ax.add_feature(cartopy.feature.COASTLINE)
            ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=cartopy.crs.PlateCarree())

            pcm= ax.pcolormesh(lonm,latm, ds[dataset],transform=cartopy.crs.PlateCarree(),zorder=0, cmap=cmap, vmin=vmin, vmax=vmax)
            
            letters='abcdefghijklmnopqrs'
            plt.text(-177.5, 35, letters[i] + ') ' + dataset, fontsize=12)

            if i== 0 or i==4 or i==8 or i==12 or i==16:
                ax.set_yticks([-20, 0, 20])
                plt.text(-235,-25, 'latitude', rotation=90)

            if (len(ds.data_vars)<=4) or (len(ds.data_vars)>4 and i>=(len(ds.data_vars)-4)) :
                plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.35, aspect=50, fraction=0.055,label = ds[dataset].units)
                plt.text(-25,-70, 'longitude',fontsize=8)
                ax.set_xticks([-180, -90, 0, 90, 180])
                ax.set_xticklabels([0, 90, 180, 270, 360])
            else:
                plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.05, aspect=50, fraction=0.055)

        plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.90, wspace=0.1) 



def plot_timeseries_multiple(ds, timeMean_option, title='', ylabel='', ymin = None, ymax=None):

    n_da = len(ds.data_vars)
    letters='abcdefghijklmnopqrs'
    
    if n_da == 1:
        f, ax = plt.subplots(figsize=(15.5, 4.5))
        plt.suptitle(title, fontsize=12, y=0.90)
        dataset = list(ds.data_vars.keys())[0]
        y = ds[dataset]
        ax.set_title(letters[0] + ') ' + dataset)
        ax.set_ylabel(ylabel + ' ['+y.units+']')
        ax.set_xlabel(list(y.dims)[0])
        plot_timeMean_option(y, ax, timeMean_option)
        
        
    elif n_da == 2:
        f, ax = plt.subplots(figsize=(15.5, 4.5))
        plt.suptitle(title, fontsize=12, y=0.99)
        dataset = list(ds.data_vars.keys())[0]
        y = ds[dataset]
        ax.set_title(letters[0] + ') ' + dataset)
        ax.set_ylabel(ylabel + ' ['+y.units+']')
        ax.set_xlabel(list(y.dims)[0])
        plot_timeMean_option(y, ax, timeMean_option)
        
        f, ax = plt.subplots(figsize=(15,5))
        dataset = list(ds.data_vars.keys())[1]
        y = ds[dataset]
        ax.set_title(letters[1] + ') ' + dataset)
        ax.set_ylabel(ylabel + ' ['+y.units+']')
        ax.set_xlabel(list(y.dims)[0])
        plot_timeMean_option(y, ax, timeMean_option)
        
    else:
        n_cols = 4
        n_rows = (n_da + n_cols - 1) // n_cols
        figsize=(22, (15/5)*n_rows)

        fig= plt.figure(figsize = figsize)
        fig.suptitle(title, fontsize=18, y=0.99) #y=0.89

        for i, dataset in enumerate(list(ds.data_vars.keys())):

            ax= fig.add_subplot(n_rows, n_cols, i + 1)
            ax.set_title(letters[i] + ') ' + dataset)
            y = ds[dataset]
            ax.set_ylim([ymin, ymax])
            plot_timeMean_option(y, ax, timeMean_option)

            if i== 0 or i==4 or i==8 or i==12 or i==16:
                ax.set_ylabel(ylabel + ' ['+y.units+']')

            if (len(ds.data_vars)<=4) or (len(ds.data_vars)>4 and i>=(len(ds.data_vars)-4)) :
                ax.set_xlabel(list(y.dims)[0])


            
            if i ==0 and timeMean_option=='season':
                ax.legend(bbox_to_anchor=(1.75, 0.95), fontsize=5)


            
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.775, wspace=0.15, hspace=0.4)





def boxplotColor(ds, labels='', colors='', title='', ylabel='', ymin = None, ymax=None):
    
    plt.figure(figsize=(4,6))
    plt.xlim(0,1)
    plt.ylim(ymin, ymax)

    y= []
    for dataset in list(ds.data_vars.keys()):
        y = np.append(y, ds[dataset])
      
    plt.boxplot(y,vert=True, positions= [0.3], patch_artist=True, medianprops = dict(color="b",linewidth=1),boxprops = dict(color="b",facecolor='w',zorder=0)
                ,sym='+',flierprops = dict(color="r"))

    x = np.linspace(0.3-0.025, 0.3+0.025, len(y))
    plt.scatter(x, y, c='k', alpha=0.4)


    if labels:
        for i, label in enumerate(labels):
            dataset_idx= list(ds.data_vars.keys()).index(label)
            plt.scatter(x[dataset_idx],y[dataset_idx],c=colors[i], label=label)
            plt.legend(bbox_to_anchor=(0.6,0.5,0.4,0.4))


    plt.xticks([])
    sns.despine(top=True, right=True, left=False, bottom=True)
    plt.title(title)
    plt.ylabel(ylabel + ' [' + ds.units + ']')
    











# ---------------------------------------------------------------- plot related functions ----------------------------------------------------------------



def to_monthly(da):
    year = da.time.dt.year
    month = da.time.dt.month

    # assign new coords
    da = da.assign_coords(year=("time", year.data), month=("time", month.data))

    # reshape the array to (..., "month", "year")
    return da.set_index(time=("year", "month")).unstack("time")



def resample(var, timeMean_option):
    if timeMean_option == 'annual':
        var_resampled = var.resample(time='Y').mean(dim='time', keep_attrs=True)
    
    if timeMean_option == 'season':
        y = var.resample(time='QS-DEC').mean(dim="time")
        y = to_monthly(y)
        y = y.rename({'month':'season'})
        var_resampled = y.assign_coords(season = ["MAM", "JJA", "SON", "DJF"])

    if timeMean_option == 'season_mean':
        y = var.resample(time='QS-DEC').mean(dim="time")
        y = to_monthly(y)
        y = y.rename({'month':'season'})
        y = y.assign_coords(season = ["MAM", "JJA", "SON", "DJF"])
        var_resampled= (y.mean(dim='year') - y.mean(dim='year').mean(dim='season'))


    if timeMean_option == 'month_mean':
        y = to_monthly(var)
        var_resampled = y.assign_coords(month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul','aug', 'sep', 'oct', 'nov', 'dec'])

    if timeMean_option == 'month':
        var_resampled = var.resample(time='M').mean(dim='time', keep_attrs=True)

    if timeMean_option == 'day':
        var_resampled  = var

    return var_resampled





















