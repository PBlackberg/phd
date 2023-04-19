import xarray as xr
import numpy as np
from scipy import stats

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



def plot_scatter(x, y ,ax='', title = '', ylabel='', xlabel=''):
    if not ax:
        f, ax = plt.subplots(figsize = (12.5,8))
        plt.ylabel(ylabel + ' [' + y.units +']')
        plt.xlabel(xlabel + ' ['+ x.units +']')

    plt.scatter(x,y,facecolors='none', edgecolor='k')
    res= stats.pearsonr(x,y)      
    if res[1]<=0.05:
        ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', xytext=(0.05, 0.875), textcoords='axes fraction')

    plt.title(title)




def plot_timeseries(y, variable_name, series_type):
    plt.figure(figsize=(25,5))
    plt.plot(y)
    plt.axhline(y=y.mean(dim='time'), color='k')
    plt.title(variable_name + ', '+ series_type + ', ' + model + ', ' + experiment)
    plt.ylabel(variable_name + ' ['+y.units+']')
    plt.xlabel(series_type)



def plot_bins(x,y):
    plt.figure(figsize=(10,5))

    bin_width = (x.max() - x.min())/100
    bin_end = x.max()
    bins = np.arange(0, bin_end+bin_width, bin_width)

    areaFrac_bins = []
    for i in np.arange(0,len(bins)-1):
        areaFrac_bins = np.append(areaFrac_bins, y.where((x>=bins[i]) & (x<=bins[i+1])).mean())
    plt.plot(areaFrac_bins)

    # plt.title(area_option + ' and ' + org_option + ', ' + bin_type + ', ' + model + ', ' + experiment)
    # plt.ylabel(area_option + ' [' + y.units +']')
    # plt.xlabel(org_option + ' ['+ x.units +']')



def plot_sceneThreshold(scene_background, scene, cmap_background, cmap, title, fig_width=17.5 ,fig_height=8):
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




# ---------------------------------------------------------------- figure with subplots ----------------------------------------------------------------


def plot_scatter_multiple(ds_x, ds_y, timeMean_option, title='', ylabel='', xlabel='', ymin = None, ymax=None, xmin = None, xmax = None):

    n_da = len(ds_y.data_vars)
    letters='abcdefghijklmnopqrs'
    
    if n_da == 1:
        dataset = list(ds_y.data_vars.keys())[0]
        y = ds_y[dataset]
        x = ds_x[dataset]
        title_sub = letters[0] + ') ' + dataset
        plot_scatter(x,y ,ax='', title=title_sub, ylabel='', xlabel='')
        plt.suptitle(title, fontsize=12, y=0.90)

    elif n_da == 2:
        dataset = list(ds_y.data_vars.keys())[0]
        y = ds_y[dataset]
        x = ds_x[dataset]
        title = letters[0] + ') ' + dataset
        plot_scatter(x,y ,ax='', title = title_sub, ylabel='', xlabel='')
        plt.suptitle(title, fontsize=12, y=0.90)
        
        dataset = list(ds_y.data_vars.keys())[1]
        y = ds_y[dataset]
        x = ds_x[dataset]
        title_sub = letters[1] + ') ' + dataset
        plot_scatter(x,y ,ax='', title = title_sub, ylabel='', xlabel='')
        
    else:
        n_cols = 4
        n_rows = (n_da + n_cols - 1) // n_cols
        figsize=(22, (15/5)*n_rows)

        fig= plt.figure(figsize = figsize)
        fig.suptitle(title, fontsize=18, y=0.85)

        for i, dataset in enumerate(list(ds_y.data_vars.keys())):
            ax= fig.add_subplot(n_rows, n_cols, i + 1)
            # ax.set_ylim([ymin, ymax])
            # ax.set_xlim([xmin, xmax])
            
            y = ds_y[dataset]
            x = ds_x[dataset]

            title_sub = letters[i] + ') ' + dataset
            plot_scatter(x,y ,ax=ax, title = title_sub, ylabel='', xlabel='')

            if i== 0 or i==4 or i==8 or i==12 or i==16:
                ax.set_ylabel(ylabel + ' ['+y.units+']')

            if (len(ds_y.data_vars)<=4) or (len(ds_y.data_vars)>4 and i>=(len(ds_y.data_vars)-4)) :
                ax.set_xlabel(xlabel)

            if i ==0 and timeMean_option=='season':
                ax.legend(bbox_to_anchor=(1.75, 0.95), fontsize=5)
            
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.775, wspace=0.15, hspace=0.4)





def plot_bins_multiple(ds_x, ds_y, timeMean_option, title='', ylabel='', xlabel='', ymin = None, ymax=None, xmin = None, xmax = None):

    n_da = len(ds_y.data_vars)
    letters='abcdefghijklmnopqrs'
    
    if n_da == 1:
        dataset = list(ds_y.data_vars.keys())[0]
        y = ds_y[dataset]
        x = ds_x[dataset]
        title_sub = letters[0] + ') ' + dataset
        plot_bins(x,y)
        plt.suptitle(title, fontsize=12, y=0.90)

    elif n_da == 2:
        dataset = list(ds_y.data_vars.keys())[0]
        y = ds_y[dataset]
        x = ds_x[dataset]
        title = letters[0] + ') ' + dataset
        plot_bins(x,y)
        plt.suptitle(title, fontsize=12, y=0.90)
        
        dataset = list(ds_y.data_vars.keys())[1]
        y = ds_y[dataset]
        x = ds_x[dataset]
        title_sub = letters[1] + ') ' + dataset
        plot_scatter(y,x ,ax='', title = title_sub, ylabel='', xlabel='')
        plt.suptitle(title, fontsize=12, y=0.90)
        
    else:
        n_cols = 4
        n_rows = (n_da + n_cols - 1) // n_cols
        figsize=(22, (30/5)*n_rows)

        fig= plt.figure(figsize = figsize)
        fig.suptitle(title, fontsize=18, y=0.85)


        for j, dataset in enumerate(list(ds_y.data_vars.keys())):
            y = ds_y[dataset]
            x = ds_x[dataset]

            bin_width = (x.max() - x.min())/100
            bin_end = x.max()
            bins = np.arange(0, bin_end+bin_width, bin_width)

            areaFrac_bins = []
            for i in np.arange(0,len(bins)-1):
                areaFrac_bins = np.append(areaFrac_bins, y.where((x>=bins[i]) & (x<=bins[i+1])).mean())

            ax= fig.add_subplot(4, 4, j + 1)
            ax.plot(areaFrac_bins)


            ax.set_ylim([ymin, ymax])
            ax.set_xlim([xmin, xmax])
            ax.set_title(letters[j] + ') ' + dataset)

            

            if j== 0 or j==4 or j==8 or j==12 or j==16:
                ax.set_ylabel(ylabel + ' ['+y.units+']')

            if (len(ds_y.data_vars)<=4) or (len(ds_y.data_vars)>4 and j>=(len(ds_y.data_vars)-4)) :
                ax.set_xlabel(xlabel)

            if j ==0 and timeMean_option=='season':
                ax.legend(bbox_to_anchor=(1.75, 0.95), fontsize=5)
            
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.775, wspace=0.15, hspace=0.4)














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





















































