import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import cartopy

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


def plot_scene(scene, cmap='Reds', title='', vmin=None, vmax=None,fig_width=20 ,fig_height=10):
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


def plot_scenes_tog(home, models, obs, experiment, 
                    scene_type, variable_file, 
                    variable_option, title,
                    vmin=None, vmax=None):
    
    fig= plt.figure(figsize=(22,12))
    fig.suptitle(title, fontsize=18, y=0.89)

    for i, model in enumerate(models):
        tas_historical = xr.open_dataset(home + '/Documents/data/cmip5/' + model + '/' + model + '_tas_sMean_historical.nc').tas_sMean.mean(dim='time',keep_attrs=True)
        tas_rcp = xr.open_dataset(home + '/Documents/data/cmip5/' + model + '/' + model + '_tas_sMean_rcp85.nc').tas_sMean.mean(dim='time',keep_attrs=True)
        tas_diff = tas_rcp - tas_historical


        if variable_option == 'tas':
            var_historical = xr.open_dataset(home + '/Documents/data/cmip5/' + 'ds' + '/' + model + '_' + variable_file + '_historical.nc')[variable_option].mean(dim='time',keep_attrs=True)
            var_rcp = xr.open_dataset(home + '/Documents/data/cmip5/' + 'ds' + '/' + model + '_' + variable_file + '_rcp85.nc')[variable_option].mean(dim='time',keep_attrs=True)
        else:
            var_historical = xr.open_dataset(home + '/Documents/data/cmip5/' + model + '/' + model + '_' + variable_file + '_historical.nc')[variable_option].mean(dim='time',keep_attrs=True)
            var_rcp = xr.open_dataset(home + '/Documents/data/cmip5/' + model + '/' + model + '_' + variable_file + '_rcp85.nc')[variable_option].mean(dim='time',keep_attrs=True)

        var_diffData = (var_rcp.data - var_historical.data)
        var_diff = xr.DataArray(
            data  = var_diffData,
            dims=['lat', 'lon'],
            coords={'lat': var_historical.lat.data, 'lon': var_historical.lon.data})

        var_diffTas = var_diff/tas_diff.values

        lat = var_historical.lat
        lon = var_historical.lon
        lonm,latm = np.meshgrid(lon,lat)

        ax= fig.add_subplot(5,4,i+1, projection=cartopy.crs.PlateCarree(central_longitude=180))
        ax.add_feature(cartopy.feature.COASTLINE)
        ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=cartopy.crs.PlateCarree())

        if (scene_type == 'experiment') and (experiment == 'historical'):
            pcm= ax.pcolormesh(lonm,latm, var_historical,transform=cartopy.crs.PlateCarree(),zorder=0, cmap="Blues", vmin=vmin, vmax=vmax)

        if (scene_type == 'experiment') and (experiment == 'rcp85'):
            pcm= ax.pcolormesh(lonm,latm, var_rcp,transform=cartopy.crs.PlateCarree(),zorder=0, cmap="Blues", vmin=vmin, vmax=vmax)

        if scene_type == 'diff':
            pcm= ax.pcolormesh(lonm,latm, var_diff,transform=cartopy.crs.PlateCarree(),zorder=0, cmap="RdBu", vmin=-np.max(var_diff.values), vmax=np.max(var_diff.values))

        if scene_type == 'diff_tas':
            pcm= ax.pcolormesh(lonm,latm, var_diffTas,transform=cartopy.crs.PlateCarree(),zorder=0, cmap="RdBu", vmin=-np.max(var_diffTas.values), vmax=np.max(var_diffTas.values))


        letters='abcdefghijklmnopqrs'
        plt.text(-177.5, 32.5, letters[i-1] + ') ' + model, fontsize=12)

        if i== 0 or i==4 or i==8 or i==12 or i==16:
            ax.set_yticks([-20, 0, 20])
            plt.text(-235,-25, 'latitude', rotation=90)


        if i>=16:
            plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.35, aspect=50, fraction=0.055,label = 'mm/day')
            plt.text(-25,-70, 'longitude',fontsize=8)
            # plt.text(-25,-135, 'mm/day',fontsize=10)
            ax.set_xticks([-180, -90, 0, 90, 180])
            ax.set_xticklabels([0, 90, 180, 270, 360])
        else:
            plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.05, aspect=50, fraction=0.055)


    if obs:
        if (scene_type == 'experiment') and (experiment == 'historical' or experiment == 'rcp85'):
            var_obs = xr.open_dataset(home + '/Documents/data/obs/GPCP' + '/' + obs + '_' + variable_file + '.nc')[variable_option].mean(dim='time',keep_attrs=True)
            var_obs.attrs['units']= 'mm/day'
            lat = var_obs.lat
            lon = var_obs.lon
            lonm,latm = np.meshgrid(lon,lat)

            ax= fig.add_subplot(5,4,20, projection=cartopy.crs.PlateCarree(central_longitude=180))
            ax.add_feature(cartopy.feature.COASTLINE)
            ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=cartopy.crs.PlateCarree())
            pcm= ax.pcolormesh(lonm,latm, var_obs,transform=cartopy.crs.PlateCarree(),zorder=0, cmap="Blues", vmin=vmin, vmax=vmax)

            plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.35, aspect=50, fraction=0.055, label = var_obs.units)
            plt.text(-25,-70, 'longitude',fontsize=8)
            # plt.text(-75,-135, var_obs.units,fontsize=10)
            ax.set_xticks([-180, -90, 0, 90, 180])
            ax.set_xticklabels([0, 90, 180, 270, 360])

            plt.text(-177.5, 32.5, letters[i-1] + ') ' + obs, fontsize=12)


    #plt.text(-30,725, 'Difference in ' + rxday_option + '/K (time mean)',fontsize=18)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.90, wspace=0.1) #, hspace=0.2)



def to_monthly(da):
    year = da.time.dt.year
    month = da.time.dt.month

    # assign new coords
    da = da.assign_coords(year=("time", year.data), month=("time", month.data))

    # reshape the array to (..., "month", "year")
    return da.set_index(time=("year", "month")).unstack("time")


def plot_timeseries(y, variable_name='', series_type='', title='', xmin = None, ymin = None):
    plt.figure(figsize=(25,5))
    plt.plot(y)
    plt.axhline(y=y.mean(dim='time'), color='k')
    plt.title(title)
    plt.ylabel(variable_name + ' ['+y.units+']')
    plt.xlabel(series_type)
    plt.ylim([xmin,ymin])



def plot_timeseries_tog(models, obs, home, experiment, 
                        variable_file, variable_option, 
                        timeMean_option, title,
                        vmin=None, vmax=None):
    
    f, axes = plt.subplots(nrows=5, ncols=4, figsize = (22,14))
    f.suptitle(title, fontsize=18, y=0.95)

    for model, ax in zip(models, axes.ravel()):

        if variable_option == 'tas':
            tas = xr.open_dataset(home + '/Documents/data/cmip5/' + 'ds' + '/' + model + '_' + variable_file + '_' + experiment + '.nc')[variable_option]
            aWeights = np.cos(np.deg2rad(tas.lat))
            y = tas.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)
        else:
            y = xr.open_dataset(home + '/Documents/data/cmip5/' + model + '/' + model + '_' + variable_file + '_' + experiment + '.nc')[variable_option]


        if timeMean_option == 'year':

            if variable_file == 'pr_rxday':
                y.attrs['units']= 'mm/day'
                aWeights = np.cos(np.deg2rad(y.lat))
                ax.plot(y.weighted(aWeights).mean(dim=('lat','lon')))
                ax.axhline(y=y.weighted(aWeights).mean(dim=('time','lat','lon')), color='k')
            else:
                ax.plot(y.resample(time='Y').mean(dim='time'))
                ax.axhline(y=y.resample(time='Y').mean(dim='time').mean(dim='time'), color='k')

        if timeMean_option == 'month':
            ax.plot(y.resample(time='M').mean(dim='time'))
            ax.axhline(y=y.resample(time='M').mean(dim='time').mean(dim='time'), color='k')

        if timeMean_option == 'day':
            ax.plot(y)
            ax.axhline(y=y.mean(dim='time'), color='k')
        
        if timeMean_option == 'season':
            y = y.resample(time='QS-DEC').mean(dim="time", keep_attrs=True)
            y = to_monthly(y)
            y = y.rename({'month':'season'})
            y = y.assign_coords(season = ["MAM", "JJA", "SON", "DJF"])

            ax.plot(y, label = y.season.values)

            if model == 'CESM1-BGC':
                ax.legend(bbox_to_anchor=(1.75, 0.95), fontsize=15)

        if timeMean_option == 'season_mean':
            y = y.resample(time='QS-DEC').mean(dim="time", keep_attrs=True)
            y = to_monthly(y)
            y = y.rename({'month':'season'})
            y = y.assign_coords(season = ["MAM", "JJA", "SON", "DJF"])

            (y.mean(dim='year') - y.mean(dim='year').mean(dim='season')).to_series().plot.bar(ax=ax)
            ax.axhline(y=0, color='k',linestyle='--')

            ax.set_xlabel('')
            ax.set_xticklabels(y.season.values, rotation=30, ha='right')

        if timeMean_option == 'month_mean':
            y = to_monthly(y)
            y = y.assign_coords(month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul','aug', 'sep', 'oct', 'nov', 'dec'])

            ax.plot(y.mean(dim='year'))
            ax.set_xticks(np.arange(0,12))
            ax.set_xticklabels(y.month.values)


        ax.set_title(model)
        ax.set_ylim([vmin, vmax])

        if model == 'MRI-CGCM3' or model == 'CESM1-BGC':
            ax.set_xlabel(timeMean_option)

        if model == 'IPSL-CM5A-MR' or model == 'CNRM-CM5' or model == 'FGOALS-g2' or model == 'NorESM1-M' or model == 'MRI-CGCM3': 
            ax.set_ylabel(variable_option + '[' + y.units + ']')


    if obs:
        ax = axes[-1,-1]
        y = xr.open_dataset(home + '/Documents/data/obs/' + obs + '/' + obs + '_' + variable_file + '.nc')[variable_option]

        if timeMean_option == 'year':
            if variable_file == 'pr_rxday':
                y.attrs['units']= 'mm/day'
                aWeights = np.cos(np.deg2rad(y.lat))
                ax.plot(y.weighted(aWeights).mean(dim=('lat','lon')))
                ax.axhline(y=y.weighted(aWeights).mean(dim=('time','lat','lon')), color='k')
            else:
                ax.plot(y.resample(time='Y').mean(dim='time'))
                ax.axhline(y=y.resample(time='Y').mean(dim='time').mean(dim='time'), color='k')

        if timeMean_option == 'month':
            ax.plot(y.resample(time='M').mean(dim='time'))
            ax.axhline(y=y.resample(time='M').mean(dim='time').mean(dim='time'), color='k')

        if timeMean_option == 'day':
            ax.plot(y)
            ax.axhline(y=y.mean(dim='time'), color='k')
        
        if timeMean_option == 'season':
            y = y.resample(time='QS-DEC').mean(dim="time", keep_attrs=True)
            y = to_monthly(y)
            y = y.rename({'month':'season'})
            y = y.assign_coords(season = ["MAM", "JJA", "SON", "DJF"])

            ax.plot(y, label = y.season.values)

        if timeMean_option == 'season_mean':
            y = y.resample(time='QS-DEC').mean(dim="time", keep_attrs=True)
            y = to_monthly(y)
            y = y.rename({'month':'season'})
            y = y.assign_coords(season = ["MAM", "JJA", "SON", "DJF"])

            (y.mean(dim='year') - y.mean(dim='year').mean(dim='season')).to_series().plot.bar(ax=ax)
            ax.axhline(y=0, color='k',linestyle='--')

            ax.set_xlabel('')
            ax.set_xticklabels(y.season.values, rotation=30, ha='right')

        if timeMean_option == 'month_mean':
            y = to_monthly(y)
            y = y.assign_coords(month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul','aug', 'sep', 'oct', 'nov', 'dec'])

            ax.plot(y.mean(dim='year'))
            ax.set_xticks(np.arange(0,12))
            ax.set_xticklabels(y.month.values)
            
        ax.set_title(obs)
        ax.set_ylim([vmin, vmax])
        ax.set_xlabel(timeMean_option)

    else:
        axes[-1, -1].remove()


    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.15, hspace=0.4)
    axes[-1, 2].remove()





def boxplot(y, title, text_ylabel):
    plt.figure(figsize=(4,6))
    plt.xlim(0,1)

    plt.boxplot(y,vert=True, positions= [0.3], patch_artist=True, medianprops = dict(color="b",linewidth=1),boxprops = dict(color="b",facecolor='w',zorder=0)
                ,sym='+',flierprops = dict(color="r"))

    x = np.linspace(0.3-0.025, 0.3+0.025, len(y))
    plt.scatter(x, y, c='k', alpha=0.4)

    plt.xticks([])
    plt.title(title)
    plt.ylabel(text_ylabel)
    plt.legend(bbox_to_anchor=(0.6,0.5,0.4,0.4))

    sns.despine(top=True, right=True, left=False, bottom=True)




def boxplotColor(home, models, variable_file, variable_option, box_type, experiment, obs, title, text_ylabel):
    y= []
    for model in models:
        tas_historical = xr.open_dataset(home + '/Documents/data/cmip5/' + model + '/' + model + '_tas_sMean_historical.nc').tas_sMean.mean(dim='time', keep_attrs=True)
        tas_rcp = xr.open_dataset(home + '/Documents/data/cmip5/' + model + '/' + model + '_tas_sMean_rcp85.nc').tas_sMean.mean(dim='time', keep_attrs=True)
        tas_diff = tas_rcp - tas_historical

        var_historical = xr.open_dataset(home + '/Documents/data/cmip5/' + model + '/' + model + '_' + variable_file + '_historical.nc')[variable_option].mean(dim='time', keep_attrs=True)
        var_rcp = xr.open_dataset(home + '/Documents/data/cmip5/' + model + '/' + model + '_' + variable_file + '_rcp85.nc')[variable_option].mean(dim='time', keep_attrs=True)
        
        if box_type == 'experiment':
            if variable_file == 'pr_rxday':
                aWeights = np.cos(np.deg2rad(var_historical.lat))
                var = xr.open_dataset(home + '/Documents/data/cmip5/' + model + '/' + model + '_' + variable_file + '_' + experiment + '.nc')[variable_option].weighted(aWeights).mean(dim=('time', 'lat', 'lon'), keep_attrs=True)
                y = np.append(y, var)
            else:
                var = xr.open_dataset(home + '/Documents/data/cmip5/' + model + '/' + model + '_' + variable_file + '_' + experiment + '.nc')[variable_option].mean(dim='time', keep_attrs=True)
                y = np.append(y, var)


        if box_type == 'diff':
            if variable_file == 'pr_rxday':
                aWeights = np.cos(np.deg2rad(var_historical.lat))
                var_diff = (var_rcp.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True) - var_historical.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True))
                y = np.append(y, var_diff)

            else:
                var_diff = var_rcp - var_historical
                y = np.append(y, var_diff)

        
        if box_type == 'diff_tas':
            if variable_file == 'pr_rxday':
                aWeights = np.cos(np.deg2rad(var_historical.lat))
                var_diff = (var_rcp.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True) - var_historical.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True))
                var_diffTas = var_diff/tas_diff.values
                y = np.append(y, var_diffTas)
            else:
                var_diff = var_rcp - var_historical
                var_diffTas = var_diff/tas_diff.values
                y = np.append(y, var_diffTas)

    if obs:
        var_obs = xr.open_dataset(home + '/Documents/data/obs/' + obs + '/' + obs +'_' + variable_file + '.nc')[variable_option].mean(dim='time')

        if box_type == 'experiment':
            if variable_file == 'pr_rxday':
                aWeights = np.cos(np.deg2rad(var_obs.lat))
                var_obs = var_obs.weighted(aWeights).mean(dim=('lat', 'lon'), keep_attrs=True)
                y = np.append(y, var)

            else:
                y = np.append(y, var_obs)            
    

    plt.figure(figsize=(4,6))
    plt.xlim(0,1)

    plt.boxplot(y,vert=True, positions= [0.3], patch_artist=True, medianprops = dict(color="b",linewidth=1),boxprops = dict(color="b",facecolor='w',zorder=0)
                ,sym='+',flierprops = dict(color="r"))

    x = np.linspace(0.3-0.025, 0.3+0.025, len(y))
    plt.scatter(x, y, c='k', alpha=0.4)
    #plt.scatter(np.ones(len(y)), y)


    if box_type == 'experiment' and experiment == 'historical':
        x_leg = [models.index('IPSL-CM5A-MR'),models.index('FGOALS-g2'), models.index('bcc-csm1-1'), -1]
        label = ['IPSL-CM5A-MR','FGOALS-g2','bcc-csm1-1', 'GPCP']
        colors = ['r','darkred','b', 'g']
    else:
        x_leg = [models.index('IPSL-CM5A-MR'),models.index('FGOALS-g2'), models.index('bcc-csm1-1')]
        label = ['IPSL-CM5A-MR','FGOALS-g2','bcc-csm1-1']
        colors = ['r','darkred','b']

    j=0
    for i in x_leg:
        plt.scatter(x[i],y[i],c=colors[j], label=label[j])
        j+=1


    plt.xticks([])
    plt.title(title)
    plt.ylabel(text_ylabel)
    plt.legend(bbox_to_anchor=(0.6,0.5,0.4,0.4))

    sns.despine(top=True, right=True, left=False, bottom=True)
    



