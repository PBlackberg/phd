import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
import cartopy

import os
home = os.path.expanduser("~")

import timeit

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 


models = [
    # 'IPSL-CM5A-MR', # 1
    # 'GFDL-CM3',     # 2
    # 'GISS-E2-H',    # 3
    # 'bcc-csm1-1',   # 4
    # 'CNRM-CM5',     # 5
    # 'CCSM4',        # 6
    # 'HadGEM2-AO',   # 7
    # 'BNU-ESM',      # 8
    # 'EC-EARTH',     # 9
    # 'FGOALS-g2',    # 10
    # 'MPI-ESM-MR',   # 11
    # 'CMCC-CM',      # 12
    # 'inmcm4',       # 13
    # 'NorESM1-M',    # 14
    # 'CanESM2',      # 15
    # 'MIROC5',       # 16
    # 'HadGEM2-CC',   # 17
    # 'MRI-CGCM3',    # 18
    # 'CESM1-BGC'     # 19
    ]


experiments = [
    'historical',
    # 'rcp85'
    ]
experiment = experiments[0]


observations = [
    'GPCP',
    # 'IMERG'
    ]
original_resolution = True


rome_options = [
    'rome',
    # 'rome_n'
    ]
rome_option = rome_options[0]



percentile_options = [
    # 'pr95',
    # 'pr97',
    'pr99',
    # 'pr999',
    ]
percentile_option = percentile_options[0]


for model in models:
    print(model, 'started')
    start = timeit.default_timer()

    folder = home + '/Documents/data/cmip5/ds'
    fileName = model + '_precip_' + experiment + '.nc'
    path = folder + '/' + fileName
    ds = xr.open_dataset(path)
    precip = ds.precip
    precip.attrs['units']= 'mm/day'


    folder = home + '/Documents/data/cmip5/' + model
    fileName = model + '_prPercentiles_' + experiment + '.nc'
    path = folder + '/' + fileName
    pr_percentiles = xr.open_dataset(path)


    conv_threshold = pr_percentiles.pr97.mean(dim=('time'))


    folder = home + '/Documents/data/cmip5/' + model
    fileName = model + '_rome_' + experiment + '.nc'
    path = folder + '/' + fileName
    rome = xr.open_dataset(path)


    fig= plt.figure(figsize=(20,7.5))

    lat = precip.lat
    lon = precip.lon
    lonm,latm = np.meshgrid(lon,lat)

    rome_prctile = np.percentile(rome[rome_option],99.5)
    x2= np.argwhere(rome[rome_option].data>=rome_prctile)

    def animate(frame):    
        ax = fig.add_subplot(projection=ccrs.PlateCarree(central_longitude=180))

        ax.add_feature(cfeat.COASTLINE)
        ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())

        pr_day = precip.isel(time=x2[frame][0])
        extreme_percentileDay = pr_percentiles[percentile_option].isel(time=x2[frame][0]).data

        pcm= ax.pcolormesh(lonm,latm, pr_day.where(pr_day>conv_threshold),transform=ccrs.PlateCarree(),zorder=0, cmap='Blues', vmin=15, vmax=80)
        ax.pcolormesh(lonm,latm, pr_day.where(pr_day>extreme_percentileDay),transform=ccrs.PlateCarree(), cmap='Reds')

        ax.set_title(model + ': location of precipitation extremes')
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')

        ax.set_yticks([-20, 0, 20])
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xticklabels([0, 90, 180, 270, 360])

        plt.colorbar(pcm, ax=ax, orientation='horizontal',pad=0.10, aspect=50, fraction=0.055, label = ' pr97 [mm/day]')
        plt.close()
    

    ani = animation.FuncAnimation(
        fig,             # figure
        animate,         # name of the function above
        frames=len(x2),  # Could also be iterable or list
        interval=500     # ms between frames
        )


    folder = home + '/Documents/log/analysis/animations'
    fileName = model + '_location_high_pr_percentile_' + experiment + '.mp4'
    path = folder + '/' + fileName
    ani.save(path)

    stop = timeit.default_timer()
    print('it takes {} minutes to crete annimation for model: {}'.format((stop-start)/60, model))







for obs in observations:
    print(obs, 'started')
    start = timeit.default_timer()

    folder = home + '/Documents/data/obs/ds'
    fileName = obs + '_precip.nc'
    path = folder + '/' + fileName
    precip = xr.open_dataset(path)['precip']
    precip.attrs['units']= 'mm/day'


    folder = home + '/Documents/data/obs/' + obs
    fileName = obs + '_prPercentiles.nc'
    path = folder + '/' + fileName
    pr_percentiles = xr.open_dataset(path)
    conv_threshold = pr_percentiles.pr97.mean(dim=('time'))


    folder = home + '/Documents/data/obs/' + obs
    fileName = obs + '_rome.nc'
    path = folder + '/' + fileName
    rome = xr.open_dataset(path)


    fig= plt.figure(figsize=(20,7.5))

    lat = precip.lat
    lon = precip.lon
    lonm,latm = np.meshgrid(lon,lat)

    rome_prctile = np.percentile(rome[rome_option].sel(time=slice('1998-01', '2021-12')),99)
    x2= np.argwhere(rome[rome_option].sel(time=slice('1998-01', '2021-12')).data>=rome_prctile)

    def animate(frame):    
        ax = fig.add_subplot(projection=ccrs.PlateCarree(central_longitude=180))

        ax.add_feature(cfeat.COASTLINE)
        ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())

        pr_day = precip.isel(time=x2[frame][0])
        extreme_percentileDay = pr_percentiles[percentile_option].isel(time=x2[frame][0]).data

        pcm= ax.pcolormesh(lonm,latm, pr_day.where(pr_day>conv_threshold),transform=ccrs.PlateCarree(),zorder=0, cmap='Blues') # vmin=15, vmax=80)
        ax.pcolormesh(lonm,latm, pr_day.where(pr_day>extreme_percentileDay),transform=ccrs.PlateCarree(), cmap='Reds')

        ax.set_title(obs + ': location of precipitation extremes')
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')

        ax.set_yticks([-20, 0, 20])
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xticklabels([0, 90, 180, 270, 360])

        plt.colorbar(pcm, ax=ax, orientation='horizontal',pad=0.10, aspect=50, fraction=0.055, label = ' pr97 [mm/day]')
        plt.close()
    

    ani = animation.FuncAnimation(
        fig,             # figure
        animate,         # name of the function above
        frames=len(x2),  # Could also be iterable or list
        interval=500     # ms between frames
        )


    folder = home + '/Documents/log/analysis/animations'
    fileName = obs + '_location_high_pr_percentile.mp4'
    path = folder + '/' + fileName
    ani.save(path)

    stop = timeit.default_timer()
    print('it takes {} minutes to crete annimation for model: {}'.format((stop-start)/60, obs))








