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




for model in models:
    print(model, 'started')
    start = timeit.default_timer()

    folder = home + '/Documents/data/cmip5/ds/' + model
    fileName = model + '_precip_' + experiment + '.nc'
    path = folder + '/' + fileName
    precip = xr.open_dataset(path)['precip']
    precip.attrs['units']= 'mm/day'


    folder = home + '/Documents/data/cmip5/' + model
    fileName = model + '_pr_percentiles_' + experiment + '.nc'
    path = folder + '/' + fileName
    pr_percentiles = xr.open_dataset(path)


    folder = home + '/Documents/data/cmip5/' + model
    fileName = model + '_rome_' + experiment + '.nc'
    path = folder + '/' + fileName
    rome = xr.open_dataset(path)


    fig= plt.figure(figsize=(20,7.5))

    lat = precip.lat
    lon = precip.lon
    lonm,latm = np.meshgrid(lon,lat)
    conv_threshold = pr_percentiles['pr97'].mean(dim=('time'))

    rome_threshold = 0.5
    rome_prctile = np.percentile(rome[rome_option],rome_threshold)
    x_rome_low= np.squeeze(np.argwhere(rome[rome_option].data<=rome_prctile))

    lower_threshold = 49.75
    upper_threshold = 50.25
    rome_prctile_lower = np.percentile(rome[rome_option], lower_threshold)
    rome_prctile_upper = np.percentile(rome[rome_option], upper_threshold)
    x_rome_med = np.squeeze(np.argwhere((rome[rome_option].data >= rome_prctile_lower) & (rome[rome_option].data <= rome_prctile_upper)))

    rome_threshold = 99.5
    rome_prctile = np.percentile(rome[rome_option],rome_threshold)
    x_rome_high= np.squeeze(np.argwhere(rome[rome_option].data>=rome_prctile))

    x_rome = np.concatenate((x_rome_low, x_rome_med, x_rome_high))

    def animate(frame):    
        ax = fig.add_subplot(projection=cartopy.crs.PlateCarree(central_longitude=180))

        ax.add_feature(cartopy.feature.COASTLINE)
        ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=cartopy.crs.PlateCarree())

        pr_day = precip.isel(time=x_rome[frame])

        pcm= ax.pcolormesh(lonm,latm, pr_day.where(pr_day>conv_threshold),transform=cartopy.crs.PlateCarree(),zorder=0, cmap='Blues', vmin=0, vmax=80)

        ax.set_title(model + ': gradually increasing DOC')
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')

        ax.set_yticks([-20, 0, 20])
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xticklabels([0, 90, 180, 270, 360])

        plt.colorbar(pcm, ax=ax, orientation='horizontal',pad=0.10, aspect=50, fraction=0.055, label = '[mm/day]')
        plt.close()
    

    ani = animation.FuncAnimation(
        fig,                    # figure
        animate,                # name of the function above
        frames=len(x_rome),     # Could also be iterable or list
        interval=500            # ms between frames
        )


    folder = home + '/Documents/log/analysis/animations/conv_doc_transition'
    fileName = model + '_doc_transition' + experiment + '.mp4'
    path = folder + '/' + fileName
    ani.save(path)

    stop = timeit.default_timer()
    print('it takes {} minutes to crete annimation for model: {}'.format((stop-start)/60, model))





for obs in observations:
    print(obs, 'started')
    start = timeit.default_timer()

    folder = home + '/Documents/data/obs/ds'
    fileName = obs + '_precip.nc'
    if original_resolution:
        fileName = obs + '_precip_orig.nc'
    path = folder + '/' + fileName
    precip = xr.open_dataset(path)['precip']
    precip.attrs['units']= 'mm/day'


    folder = home + '/Documents/data/obs/' + obs
    fileName = obs + '_prPercentiles.nc'
    if original_resolution:
        folder = home + '/Documents/data/obs/GPCP_orig'
        fileName = obs + '_prPercentiles_orig.nc'
    path = folder + '/' + fileName
    pr_percentiles = xr.open_dataset(path)


    folder = home + '/Documents/data/obs/' + obs
    fileName = obs + '_rome.nc'
    if original_resolution:
        folder = home + '/Documents/data/obs/GPCP_orig'
        fileName = obs + '_rome_orig.nc'
    path = folder + '/' + fileName
    rome = xr.open_dataset(path)


    fig= plt.figure(figsize=(20,7.5))

    lat = precip.lat
    lon = precip.lon
    lonm,latm = np.meshgrid(lon,lat)
    conv_threshold = pr_percentiles['pr97'].mean(dim=('time'))


    rome_threshold = 0.5
    rome_prctile = np.percentile(rome[rome_option],rome_threshold)
    x_rome_low= np.squeeze(np.argwhere(rome[rome_option].data<=rome_prctile))

    lower_threshold = 49.75
    upper_threshold = 50.25
    rome_prctile_lower = np.percentile(rome[rome_option], lower_threshold)
    rome_prctile_upper = np.percentile(rome[rome_option], upper_threshold)
    x_rome_med = np.squeeze(np.argwhere((rome[rome_option].data >= rome_prctile_lower) & (rome[rome_option].data <= rome_prctile_upper)))

    rome_threshold = 99.5
    rome_prctile = np.percentile(rome[rome_option],rome_threshold)
    x_rome_high= np.squeeze(np.argwhere(rome[rome_option].data>=rome_prctile))

    x_rome = np.concatenate((x_rome_low, x_rome_med, x_rome_high))


    def animate(frame):    
        ax = fig.add_subplot(projection=cartopy.crs.PlateCarree(central_longitude=180))

        ax.add_feature(cartopy.feature.COASTLINE)
        ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=cartopy.crs.PlateCarree())

        pr_day = precip.isel(time=x_rome[frame])

        pcm= ax.pcolormesh(lonm,latm, pr_day.where(pr_day>conv_threshold),transform=cartopy.crs.PlateCarree(),zorder=0, cmap='Blues',vmin=0, vmax=80)

        ax.set_title(obs + ': gradually increasing DOC')
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')

        ax.set_yticks([-20, 0, 20])
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xticklabels([0, 90, 180, 270, 360])

        plt.colorbar(pcm, ax=ax, orientation='horizontal',pad=0.10, aspect=50, fraction=0.055, label = '[mm/day]')
        plt.close()
    

    ani = animation.FuncAnimation(
        fig,                    # figure
        animate,                # name of the function above
        frames=len(x_rome),     # Could also be iterable or list
        interval=500            # ms between frames
        )


    folder = home + '/Documents/log/analysis/animations/conv_doc_transition'
    fileName = obs + '_doc_transition.mp4'
    if original_resolution:
        fileName = obs + '_orig_location_high_pr_percentile.mp4'
    path = folder + '/' + fileName
    ani.save(path)

    stop = timeit.default_timer()
    print('it takes {} minutes to crete annimation for obs: {}'.format((stop-start)/60, obs))




































