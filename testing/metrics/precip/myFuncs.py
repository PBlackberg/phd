import xarray as xr
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
import cartopy

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import os
home = os.path.expanduser("~")



# -------------------------------------------------------------------------------------- basic plot functions --------------------------------------------------------------------------------------------------------------- #

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
        if len(y)<1000:
            pass
        else:
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



def plot_scatter(x,y,ax):
    ax.scatter(x,y,facecolors='none', edgecolor='k')
    res= stats.pearsonr(x,y)
    if res[1]<=0.05:
        ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', xytext=(0.8, 0.875), textcoords='axes fraction') # xy=(0.2, 0.1), xytext=(0.05, 0.875)














# ------------------------------------------------------------------------------------- common operations functions --------------------------------------------------------------------------------------------------- #


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



def regrid_conserv(M_in):
    # dimensions of model to regrid to
    folder1 = '/g/data/al33/replicas/CMIP5/combined/LASG-CESS/FGOALS-g2/historical/day/atmos/day/r1i1p1/v20161204/pr'
    fileName1 = 'pr_day_FGOALS-g2_historical_r1i1p1_19970101-19971231.nc'
    path1 = folder1 + '/' + fileName1

    folder2 = '/Users/cbla0002/Documents/data/CMIP5/ds_cmip5/FGOALS-g2'
    fileName2 = 'FGOALS-g2_precip_historical.nc'
    path2 = folder2 + '/' + fileName2
    

    try:
        M_out = xr.open_dataset(path1)['pr'].sel(lat=slice(-30,30))
    except FileNotFoundError:
        try:
            M_out = xr.open_dataset(path2)['precip'].sel(lat=slice(-30,30))
        except FileNotFoundError:
            print(f"Error: no file at {path1} or {path2}")


    # dimensions
    dlat = M_in.lat.data[1]-M_in.lat.data[0]
    dlon = M_in.lon.data[1]-M_in.lon.data[0]
    latBnds = (M_in.lat.data-(dlat/2), M_in.lat.data+(dlat/2))
    lonBnds = (M_in.lon.data-(dlon/2), M_in.lon.data+(dlon/2))
    lat = np.mean(latBnds, axis=0)
    lon = np.mean(lonBnds, axis=0)
    # area of gridboxes as fraction of earth surface area
    area_wlat = np.cos(np.deg2rad(lat))*dlat*np.pi/(4*180^2)

    dlat_n = M_out.lat.data[1]-M_out.lat.data[0]
    dlon_n = M_out.lon.data[1]-M_out.lon.data[0]
    latBnds_n = (M_out.lat.data-(dlat_n/2), M_out.lat.data+(dlat_n/2))
    lonBnds_n = (M_out.lon.data-(dlon_n/2), M_out.lon.data+(dlon_n/2))
    lat_n = np.mean(latBnds_n, axis=0)
    lon_n = np.mean(lonBnds_n, axis=0)

    # weights
    Wlat = np.zeros([len(lat_n), len(lat)])
    for i in np.arange(0,len(lat_n)):
        latBoxMin_n = latBnds_n[0][i]
        latBoxMax_n = latBnds_n[1][i]

        # gridboxes that are atleast partially overlapping with iteration gridbox
        J = (latBnds[0]<=latBoxMax_n)*(latBnds[1]>= latBoxMin_n)*area_wlat

        # including fractional area component contribution
        I = J*(latBnds[1]-latBoxMin_n)/dlat
        K = J*(latBoxMax_n-latBnds[0])/dlat
        II = np.min([I,J,K], axis=0)

        # weights from individual gridboxes contributing to the new gridbox as fraction of the total combined area contribution
        Wlat[i,:] = II/np.sum(II)

    Wlat = xr.DataArray(
        data = Wlat,
        dims = ['lat_n', 'lat']
        )

    Wlon = np.zeros([len(lon_n), len(lon)])
    for i in np.arange(0,len(lon_n)):
        lonBoxMin_n = lonBnds_n[0][i]
        lonBoxMax_n = lonBnds_n[1][i]

        # gridboxes that are atleast partially overlapping with iteration gridbox
        J = (lonBnds[0]<=lonBoxMax_n)*(lonBnds[1]>= lonBoxMin_n)*1

        # Including fractional area component contribution
        I = J*(lonBnds[1]-lonBoxMin_n)/dlon
        K = J*(lonBoxMax_n-lonBnds[0])/dlon
        L = J*(lonBoxMax_n-lonBnds[0]+360)/dlon
        II = np.min([I,J,K,L], axis=0)

        # weights from individual gridboxes contributing to the new gridbox as fraction of the total combined area contribution
        Wlon[i,:] = II/np.sum(II)

    Wlon = xr.DataArray(
        data = Wlon,
        dims = ['lon_n', 'lon']
        )

    # interpolation
    if ('plev' or 'lev') in M_in.dims:
        if 'lev' in M_in.dims:
            M_n = M_n.rename({'lev': 'plev'})

        M_n = xr.DataArray(
            data = np.zeros([len(M_in.time.data), len(M_in.plev.data), len(lat_n), len(lon_n)]),
            dims = ['time', 'plev', 'lat_n', 'lon_n'],
            coords = {'time': M_in.time.data, 'plev': M_in.plev.data, 'lat_n': M_out.lat.data, 'lon_n': M_out.lon.data},
            attrs = M_in.attrs
            )

        for day in np.arange(0,len(M_in.time.data)):
            
            M_Wlat = xr.DataArray(
            data = np.zeros([len(M_in.plev), len(lat_n), len(lon)]),
            dims = ['plev', 'lat_n', 'lon']
            )

            for i in range(0, len(Wlat.lat_n)):
                M_Wlat[:,i,:] = (M_in.isel(time=day) * Wlat[i,:]).sum(dim='lat', skipna=True) / (M_in.isel(time=day).notnull()*1*Wlat[i,:]).sum(dim='lat')
                
            for i in range(0, len(Wlon.lon_n)):
                M_n[day,:,:,i] = (M_Wlat * Wlon[i,:]).sum(dim='lon', skipna=True) / (M_Wlat.notnull()*1*Wlon[i,:]).sum(dim='lon')


    else:
        M_n = xr.DataArray(
            data = np.zeros([len(M_in.time.data), len(lat_n), len(lon_n)]),
            dims = ['time', 'lat_n', 'lon_n'],
            coords = {'time': M_in.time.data, 'lat_n': M_out.lat.data, 'lon_n': M_out.lon.data},
            attrs = M_in.attrs
            )

        for day in np.arange(0,len(M_in.time.data)):

            M_Wlat = xr.DataArray(
            data = np.zeros([len(lat_n), len(lon)]),
            dims = ['lat_n', 'lon']
            )

            for i in range(0, len(Wlat.lat_n)):
                M_Wlat[i,:] = (M_in.isel(time=day) * Wlat[i,:]).sum(dim='lat', skipna=True) / (M_in.isel(time=day).notnull()*1*Wlat[i,:]).sum(dim='lat')
                
            for i in range(0, len(Wlon.lon_n)):
                M_n[day,:,i] = (M_Wlat * Wlon[i,:]).sum(dim='lon', skipna=True) / (M_Wlat.notnull()*1*Wlon[i,:]).sum(dim='lon')


    M_n = M_n.rename({'lat_n': 'lat', 'lon_n': 'lon'})
    
    return M_n







































# -----------------------------------------------------------------------   Format for multiple plots   ----------------------------------------------------------------------------------- #

# foramt for multiple plots
# def plot_format_multiple(datasets, variable, timeMean_options, experiments, data):
#     absolute_limits = True
#     quantile_low = 0
#     quantile_high = 1
#     if absolute_limits:
#         vmin, vmax = [], []
#         for dataset in datasets:



#             vmin = np.append(vmin, np.quantile(y, quantile_low))
#             vmax = np.append(vmax, np.quantile(y, quantile_high))

#         vmin = np.min(vmin)
#         vmax = np.max(vmax)

#     else:
#         vmin, vmax = None, None 


#     fig= plt.figure(figsize=(22.5,17.5))
#     title = '{} spatial mean of {} field from model:{}, experiment:{}'.format(timeMean_options[0], variable, dataset, experiments[0])

#     fig.suptitle(title, fontsize=18, y=0.95)

#     for i, dataset in enumerate(datasets):
#         ax= fig.add_subplot(5,4,i+1)
#         title = dataset


#         plot_timeseries(y, title=title, timeMean_option=timeMean_options, ax=ax, ymin=vmin, ymax=vmax)

#         if (len(datasets)-i)<=4:
#             xlabel = '{} [{} - {}]'.format(timeMean_options[0], str(data.isel(time=0).coords['time'].values)[:10], str(data.isel(time=-1).coords['time'].values)[:10])
#             plt.xlabel(xlabel)

#         if i== 0 or i==4 or i==8 or i==12 or i==16:
#             ylabel = 'Relative humidity [{}]'.format('%')
#             plt.ylabel(ylabel)

#     plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.15, hspace=0.3)




























