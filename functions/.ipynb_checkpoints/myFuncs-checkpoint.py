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



def plot_timeseries(y, timeMean_option='', title='', ax='', ymin=None, ymax=None, fig_width=20, fig_height=5):

    if not ax:
        f, ax = plt.subplots(figsize=(fig_width, fig_height))

    if timeMean_option == 'seasonal':
        ax.plot(y, label = y.season.values)
    else:
        ax.plot(y)
        ax.axhline(y=y.mean(dim='time'), color= 'k',  linestyle="--")
    
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


def plot_scatter(x,y,ax, color='k'):
    ax.scatter(x,y,facecolors='none', edgecolor=color)
    res= stats.pearsonr(x,y)
    if res[1]<=0.05:
        ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', xytext=(0.8, 0.875), textcoords='axes fraction') # xy=(0.2, 0.1), xytext=(0.05, 0.875)


def plot_bins(x,y, ax, color='k'):    
    bin_width = (x.max() - x.min())/100
    bin_end = x.max()
    bins = np.arange(0, bin_end+bin_width, bin_width)

    y_bins = []
    for i in np.arange(0,len(bins)-1):
        y_bins = np.append(y_bins, y.where((x>=bins[i]) & (x<=bins[i+1])).mean())
    ax.plot(bins[:-1], y_bins, color)

    res= stats.pearsonr(x,y)
    if res[1]<=0.05:
        ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', xytext=(0.8, 0.875), textcoords='axes fraction')







# ------------------------------------------------------------------------------------- common operations functions --------------------------------------------------------------------------------------------------- #


def to_monthly(da):
    year = da.time.dt.year
    month = da.time.dt.month

    # assign new coords
    da = da.assign_coords(year=("time", year.data), month=("time", month.data))

    # reshape the array to (..., "month", "year")
    return da.set_index(time=("year", "month")).unstack("time")



def get_dsvariable(variable, dataset, experiment = 'historical', home = os.path.expanduser("~") + '/Documents', resolution='regridded'):

    folder = '{}/data/CMIP5/ds_cmip5_{}/{}'.format(home, resolution, dataset)
    filename = dataset + '_' + variable + '_' + experiment + '_' + resolution + '.nc'
    path_cmip5 = os.path.join(folder, filename)

    folder = '{}/data/CMIP6/ds_cmip6_{}/{}'.format(home, resolution, dataset)
    filename = dataset + '_' + variable + '_' + experiment + '_' + resolution + '.nc'
    path_cmip6 = os.path.join(folder, filename)

    folder = '{}/data/obs/ds_obs_{}/{}'.format(home, resolution, dataset)
    filename = dataset + '_' + variable + '_' + resolution + '.nc'
    path_obs = os.path.join(folder, filename)

    try:
        ds = xr.open_dataset(path_cmip5)
    except FileNotFoundError:
        try:
            ds = xr.open_dataset(path_cmip6)
        except FileNotFoundError:
            try:
                ds = xr.open_dataset(path_obs)
            except FileNotFoundError:
                print(f"Error: no file at {path_cmip5}, {path_cmip6}, or {path_obs}")
    return ds

                        

def get_metric(metric, dataset, experiment='historical', home=os.path.expanduser("~") + '/Documents', resolution='regridded'):

    folder = '{}/data/CMIP5/metrics_cmip5_{}/{}'.format(home, resolution, dataset)
    filename = dataset + '_' + metric + '_' + experiment + '_' + resolution + '.nc'
    path_cmip5 = os.path.join(folder, filename)

    folder = '{}/data/CMIP6/metrics_cmip6_{}/{}'.format(home, resolution, dataset)
    filename = dataset + '_' + metric + '_' + experiment + '_' + resolution + '.nc'
    path_cmip6 = os.path.join(folder, filename)

    folder = '{}/data/obs/ds_obs_{}/{}'.format(home, resolution, dataset)
    filename = dataset + '_' + metric + '_' + resolution + '.nc'
    path_obs = os.path.join(folder, filename)

    try:
        ds = xr.open_dataset(path_cmip5)
    except FileNotFoundError:
        try:
            ds = xr.open_dataset(path_cmip6)
        except FileNotFoundError:
            try:
                ds = xr.open_dataset(path_obs)
            except FileNotFoundError:
                print(f"Error: no file at {path_cmip5}, {path_cmip6}, or {path_obs}")
    return ds



def regrid_conserv(M_in):
    # dimensions of model to regrid to
    folder = '/g/data/al33/replicas/CMIP5/combined/LASG-CESS/FGOALS-g2/historical/day/atmos/day/r1i1p1/v20161204/pr'
    fileName = 'pr_day_FGOALS-g2_historical_r1i1p1_19970101-19971231.nc'
    path1 = folder + '/' + fileName

    folder = '/Users/cbla0002/Documents/data/CMIP5/ds_cmip5_orig/FGOALS-g2'
    fileName = 'FGOALS-g2_precip_historical_orig.nc'
    path2 = folder + '/' + fileName
    
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
    if 'lev' in M_in.dims or 'plev' in M_in.dims:
        if 'lev' in M_in.dims:
            M_in = M_in.rename({'lev': 'plev'})

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



def save_file(dataset, folder, filename):

    os.makedirs(folder, exist_ok=True)
    path = folder + '/' + filename

    if os.path.exists(path):
        os.remove(path)    
    dataset.to_netcdf(path)




def resample_timeMean(y, timeMean_option=''):

    if timeMean_option == 'annual':
        if len(y)<100:
            pass
        else:
            y = y.resample(time='Y').mean(dim='time', keep_attrs=True)

    if timeMean_option == 'seasonal':
        if len(y)<100:
            pass
        else:
            y = y.resample(time='QS-DEC').mean(dim="time")
            y = to_monthly(y)
            y = y.rename({'month':'season'})
            y = y.assign_coords(season = ["MAM", "JJA", "SON", "DJF"])
            y = y.isel(year=slice(1, None))

    if timeMean_option == 'monthly':
        if len(y)<1000:
            pass
        else:
            y = y.resample(time='M').mean(dim='time', keep_attrs=True)

    if timeMean_option == 'daily' or not timeMean_option:
        pass
    return y







def onePlus(a):
    return a+1







# --------------------------------------------------------------------------   commonly used variables   ---------------------------------------------------------------------------------- #

institutes_cmip5 = {
    'IPSL-CM5A-MR':'IPSL',
    'GFDL-CM3':'NOAA-GFDL',
    'GISS-E2-H':'NASA-GISS',
    'bcc-csm1-1':'BCC',
    'CNRM-CM5':'CNRM-CERFACS',
    'CCSM4':'NCAR',
    'HadGEM2-AO':'NIMR-KMA',
    'BNU-ESM':'BNU',
    'EC-EARTH':'ICHEC',
    'FGOALS-g2':'LASG-CESS',
    'MPI-ESM-MR':'MPI-M',
    'CMCC-CM':'CMCC',
    'inmcm4':'INM',
    'NorESM1-M':'NCC',
    'CanESM2':'CCCma',
    'MIROC5':'MIROC',
    'HadGEM2-CC':'MOHC',
    'MRI-CGCM3':'MRI',
    'CESM1-BGC':'NSF-DOE-NCAR'
    }

institutes_cmip6 = {
    'TaiESM1':'AS-RCEC',
    'BCC-CSM2-MR':'BCC',
    'FGOALS-g3':'CAS',
    'CNRM-CM6-1':'CNRM-CERFACS',
    'MIROC6':'MIROC',
    'MPI-ESM1-2-HR':'MPI-M',
    'GISS-E2-1-H':'NASA-GISS',
    'NorESM2-MM':'NCC',
    'GFDL-CM4':'NOAA-GFDL',
    'CanESM5':'CCCma',
    'CMCC-ESM2':'CMCC',
    'UKESM1-0-LL':'MOHC',
    'MRI-ESM2-0':'MRI',
    'CESM2':'NCAR',
    'NESM3':'NUIST'
    }


institutes = {**institutes_cmip5, **institutes_cmip6}





























