import xarray as xr
import os
import numpy as np

import matplotlib.pyplot as plt
import cartopy

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 





# -------------------------------------------------------------------------------------- plotfuncs -----------------------------------------------------------------------------------------

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









# ----------------------------------------------------------------------------------- common operations -----------------------------------------------------------------------------------


def save_file(dataset, folder, fileName):
    
    os.makedirs(folder, exist_ok=True)
    path = folder + '/' + fileName

    if os.path.exists(path):
        os.remove(path)    
    
    dataset.to_netcdf(path)



def to_monthly(da):
    year = da.time.dt.year
    month = da.time.dt.month

    # assign new coords
    da = da.assign_coords(year=("time", year.data), month=("time", month.data))

    # reshape the array to (..., "month", "year")
    return da.set_index(time=("year", "month")).unstack("time")







# ------------------------------------------------------------------------------------- bigger functions -------------------------------------------------------------------------------------

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







