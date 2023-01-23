import xarray as xr
import numpy as np
import os
import intake
import xesmf as xe
import scipy
from scipy.interpolate import griddata




def regrid_conserv_np(M_in, M_out):

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

    # interpolation
    M_n = np.zeros([len(M_in.time.data), len(lat_n), len(lon_n)])
    for day in np.arange(0,len(M_in.time.data)):
        M_Wlat = np.zeros([len(lat_n), len(lon)])

        for i in range(0, np.shape(Wlat)[0]):
            M_Wlat[i,:] = np.nansum(M_in.isel(time=day) * np.vstack(Wlat[i,:]),axis=0)/np.sum(~np.isnan(M_in.isel(time=day))*1*np.vstack(Wlat[i,:]),axis=0)
            
        for i in range(0, np.shape(Wlon)[0]):
            M_n[day,:,i] = np.nansum(M_Wlat * Wlon[i,:],axis=1)/np.sum(~np.isnan(M_Wlat)*1*Wlon[i,:], axis=1)

    return M_n



def get_gpcp():

    path_gen = '/g/data/ia39/aus-ref-clim-data-nci/gpcp/data/day/v1-3'
    years = range(1996,2023)
    folders = [f for f in os.listdir(path_gen) if (f.isdigit() and int(f) in years)]
    folders = sorted(folders, key=int)

    path_fileList = []
    for folder in folders:
        path_folder = os.path.join(path_gen, folder)
        files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
        files = sorted(files, key=lambda x: x[x.index("y_d")+1:x.index("_c")])

        for file in files:
            path_fileList = np.append(path_fileList, os.path.join(path_folder, file))

    ds = xr.open_mfdataset(path_fileList, combine='by_coords')
    precip = ds.precip.sel(latitude=slice(-35,35))
    precip = precip.rename({'latitude': 'lat', 'longitude': 'lon'})
    valid_range = [0, 250]
    precip = precip.where((precip >= valid_range[0]) & (precip <= valid_range[1]), np.nan)
    precip = precip.where(precip.sum(dim =('lat','lon')) != 0, np.nan)
    threshold = 0.5
    precip = precip.where(precip.isnull().sum(dim=('lat','lon'))/(precip.shape[1]*precip.shape[2]) < threshold, other=np.nan)
    precip = precip.dropna('time', how='all')
    nb_nan = precip.isnull().sum(dim=('lat', 'lon'))
    nan_days =np.nonzero(nb_nan.data)[0]
    for day in nan_days:
        time_slice = precip.isel(time=day)
        nan_indices = np.argwhere(np.isnan(time_slice.values))
        nonnan_indices = np.argwhere(~np.isnan(time_slice.values))
        interpolated_values = griddata(nonnan_indices, time_slice.values[~np.isnan(time_slice.values)], nan_indices, method='linear')
        time_slice.values[nan_indices[:, 0], nan_indices[:, 1]] = interpolated_values

    M_out = xr.open_dataset('/g/data/k10/cb4968/data/cmip5/FGOALS-g2/FGOALS-g2_ds_regid_historical.nc')['pr']
    precip_n = regrid_conserv_np(precip, M_out)

    precip_n = xr.DataArray(
        data = precip_n,
        dims = ['time', 'lat', 'lon'],
        coords = {'time': precip.time.data, 'lat': M_out.lat.data, 'lon': M_out.lon.data},
        attrs = precip.attrs
        )

    ds_gpcp = xr.Dataset(
        data_vars = {'precip': precip_n}
        )

    return ds_gpcp







if __name__ == '__main__':

    import matplotlib.pyplot as plt

    ds_gpcp = get_gpcp()


    

    folder = '/g/data/k10/cb4968/data/obs/ds'
    save_gpcp = False


    def save_file(dataSet, folder, fileName):
        os.makedirs(folder, exist_ok=True)
        path = folder + '/' + fileName

        if os.path.exists(path):
            os.remove(path)    
        
        dataSet.to_netcdf(path)

    if save_gpcp:
        fileName = 'GPCP_precip.nc'
        dataSet = ds_gpcp
        save_file(dataSet, folder, fileName)












