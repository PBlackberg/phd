import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt



experiments=[
    'historical',
    # 'rcp85'
]
experiment = experiments[0]

path_gen = '/g/data/al33/replicas/CMIP5/combined/NOAA-GFDL/GFDL-CM3/' + experiment + '/mon/atmos/Amon'
ensemble = 'r1i1p1'
version = os.listdir(os.path.join(path_gen, ensemble))[-1]
path_folder =  os.path.join(path_gen, ensemble, version,'cl')

if experiment == 'historical':
    yearEnd_first = 1970
    yearStart_last = 1999

if experiment == 'rcp85':
    yearEnd_first = 2070
    yearStart_last = 2099


files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
files = sorted(files, key=lambda x: x[x.index(".nc")-13:x.index(".nc")-9])
files = [f for f in files if int(f[f.index(".nc")-13:f.index(".nc")-9]) <= yearStart_last and int(f[f.index(".nc")-6:f.index(".nc")-2]) >= yearEnd_first]

path_fileList = []
for file in files:
    path_fileList = np.append(path_fileList, os.path.join(path_folder, file))

ds = xr.open_mfdataset(path_fileList, combine='by_coords')


pressureLevels = ds.a*ds.p0 + ds.b*ds.ps


pressureLevels_low = xr.where((pressureLevels<=1000e2) & (pressureLevels>=600), 1, 0)
pressureLevels_high = xr.where((pressureLevels<=250e2) & (pressureLevels>=100), 1, 0)




cl = ds.cl



cloud_low = cl*pressureLevels_low
cloud_low = cloud_low.max(dim='lev')

cloud_high = cl*pressureLevels_high
cloud_high = cloud_high.max(dim='lev')





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


M_out = xr.open_dataset('/g/data/k10/cb4968/data/cmip5/FGOALS-g2/FGOALS-g2_ds_regid_historical.nc')['pr']




cloud_low_n = regrid_conserv_np(cloud_low, M_out)
cloud_low_n = xr.DataArray(
    data = cloud_low_n,
    dims = ['time', 'lat', 'lon'],
    coords = {'time': cloud_low.time.data, 'lat': M_out.lat.data, 'lon': M_out.lon.data},
    attrs = {'description': 'Maximum cloud fraction (%) from plev: 1000-600 hpa'}
    )
cloud_low_n


cloud_high_n = regrid_conserv_np(cloud_high, M_out)
cloud_high_n = xr.DataArray(
    data = cloud_high_n,
    dims = ['time', 'lat', 'lon'],
    coords = {'time': cloud_high.time.data, 'lat': M_out.lat.data, 'lon': M_out.lon.data},
    attrs = {'description': 'Maximum cloud fraction (%) from plev: 250-100 hpa'}
    )


ds_clouds = xr.Dataset(
    data_vars = {
        'cloud_low': cloud_low_n, 
        'cloud_high':cloud_high_n},
    attrs = {'description': 'Metric defined as maximum cloud fraction (%) from specified pressure level intervals'}
    )





def save_file(dataset, folder, fileName):
    
    os.makedirs(folder, exist_ok=True)
    path = folder + '/' + fileName

    if os.path.exists(path):
        os.remove(path)    
    
    dataset.to_netcdf(path)




save = True
if save:
    folder = '/g/data/k10/cb4968/data/cmip5/ds'
    fileName = 'GFDL-CM3_cl_' + experiment + '.nc'

    save_file(ds_clouds, folder, fileName)
