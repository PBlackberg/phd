import numpy as np
import xarray as xr
import scipy
import timeit
import os
home = os.path.expanduser("~")

from metrics.get_variables.myFuncs import *


# ---------------------------------------------------------------------- function for regridding data ----------------------------------------------------------------------- #

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



def save_file(dataset, folder, fileName):
    os.makedirs(folder, exist_ok=True)
    path = folder + '/' + fileName

    if os.path.exists(path):
        os.remove(path)    
    
    dataset.to_netcdf(path)





# --------------------------------------------------------------------------------- apply regridding ------------------------------------------------------------------------------- #



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    models_cmip5 = [
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

    models_cmip6 =[
        # 'TaiESM1',        # 1 rcp monthly
        # 'BCC-CSM2-MR',    # 2 rcp monthly   
        'FGOALS-g3',      # 3 rcp 0463 - 0614
        'CNRM-CM6-1',     # 4 rcp 1850-1999
        'MIROC6',         # 5 rcp 3200 - 3340
        # 'MPI-ESM1-2-HR',  # 6 rcp 1850 - 2014
        'NorESM2-MM',     # 7 rcp 0001 - 0141
        'GFDL-CM4',       # 8 rcp 0001 - 0141 (gr2)
        'CanESM5',        # 9 rcp 1850 - 2000
        # 'CMCC-ESM2',      # 10 rcp monthly
        'UKESM1-0-LL',    # 11 rcp 1850 - 1999
        'MRI-ESM2-0',     # 12 rcp 1850 - 2000
        'CESM2',          # 13 rcp 0001 - 0990  (multiple fill values (check if all get converted to NaN), for historical)
        'NESM3',          # 14 rcp 1850-2014
        ]


    observations = [
        # 'GPCP'
        ]

    datasets = models_cmip5 + models_cmip6 + observations


    resolutions = [
        'original'
        ]
        
    experiments = [
                'historical',
                # 'rcp85',
                # 'abrupt-4xCO2'
                ]

    institutes = {
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


    for dataset in datasets:
        print(dataset, 'started') 
        start = timeit.default_timer()
        for experiment in experiments:
            print(experiment) 


            # precip
            if dataset == 'GPCP':
                # ds = cf.matrix3d
                # ds = get_pr(institutes[dataset], dataset, experiment)  
                ds = get_dsvariable('precip', dataset, experiment, resolution=resolutions[0])
            else:
                # ds = cf.matrix3d
                # ds = get_pr(institutes[dataset], dataset, experiment)  
                ds = get_dsvariable('precip', dataset, experiment, resolution=resolutions[0])
            
            data = ds['precip']




            # regridding
            print('loaded data')
            data_regrid = regrid_conserv(data)
            print('finished regridding')



            # save if necessary
            save_pr = True



            # select folder
            if np.isin(models_cmip5, dataset).any():
                folder_save = home + '/Documents/data/cmip5/ds_cmip5/' + dataset

            if np.isin(models_cmip6, dataset).any():
                folder_save = home + '/Documents/data/cmip6/ds_cmip6/' + dataset

            if np.isin(observations, dataset).any():
                folder_save = home + '/Documents/data/obs/ds_obs/' + dataset



            # save
            if save_pr:
                fileName = dataset + '_precip_' + experiment + '.nc'
                ds_pr = xr.Dataset(
                    data_vars = {'precip': data_regrid},
                    attrs = ds.attrs
                    )
                save_file(ds_pr, folder_save, fileName)


        stop = timeit.default_timer()
        print('script took: {} minutes to finsih'.format((stop-start)/60))
























































