import xarray as xr
import xesmf as xe
import numpy as np
import os
import scipy



# ---------------------------- functions to process the data ---------------------------------


def concat_files(path_folder, experiment):
    if experiment == 'historical':
        yearEnd_first = 1970
        yearStart_last = 1999

    if experiment == 'rcp85':
        yearEnd_first = 2070
        yearStart_last = 2099

    files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
    if 'Amon' in path_folder:
        files = sorted(files, key=lambda x: x[x.index(".nc")-13:x.index(".nc")-9])
        files = [f for f in files if int(f[f.index(".nc")-13:f.index(".nc")-9]) <= yearStart_last and int(f[f.index(".nc")-6:f.index(".nc")-2]) >= yearEnd_first]
    else:
        files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
        files = sorted(files, key=lambda x: x[x.index(".nc")-17:x.index(".nc")-13])
        files = [f for f in files if int(f[f.index(".nc")-17:f.index(".nc")-13]) <= yearStart_last and int(f[f.index(".nc")-8:f.index(".nc")-4]) >= yearEnd_first]


    path_fileList = []
    for file in files:
        path_fileList = np.append(path_fileList, os.path.join(path_folder, file))

    ds = xr.open_mfdataset(path_fileList, combine='by_coords').sel(time=slice(str(yearEnd_first), str(yearStart_last)),lat=slice(-35,35))

    return ds



def regrid_conserv_xesmf(ds_in):
    folder = '/g/data/al33/replicas/CMIP5/combined/LASG-CESS/FGOALS-g2/historical/day/atmos/day/r1i1p1/v20161204/pr'
    fileName = 'pr_day_FGOALS-g2_historical_r1i1p1_19970101-19971231.nc'
    ds_out = xr.open_dataset(folder + '/' + fileName).sel(lat=slice(-30,30))
    regridder = xe.Regridder(ds_in.isel(time=0), ds_out, 'conservative', periodic=True)

    return regridder



def regrid_conserv(M_in):
    # dimensions of model to regrid to
    folder = '/g/data/al33/replicas/CMIP5/combined/LASG-CESS/FGOALS-g2/historical/day/atmos/day/r1i1p1/v20161204/pr'
    fileName = 'pr_day_FGOALS-g2_historical_r1i1p1_19970101-19971231.nc'
    M_out = xr.open_dataset(folder + '/' + fileName)['pr'].sel(lat=slice(-30,30))

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




# --------------------------------- functions to get the data -------------------------------------------


def get_pr(institute, model, experiment):

    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute +'/'+ model +'/'+ experiment +'/day/atmos/day'
    ensemble = 'r1i1p1'

    if experiment == 'historical' and model == 'GISS-E2-H':
        ensemble = 'r6i1p1'

    if experiment == 'rcp85' and model == 'GISS-E2-H':
        ensemble = 'r2i1p1'

    version = os.listdir(os.path.join(path_gen, ensemble))[-1]
    variable = 'pr'
    path_folder =  os.path.join(path_gen, ensemble, version, variable)
    
    ds = concat_files(path_folder, experiment)
    regridder = regrid_conserv_xesmf(ds)
    
    precip = ds['pr']*60*60*24
    precip_n = regridder(precip)
    precip_n.attrs['units']= 'mm day' + chr(0x207B) + chr(0x00B9)

    ds_pr = xr.Dataset(
        data_vars = {'precip': precip_n},
        attrs = ds.attrs
        )

    return ds_pr


def get_tas(institute, model, experiment):

    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute +'/'+ model +'/'+ experiment +'/day/atmos/day'
    ensemble = 'r1i1p1'

    if experiment == 'historical' and model == 'GISS-E2-H':
        ensemble = 'r6i1p1'
    
    if experiment == 'rcp85' and model == 'GISS-E2-H':
        ensemble = 'r2i1p1'

    if model == 'EC-EARTH':
        ensemble = 'r6i1p1'

    version = os.listdir(os.path.join(path_gen, ensemble))[-1]
    variable = 'tas'
    path_folder =  os.path.join(path_gen, ensemble, version, variable)
    ds = concat_files(path_folder, experiment)
    regridder = regrid_conserv_xesmf(ds)

    tas = ds['tas']-273.15
    tas_n = regridder(tas)
    tas_n.attrs['units']= '\u00B0C'


    ds_tas = xr.Dataset(
        data_vars = {'tas': tas_n},
        attrs = ds.attrs
        )

    return ds_tas


def get_pw(institute, model, experiment):

    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute +'/'+ model +'/'+ experiment +'/day/atmos/day'
    ensemble = 'r1i1p1'

    if experiment == 'historical' and model == 'GISS-E2-H':
        ensemble = 'r6i1p1'

    if experiment == 'rcp85' and model == 'GISS-E2-H':
        ensemble = 'r2i1p1'

    if model == 'CCSM4':
        ensemble = 'r5i1p1'

    version = os.listdir(os.path.join(path_gen, ensemble))[-1]
    variable = 'hus'
    path_folder =  os.path.join(path_gen, ensemble, version, variable)
    ds = concat_files(path_folder, experiment)
    regridder = regrid_conserv_xesmf(ds)

    hus = ds['hus'].sel(plev=slice(850e2,0)) # free troposphere
    hus_n = regridder(hus).fillna(0)

    g = 9.8
    pw_n = xr.DataArray(
        data= -scipy.integrate.simpson(hus_n.data, hus_n.plev.data, axis=1, even='last')/g,
        dims=['time','lat', 'lon'],
        coords={'time': hus_n.time.data, 'lat': hus_n.lat.data, 'lon': hus_n.lon.data},
        attrs={'units':'mm',
               'Description': 'precipitable water from 850-0 hpa'}
        )

    ds_pw = xr.Dataset(
        data_vars = {'pw': pw_n},
        attrs = {'description': 'Precipitable water calculated as the vertically integrated specific humidity (simpson\'s method)'}
        )

    return ds_pw



def get_hur(institute, model, experiment):
    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute + '/' + model + '/' + experiment + '/mon/atmos/Amon'
    ensemble = 'r1i1p1'

    # if experiment == 'historical' and model == 'GISS-E2-H':
    #     ensemble = 'r6i1p1'

    # if experiment == 'rcp85' and model == 'GISS-E2-H':
    #     ensemble = 'r2i1p1'

    # if model == 'CCSM4':
    #     ensemble = 'r5i1p1'

    version = os.listdir(os.path.join(path_gen, ensemble))[-1]
    variable = 'hur'
    path_folder =  os.path.join(path_gen, ensemble, version, variable)
    ds = concat_files(path_folder, experiment)
    regridder = regrid_conserv_xesmf(ds)

    hur = ds['hur'].sel(plev=slice(850e2,0))*100 # free troposphere
    hur_n = regridder(hur).fillna(0) 
    hur_n = (ds.hur_n * ds.plev).sum(dim='plev') / ds.plev.sum(dim='plev')
    hur_n.attrs['units']= '%'
    hur_n.attrs['Description'] = 'weighted mean relative humidity from 850-0 hpa'

    ds_hur = xr.Dataset(
        data_vars = {'hur': hur_n},
        attrs = {'weighted mean relative humidity'}
        )

    return ds_hur



def get_wap500(institute, model, experiment):

    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute +'/'+ model +'/'+ experiment +'/day/atmos/day'
    ensemble = 'r1i1p1'
    version = os.listdir(os.path.join(path_gen, ensemble))[-1]
    variable = 'wap'
    path_folder =  os.path.join(path_gen, ensemble, version, variable)
    ds = concat_files(path_folder, experiment)
    regridder = regrid_conserv_xesmf(ds)

    wap500 = ds['wap'].sel(plev=500e2)*60*60*24    
    wap500_n = regridder(wap500)
    wap500_n.attrs['units']= 'Pa day' + chr(0x207B) + chr(0x00B9)

    ds_wap500 = xr.Dataset(
        data_vars = {'wap500': wap500_n}
        )

    return ds_wap500



def get_clouds(institute, model, experiment):
    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute + '/' + model + '/' + experiment + '/mon/atmos/Amon'
    ensemble = 'r1i1p1'
    version = os.listdir(os.path.join(path_gen, ensemble))[-1]
    variable = 'cl'
    path_folder =  os.path.join(path_gen, ensemble, version, variable)
    ds = concat_files(path_folder, experiment)
    regridder = regrid_conserv_xesmf(ds)
    
    clouds = ds['cl']*100
    clouds_n = regridder(clouds)

    pressureLevels = ds.a*ds.p0 + ds.b*ds.ps
    pressureLevels_n = regridder(pressureLevels)

    pressureLevels_low = xr.where((pressureLevels_n<=10000e2) & (pressureLevels_n>=600), 1, 0)
    cloud_low = clouds_n*pressureLevels_low
    cloud_low = cloud_low.max(dim='lev')
    cloud_low.attrs['units'] = '%'
    cloud_low.attrs['description'] = 'Maximum cloud fraction (%) from plev: 1000-600 hpa'

    pressureLevels_high = xr.where((pressureLevels_n<=250e2) & (pressureLevels_n>=100), 1, 0)
    cloud_high = clouds_n*pressureLevels_high
    cloud_high = cloud_high.max(dim='lev')
    cloud_high.attrs['units'] = '%'
    cloud_high.attrs['description'] = 'Maximum cloud fraction (%) from plev: 250-100 hpa'

    ds_clouds = xr.Dataset(
        data_vars = {
            'cloud_low': cloud_low, 
            'cloud_high': cloud_high},
        attrs = {'description': 'Metric defined as maximum cloud fraction (%) from specified pressure level intervals'}
        )
    
    return ds_clouds





if __name__ == '__main__':

    import matplotlib.pyplot as plt

    models = [
            # 'IPSL-CM5A-MR', # 1
            'GFDL-CM3',     # 2
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
                'rcp85'
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


    for model in models:
        for experiment in experiments:

            ds_pr = get_pr(institutes[model], model, experiment)
            ds_tas = get_tas(institutes[model], model, experiment)
            ds_pw = get_pw(institutes[model], model, experiment)
            ds_hur = get_hur(institutes[model], model, experiment)
            ds_wap500 = get_wap500(institutes[model], model, experiment)
            ds_clouds = get_clouds(institutes[model], model, experiment)


            save_pr = True
            save_tas = True
            save_pw = True
            save_hur = True
            save_wap500 = True
            save_cl = True
            
            folder = '/g/data/k10/cb4968/data/cmip5/ds/'
            
            if save_pr:
                fileName = model + '_precip_' + experiment + '.nc'
                save_file(ds_pr, folder, fileName)
                
            if save_tas:
                fileName = model + '_tas_' + experiment + '.nc'
                save_file(ds_tas, folder, fileName)

            if save_pw:
                fileName = model + '_pw_' + experiment + '.nc'
                save_file(ds_pw, folder, fileName)

            if save_hur:
                fileName = model + '_hur_' + experiment + '.nc'
                save_file(ds_hur, folder, fileName)
                
            if save_wap500:
                fileName = model + '_wap500_' + experiment + '.nc'
                save_file(ds_wap500, folder, fileName)

            if save_cl:
                fileName = model + '_clMax_' + experiment + '.nc'
                save_file(ds_clouds, folder, fileName)



















