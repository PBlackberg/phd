import xarray as xr
import numpy as np
import os
import intake
import xesmf as xe
import scipy



def regrid_conserv_xesmf(ds_in, path_dsOut='/g/data/k10/cb4968/data/cmip5/FGOALS-g2/FGOALS-g2_ds_regid_historical.nc', model_dsOut='FGOALS-g2'):

    if path_dsOut:
        ds_out = xr.open_dataset(path_dsOut)
        regrid = xe.Regridder(ds_in, ds_out, 'conservative', periodic=True)
    
    else:
        ds_dict = intake.cat.nci['esgf'].cmip5.search(
                                        model_id = model_dsOut, 
                                        experiment = 'historical',
                                        time_frequency = 'day', 
                                        realm = 'atmos', 
                                        ensemble = 'r1i1p1', 
                                        variable= 'pr').to_dataset_dict()

        ds_out = ds_dict[list(ds_dict.keys())[-1]].sel(time='1970-01-01', lon=slice(0,360),lat=slice(-30,30))

        # ds_regrid= ds_dict[list(ds_dict.keys())[-1]].sel(time='1970-01-01', lon=slice(0,360),lat=slice(-30,30))
        # ds_regrid.to_netcdf(path_saveDsOut)
        # ds_out = xr.open_dataset(path_saveDsOut)

        regrid = xe.Regridder(ds_in, ds_out, 'conservative', periodic=True)
        
    return regrid(ds_in)



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



def get_pr(model, experiment):

    if experiment == 'historical':
        period=slice('1970-01','1999-12')
        ensemble = 'r1i1p1'

        if model == 'GISS-E2-H':
            ensemble = 'r6i1p1'

    if experiment == 'rcp85':
        period=slice('2070-01','2099-12')
        ensemble = 'r1i1p1'

        if model == 'GISS-E2-H':
            ensemble = 'r2i1p1'

    ds_dict = intake.cat.nci['esgf'].cmip5.search(
                                    model_id = model, 
                                    experiment = experiment,
                                    time_frequency = 'day', 
                                    realm = 'atmos', 
                                    ensemble = ensemble, 
                                    variable= 'pr').to_dataset_dict()

    if not (model == 'CanESM2' and experiment == 'historical'):
        ds_orig =ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-35,35))

    else:
        ds_orig =ds_dict[list(ds_dict.keys())[-1]].isel(time=slice(43800, 43800+10950)).sel(lon=slice(0,360),lat=slice(-35,35))

    precip = regrid_conserv_xesmf(ds_orig).pr
    ds_pr = xr.Dataset(
        data_vars = {'precip': precip}
        )
            
    return ds_pr



def get_tas(model, experiment):

    if experiment == 'historical':
        period=slice('1970-01','1999-12')
        ensemble = 'r1i1p1'

        if model == 'GISS-E2-H':
            ensemble = 'r6i1p1'

        if model == 'EC-EARTH':
            ensemble = 'r6i1p1'


    if experiment == 'rcp85':
        period=slice('2070-01','2099-12')
        ensemble = 'r1i1p1'

        if model == 'GISS-E2-H':
            ensemble = 'r2i1p1'

        if model == 'EC-EARTH':
            ensemble = 'r6i1p1'

    ds_dict = intake.cat.nci['esgf'].cmip5.search(
                                        model_id = model, 
                                        experiment = experiment,
                                        time_frequency = 'mon', 
                                        realm = 'atmos', 
                                        ensemble = ensemble, 
                                        variable= 'tas').to_dataset_dict()


    if not (model == 'FGOALS-g2' or model == 'CNRM-CM5'):
        ds_orig =ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-35,35))


    elif model == 'FGOALS-g2' and experiment == 'historical':
        ds_orig =ds_dict[list(ds_dict.keys())[-1]].isel(time=slice(12*120, 12*120 + 12*30)).sel(lon=slice(0,360), lat=slice(-35,35))

    elif model == 'FGOALS-g2' and experiment == 'rcp85':
        ds_orig =ds_dict[list(ds_dict.keys())[-1]].isel(time=slice(12*64, 12*64 + 12*30)).sel(lon=slice(0,360), lat=slice(-35,35))


    elif model == 'CNRM-CM5' and experiment == 'historical':
        ds_orig =ds_dict[list(ds_dict.keys())[-1]].isel(time=slice(12*120, 12*120 + 12*30)).sel(lon=slice(0,360), lat=slice(-35,35))
        
    elif model == 'CNRM-CM5' and experiment == 'rcp85':
        ds_orig =ds_dict[list(ds_dict.keys())[-1]].isel(time=slice(12*64, 12*64 + 12*30)).sel(lon=slice(0,360), lat=slice(-35,35))


    tas = regrid_conserv_xesmf(ds_orig).tas

    ds_tas = xr.Dataset(
        data_vars = {'tas': tas}
        )
            
    return ds_tas




def get_pw(model, experiment):

    if experiment == 'historical':
        period=slice('1970-01','1999-12')
        ensemble = 'r1i1p1'

        if model == 'GISS-E2-H':
            ensemble = 'r6i1p1'

        if model == 'CCSM4':
            ensemble = 'r5i1p1'

    if experiment == 'rcp85':
        period=slice('2070-01','2099-12')
        ensemble = 'r1i1p1'

        if model == 'GISS-E2-H':
            ensemble = 'r2i1p1'

        if model == 'CCSM4':
            ensemble = 'r5i1p1'

    ds_dict = intake.cat.nci['esgf'].cmip5.search(
                                            model_id = model, 
                                            experiment = experiment,
                                            time_frequency = 'day', 
                                            realm = 'atmos', 
                                            ensemble = ensemble, 
                                            variable= 'hus').to_dataset_dict()

    if not model == 'CanESM2':
        ds_orig =ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-35,35))

    elif (model == 'CanESM2' and experiment == 'historical'):
        ds_orig =ds_dict[list(ds_dict.keys())[-1]].isel(time=slice(43800, 43800+10950)).sel(lon=slice(0,360),lat=slice(-35,35))

    elif (model == 'CanESM2' and experiment == 'rcp85'):
        ds_orig = ds_dict[list(ds_dict.keys())[-1]].isel(time=slice(365*64,365*94)).sel(lon=slice(0,360),lat=slice(-35,35))

    ds_hus = regrid_conserv_xesmf(ds_orig)



    hus = ds_hus.hus.fillna(0)
    pw = xr.DataArray(
        data=-scipy.integrate.simpson(hus, ds_hus.plev.data, axis=1, even='last'),
        dims=['time','lat', 'lon'],
        coords={'time': ds_hus.time.data, 'lat': ds_hus.lat.data, 'lon': ds_hus.lon.data},
        attrs={'units':'mm/day',
               'Description': 'total column precipitable water'}
        )
    
    pw_lower = xr.DataArray(
        data=-scipy.integrate.simpson(hus.sel(plev = slice(1000e2, 500e2)), hus.plev.data, axis=1, even='last'),
        dims=['time','lat', 'lon'],
        coords={'time': hus.time.data, 'lat': hus.lat.data, 'lon': hus.lon.data},
        attrs={'units':'mm/day',
                'Description': '1000-500 hpa precipitable water'}
        )

    pw_upper = xr.DataArray(
        data=-scipy.integrate.simpson(hus.sel(plev = slice(500e2, 10e2)), hus.plev.data, axis=1, even='last'),
        dims=['time','lat', 'lon'],
        coords={'time': hus.time.data, 'lat': hus.lat.data, 'lon': hus.lon.data},
        attrs={'units':'mm/day',
                'Description': '500-0 hpa precipitable water'}
        )
    
    ds_pw = xr.DataSet(
        data_vars = {'pw':pw, 'pw_lower':pw_lower, 'pw_upper':pw_upper},
        attrs = {'description': 'Precipitable water calculated as the vertically integrated specific humidity (simpson\'s method)'}
        )

    return ds_pw



def get_wap500(institute, model, experiment):

    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute +'/'+ model +'/'+ experiment +'/day/atmos/day/'
    ensemble = 'r1i1p1'
    version = os.listdir(os.path.join(path_gen, ensemble))[-1]
    path_folder =  os.path.join(path_gen, ensemble, version,'/wap')

    if experiment == 'historical':
        yearEnd_first = 1970
        yearStart_last = 1999

    if experiment == 'rcp85':
        yearEnd_first = 2070
        yearStart_last = 2099


    files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
    files = sorted(files, key=lambda x: x[x.index(".nc")-17:x.index(".nc")-13])
    files = [f for f in files if int(f[f.index(".nc")-17:f.index(".nc")-13]) <= yearStart_last and int(f[f.index(".nc")-8:f.index(".nc")-4]) >= yearEnd_first]

    path_fileList = []
    for file in files:
        path_fileList = np.append(path_fileList, os.path.join(path_folder, file))

    ds = xr.open_mfdataset(path_fileList, combine='by_coords')
    wap500 = ds.wap.sel(plev=500e2, time=slice(yearEnd_first, yearStart_last),lat=slice(-35,35))

    M_out = xr.open_dataset('/g/data/k10/cb4968/data/cmip5/FGOALS-g2/FGOALS-g2_ds_regid_historical.nc')['pr']
    wap500_n = regrid_conserv_np(wap500, M_out)

    wap500_n = xr.DataArray(
        data = wap500_n,
        dims = ['time', 'lat', 'lon'],
        coords = {'time': wap500.time.data, 'lat': M_out.lat.data, 'lon': M_out.lon.data},
        attrs = wap500.attrs
        )

    ds_wap500 = xr.Dataset(
        data_vars = {'wap500': wap500_n}
        )

    return ds_wap500




def get_clouds(institute, model, experiment):
    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute + '/' + model + '/' + experiment + '/mon/atmos/Amon'
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
    clouds = ds.cl

    cloud_low = clouds*pressureLevels_low
    cloud_low = cloud_low.max(dim='lev').sel(time=slice(yearEnd_first, yearStart_last),lat=slice(-35,35))

    cloud_high = clouds*pressureLevels_high
    cloud_high = cloud_high.max(dim='lev').sel(time=slice(yearEnd_first, yearStart_last),lat=slice(-35,35))

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
            'cloud_high': cloud_high_n},
        attrs = {'description': 'Metric defined as maximum cloud fraction (%) from specified pressure level intervals'}
        )
    

    return ds_clouds




def save_file(dataset, folder, fileName):
    os.makedirs(folder, exist_ok=True)
    path = folder + '/' + fileName

    if os.path.exists(path):
        os.remove(path)    
    
    dataSet.to_netcdf(path)




if __name__ == '__main__':

    import matplotlib.pyplot as plt

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
    

    models = [
            # 'IPSL-CM5A-MR', # 1
             'GFDL-CM3',     # 2
            # 'GISS-E2-H',    # 3
            # 'bcc-csm1-1',   # 4
            # 'CNRM-CM5',     # 5
            # 'CCSM4',        # 6 # cannot concatanate files for rcp85 run
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


    for model in models:
        for experiment in experiments:

            ds_pr = get_pr(model, experiment)
            # plot_snapshot(ds_pr.pr.isel(time=0), 'Blues', 'pr_day', model)
            # plt.show()

            ds_pw = get_pw(model, experiment)
            ds_tas = get_tas(model, experiment)
            ds_wap500 = get_wap500(institutes[model], model, experiment)
            ds_clouds = get_clouds(institutes[model], model, experiment)




            folder = '/g/data/k10/cb4968/data/cmip5/ds/'
            save_precip = False
            save_pw = False
            save_tas = False
            save_wap500 = False
            save_cl = False

            if save_precip:
                fileName = model + '_precip_' + experiment + '.nc'
                dataset = ds_pr
                save_file(dataset, folder, fileName)

            if save_pw:
                fileName = model + '_pw_' + experiment + '.nc'
                dataset = ds_pw
                save_file(dataset, folder, fileName)

            if save_tas:
                fileName = model + '_tas_' + experiment + '.nc'
                dataset = ds_tas
                save_file(dataset, folder, fileName)

            if save_wap500:
                fileName = model + '_wap500_' + experiment + '.nc'
                dataset = ds_tas
                save_file(dataset, folder, fileName)

            if save_cl:
                fileName = model + '_cl_' + experiment + '.nc'
                dataset = ds_tas
                save_file(dataset, folder, fileName)



















