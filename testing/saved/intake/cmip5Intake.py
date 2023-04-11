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




def save_file(dataset, folder, fileName):
    os.makedirs(folder, exist_ok=True)
    path = folder + '/' + fileName

    if os.path.exists(path):
        os.remove(path)    
    
    dataset.to_netcdf(path)

    

if __name__ == '__main__':

    import matplotlib.pyplot as plt


    models = [
            # 'IPSL-CM5A-MR', # 1
             'GFDL-CM3',      # 2
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












