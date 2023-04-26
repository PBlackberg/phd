import intake
import scipy
import xarray as xr
import xesmf as xe



def regrid_conserv(ds_in, haveDsOut, path='/g/data/k10/cb4968/data/cmip5/FGOALS-g2/FGOALS-g2_ds_regid_historical.nc', modelDsOut='FGOALS-g2'):

    if haveDsOut:
        ds_out = xr.open_dataset(path)
        regrid = xe.Regridder(ds_in, ds_out, 'conservative', periodic=True)
    
    else:
        ds_dict = intake.cat.nci['esgf'].cmip5.search(
                                        model_id = modelDsOut, 
                                        experiment = 'historical',
                                        time_frequency = 'day', 
                                        realm = 'atmos', 
                                        ensemble = 'r1i1p1', 
                                        variable= 'pr').to_dataset_dict()

        ds_regrid = ds_dict[list(ds_dict.keys())[-1]].sel(time='1970-01-01', lon=slice(0,360),lat=slice(-30,30))
        ds_regrid.to_netcdf(path)

        ds_out = xr.open_dataset(path)
        regrid = xe.Regridder(ds_in, ds_out, 'conservative', periodic=True)
        
    return regrid(ds_in)



def get_hus(model, experiment):

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


    haveDsOut = True
    ds_hus = regrid_conserv(ds_orig, haveDsOut) # path='', model ='')



    # da = ds_hus.hus.fillna(0)
    # hus_vInt = xr.DataArray(
    #     data=-scipy.integrate.simpson(da, ds_hus.plev.data, axis=1, even='last'),
    #     dims=['time','lat', 'lon'],
    #     coords={'time': ds_hus.time.data, 'lat': ds_hus.lat.data, 'lon': ds_hus.lon.data},
    #     attrs={'units':'mm/day',
    #            'Description': 'precipitable water'}
    #     )


    return ds_hus






if __name__ == '__main__':


    from functions.myFuncs import *
    from myPlots import *

    import numpy as np
    import matplotlib.pyplot as plt


    models = [
        # 'IPSL-CM5A-MR', # 1 # super slow for some reason
         'GFDL-CM3',     # 2
        # 'GISS-E2-H',    # 3
        # 'bcc-csm1-1',   # 4
        # 'CNRM-CM5',     # 5
        # 'CCSM4',        # 6 # cannot concatanate files for rcp
        # 'HadGEM2-AO',   # 7
        # 'BNU-ESM',      # 8
        # 'EC-EARTH',     # 9
        # 'FGOALS-g2',    # 10
        # 'MPI-ESM-MR',   # 11
        # 'CMCC-CM',      # 12
        # 'inmcm4',       # 13
        # 'NorESM1-M',    # 14
        # 'CanESM2',      # 15 # rcp scenario: slicing with .sel does not work, says it 'contains no datetime objects'
        # 'MIROC5',       # 16
        # 'HadGEM2-CC',   # 17
        # 'MRI-CGCM3',    # 18
        # 'CESM1-BGC'     # 19
        ]


    experiments = [
                'historical',
                'rcp85'
                ]


    for model in models:
        for experiment in experiments:
            

            ds_hus = get_hus(model, experiment)


            # plot_snapshot(hus_vInt.isel(time=0), 'Greens', 'precipitable water', model)
            # plt.show()
            # plot_snapshot(hus_vInt.mean(dim=('time'), keep_attrs=True), 'Greens', 'time mean precipitable water', model)
            # plt.show()



            saveit = False
            if saveit:
                folder = '/g/data/k10/cb4968/data/cmip5/ds'
                fileName = model + '_hus_' + experiment + '.nc'
                dataset = ds_hus
                save_file(dataset, folder, fileName)









