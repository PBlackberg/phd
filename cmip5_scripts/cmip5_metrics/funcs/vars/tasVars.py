import intake

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

import myFuncs
import myPlots


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

    elif model == 'FGOALS-g2':
        ds_orig =ds_dict[list(ds_dict.keys())[-1]].isel(time=slice(12*120, 12*120 + 12*30)).sel(lon=slice(0,360), lat=slice(-35,35))
        
    elif model == 'CNRM-CM5':
        ds_orig =ds_dict[list(ds_dict.keys())[-1]].isel(time=slice(12*64, 12*64 + 12*30)).sel(lon=slice(0,360), lat=slice(-35,35))



    haveDsOut = True
    tas = myFuncs.regrid_conserv(ds_orig, haveDsOut).tas-273.15
    tas.attrs['units']= 'deg (C)'


    ds_tas = xr.Dataset(
        data = {'tas': tas}
    )
            

    return ds_tas




if __name__ == '__main__':

    models = [
            # 'IPSL-CM5A-MR', # 1
            'GFDL-CM3',     # 2
            # 'GISS-E2-H',    # 3
            # 'bcc-csm1-1',   # 4
            # 'CNRM-CM5',     # 5
            # 'CCSM4',        # 6 # cannot concatanate files for historical run
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



    for model in models:
        for experiment in experiments:
            
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



            ds_tas = get_tas(model, experiment)



            myPlots.plot_snapshot(ds_tas.tas.isel(time=0), 'Reds', 'surface temperature', model)
            plt.show()
            myPlots.plot_snapshot(ds_tas.tas.mean(dim='time', keep_attrs=True), 'Reds', 'surface temperature', model)
            plt.show()



            saveit = True
            if saveit:
                folder = '/g/data/k10/cb4968/data/cmip5/ds' #+ model
                fileName = model + '_tas_' + experiment + '.nc'
                dataset = xr.Dataset({'tas': ds_tas.tas})
                myFuncs.save_file(dataset, folder, fileName)





































