import intake
import xarray as xr



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

    haveDsOut = True
    precip = regrid_conserv(ds_orig, haveDsOut).pr*60*60*24
    precip.attrs['units']= 'mm/day'


    ds_pr = xr.Dataset(
        data_vars = {'precip': precip}
    )
            
    return ds_pr






if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt

    from myFuncs import *
    from myPlots import *


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
                'rcp85'
                ]


    for model in models:
        for experiment in experiments:

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



            ds_pr = get_pr(model, experiment)


            
            plot_snapshot(ds_pr.pr.isel(time=0), 'Blues', 'pr_day', model)
            plt.show()
            plot_snapshot(ds_pr.pr.mean(dim=('time'), keep_attrs=True), 'Blues','pr_mean', model)
            plt.show()


            saveit = False
            if saveit:
                folder = '/g/data/k10/cb4968/data/cmip5/ds/'
                fileName = model + '_precip_' + experiment + '.nc'
                dataset = xr.Dataset({'precip': ds_pr.pr})
                save_file(dataset, folder, fileName)





