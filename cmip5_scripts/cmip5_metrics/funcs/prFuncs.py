import xarray as xr
import pandas as pd

def calc_rxday(precip):
    rx1day = precip.resample(time='Y').max(dim='time')
    # rx1day_tMean = rx1day.mean(dim=('time'),keep_attrs=True)
    # rx1day_sMean = rx1day.mean(dim=('lat','lon'),keep_attrs=True)


    precip5day = precip.resample(time='5D').mean(dim='time')
    rx5day = precip5day.resample(time='Y').max(dim='time')


    rxday = xr.Dataset(
        data_vars = {'rx1day': rx1day, 
                     'rx5day': rx5day}
        )

    return rxday



def calc_pr_percentiles(precip):

    pr95 = precip.quantile(0.95,dim=('lat','lon'),keep_attrs=True)
    pr95 = xr.DataArray(
        data = pr95.data,
        dims = ['time'],
        coords = {'time': precip.time.data}, 
        attrs = {'units':'mm/day'}
        )

    pr97 = precip.quantile(0.97,dim=('lat','lon'),keep_attrs=True)
    pr97 = xr.DataArray(
        data = pr97.data,
        dims = ['time'],
        coords = {'time': precip.time.data},
        attrs = {'units':'mm/day'}
        )

    pr99 = precip.quantile(0.99,dim=('lat','lon'),keep_attrs=True)
    pr99 = xr.DataArray(
        data = pr99.data,
        dims = ['time'],
        coords = {'time': precip.time.data},
        attrs = {'units':'mm/day'}
        )

    pr999 = precip.quantile(0.999,dim=('lat','lon'),keep_attrs=True)
    pr999 = xr.DataArray(
        data = pr999.data,
        dims = ['time'],
        coords = {'time': precip.time.data},
        attrs = {'units':'mm/day'}
        )
        

    pr_percentiles = xr.Dataset(
        data_vars = {'pr95': pr95, 
                     'pr97': pr97, 
                     'pr99': pr99, 
                     'pr999': pr999}
        ) 

    return pr_percentiles




if __name__ == '__main__':

    import numpy as np

    from vars.prVars import *
    from vars.myFuncs import *
    from vars.myPlots import *


    models = [
            # 'IPSL-CM5A-MR', # 1
            'GFDL-CM3',     
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
            # 'CanESM2',      # 15 # slicing with .sel does not work, 'contains no datetime objects'
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

            haveData = False
            if haveData:
                folder = '/g/data/k10/cb4968/data/cmip5/ds'
                fileName = model + '_precip_' + experiment + '.nc'
                path = folder + '/' + fileName
                precip = xr.open_dataset(path).precip
            else:
                precip = get_pr(model, experiment).precip



            rx1day = precip.resample(time='Y').max(dim='time')
            precip5day = precip.resample(time='5D').mean(dim='time')
            rx5day = precip5day.resample(time='Y').max(dim='time')

            rxday = xr.Dataset(
                data_vars = {'rx1day': rx1day, 
                             'rx5day': rx5day}
                )

            saveit =False
            if saveit:
                fileName = model + '_pr_rxday_' + experiment + '.nc'
                dataSet = rxday
                myFuncs.save_file(dataSet, folder, fileName)




            pr95 = precip.quantile(0.95,dim=('lat','lon'),keep_attrs=True)
            pr95 = xr.DataArray(
                data = pr95.data,
                dims = ['time'],
                coords = {'time': precip.time.data},
                attrs = {'units':'mm/day'}
                )

            pr97 = precip.quantile(0.97,dim=('lat','lon'),keep_attrs=True)
            pr97 = xr.DataArray(
                data = pr97.data,
                dims = ['time'],
                coords = {'time': precip.time.data},
                attrs = {'units':'mm/day'}
                )

            pr99 = precip.quantile(0.99,dim=('lat','lon'),keep_attrs=True)
            pr99 = xr.DataArray(
                data = pr99.data,
                dims = ['time'],
                coords = {'time': precip.time.data},
                attrs = {'units':'mm/day'}
                )

            pr999 = precip.quantile(0.999,dim=('lat','lon'),keep_attrs=True)
            pr999 = xr.DataArray(
                data = pr999.data,
                dims = ['time'],
                coords = {'time': precip.time.data},
                attrs = {'units':'mm/day'}
                )
            
            pr_percentiles = xr.Dataset(
                data_vars = {'pr95': pr95, 
                             'pr97': pr97, 
                             'pr99': pr99, 
                             'pr999': pr999}
                ) 

            saveit =False
            if saveit:
                fileName = model + '_pr_percentiles_' + experiment + '.nc'
                dataSet = pr_percentiles
                myFuncs.save_file(dataSet, folder, fileName)








