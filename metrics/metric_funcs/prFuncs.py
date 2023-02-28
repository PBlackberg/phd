import xarray as xr
import os
# from var_funcs.cmip5Vars import *
from os.path import expanduser
home = expanduser("~")


def calc_rxday(precip):
    rx1day = precip.resample(time='Y').max(dim='time')
    rx1day.attrs['units']= 'mm day' + chr(0x207B) + chr(0x00B9)

    precip5day = precip.resample(time='5D').mean(dim='time')
    rx5day = precip5day.resample(time='Y').max(dim='time')
    rx5day.attrs['units']= 'mm day' + chr(0x207B) + chr(0x00B9)

    ds_rxday = xr.Dataset(
        data_vars = {'rx1day': rx1day, 
                     'rx5day': rx5day}
        )
    
    return ds_rxday



def calc_pr_percentiles(precip):

    pr95 = precip.quantile(0.95,dim=('lat','lon'),keep_attrs=True)
    pr95 = xr.DataArray(
        data = pr95.data,
        dims = ['time'],
        coords = {'time': precip.time.data}, 
        attrs = {'units':'mm day' + chr(0x207B) + chr(0x00B9)}
        )

    pr97 = precip.quantile(0.97,dim=('lat','lon'),keep_attrs=True)
    pr97 = xr.DataArray(
        data = pr97.data,
        dims = ['time'],
        coords = {'time': precip.time.data},
        attrs = {'units':'mm day' + chr(0x207B) + chr(0x00B9)}
        )

    pr99 = precip.quantile(0.99,dim=('lat','lon'),keep_attrs=True)
    pr99 = xr.DataArray(
        data = pr99.data,
        dims = ['time'],
        coords = {'time': precip.time.data},
        attrs = {'units':'mm day' + chr(0x207B) + chr(0x00B9)}
        )

    pr999 = precip.quantile(0.999,dim=('lat','lon'),keep_attrs=True)
    pr999 = xr.DataArray(
        data = pr999.data,
        dims = ['time'],
        coords = {'time': precip.time.data},
        attrs = {'units':'mm day' + chr(0x207B) + chr(0x00B9)}
        )
        

    ds_prPercentiles = xr.Dataset(
        data_vars = {'pr95': pr95, 
                     'pr97': pr97, 
                     'pr99': pr99, 
                     'pr999': pr999}
        ) 

    return ds_prPercentiles



def F_pr10(precip):
    F_pr10 = ((precip>10)*1).sum(dim=('lat','lon'))
    F_pr10.attrs['units'] = 'Nb'

    ds_F_pr10 = xr.Dataset(
    data_vars = {'F_pr10': F_pr10},
    attrs = {'description': 'Number of gridboxes in daily scene exceeding 10 mm/day'}
        )

    return ds_F_pr10


def save_file(dataset, folder, fileName):
    os.makedirs(folder, exist_ok=True)
    path = folder + '/' + fileName

    if os.path.exists(path):
        os.remove(path)    
    
    dataset.to_netcdf(path)



if __name__ == '__main__':


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

            # precip = get_pr(institutes[model], model, experiment).precip
            precip = xr.open_dataset(home + '/Documents/data/cmip5/ds/' + model + '/' + model + '_precip_' + experiment + '.nc')['precip']

            ds_rxday = calc_rxday(precip)
            ds_prPercentiles = calc_pr_percentiles(precip)
            ds_F_pr10 = F_pr10(precip)


            save_rxday = False
            save_prPercentiles = False
            save_F_pr10 = False

            folder_save = '/g/data/k10/cb4968/data/cmip5/'+ model

            if save_rxday:
                fileName = model + '_rxday_' + experiment + '.nc'
                dataSet = ds_rxday
                save_file(dataSet, folder_save, fileName)

            if save_prPercentiles:
                fileName = model + '_prPercentiles_' + experiment + '.nc'
                dataSet = ds_prPercentiles
                save_file(dataSet, folder_save, fileName)


            if save_F_pr10 :
                fileName = model + '_F_pr10_' + experiment + '.nc'
                dataSet = ds_F_pr10
                save_file(dataSet, folder_save, fileName)







