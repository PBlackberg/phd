import numpy as np
import xarray as xr
import timeit
from variables.cmip5Vars import *


def snapshot(var):
    return var.isel(time=0)


def calc_tMean(var):
    return var.mean(dim='time', keep_attrs=True)


def calc_sMean(var):
    aWeights = np.cos(np.deg2rad(var.lat))
    return var.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)






if __name__ == '__main__':

    import matplotlib.pyplot as plt

    models = [
            # 'IPSL-CM5A-MR', # 1
            # 'GFDL-CM3',     # 2
            # 'GISS-E2-H',    # 3
            # 'bcc-csm1-1',   # 4
            'CNRM-CM5',     # 5
            'CCSM4',        # 6
            'HadGEM2-AO',   # 7
            'BNU-ESM',      # 8
            'EC-EARTH',     # 9
            'FGOALS-g2',    # 10
            'MPI-ESM-MR',   # 11
            'CMCC-CM',      # 12
            'inmcm4',       # 13
            'NorESM1-M',    # 14
            'CanESM2',      # 15 
            'MIROC5',       # 16
            'HadGEM2-CC',   # 17
            'MRI-CGCM3',    # 18
            'CESM1-BGC'     # 19
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
        print(model, 'started') 
        start = timeit.default_timer()
        
        for experiment in experiments:

            precip = get_pr(institutes[model], model, experiment)['precip']
            tas = get_tas(institutes[model], model, experiment)['tas']
            pw = get_pw(institutes[model], model, experiment)['pw']
            hur = get_hur(institutes[model], model, experiment)['hur']
            cloud_low = get_clouds(institutes[model], model, experiment)['cloud_low']
            cloud_high = get_clouds(institutes[model], model, experiment)['cloud_high']
            wap500 = get_wap500(institutes[model], model, experiment)['wap500']
            

            save_pr = True
            save_tas = True
            save_pw = True
            save_hur = True
            save_cl = True
            save_wap500 = True
            
            folder = '/g/data/k10/cb4968/data/cmip5/' + model

            if save_pr:
                fileName = model + '_precip_tMean_' + experiment + '.nc'
                dataset = xr.Dataset(
                    data_vars = {'precip_tMean': calc_tMean(precip),
                                'precip_snapshot': snapshot(precip)}
                        )
                save_file(dataset, folder, fileName)

                fileName = model + '_precip_sMean_' + experiment + '.nc'
                dataset = xr.Dataset(
                    data_vars = {'precip_sMean': calc_sMean(precip)}
                        )
                save_file(dataset, folder, fileName)


            if save_tas:
                fileName = model + '_tas_tMean_' + experiment + '.nc'
                dataset = xr.Dataset(
                    data_vars = {'tas_tMean': calc_tMean(tas),
                                'tas_snapshot': snapshot(tas)}
                        )
                save_file(dataset, folder, fileName)

                fileName = model + '_tas_sMean_' + experiment + '.nc'
                dataset = xr.Dataset(
                    data_vars = {'tas_sMean': calc_sMean(tas)}
                        )
                save_file(dataset, folder, fileName)


            if save_pw:
                fileName = model + '_pw_tMean_' + experiment + '.nc'
                dataset = xr.Dataset(
                    data_vars = {'pw_tMean': calc_tMean(pw),
                                'pw_snapshot': snapshot(pw)}
                        )
                save_file(dataset, folder, fileName)

                fileName = model + '_pw_sMean_' + experiment + '.nc'
                dataset = xr.Dataset(
                    data_vars = {'pw_sMean': calc_sMean(pw)}
                        )
                save_file(dataset, folder, fileName)


            if save_hur:
                fileName = model + '_hur_tMean_' + experiment + '.nc'
                dataset = xr.Dataset(
                    data_vars = {'hur_tMean': calc_tMean(hur),
                                'hur_snapshot': snapshot(hur)}
                        )
                save_file(dataset, folder, fileName)

                fileName = model + '_hur_sMean_' + experiment + '.nc'
                dataset = xr.Dataset(
                    data_vars = {'hur_sMean': calc_sMean(hur)}
                        )
                save_file(dataset, folder, fileName)


            if save_cl:
                if model == 'CNRM-CM5' or model == 'CCSM4':
                    pass
                else:
                    fileName = model + '_clouds_tMean_' + experiment + '.nc'
                    dataset = xr.Dataset(
                        data_vars = {'cloud_low_tMean': calc_tMean(cloud_low),
                                    'cloud_low_snapshot': snapshot(cloud_low),
                                    'cloud_high_tMean': calc_tMean(cloud_high),
                                    'cloud_high_snapshot': snapshot(cloud_high)}
                            )
                    save_file(dataset, folder, fileName)

                    fileName = model + '_clouds_sMean_' + experiment + '.nc'
                    dataset = xr.Dataset(
                        data_vars = {'cloud_low_sMean': calc_sMean(cloud_low),
                                     'cloud_high_sMean': calc_sMean(cloud_high)}
                            )
                    save_file(dataset, folder, fileName)


            if save_wap500:
                
                if model == 'GISS-E2-H':
                    pass
                else:
                    fileName = model + '_wap500_tMean_' + experiment + '.nc'
                    dataset = xr.Dataset(
                        data_vars = {'wap500_tMean': calc_tMean(wap500),
                                    'wap500_snapshot': snapshot(wap500)}
                            )
                    save_file(dataset, folder, fileName)

                    fileName = model + '_wap500_sMean_' + experiment + '.nc'
                    dataset = xr.Dataset(
                        data_vars = {'wap500_sMean': calc_sMean(wap500)}
                            )
                    save_file(dataset, folder, fileName)

        
        stop = timeit.default_timer()
        print('model: {} took {} minutes to finsih'.format(model, (stop-start)/60))



