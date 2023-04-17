import numpy as np
import xarray as xr
import timeit
from get_variables.cmip5_variables import *
import myFuncs
import matplotlib.pyplot as plt
import os
# home = os.path.expanduser("~")
# import constructted_fields as cf

def calc_snapshot(var):
    return var.isel(time=0)

def calc_tMean(var):
    return var.mean(dim='time', keep_attrs=True)

def calc_sMean(var):
    aWeights = np.cos(np.deg2rad(var.lat))
    return var.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)


if __name__ == '__main__':

    models = [
            'IPSL-CM5A-MR', # 1
            'GFDL-CM3',     # 2
            'GISS-E2-H',    # 3
            'bcc-csm1-1',   # 4
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
        print(model, 'started') 
        start = timeit.default_timer()

        for experiment in experiments:
            print(experiment, 'started') 

            calc_pr = False
            calc_tas = False
            calc_hus = False
            calc_hur = False
            calc_wap = False
            calc_cl = False

            if calc_pr and data_exist(model, experiment, 'precip') == 'yes':
                # precip = cf.matrix3d
                # precip = get_dsvariable('precip', model, experiment)['precip']
                precip = get_pr(institutes[model], model, experiment)['precip']     

                pr_snapshot = calc_snapshot(precip)
                pr_tMean = calc_tMean(precip)
                pr_sMean = calc_sMean(precip)


                



            if variable == 'tas' and data_exist(model, experiment, variable) == 'yes':
                tas = get_tas(institutes[model], model, experiment)['tas']
                # tas = xr.open_dataset(home + '/Documents/data/cmip5/ds/' + model + '/' + model + '_tas_' + experiment + '.nc')['tas']

                snapshot = calc_snapshot(tas)
                tMean = calc_tMean(tas)
                sMean = calc_sMean(tas)

                if save_tas:
                    fileName = model + '_tas_snapshot_' + experiment + '.nc'
                    ds_snapshot = xr.Dataset(
                        data_vars ={'tas_snapshot':snapshot})
                    save_file(ds_snapshot, folder_save, fileName)

                    fileName = model + '_tas_tMean_' + experiment + '.nc'
                    ds_tMean = xr.Dataset(
                        data_vars ={'tas_tMean':tMean})
                    save_file(ds_tMean, folder_save, fileName)

                    fileName = model + '_tas_sMean_' + experiment + '.nc'
                    ds_sMean = xr.Dataset(
                        data_vars ={'tas_sMean':sMean})
                    save_file(ds_sMean, folder_save, fileName)


            if variable == 'pw' and data_exist(model, experiment, variable) == 'yes':
                pw = get_pw(institutes[model], model, experiment)['pw']
                # pw = xr.open_dataset(home + '/Documents/data/cmip5/ds/' + model + '/' + model + '_pw_' + experiment + '.nc')['pw']


                snapshot = calc_snapshot(pw)
                tMean = calc_tMean(pw)
                sMean = calc_sMean(pw)

                if save_pw:
                    fileName = model + '_pw_snapshot_' + experiment + '.nc'
                    ds_snapshot = xr.Dataset(
                        data_vars ={'pw_snapshot':snapshot})
                    save_file(ds_snapshot, folder_save, fileName)

                    fileName = model + '_pw_tMean_' + experiment + '.nc'
                    ds_tMean = xr.Dataset(
                        data_vars ={'pw_tMean':tMean})
                    save_file(ds_tMean, folder_save, fileName)

                    fileName = model + '_pw_sMean_' + experiment + '.nc'
                    ds_sMean = xr.Dataset(
                        data_vars ={'pw_sMean':sMean})
                    save_file(ds_sMean, folder_save, fileName)


            if variable == 'hur' and data_exist(model, experiment, variable) == 'yes':
                hur = get_hur(institutes[model], model, experiment)['hur']
                # hur = xr.open_dataset(home + '/Documents/data/cmip5/ds/' + model + '/' + model + '_hur_' + experiment + '.nc')['hur']

                snapshot = calc_snapshot(hur)
                tMean = calc_tMean(hur)
                sMean = calc_sMean(hur)

                if save_hur:
                    fileName = model + '_hur_snapshot_' + experiment + '.nc'
                    ds_snapshot = xr.Dataset(
                        data_vars ={'hur_snapshot':snapshot})
                    save_file(ds_snapshot, folder_save, fileName)

                    fileName = model + '_hur_tMean_' + experiment + '.nc'
                    ds_tMean = xr.Dataset(
                        data_vars ={'hur_tMean':tMean})
                    save_file(ds_tMean, folder_save, fileName)

                    fileName = model + '_hur_sMean_' + experiment + '.nc'
                    ds_sMean = xr.Dataset(
                        data_vars ={'hur_sMean':sMean})
                    save_file(ds_sMean, folder_save, fileName)


            if variable == 'wap500' and data_exist(model, experiment, variable) == 'yes':
                wap500 = get_wap500(institutes[model], model, experiment)['wap500']
                # wap500 = xr.open_dataset(home + '/Documents/data/cmip5/ds/' + model + '/' + model + '_wap500_' + experiment + '.nc')['wap500']

                snapshot = calc_snapshot(wap500)
                tMean = calc_tMean(wap500)
                areaFrac = calc_wapArea(wap500, 'area_ascent')

                if save_wap500:
                    fileName = model + '_wap500_snapshot_' + experiment + '.nc'
                    ds_snapshot = xr.Dataset(
                        data_vars ={'wap500_snapshot':snapshot})
                    save_file(ds_snapshot, folder_save, fileName)

                    fileName = model + '_wap500_tMean_' + experiment + '.nc'
                    ds_tMean = xr.Dataset(
                        data_vars ={'wap500_tMean':tMean})
                    save_file(ds_tMean, folder_save, fileName)

                    fileName = model + '_wap500_ascentArea_' + experiment + '.nc'
                    ds_sMean = xr.Dataset(
                        data_vars ={'wap500_ascentArea': areaFrac})
                    save_file(ds_sMean, folder_save, fileName)


            if variable == 'cloud_low' and data_exist(model, experiment, variable) == 'yes':
                cloud_low = get_clouds(institutes[model], model, experiment)['cloud_low']
                # cloud_low = xr.open_dataset(home + '/Documents/data/cmip5/ds/' + model + '/' + model + '_cloud_low_' + experiment + '.nc')['cloud_low']

                snapshot = calc_snapshot(cloud_low)
                tMean = calc_tMean(cloud_low)
                sMean = calc_sMean(cloud_low)

                if save_cloud_low:
                    fileName = model + '_cloud_low_snapshot_' + experiment + '.nc'
                    ds_snapshot = xr.Dataset(
                        data_vars ={'cloud_low_snapshot':snapshot})
                    save_file(ds_snapshot, folder_save, fileName)

                    fileName = model + '_cloud_low_tMean_' + experiment + '.nc'
                    ds_tMean = xr.Dataset(
                        data_vars ={'cloud_low_tMean':tMean})
                    save_file(ds_tMean, folder_save, fileName)

                    fileName = model + '_cloud_low_sMean_' + experiment + '.nc'
                    ds_sMean = xr.Dataset(
                        data_vars ={'cloud_low_sMean':sMean})
                    save_file(ds_sMean, folder_save, fileName)


            if variable == 'cloud_high' and data_exist(model, experiment, variable) == 'yes':
                cloud_high = get_clouds(institutes[model], model, experiment)['cloud_high']
                # cloud_high = xr.open_dataset(home + '/Documents/data/cmip5/ds/' + model + '/' + model + '_cloud_high_' + experiment + '.nc')['cloud_high']

                snapshot = calc_snapshot(cloud_high)
                tMean = calc_tMean(cloud_high)
                sMean = calc_sMean(cloud_high)

                if save_cloud_low:
                    fileName = model + '_cloud_high_snapshot_' + experiment + '.nc'
                    ds_snapshot = xr.Dataset(
                        data_vars ={'cloud_high_snapshot':snapshot})
                    save_file(ds_snapshot, folder_save, fileName)

                    fileName = model + '_cloud_high_tMean_' + experiment + '.nc'
                    ds_tMean = xr.Dataset(
                        data_vars ={'cloud_high_tMean':tMean})
                    save_file(ds_tMean, folder_save, fileName)

                    fileName = model + '_cloud_high_sMean_' + experiment + '.nc'
                    ds_sMean = xr.Dataset(
                        data_vars ={'cloud_high_sMean':sMean})
                    save_file(ds_sMean, folder_save, fileName)







                save_pr = False
                save_tas = False
                save_hus = False
                save_hur = False
                save_wap = False
                save_cl = False

                # folder_save = home + '/Documents/data/cmip5/metrics_cmip5/' + model
                folder_save = '/g/data/k10/cb4968/data/cmip5/' + model


                if save_pr:
                    fileName = model + '_pr_snapshot_' + experiment + '.nc'
                    ds_snapshot = xr.Dataset(
                        data_vars ={'pr_snapshot': pr_snapshot})
                    save_file(ds_snapshot, folder_save, fileName)

                    fileName = model + '_pr_tMean_' + experiment + '.nc'
                    ds_tMean = xr.Dataset(
                        data_vars ={'pr_tMean':pr_tMean})
                    save_file(ds_tMean, folder_save, fileName)

                    fileName = model + '_pr_sMean_' + experiment + '.nc'
                    ds_sMean = xr.Dataset(
                        data_vars ={'pr_sMean':pr_sMean})
                    save_file(ds_sMean, folder_save, fileName)


                if save_pr:
                    fileName = model + '_pr_snapshot_' + experiment + '.nc'
                    ds_snapshot = xr.Dataset(
                        data_vars ={'pr_snapshot': pr_snapshot})
                    save_file(ds_snapshot, folder_save, fileName)

                    fileName = model + '_pr_tMean_' + experiment + '.nc'
                    ds_tMean = xr.Dataset(
                        data_vars ={'pr_tMean':pr_tMean})
                    save_file(ds_tMean, folder_save, fileName)

                    fileName = model + '_pr_sMean_' + experiment + '.nc'
                    ds_sMean = xr.Dataset(
                        data_vars ={'pr_sMean':pr_sMean})
                    save_file(ds_sMean, folder_save, fileName)









        stop = timeit.default_timer()
        print('model: {} took {} minutes to finsih'.format(model, (stop-start)/60))





