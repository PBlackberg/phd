import numpy as np
import xarray as xr
import scipy
import timeit
import os
home = os.path.expanduser("~")

import myFuncs
from get_variables.cmip5_variables import *
import constructed_fields as cf

def snapshot(var):
    return var.isel(time=0)

def tMean(var):
    return var.mean(dim='time', keep_attrs=True)

def sMean(var):
    aWeights = np.cos(np.deg2rad(var.lat))
    return var.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)

def vertical_mean(var):
    var = var.sel(plev=slice(850e2,0)) # free troposphere (most values at 1000 hPa over land are NaN)
    return (var * var.plev).sum(dim='plev') / var.plev.sum(dim='plev')

def vertical_integral(var):
    var = var.sel(plev=slice(850e2,0)) # free troposphere (most values at 1000 hPa over land are NaN)
    var.fillna(0) # mountains will be NaN for larger values as well, so setting them to zero
    g = 9.8
    var = xr.DataArray(
        data= -scipy.integrate.simpson(var.data, var.plev.data, axis=1, even='last')/g,
        dims=['time','lat', 'lon'],
        coords={'time': var.time.data, 'lat': var.lat.data, 'lon': var.lon.data}
        )
    return var

def in_descent(var, dataset, experiment):
    wap500 = get_dsvariable('wap', dataset, experiment, resolution=resolutions[0])['wap'].sel(plev = 5e4)

    if len(var)<1000:
        wap500 = resample_timeMean(wap500, 'monthly')
        wap500 = wap500.assign_coords(time=data.time)
    
    return var.where(wap500>0, np.nan)




if __name__ == '__main__':
    import matplotlib.pyplot as plt

    datasets = [
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

    observations = [
        'GPCP'
        ]
    
    datasets = datasets + observations


    resolutions = [
        # 'original',
        'regridded'
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
    

    for dataset in datasets:
        print(dataset, 'started') 
        start = timeit.default_timer()
        for experiment in experiments:
            print(experiment) 


            in_descent_regions = True

            # precip
            if data_exist(dataset, experiment, 'precip') == 'yes':

                if dataset == 'GPCP':
                    # ds = cf.matrix3d
                    # ds = get_pr(institutes[dataset], dataset, experiment)  
                    ds = get_dsvariable('precip', dataset, experiment, resolution=resolutions[0])
                else:
                    # ds = cf.matrix3d
                    # ds = get_pr(institutes[dataset], dataset, experiment)  
                    ds = get_dsvariable('precip', dataset, experiment, resolution=resolutions[0])
                
                data = ds['precip']

                if in_descent_regions:
                    data = in_descent(data)
                
                pr_snapshot = snapshot(data)
                pr_tMean = tMean(data)
                pr_sMean = sMean(data)

            # tas
            if data_exist(dataset, experiment, 'tas') == 'yes':
                # ds = cf.matrix3d
                # ds = get_tas(institutes[dataset], dataset, experiment)
                ds = get_dsvariable('tas', dataset, experiment, resolution=resolutions[0])

                data = ds['tas']

                if in_descent_regions:
                    data = in_descent(data)
                
                tas_snapshot = snapshot(data)
                tas_tMean = tMean(data)
                tas_sMean = sMean(data)

            # hus
            if data_exist(dataset, experiment, 'hus') == 'yes':
                # ds = cf.matrix4d
                # ds = get_tas(institutes[dataset], dataset, experiment)
                ds = get_dsvariable('hus', dataset, experiment, resolution=resolutions[0])

                data = ds['hus']
                data = vertical_integral(data)

                if in_descent_regions:
                    data = in_descent(data)

                hus_snapshot = snapshot(data)
                hus_tMean = tMean(data)
                hus_sMean = sMean(data)

            # hur
            if data_exist(dataset, experiment, 'hur') == 'yes':
                # ds = cf.matrix4d
                # ds = get_tas(institutes[dataset], dataset, experiment)
                ds = get_dsvariable('hur', dataset, experiment, resolution=resolutions[0])
                
                data = ds['hur']
                data = vertical_mean(data)

                if in_descent_regions:
                    data = in_descent(data)
                
                hur_snapshot = snapshot(data)
                hur_tMean = tMean(data)
                hur_sMean = sMean(data)

            # wap
            if data_exist(dataset, experiment, 'wap') == 'yes':
                # ds = cf.matrix4d
                # ds = get_wap(institutes[dataset], dataset, experiment)
                ds = get_dsvariable('wap', dataset, experiment, resolution=resolutions[0])

                data = ds['wap'].sel(plev = 500e2)

                wap_snapshot = snapshot(data)
                wap_tMean = tMean(data)
                
                data_a = xr.where(data<0, 1, 0)
                wap_a_snapshot = snapshot(data_a)
                wap_a_tMean = tMean(data_a)
                wap_a_sMean = data_a.sum(dim=('lat','lon'))*(100/(len(data_a['lat'])*len(data_a['lon'])))

                data_d = xr.where(data>0, 1, 0)
                wap_d_snapshot = snapshot(data_d)
                wap_d_tMean = tMean(data_d)
                wap_d_sMean = data_d.sum(dim=('lat','lon'))*(100/(len(data_a['lat'])*len(data_a['lon'])))

            # cl
            if data_exist(dataset, experiment, 'cl') == 'yes':
                # ds, ds_p_hybridsigma = cf.ds_matrix4d, cf.ds_matrix4d
                # ds, ds_p_hybridsigma = get_cl(institutes[dataset], dataset, experiment)
                ds, ds_p_hybridsigma = get_dsvariable('wap', dataset, experiment, resolution=resolutions[0]), get_dsvariable('wap', dataset, experiment, resolution=resolutions[0])

                data = ds['cl']
                data_p_hybridsigma = ds['p_hybridsigma']
                
                data_low = xr.where((data_p_hybridsigma<=2000e2) & (data_p_hybridsigma>=600e2), 1, 0)

                if in_descent_regions:
                    data = in_descent(data)
                cl_low_snapshot = snapshot(data_low)
                cl_low_tMean = tMean(data_low)
                cl_low_sMean = sMean(data_low)

                data_high = xr.where((data_p_hybridsigma<=250e2) & (data_p_hybridsigma>=100e2), 1, 0)
                if in_descent_regions:
                    data = in_descent(data)
                cl_high_snapshot = snapshot(data_high)
                cl_high_tMean = tMean(data_high)
                cl_high_sMean = sMean(data_high)




                save_pr = False
                save_tas = False
                save_hus = False
                save_hur = False
                save_wap = False
                save_cl = False


                if dataset == 'GPCP':
                    # folder_save = '/g/data/k10/cb4968/data/obs/'+ dataset
                    folder_save = home + '/Documents/data/obs/' + dataset
                else:
                    # folder_save = home + '/Documents/data/cmip5/metrics_cmip5/' + dataset
                    folder_save = '/g/data/k10/cb4968/data/cmip5/' + dataset



                if save_pr and data_exist(dataset, experiment, 'precip') == 'yes':
                    fileName = dataset + '_pr_snapshot_' + experiment + '.nc'
                    ds_snapshot = xr.Dataset(
                        data_vars ={'pr_snapshot': pr_snapshot}
                        )
                    save_file(ds_snapshot, folder_save, fileName)

                    fileName = dataset + '_pr_tMean_' + experiment + '.nc'
                    ds_tMean = xr.Dataset(
                        data_vars ={'pr_tMean':pr_tMean}
                        )
                    save_file(ds_tMean, folder_save, fileName)

                    fileName = dataset + '_pr_sMean_' + experiment + '.nc'
                    ds_sMean = xr.Dataset(
                        data_vars ={'pr_sMean':pr_sMean}
                        )
                    save_file(ds_sMean, folder_save, fileName)


                if save_tas and data_exist(dataset, experiment, 'tas') == 'yes':
                    fileName = dataset + '_tas_snapshot_' + experiment + '.nc'
                    ds_snapshot = xr.Dataset(
                        data_vars ={'tas_snapshot': tas_snapshot}
                        )
                    save_file(ds_snapshot, folder_save, fileName)

                    fileName = dataset + '_tas_tMean_' + experiment + '.nc'
                    ds_tMean = xr.Dataset(
                        data_vars ={'tas_tMean': tas_tMean}
                        )
                    save_file(ds_tMean, folder_save, fileName)

                    fileName = dataset + '_tas_sMean_' + experiment + '.nc'
                    ds_sMean = xr.Dataset(
                        data_vars ={'tas_sMean': tas_sMean}
                        )
                    save_file(ds_sMean, folder_save, fileName)


                if save_hus and data_exist(dataset, experiment, 'hus') == 'yes':
                    fileName = dataset + '_hus_snapshot_' + experiment + '.nc'
                    ds_snapshot = xr.Dataset(
                        data_vars ={'hus_snapshot': hus_snapshot}
                        )
                    save_file(ds_snapshot, folder_save, fileName)

                    fileName = dataset + '_hus_tMean_' + experiment + '.nc'
                    ds_tMean = xr.Dataset(
                        data_vars ={'hus_tMean':hus_tMean}
                        )
                    save_file(ds_tMean, folder_save, fileName)

                    fileName = dataset + '_hus_sMean_' + experiment + '.nc'
                    ds_sMean = xr.Dataset(
                        data_vars ={'hus_sMean': hus_sMean}
                        )
                    save_file(ds_sMean, folder_save, fileName)


                if save_hur and data_exist(dataset, experiment, 'hur') == 'yes':
                    fileName = dataset + '_hur_snapshot_' + experiment + '.nc'
                    ds_snapshot = xr.Dataset(
                        data_vars ={'hur_snapshot':hur_snapshot}
                        )
                    save_file(ds_snapshot, folder_save, fileName)

                    fileName = dataset + '_hur_tMean_' + experiment + '.nc'
                    ds_tMean = xr.Dataset(
                        data_vars ={'hur_tMean':hur_tMean}
                        )
                    save_file(ds_tMean, folder_save, fileName)

                    fileName = dataset + '_hur_sMean_' + experiment + '.nc'
                    ds_sMean = xr.Dataset(
                        data_vars ={'hur_sMean':hur_sMean}
                        )
                    save_file(ds_sMean, folder_save, fileName)


                if save_wap and data_exist(dataset, experiment, 'wap') == 'yes':
                    fileName = dataset + '_wap_snapshot_' + experiment + '.nc'
                    ds_snapshot = xr.Dataset(
                        data_vars ={'wap_snapshot':wap_snapshot,
                                    'wap_a_snapshot':wap_a_snapshot,
                                    'wap_d_snapshot':wap_d_snapshot}
                        )
                    save_file(ds_snapshot, folder_save, fileName)

                    fileName = dataset + '_wap_tMean_' + experiment + '.nc'
                    ds_tMean = xr.Dataset(
                        data_vars ={'wap_tMean':wap_tMean,
                                    'wap_tMean':wap_tMean,
                                    'wap_tMean':wap_tMean}
                        )
                    save_file(ds_tMean, folder_save, fileName)

                    fileName = dataset + '_wap_sMean_' + experiment + '.nc'
                    ds_sMean = xr.Dataset(
                        data_vars ={'wap_a_sMean': wap_a_sMean,
                                    'wap_d_sMean': wap_d_sMean}
                        )
                    save_file(ds_sMean, folder_save, fileName)


                if save_cl and data_exist(dataset, experiment, 'cl') == 'yes':
                    fileName = dataset + '_cl_snapshot_' + experiment + '.nc'
                    ds_snapshot = xr.Dataset(
                        data_vars ={'cl_low_snapshot': cl_low_snapshot,
                                    'cl_high_snapshot': cl_high_snapshot}
                        )
                    save_file(ds_snapshot, folder_save, fileName)

                    fileName = dataset + '_cl_tMean_' + experiment + '.nc'
                    ds_tMean = xr.Dataset(
                        data_vars ={'cl_low_tMean':cl_low_tMean,
                                    'cl_high_tMean':cl_high_tMean}
                        )
                    save_file(ds_tMean, folder_save, fileName)

                    fileName = dataset + '_cl_sMean_' + experiment + '.nc'
                    ds_sMean = xr.Dataset(
                        data_vars ={'cl_low_sMean':cl_low_sMean,
                                    'cl_high_sMean':cl_low_sMean}
                        )
                    save_file(ds_sMean, folder_save, fileName)


        stop = timeit.default_timer()
        print('dataset: {} took {} minutes to finsih'.format(dataset, (stop-start)/60))





