import intake
import xarray as xr
import numpy as np
import timeit

import myFuncs
import cmip5_metrics.prFuncs as prFuncs
import cmip5_metrics.aggFuncs as aggFuncs
import cmip5_metrics.husFuncs as husFuncs


models = [
        # 'IPSL-CM5A-MR', # 1
         'GFDL-CM3',     # 2
        # 'GISS-E2-H',    # 3
        # 'bcc-csm1-1',   # 4
        # 'CNRM-CM5',     # 5
        # #'CCSM4',        # 6 # cannot concatanate files for historical run
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

variables = [
            'pr',
            'tas', 
            'hus'
            ]

experiments = [
                'historical', 
                'rcp85'
            ]

metricFiles = {
                    'pr_examples':True, 
                    'pr_rxday':True, 
                    'pr_percentiles':True, 
                    'numberIndex': True, 
                    'pwad':True, 
                    'rome':True, 
                    'rome_n':True,
                    'tas_examples':True,
                    'tas_annual':True,
                    'hus_examples':True,
                    'hus_sMean':True
    }



for model in models:
    folder = '/g/data/k10/cb4968/cmip5/' + model
    start = timeit.default_timer()

    for var in variables:
        for experiment in experiments:


# ------------------------------ load variable data ---------------------------------------------

            if not var =='tas':
                time_frequency = 'day'
            else:
                time_frequency = 'mon'
            
            if experiment == 'historical':
                period=slice('1970-01','1999-12')
                ensemble = 'r1i1p1'

                if model == 'GISS-E2-H':
                    ensemble = 'r6i1p1'

                if (var=='tas' and model == 'EC-EARTH'):
                    ensemble = 'r6i1p1'

                if (var == 'hus' and model == 'CCSM4'):
                    ensemble = 'r5i1p1'


            if experiment == 'rcp85':
                period=slice('2070-01','2099-12')
                ensemble = 'r1i1p1'

                if model == 'GISS-E2-H':
                    ensemble = 'r2i1p1'

                if model == 'EC-EARTH':
                    ensemble = 'r6i1p1'

                if model == (var == 'hus' and model == 'CCSM4'):
                    ensemble = 'r5i1p1'


            
            ds_dict = intake.cat.nci['esgf'].cmip5.search(
                                            model_id = model, 
                                            experiment = experiment,
                                            time_frequency = 'day', 
                                            realm = 'atmos', 
                                            ensemble = ensemble, 
                                            variable= var).to_dataset_dict()

            if not (model == 'CanESM2' and experiment == 'historical'):
                ds_orig =ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-35, 35))
            else:
                ds_orig =ds_dict[list(ds_dict.keys())[-1]].isel(time=slice(43800, 43800+10950)).sel(lon=slice(0,360),lat=slice(-35, 35))


            haveDsOut = True
            ds_regrid = myFuncs.regrid_conserv(ds_orig, haveDsOut)





# ------------------------------- Calculate metrics ---------------------------------------------

            if var == 'pr':
                precip = ds_regrid.pr * 60*60*24
                precip.attrs['units']= 'mm/day'
                
                if metricFiles['pr_examples']:
                    fileName = model + '_pr_examples_' + experiment + '.nc'
                    dataSet = xr.Dataset(
                                        {'pr_day': precip.isel(time=0), 
                                         'pr_tMean': precip.mean(dim='time', keep_attrs=True)})
                    myFuncs.save_file(dataSet, folder, fileName)


                if metricFiles['pr_rxday']:
                    fileName = model + '_pr_rxday_' + experiment + '.nc'
                    dataSet = xr.Dataset(
                                        {'rx1day': precip.resample(time='Y').max(dim='time', keep_attrs=True), 
                                         'rx5day': precip.resample(time='5D').mean(dim='time', keep_attrs=True).resample(time='Y').max(dim='time')})
                    myFuncs.save_file(dataSet, folder, fileName)


                if metricFiles['pr_percentiles']:
                    fileName = model + '_pr_percentiles_' + experiment + '.nc'
                    pr95, pr97, pr99, pr999 = prFuncs.calc_percentiles(precip)
                    dataSet = xr.Dataset(
                            {'pr95': pr95, 
                             'pr97': pr97, 
                             'pr99': pr99, 
                             'pr999': pr999}) 
                    myFuncs.save_file(dataSet, folder, fileName)



                if (metricFiles['numberIndex'] or metricFiles['rome'] or metricFiles['rome_n']):
                    listOfdays = np.arange(0,len(precip.time.data))
                    conv_threshold = precip.quantile(0.97,dim=('lat','lon'),keep_attrs=True).mean(dim='time')


                    if metricFiles['numberIndex']:
                        fileName = model + '_number_index_' + experiment + '.nc'
                        numberIndex, areaf = aggFuncs.calc_numberIndex(precip, listOfdays, conv_threshold)

                        dataSet = xr.Dataset(
                                            {'numberIndex': numberIndex,
                                             'areaf':areaf})
                        myFuncs.save_file(dataSet, folder, fileName)


                    if metricFiles['pwad']:
                        fileName = model + '_pwad_' + experiment + '.nc'
                        o_area, o_pr = aggFuncs.calc_area_pr(precip, listOfdays, conv_threshold)

                        dataSet = xr.Dataset(
                                            {'o_area': o_area,
                                             'o_pr':o_pr})
                        myFuncs.save_file(dataSet, folder, fileName)


                    if metricFiles['rome']:
                        fileName = model + '_rome_' + experiment + '.nc'
                        dataSet = xr.Dataset(
                                            {'rome': aggFuncs.calc_rome(precip, listOfdays, conv_threshold)})
                        myFuncs.save_file(dataSet, folder, fileName)


                    if metricFiles['rome_n']:
                        n = 8
                        fileName = model + '_rome_n_' + experiment + '.nc'
                        dataSet = xr.Dataset(
                                            {'rome_n': aggFuncs.calc_rome_n(n, precip, listOfdays, conv_threshold)})
                        myFuncs.save_file(dataSet, folder, fileName)

                    



            if var == 'tas':
                tas = ds_regrid.tas-273.15
                tas.attrs['units']= 'deg (C)'
                    

                if metricFiles['tas_examples']:
                    fileName = model + '_tas_' + experiment + '.nc'
                    dataSet = xr.Dataset({
                        'tas_day': tas.isel(time=0), 
                        'tas_tMean': tas.mean(dim='time', keep_attrs=True)})
                    myFuncs.save_file(dataSet, folder, fileName)


                if metricFiles['tas_annual']:                
                    fileName = model + '_tas_annual_' + experiment + '.nc'
                    path = folder + '/' + fileName 
                    dataSet = xr.Dataset({'tas_annual': tas.resample(time='Y').mean(dim='time', keep_attrs=True).weighted(np.cos(np.deg2rad(tas.lat))).mean(dim=('lat','lon'), keep_attrs=True)})
                    myFuncs.save_file(dataSet, folder, fileName)





            if var == 'hus':
                hus = ds_regrid.hus


                if metricFiles['hus_examples']:
                    fileName = model + '_hus_examples' + experiment + '.nc'

                    hus_day, hus_tMean = husFuncs.get_hus_snapshot_tMean(hus)
                    dataSet = xr.Dataset(
                        {'hus_day': hus_day, 
                        'hus_tMean': hus_tMean})
                    myFuncs.save_file(dataSet, folder, fileName)



                if metricFiles['hus_sMean']:
                    fileName = model + '_hus_sMean' + experiment + '.nc'

                    hus_sMean = husFuncs.calc_hus_sMean(hus)
                    fileName = model + '_hus_examples' + experiment + '.nc'
                    dataSet = xr.Dataset(
                        {'hus_sMean': hus_sMean})
                    myFuncs.save_file(dataSet, folder, fileName)





    stop = timeit.default_timer()
    print('model: {} took {} minutes to finsih'.format(model, (stop-start)/60))















