import intake
import xarray as xr
import numpy as np
import timeit

import myFuncs
import mseFuncs
import lwFuncs
import swFuncs
import sefFuncs


models = [
        'MPI-ESM1-2-HR', # 1
        # 'ACCESS',     # 2
        ]

variables = [
            'mse',
            'lw', 
            'sw',
            'sef'
            ]

experiments = [
                'historical', 
                # 'rcp85'
            ]

metricFiles = {
                    'mse_test':True, 
                    'mse_tMean':True, 
                    'mse_var': True, 
                    'lw_test':True, 
                    'lw_tMean':True,
                    'lw_anom':True,
                    'sw_test':True,
                    'sw_tMean':True,
                    'sw_anom':True,
                    'sef_test':True,
                    'sef_tMean':True,
                    'sef_anom':True
    }



for model in models:
    folder = '/g/data/k10/cb4968/data/cmip6/' + model
    start = timeit.default_timer()

    for var in variables:
        for experiment_id in experiments:

            if experiment_id == 'historical':
                period=slice('1970-01','1999-12')
                member_id='r1i1p1f1'
                
            elif experiment_id == 'rcp85':
                period=slice('2070-01','2099-12')
                member_id='r1i1p1f1'




# ------------------------------------ mse ----------------------------------------------------


            if var == 'mse':
                table_id='day'

                variable_id = 'ta'
                ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                                source_id=model, 
                                                experiment_id=experiment_id, 
                                                member_id=member_id, 
                                                variable_id=variable_id, 
                                                table_id=table_id).to_dataset_dict()
                ta = ds_dict[list(ds_dict.keys())[-1]].ta.sel(time=period, lon=slice(0,360),lat=slice(-30,30))-273.15
                ta.attrs['units']= 'C\xb0'

                variable_id = 'zg'
                ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                                source_id=model, 
                                                experiment_id=experiment_id, 
                                                member_id=member_id, 
                                                variable_id=variable_id, 
                                                table_id=table_id).to_dataset_dict()

                zg = ds_dict[list(ds_dict.keys())[-1]].zg.sel(time=period, lon=slice(0,360),lat=slice(-30,30))

                variable_id = 'hus'
                ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                                source_id=model, 
                                                experiment_id=experiment_id, 
                                                member_id=member_id, 
                                                variable_id=variable_id, 
                                                table_id=table_id).to_dataset_dict() 
                                                
                hus = ds_dict[list(ds_dict.keys())[-1]].hus.sel(time=period, lon=slice(0,360),lat=slice(-30,30))


                c_p = 1.005
                L_v = 2.256e6
                mse = (c_p*ta + zg + L_v*hus)

                if metricFiles['mse_test']:
                    fileName = model + '_mse_test_' + experiment_id + '.nc'
                    dataSet = xr.Dataset({
                                        'mse_test': mse.isel(time=slice(0,4)), 
                                        'ta_test': ta.isel(time=slice(0,4)), 
                                        'zg_test': zg.isel(time=slice(0,4)),
                                        'hus_test': hus.isel(time=slice(0,4))})
                    myFuncs.save_file(dataSet, folder, fileName)



                if metricFiles['mse_tMean']:
                    fileName = model + '_mse_tMean_' + experiment_id + '.nc'
                    dataSet = xr.Dataset({'mse_tMean': mseFuncs.mse_tMean(mse)})
                    myFuncs.save_file(dataSet, folder, fileName)



                if metricFiles['mse_var']:
                    fileName = model + '_mse_var_' + experiment_id + '.nc'
                    dataSet = xr.Dataset({
                                        'mse_var': mseFuncs.mse_var(mse),
                                        'mse_anom': mseFuncs.mse_anom(mse),
                                        'mse_sMean': mseFuncs.mse_sMean(mse)})
                    myFuncs.save_file(dataSet, folder, fileName)






# ------------------------------------ lw ----------------------------------------------------


            if var == 'lw':
                
                table_id='E3hr'
                variable_id = 'rlut'
                ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                                source_id=model, 
                                                experiment_id=experiment_id, 
                                                member_id=member_id, 
                                                variable_id=variable_id, 
                                                table_id=table_id).to_dataset_dict()

                rlut_3hr = ds_dict[list(ds_dict.keys())[-1]].rlut.sel(time=period, lon=slice(0,360),lat=slice(-30,30))
                rlut = rlut_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)


                table_id='3hr'
                variable_id = 'rlds'
                ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                                source_id=model, 
                                                experiment_id=experiment_id, 
                                                member_id=member_id, 
                                                variable_id=variable_id, 
                                                table_id=table_id).to_dataset_dict()

                rlds_3hr = ds_dict[list(ds_dict.keys())[-1]].rlds.sel(time=period, lon=slice(0,360),lat=slice(-30,30))
                rlds = rlds_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)



                table_id='3hr'
                variable_id = 'rlus'
                ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                                source_id=model, 
                                                experiment_id=experiment_id, 
                                                member_id=member_id, 
                                                variable_id=variable_id, 
                                                table_id=table_id).to_dataset_dict()

                rlus_3hr = ds_dict[list(ds_dict.keys())[-1]].rlus.isel(time=slice(43800*8+8*28+8, (43800+10950)*8+8*36)).sel(lon=slice(0,360),lat=slice(-30,30))
                rlus = rlus_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)




                netlw = rlus - rlds - rlut


                if metricFiles['lw_test']:
                    fileName = model + '_lw_test_' + experiment_id + '.nc'
                    dataSet = xr.Dataset({
                                        'netlw_test': netlw.isel(time=slice(0,4)), 
                                        'rlut_test': rlut.isel(time=slice(0,4)), 
                                        'rlds_test': rlds.isel(time=slice(0,4)),
                                        'rlus_test': rlus.isel(time=slice(0,4))})
                    myFuncs.save_file(dataSet, folder, fileName)


                if metricFiles['lw_tMean']:
                    fileName = model + '_lw_tMean_' + experiment_id + '.nc'
                    dataSet = xr.Dataset({'lw_tMean': swFuncs.lw_tMean(mse)})
                    myFuncs.save_file(dataSet, folder, fileName)



                if metricFiles['lw_anom']:
                    fileName = model + '_lw_anom_' + experiment_id + '.nc'
                    dataSet = xr.Dataset({
                                        'lw_anom': swFuncs.lw_anom(mse),
                                        'lw_sMean': swFuncs.lw_sMean(mse)})
                    myFuncs.save_file(dataSet, folder, fileName)






# ------------------------------------ sw ----------------------------------------------------


            if var == 'sw':
                
                table_id='E3hr'
                variable_id = 'rsut'
                ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                                source_id=model, 
                                                experiment_id=experiment_id, 
                                                member_id=member_id, 
                                                variable_id=variable_id, 
                                                table_id=table_id).to_dataset_dict()

                rsut_3hr = ds_dict[list(ds_dict.keys())[-1]].rsut.sel(time=period, lon=slice(0,360),lat=slice(-30,30))
                rsut = rlut_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)


                table_id='3hr'
                variable_id = 'rsds'
                ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                                source_id=model, 
                                                experiment_id=experiment_id, 
                                                member_id=member_id, 
                                                variable_id=variable_id, 
                                                table_id=table_id).to_dataset_dict()

                rsds_3hr = ds_dict[list(ds_dict.keys())[-1]].rsds.sel(time=period, lon=slice(0,360),lat=slice(-30,30))
                rsds = rsds_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)



                table_id='3hr'
                variable_id = 'rsus'
                ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                                source_id=model, 
                                                experiment_id=experiment_id, 
                                                member_id=member_id, 
                                                variable_id=variable_id, 
                                                table_id=table_id).to_dataset_dict()

                rsus_3hr = ds_dict[list(ds_dict.keys())[-1]].rsus.isel(time=slice(43800*8+8*28+8, (43800+10950)*8+8*36)).sel(lon=slice(0,360),lat=slice(-30,30))
                rsus = rlus_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)




                netsw = rsus - rsds - rsut


                if metricFiles['sw_test']:
                    fileName = model + '_sw_test_' + experiment_id + '.nc'
                    dataSet = xr.Dataset({
                                        'netsw_test': netsw.isel(time=slice(0,4)), 
                                        'rsut_test': rsut.isel(time=slice(0,4)), 
                                        'rsds_test': rsds.isel(time=slice(0,4)),
                                        'rsus_test': rsus.isel(time=slice(0,4))})
                    myFuncs.save_file(dataSet, folder, fileName)


                if metricFiles['sw_tMean']:
                    fileName = model + '_sw_tMean_' + experiment_id + '.nc'
                    dataSet = xr.Dataset({'sw_tMean': swFuncs.lw_tMean(mse)})
                    myFuncs.save_file(dataSet, folder, fileName)



                if metricFiles['sw_anom']:
                    fileName = model + '_sw_anom_' + experiment_id + '.nc'
                    dataSet = xr.Dataset({
                                        'sw_anom': swFuncs.lw_anom(mse),
                                        'sw_sMean': swFuncs.lw_sMean(mse)})
                    myFuncs.save_file(dataSet, folder, fileName)






# ------------------------------------ sef ----------------------------------------------------


            if var == 'sef':
        
                table_id='day'
                variable_id = 'hfls'
                ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                                source_id=model, 
                                                experiment_id=experiment_id, 
                                                member_id=member_id, 
                                                variable_id=variable_id, 
                                                table_id=table_id).to_dataset_dict()

                hfls = ds_dict[list(ds_dict.keys())[-1]].hfls.sel(time=period, lon=slice(0,360),lat=slice(-30,30))



                table_id='day'
                variable_id = 'hfss'
                ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                                source_id=model, 
                                                experiment_id=experiment_id, 
                                                member_id=member_id, 
                                                variable_id=variable_id, 
                                                table_id=table_id).to_dataset_dict()

                hfss = ds_dict[list(ds_dict.keys())[-1]].hfss.sel(time=period, lon=slice(0,360),lat=slice(-30,30))


                netsef = hfls + hfss



                if metricFiles['sef_test']:
                    fileName = model + '_sef_test_' + experiment_id + '.nc'
                    
                    hfls_test = hfls.isel(time=slice(0,4))
                    hfss_test = hfls.isel(time=slice(0,4))
                    del hfls_test.encoding['chunksizes']
                    del hfss_test.encoding['chunksizes']

                    dataSet = xr.Dataset({
                                        'netsef_test': netsef.isel(time=slice(0,4)), 
                                        'hfls_test': hfls_test, 
                                        'hfss_test': hfss_test})
                    myFuncs.save_file(dataSet, folder, fileName)


                if metricFiles['sef_tMean']:
                    fileName = model + '_sef_tMean_' + experiment_id + '.nc'
                    dataSet = xr.Dataset({'sef_tMean': sefFuncs.sef_tMean(mse)})
                    myFuncs.save_file(dataSet, folder, fileName)



                if metricFiles['sef_anom']:
                    fileName = model + '_sef_anom_' + experiment_id + '.nc'
                    dataSet = xr.Dataset({
                                        'sef_anom': sefFuncs.lw_anom(mse),
                                        'sef_sMean': sefFuncs.lw_sMean(mse)})
                    myFuncs.save_file(dataSet, folder, fileName)







    stop = timeit.default_timer()
    print('model: {} took {} minutes to finsih'.format(model, (stop-start)/60))





