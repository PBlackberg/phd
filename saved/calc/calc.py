
#                 print('finished')

# ds_rxday = xr.Dataset({'rx1day': da, 'rx5day': rx5day})

#     da.attrs['units']= 'mm day' + mF.get_super('-1')



# F_pr10.attrs['units'] = 'Nb'

# ds_F_pr10 = xr.Dataset(
# data_vars = {'F_pr10': F_pr10},
# attrs = {'description': 'Number of gridboxes in daily scene exceeding 10 mm/day'}
#     )







#     switch = {'artificial': False, 'random': False,
#               'cop': False, 'cop_mod': False, 'sic': False, 'rom_limod': False, 'rom_el': False,
#               'iorg': False, 'scai': False, 'rom': False, 'basics': True,
#               'boundary': True}



# def pr_percentiles(precip):
#     percentiles = [0.95, 0.97, 0.99]
#     ds_prPercentiles = xr.Dataset()

#     for percentile in percentiles:
#         pr = precip.quantile(percentile, dim=('lat', 'lon'), keep_attrs=True)
#         pr_data = xr.DataArray(data=pr.data, dims=['time'], coords={'time': precip.time.data})
#         pr_data.attrs['units'] = 'mm day⁻¹'
#         ds_prPercentiles[f'pr{int(percentile * 100)}'] = pr_data

#     return ds_prPercentiles




# def pr_percentiles(precip):
#     pr95 = precip.quantile(0.95,dim=('lat','lon'),keep_attrs=True)
#     pr95 = xr.DataArray(
#         data = pr95.data,
#         dims = ['time'],
#         coords = {'time': precip.time.data}, 
#         attrs = {'units':'mm day' + chr(0x207B) + chr(0x00B9)}
#         )

#     pr97 = precip.quantile(0.97,dim=('lat','lon'),keep_attrs=True)
#     pr97 = xr.DataArray(
#         data = pr97.data,
#         dims = ['time'],
#         coords = {'time': precip.time.data},
#         attrs = {'units':'mm day' + chr(0x207B) + chr(0x00B9)}
#         )

#     pr99 = precip.quantile(0.99,dim=('lat','lon'),keep_attrs=True)
#     pr99 = xr.DataArray(
#         data = pr99.data,
#         dims = ['time'],
#         coords = {'time': precip.time.data},
#         attrs = {'units':'mm day' + chr(0x207B) + chr(0x00B9)}
#         )
        
#     ds_prPercentiles = xr.Dataset(
#         data_vars = {'pr95': pr95, 
#                      'pr97': pr97, 
#                      'pr99': pr99}
#         ) 
#     return ds_prPercentiles

# def pr_MeanPercentiles(precip):
#     pr95 = precip.quantile(0.95,dim=('lat','lon'),keep_attrs=True)
#     aWeights = np.cos(np.deg2rad(precip.lat))
#     pr95Mean = precip.where(precip>= pr95).weighted(aWeights).mean(dim=('lat', 'lon'))
#     pr95Mean = xr.DataArray(
#         data = pr95Mean.data,
#         dims = ['time'],
#         coords = {'time': precip.time.data}, 
#         attrs = {'units':'mm day' + chr(0x207B) + chr(0x00B9)}
#         )

#     pr97 = precip.quantile(0.97,dim=('lat','lon'),keep_attrs=True)
#     aWeights = np.cos(np.deg2rad(precip.lat))
#     pr97Mean = precip.where(precip>= pr97).weighted(aWeights).mean(dim=('lat', 'lon'))
#     pr97Mean = xr.DataArray(
#         data = pr97Mean.data,
#         dims = ['time'],
#         coords = {'time': precip.time.data}, 
#         attrs = {'units':'mm day' + chr(0x207B) + chr(0x00B9)}
#         )
    
#     pr99 = precip.quantile(0.99,dim=('lat','lon'),keep_attrs=True)
#     aWeights = np.cos(np.deg2rad(precip.lat))
#     pr99Mean = precip.where(precip>= pr99).weighted(aWeights).mean(dim=('lat', 'lon'))
#     pr99Mean = xr.DataArray(
#         data = pr99Mean.data,
#         dims = ['time'],
#         coords = {'time': precip.time.data}, 
#         attrs = {'units':'mm day' + chr(0x207B) + chr(0x00B9)}
#         )
    
#     ds_prPercentiles = xr.Dataset(
#         data_vars = {'pr95': pr95Mean, 
#                      'pr97': pr97Mean, 
#                      'pr99': pr99Mean}
#         ) 
#     return ds_prPercentiles

# def F_pr10(precip):
#     precip = precip.resample(time='M').mean(dim='time', keep_attrs=True)
#     mask = xr.where(precip>10,1,0)
#     F_pr10 = (mask).sum(dim=('lat','lon'))
#     F_pr10.attrs['units'] = 'Nb'

#     ds_F_pr10 = xr.Dataset(
#     data_vars = {'F_pr10': F_pr10},
#     attrs = {'description': 'Number of gridboxes in daily scene exceeding 10 mm/day'}
#         )
#     return ds_F_pr10




                # # load data
                # if run_on_gadi:
                #     if dataset == 'GPCP':
                #         from obs_variables import *
                #         precip = get_GPCP(institutes[model], model, experiment)['precip']
                    
                #     if np.isin(models_cmip5, dataset).any():
                #         from cmip5_variables import *
                #         precip = get_pr(institutes[model], model, experiment)['precip']
                    
                #     if run_on_gadi and np.isin(models_cmip6, dataset).any():
                #         from cmip6_variables import *
                #         precip = get_pr(institutes[model], model, experiment)['precip']
                # else:
                #     precip = get_dsvariable('precip', dataset, experiment, timescale = 'daily')['precip']


                # # Calculate diagnostics and put into dataset
                # # ds_rxday = rxday(precip)
                # # ds_prPercentiles = pr_percentiles(precip)
                # ds_prMeanPercentiles = pr_MeanPercentiles(precip)
                # # ds_F_pr10 = F_pr10(precip)


                # # save
                # save_rxday = False
                # save_prPercentiles = False
                # save_prMeanPercentiles = True
                # save_F_pr10 = False


                # if np.isin(models_cmip5, dataset).any():
                #     project = 'cmip5'
                # elif np.isin(models_cmip6, dataset).any():
                #     project = 'cmip6'
                # elif np.isin(observations, dataset).any():
                #     project = 'obs'
                # folder_save = home + '/data/' + project + '/' + 'metrics_' + project + '_' + resolutions[0] + '/' + dataset 


                # if save_rxday:
                #     fileName = dataset + '_rxday_' + experiment + '_' + resolutions[0] + '.nc'
                #     save_file(ds_rxday, folder_save, fileName)

                # if save_prPercentiles:
                #     fileName = dataset + '_prPercentiles_' + experiment + '_' + resolutions[0] + '.nc'
                #     save_file(ds_prPercentiles, folder_save, fileName)

                # if save_prMeanPercentiles:
                #     fileName = dataset + '_prMeanPercentiles_' + experiment + '_' + resolutions[0] + '.nc'
                #     save_file(ds_prMeanPercentiles, folder_save, fileName)

                # if save_F_pr10 :
                #     fileName = dataset + '_F_pr10_' + experiment + '_' + resolutions[0] + '.nc'
                #     save_file(ds_F_pr10, folder_save, fileName)
