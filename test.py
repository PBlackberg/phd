# import numpy as np

# # Sample conv_obj matrix
# conv_obj = np.array([[1, 2, 3],
#                      [4, 5, 6],
#                      [7, 8, 9]])

# # Sample labels list
# labels = [3, 6, 9]

# # Applying the comparison
# result = (conv_obj==labels)*1

# # Print the original matrix and the result
# print("Original matrix (conv_obj):")
# print(conv_obj)
# print("\nResult of (conv_obj==labels)*1:")
# print(result)



    #         switch = {'test_sample': False, 'ocean_mask': False, '700hpa': False, '500hpa': True}
    #         var_name = 'wap'
    #         if switch['700hpa']:
    #             height = '700'
    #         if switch['500hpa']:
    #             height = '500'
    #         print(f'using {height} hpa')
    #         # dataset = mV.datasets[0]
    #         experiment = mV.experiments[0]
    #         # print(f'running {dataset} {experiment}')
    #         # da, region = vD.get_variable_data(switch = switch, var_name = var_name, dataset = dataset, experiment = experiment)


    # for dataset in mV.datasets:
    #     da, region = vD.get_variable_data(switch = switch, var_name = var_name, dataset = dataset, experiment = experiment)
    #     # ----------------------------------------------------------------------------------- visualize ----------------------------------------------------------------------------------------------------- #
    #     plot_snapshot_wap500 = False
    #     if plot_snapshot_wap500:
    #         # nan_count = np.sum(np.isnan(da.compute()))
    #         # print(f'Number of NaN values in the array: {nan_count.data}')
    #         da_mean = da.isel(time=0)
    #         fig, ax = mFp.plot_scene(da_mean, cmap = 'RdBu', ax_title = f'{dataset}', fig_title = f'{dataset} {experiment} {mV.resolutions[0]} snapshot wap_500', vmin = -60, vmax = 60)
    #         mFp.show_plot(fig, show_type = 'save_cwd', filename = f'{dataset}_{experiment}_{mV.resolutions[0]}_wap{height}_snapshot')
    #         # exit()


    #     plot_mean_wap500 = False
    #     if plot_mean_wap500:
    #         # nan_count = np.sum(np.isnan(da.compute()))
    #         # print(f'Number of NaN values in the array: {nan_count.data}')
    #         da_mean = da.mean(dim = 'time')
    #         fig, ax = mFp.plot_scene(da_mean, cmap = 'RdBu', ax_title = f'{dataset}', fig_title = f'{dataset} {experiment} {mV.resolutions[0]} time-mean wap_500', vmin = -60, vmax = 60)
    #         mFp.show_plot(fig, show_type = 'save_cwd', filename = f'{dataset}_{experiment}_{mV.resolutions[0]}_wap{height}_tMean')
    #         # exit()

    #     plot_tMean_lonMean_wap500 = False
    #     if plot_tMean_lonMean_wap500:
    #         da_mean = da.mean(dim = 'time')
    #         da_mean_lat = da_mean.mean(dim = 'lon')
    #         x = da_mean_lat.values
    #         y = da_mean_lat['lat'].values
    #         fig = plt.figure(figsize=(8, 6))
    #         plt.plot(x, y, 'k')
    #         plt.xlabel('time-lon mean wap')
    #         plt.ylabel('lat')
    #         plt.axvline(x=0, color='k', linestyle='--')
    #         plt.axhline(y=0, color='k', linestyle='--')
    #         mFp.show_plot(fig, show_type = 'save_cwd', filename = f'{dataset}_{experiment}_{mV.resolutions[0]}_wap{height}_tMean_lonMean')
    #         # exit()

    #     plot_area_descent = False
    #     if plot_area_descent:
    #         da_mean = da.mean(dim = 'time')
    #         da_mean_descent = xr.where(da_mean > 0, 1, np.nan)
    #         # print(da_mean_descent)
    #         fig, ax = mFp.plot_scene(da_mean_descent, ax_title = f'{dataset}', fig_title = f'{dataset} {experiment} {mV.resolutions[0]} time-mean wap{height}_descent')
    #         mFp.show_plot(fig, show_type = 'save_cwd', filename = f'{dataset}_{experiment}_{mV.resolutions[0]}_wap{height}_area_descent')



    #     # ----------------------------------------------------------------------------------- Calculate ----------------------------------------------------------------------------------------------------- #
    #     test_calc = False
    #     if test_calc:
    #         dims = mF.dims_class(da)
    #         width_sMean = wap_itcz_width(da)
    #         width = itcz_width(da)
    #         # descent_fraction = get_fraction_descent(da, dims)

    #         print(f'The itcz width per time step: {np.round(width_sMean[0:5].data, 2)} degrees latitude')
    #         # print(f'The itcz width is: {np.round(width.data, 2)} degrees latitude')
    #         print(f'The itcz width is: {width} degrees latitude')
    #         # print(f'The fraction of descending motion is: {np.round(descent_fraction.data, 2)} % of the tropical domain')
    #         # exit()


    #     # ------------------------------------------------------------------------------- Calculate difference ----------------------------------------------------------------------------------------------------- #
    #     test_calc = False
    #     if test_calc:
    #         da, region = vD.get_variable_data(switch = switch, var_name = var_name, dataset = dataset, experiment = mV.experiments[0])
    #         dims = mF.dims_class(da)
    #         width_hist = itcz_width(da)
    #         # width_sMean = itcz_width_sMean(da)
    #         # descent_fraction = get_fraction_descent(da, dims)
    #         # nan_count = np.sum(np.isnan(da.compute()))
    #         # print(f'Number of NaN values in the array: {nan_count.data}')

    #         da, region = vD.get_variable_data(switch = switch, var_name = var_name, dataset = dataset, experiment = mV.experiments[1])
    #         dims = mF.dims_class(da)
    #         width_warm = itcz_width(da)
    #         # width_sMean = itcz_width_sMean(da)
    #         # descent_fraction = get_fraction_descent(da, dims)
    #         # nan_count = np.sum(np.isnan(da.compute()))
    #         # print(f'Number of NaN values in the array: {nan_count.data}')

    #         width_diff = width_warm - width_hist

    #         # print(f'ITCZ width for model: {dataset} {mV.experiments[0]} is {width_hist.data} degrees')
    #         # print(f'ITCZ width for model: {dataset} {mV.experiments[0]} is {width_warm.data} degrees')
    #         print(f'Change in ITCZ width is {np.round(width_diff.data, 2)} degrees for model: {dataset}')





