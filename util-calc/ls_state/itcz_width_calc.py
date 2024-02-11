'''
# ------------------------
#  ITCZ width calculation
# ------------------------
Difference in latitude between time-mean vertical pressure velocity at 500 hpa
'''



# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np


# ------------------------
#    Calculate metric
# ------------------------
# -------------------------------------------------------------------------------------- itcz width ----------------------------------------------------------------------------------------------------- #
def itcz_width_sMean(da):
    da = da.mean(dim = ['lon'])
    itcz = xr.where(da < 0, 1, np.nan).compute()
    itcz_lats = itcz * da['lat']
    max_lats = itcz_lats.max(dim='lat', skipna=True)
    min_lats = itcz_lats.min(dim='lat', skipna=True)
    return max_lats - min_lats                                # range of lats

def itcz_width(da):
    alist = da.mean(dim = ['time', 'lon']).compute()
    itcz_lats = alist.where(alist < 0, drop = True)['lat']          # ascending region
    return itcz_lats.max() - itcz_lats.min()                        # range of lats


# ----------------------------------------------------------------------------------- area fraction of descent ----------------------------------------------------------------------------------------------------- #
def get_fraction_descent(da, dims):
    da = da.mean(dim = 'time')
    da = (xr.where(da > 0, 1, 0) * dims.aream).sum()                # area of descending region
    return (da / dims.aream.sum())*100                              # fraction of descending region

# @mF.timing_decorator()
# def get_area(da, dim):
#     ''' Area covered in domain [% of domain]. Used for area of ascent and descent (wap) '''
#     mask = xr.where(da > 0, 1, 0)
#     area = ((mask*dim.aream).sum(dim=('lat','lon')) / np.sum(dim.aream)) * 100
#     return area



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt

    import os
    import sys
    home = os.path.expanduser("~")                                        
    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import myFuncs_plots as mFp   
    import myFuncs as mF
    import myVars as mV

    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import get_data.variable_data as vD

    print('testing itcz width started')
    remove_plots = True
    mFp.remove_test_plots() if remove_plots else None


    # ----------------------------------------------------------------------------------- get data ----------------------------------------------------------------------------------------------------- #
    # ds = xr.open_dataset(f'{mV.folder_scratch}/sample_data/wap/cmip6/TaiESM1_wap_monthly_historical_regridded_144x72.nc')
    # da = ds['wap'].sel(plev = 500e2)
    # print(ds)
    # print(da)

    switch = {'test_sample': False, 'ocean_mask': False, '700hpa': False, '500hpa': True}
    var_name = 'wap'
    if switch['700hpa']:
        height = '700'
    if switch['500hpa']:
        height = '500'
    print(f'using {height} hpa')
    # dataset = mV.datasets[0]
    experiment = mV.experiments[0]
    # print(f'running {dataset} {experiment}')
    # da, region = vD.get_variable_data(switch = switch, var_name = var_name, dataset = dataset, experiment = experiment)


    for dataset in mV.datasets:
        da, region = vD.get_variable_data(switch = switch, var_name = var_name, dataset = dataset, experiment = experiment)
        # ----------------------------------------------------------------------------------- visualize ----------------------------------------------------------------------------------------------------- #
        plot_snapshot_wap500 = False
        if plot_snapshot_wap500:
            # nan_count = np.sum(np.isnan(da.compute()))
            # print(f'Number of NaN values in the array: {nan_count.data}')
            da_mean = da.isel(time=0)
            fig, ax = mFp.plot_scene(da_mean, cmap = 'RdBu', ax_title = f'{dataset}', fig_title = f'{dataset} {experiment} {mV.resolutions[0]} snapshot wap_500', vmin = -60, vmax = 60)
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'{dataset}_{experiment}_{mV.resolutions[0]}_wap{height}_snapshot')
            # exit()


        plot_mean_wap500 = False
        if plot_mean_wap500:
            # nan_count = np.sum(np.isnan(da.compute()))
            # print(f'Number of NaN values in the array: {nan_count.data}')
            da_mean = da.mean(dim = 'time')
            fig, ax = mFp.plot_scene(da_mean, cmap = 'RdBu', ax_title = f'{dataset}', fig_title = f'{dataset} {experiment} {mV.resolutions[0]} time-mean wap_500', vmin = -60, vmax = 60)
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'{dataset}_{experiment}_{mV.resolutions[0]}_wap{height}_tMean')
            # exit()

        plot_tMean_lonMean_wap500 = False
        if plot_tMean_lonMean_wap500:
            da_mean = da.mean(dim = 'time')
            da_mean_lat = da_mean.mean(dim = 'lon')
            x = da_mean_lat.values
            y = da_mean_lat['lat'].values
            fig = plt.figure(figsize=(8, 6))
            plt.plot(x, y, 'k')
            plt.xlabel('time-lon mean wap')
            plt.ylabel('lat')
            plt.axvline(x=0, color='k', linestyle='--')
            plt.axhline(y=0, color='k', linestyle='--')
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'{dataset}_{experiment}_{mV.resolutions[0]}_wap{height}_tMean_lonMean')
            # exit()

        plot_area_descent = False
        if plot_area_descent:
            da_mean = da.mean(dim = 'time')
            da_mean_descent = xr.where(da_mean > 0, 1, np.nan)
            # print(da_mean_descent)
            fig, ax = mFp.plot_scene(da_mean_descent, ax_title = f'{dataset}', fig_title = f'{dataset} {experiment} {mV.resolutions[0]} time-mean wap{height}_descent')
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'{dataset}_{experiment}_{mV.resolutions[0]}_wap{height}_area_descent')



        # ----------------------------------------------------------------------------------- Calculate ----------------------------------------------------------------------------------------------------- #
        test_calc = False
        if test_calc:
            dims = mF.dims_class(da)
            width_sMean = itcz_width_sMean(da)
            width = itcz_width(da)
            # descent_fraction = get_fraction_descent(da, dims)

            print(f'The itcz width per time step: {np.round(width_sMean[0:5].data, 2)} degrees latitude')
            # print(f'The itcz width is: {np.round(width.data, 2)} degrees latitude')
            print(f'The itcz width is: {width} degrees latitude')
            # print(f'The fraction of descending motion is: {np.round(descent_fraction.data, 2)} % of the tropical domain')
            # exit()


        # ------------------------------------------------------------------------------- Calculate difference ----------------------------------------------------------------------------------------------------- #
        test_calc = False
        if test_calc:
            da, region = vD.get_variable_data(switch = switch, var_name = var_name, dataset = dataset, experiment = mV.experiments[0])
            dims = mF.dims_class(da)
            width_hist = itcz_width(da)
            # width_sMean = itcz_width_sMean(da)
            # descent_fraction = get_fraction_descent(da, dims)
            # nan_count = np.sum(np.isnan(da.compute()))
            # print(f'Number of NaN values in the array: {nan_count.data}')

            da, region = vD.get_variable_data(switch = switch, var_name = var_name, dataset = dataset, experiment = mV.experiments[1])
            dims = mF.dims_class(da)
            width_warm = itcz_width(da)
            # width_sMean = itcz_width_sMean(da)
            # descent_fraction = get_fraction_descent(da, dims)
            # nan_count = np.sum(np.isnan(da.compute()))
            # print(f'Number of NaN values in the array: {nan_count.data}')

            width_diff = width_warm - width_hist

            # print(f'ITCZ width for model: {dataset} {mV.experiments[0]} is {width_hist.data} degrees')
            # print(f'ITCZ width for model: {dataset} {mV.experiments[0]} is {width_warm.data} degrees')
            print(f'Change in ITCZ width is {np.round(width_diff.data, 2)} degrees for model: {dataset}')





