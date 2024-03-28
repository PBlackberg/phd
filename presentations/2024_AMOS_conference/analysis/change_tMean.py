'''
# ------------------------------------------------
#  Change with warming: Conv_org, MSE, ITCZ width
# ------------------------------------------------
This script plots changes in convective object position, ITCZ width, and MSE with warming
metrics:
    pr95
    wap500 (itcz width)
    MSE variance
'''



# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np


# ------------------------------------------------------------------------------------ imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")                                        
sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars as mV                                   

sys.path.insert(0, f'{os.getcwd()}/util-data')
import get_data.variable_data   as vD
import get_data.metric_data     as mD



# ------------------------
#   Calculate metric
# ------------------------
# ------------------------------------------------------------------------------------- object heatmap --------------------------------------------------------------------------------------------------- #
def get_difference_o_heatmap():
    ''' number of occurences of object in gridbox / total number of days '''
    path = '/scratch/w40/cb4968/test/o_heatmap_difference.nc' # file with metric for all datasets
    try:
        ds = xr.open_dataset(path)
        print('loaded tMean from temp file')
    except:
        print(f'Creating xarray dataset with tMean from all datasets')
        ds = xr.Dataset()
        folder = '/scratch/w40/cb4968/metrics/conv_org/o_heatmap_95thprctile/cmip6'
        for model in mV.models_cmip6:
            da_historical = xr.open_dataset(f'{folder}/{model}_o_heatmap_95thprctile_daily_historical_regridded.nc')['o_heatmap_95thprctile']
            da_warm = xr.open_dataset(f'{folder}/{model}_o_heatmap_95thprctile_daily_ssp585_regridded.nc')['o_heatmap_95thprctile']
            ds[model] = da_warm - da_historical
        ds.to_netcdf(path, mode = 'w')
    return ds

def get_difference_model_mean_o_heatmap():
    ''' number of occurences of object in gridbox / total number of days '''
    path = '/scratch/w40/cb4968/test/o_heatmap_model_mean_difference.nc' # file with metric for all datasets
    try:
        ds = xr.open_dataset(path)
        print('loaded tMean from temp file')
    except:
        print(f'Creating xarray dataset with tMean from all datasets')
        ds = xr.Dataset()
        folder = '/scratch/w40/cb4968/metrics/conv_org/o_heatmap_95thprctile/cmip6'
        for i, model in enumerate(mV.models_cmip6):
            da = xr.open_dataset(f'{folder}/{model}_o_heatmap_95thprctile_daily_historical_regridded.nc')['o_heatmap_95thprctile']
            if i == 0:
                da_historical = da
            else:
                da_historical = da_historical + da

        for i, model in enumerate(mV.models_cmip6):
            da = xr.open_dataset(f'{folder}/{model}_o_heatmap_95thprctile_daily_ssp585_regridded.nc')['o_heatmap_95thprctile']
            if i == 0:
                da_warm = da
            else:
                da_warm = da_warm + da
        
        ds['model_mean'] = (da_warm - da_historical) / len(mV.models_cmip6)
        ds.to_netcdf(path, mode = 'w')
    return ds


# --------------------------------------------------------------------------------- vertical pressure velocity --------------------------------------------------------------------------------------------------- #
def get_wap500():
    ''' number of occurences of object in gridbox / total number of days '''
    path = '/scratch/w40/cb4968/test/wap500.nc' # file with metric for all datasets
    try:
        ds = xr.open_dataset(path)
        print('loaded tMean from temp file')
    except:
        print(f'Creating xarray dataset with tMean from all datasets')
        ds = xr.Dataset()
        # folder = '/scratch/w40/cb4968/metrics/conv_org/o_heatmap_95thprctile/cmip6'
        for model in mV.models_cmip6:
            # da_historical = xr.open_dataset(f'{folder}/{model}_o_heatmap_95thprctile_daily_historical_regridded.nc')['o_heatmap_95thprctile']
            da, region = vD.get_variable_data({'test_sample': False, 'ocean_mask': False, '500hpa': True}, var_name= 'wap', dataset = model, experiment = 'historical')
            ds[model] = da.mean(dim='time')
        ds.to_netcdf(path, mode = 'w')
    return ds

def get_model_mean_wap500():
    ''' number of occurences of object in gridbox / total number of days '''
    path = '/scratch/w40/cb4968/test/wap500_model_mean.nc' # file with metric for all datasets
    try:
        ds = xr.open_dataset(path)
        print('loaded tMean from temp file')
    except:
        print(f'Creating xarray dataset with tMean from all datasets')
        ds = xr.Dataset()
        for i, model in enumerate(mV.models_cmip6):
            da, region = vD.get_variable_data({'test_sample': False, 'ocean_mask': False, '500hpa': True}, var_name= 'wap', dataset = model, experiment = 'historical')
            da = da.mean(dim='time')
            if i == 0:
                da_historical = da
            else:
                da_historical = da_historical + da
        ds['model_mean'] = da_historical / len(mV.models_cmip6)
        ds.to_netcdf(path, mode = 'w')
    return ds


# ------------------------------------------------------------------------------------------- MSE variance --------------------------------------------------------------------------------------------------- #
def get_difference_h_anom2():
    ''' number of occurences of object in gridbox / total number of days '''
    path = '/scratch/w40/cb4968/test/mse_variance_change.nc' # file with metric for all datasets
    try:
        ds = xr.open_dataset(path)
        print('loaded tMean from temp file')
    except:
        print(f'Creating xarray dataset with tMean from all datasets')
        ds = xr.Dataset()
        folder = '/scratch/w40/cb4968/metrics/h_anom2/h_anom2_tMean/cmip6'
        for model in mV.models_cmip6:
            # print(model)
            da_historical = xr.open_dataset(f'{folder}/{model}_h_anom2_tMean_monthly_historical_regridded.nc')['h_anom2_tMean']
            da_warm = xr.open_dataset(f'{folder}/{model}_h_anom2_tMean_monthly_ssp585_regridded.nc')['h_anom2_tMean']
            ds[model] = da_warm - da_historical
        ds.to_netcdf(path, mode = 'w')
    return ds


def get_difference_model_mean_h_anom2():
    ''' number of occurences of object in gridbox / total number of days '''
    path = '/scratch/w40/cb4968/test/mse_variance_model_mean_change.nc' # file with metric for all datasets
    try:
        ds = xr.open_dataset(path)
        print('loaded tMean from temp file')
    except:
        print(f'Creating xarray dataset with tMean from all datasets')
        ds = xr.Dataset()
        folder = '/scratch/w40/cb4968/metrics/h_anom2/h_anom2_tMean/cmip6'
        for i, model in enumerate(mV.models_cmip6):
            da = xr.open_dataset(f'{folder}/{model}_h_anom2_tMean_monthly_historical_regridded.nc')['h_anom2_tMean']
            if i == 0:
                da_historical = da
            else:
                da_historical = da_historical + da
        for i, model in enumerate(mV.models_cmip6):
            da = xr.open_dataset(f'{folder}/{model}_h_anom2_tMean_monthly_ssp585_regridded.nc')['h_anom2_tMean']
            if i == 0:
                da_warm = da
            else:
                da_warm = da_warm + da
        ds['model_mean'] = (da_warm - da_historical) / len(mV.models_cmip6)
        ds.to_netcdf(path, mode = 'w')
    return ds



# --------------------------------------------------------------------------------- vertical pressure velocity --------------------------------------------------------------------------------------------------- #
def get_difference_hur700():
    ''' number of occurences of object in gridbox / total number of days '''
    path = '/scratch/w40/cb4968/test/difference_hur700.nc' # file with metric for all datasets
    try:
        ds = xr.open_dataset(path)
        print('loaded tMean from temp file')
    except:
        print(f'Creating xarray dataset with tMean from all datasets')
        ds = xr.Dataset()
        for model in mV.models_cmip6:
            da_historical, region = vD.get_variable_data({'test_sample': False, 'ocean_mask': False, '700hpa': True}, var_name= 'hur', dataset = model, experiment = 'historical')
            da_warm, region = vD.get_variable_data({'test_sample': False, 'ocean_mask': False, '700hpa': True}, var_name= 'hur', dataset = model, experiment = 'ssp585')

            # da_historical, region = vD.get_variable_data({'test_sample': False, 'ocean_mask': False, 'vMean': True}, var_name= 'hur', dataset = model, experiment = 'historical')
            # da_warm, region = vD.get_variable_data({'test_sample': False, 'ocean_mask': False, 'vMean': True}, var_name= 'hur', dataset = model, experiment = 'ssp585')
            da_historical = da_historical.mean(dim='time')
            da_warm = da_warm.mean(dim='time')
            ds[model] = da_warm - da_historical
        ds.to_netcdf(path, mode = 'w')
    return ds


def get_difference_model_mean_hur700():
    ''' number of occurences of object in gridbox / total number of days '''
    path = '/scratch/w40/cb4968/test/difference_model_mean_hur700.nc' # file with metric for all datasets
    try:
        ds = xr.open_dataset(path)
        print('loaded tMean from temp file')
    except:
        print(f'Creating xarray dataset with tMean from all datasets')
        ds = xr.Dataset()
        for i, model in enumerate(mV.models_cmip6):
            da, region = vD.get_variable_data({'test_sample': False, 'ocean_mask': False, '700hpa': True}, var_name= 'hur', dataset = model, experiment = 'historical')
            if i == 0:
                da_historical = da.mean(dim='time')
            else:
                da_historical = da_historical + da.mean(dim='time')
                
        for i, model in enumerate(mV.models_cmip6):
            da, region = vD.get_variable_data({'test_sample': False, 'ocean_mask': False, '700hpa': True}, var_name= 'hur', dataset = model, experiment = 'ssp585')
            if i == 0:
                da_warm = da.mean(dim='time')
            else:
                da_warm = da_warm + da.mean(dim='time')
        ds['model_mean'] = (da_warm - da_historical) / len(mV.models_cmip6)
        # exit()
        ds.to_netcdf(path, mode = 'w')
    return ds





# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import myFuncs_plots            as mFp
    import myFuncs                  as mF       

    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import map_plot                 as mP



# ---------------------------------------------------------------------------------- Remove temporary files --------------------------------------------------------------------------------------------------- #
    remove_temp_data, remove_plots, exit_after = True, True, False   # removes the temporary datasets and plots created in sections of this script
    if remove_temp_data:
        mF.remove_test_data()
    if remove_plots:
        mFp.remove_test_plots()
    if exit_after:
        exit()


# -------------------------------------------------------------------------------------- tMean: heatmap --------------------------------------------------------------------------------------------------- #
    plot = False                                                                     # Difference in o_heatmap
    if plot:
        units = 'Frequency of occurence'
        ds = get_difference_o_heatmap()
        # print(ds)
        fig, axes = mP.plot_dsScenes(ds, label = units, title = f'o_heatmap_change', vmin = -0.15, vmax = 0.15, cmap = 'RdBu_r', variable_list = ds.data_vars)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'o_heatmap_change.png')
        print('plotted change in object heatmap (frequency of occurence)')

    plot = False                                                                 # Difference in o_heatmap (model-mean)
    if plot:
        units = 'Frequency of occurence'
        ds = get_difference_model_mean_o_heatmap()
        # print(ds)
        fig, axes = mP.plot_dsScenes(ds, label = units, title = '', vmin = -0.15, vmax = 0.15, cmap = 'RdBu_r', variable_list = ds.data_vars)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'o_heatmap_model_mean_change.png')
        print('plotted change in object heatmap (model mean) (frequency of occurence)')


# ----------------------------------------------------------------------------------------- tMean: ITCZ (wap) --------------------------------------------------------------------------------------------------- #
    plot = False                                                                     # Difference in ITCZ_width
    if plot:
        units = 'wap [hPa/day]'
        ds = get_wap500()
        fig, axes = mP.plot_dsScenes(ds, label = units, title = '', vmin = -60, vmax = 60, cmap = 'RdBu_r', variable_list = ds.data_vars)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'wap500_historical.png')
        print('plotted wap500')

    plot = False                                                                     # Difference in ITCZ_width
    if plot:
        units = 'wap [hPa/day]'
        ds = get_model_mean_wap500()
        fig, axes = mP.plot_dsScenes(ds, label = units, title = '', vmin = -60, vmax = 60, cmap = 'RdBu_r', variable_list = ds.data_vars)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'wap500_model_mean_historical.png')
        print('plotted wap500 model mean')


# ---------------------------------------------------------------------------------------- tMean: MSE variance --------------------------------------------------------------------------------------------------- #
    plot = False                                                                     # Difference in MSE variance
    if plot:
        units = r'h [(J/kg)$^{2}$]'
        ds = get_difference_h_anom2()
        fig, axes = mP.plot_dsScenes(ds, label = units, title = '', vmin = 0, vmax = 2.5e7, cmap = 'Reds', variable_list = ds.data_vars)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'h_anom2_change.png')
        print('plotted mse variance change')


    plot = False                                                                     # Difference in model mean MSE variance
    if plot:
        units = r'h [(J/kg)$^{2}$]'
        ds = get_difference_model_mean_h_anom2()
        fig, axes = mP.plot_dsScenes(ds, label = units, title = '', vmin = None, vmax = None, cmap = 'Reds', variable_list = ds.data_vars)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'h_anom2_model_mean_change.png')
        print('plotted mse model mean variance change')



# -------------------------------------------------------------------------------------- tMean: relative humidity --------------------------------------------------------------------------------------------------- #
    plot = True                                                                     # Difference in ITCZ_width
    if plot:
        units = 'hur_700hpa [%]'
        ds = get_difference_hur700()
        fig, axes = mP.plot_dsScenes(ds, label = units, title = '', vmin = -10, vmax = 10, cmap = 'RdBu_r', variable_list = ds.data_vars)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'difference_hur.png')
        print('plotted difference hur700')

    plot = True                                                                     # Difference in ITCZ_width
    if plot:
        units = 'hur_700hpa [%]'
        ds = get_difference_model_mean_hur700()
        fig, axes = mP.plot_dsScenes(ds, label = units, title = '', vmin = None, vmax = None, cmap = 'RdBu_r', variable_list = ds.data_vars)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'hur_model_mean_difference.png')
        print('plotted hur700 model mean')






