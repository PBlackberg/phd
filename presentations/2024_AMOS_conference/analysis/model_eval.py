'''
# ---------------------
#   Model evaluation
# ---------------------
This scripts compares model data distribution with observation data distributions and ranks closeness to observations
Metrics:
From time-mean spatial distribution
    MAE
    RMSE
    ks-statistic

From spatial-mean time series
    Time-mean difference
    Variance difference
'''


# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import numpy as np
import xarray as xr
from scipy.stats import ks_2samp


# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys



# ------------------------
#   Calculate metric
# ------------------------
def get_mae(da_model, da_obs):
    ''' Mean absolute error '''
    return np.abs((da_model - da_obs)).mean()

def get_d_squared(da_model, da_obs):
    return (da_model - da_obs) ** 2

def get_rmse(da_model, da_obs):
    ''' Root Mean Squared Error '''
    return np.round(np.sqrt(get_d_squared(da_model, da_obs).mean()), 2)    

def get_ks_2sample(da_model, da_obs):
    ''' Kolmogorov-Smirnov statistic '''
    da_model_flat = da_model.values.flatten()
    da_obs_flat = da_obs.values.flatten()
    ks_statistic, p_value = ks_2samp(da_obs_flat, da_model_flat)
    return ks_statistic, p_value



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':                   
    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import myVars               as mV   
    import myFuncs              as mF     
    import myFuncs_plots        as mFp

    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import get_data.variable_data as vD

    sys.path.insert(0, f'{os.getcwd()}/util-calc')
    import ls_state.means_calc  as mC

    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import map_plot             as mP
    import bar_plot             as bP



    var_name = 'pr'
    units = 'mm/day'



    switch = {'test_sample': False, 'ocean_mask': False}



    print(f'{var_name} evaluation started')
# ---------------------------------------------------------------------------------- Remove temporary files --------------------------------------------------------------------------------------------------- #
    remove_temp_data, remove_plots, exit_after = True, True, True   # removes the temporary datasets and plots created in sections of this script
    if remove_temp_data:
        mF.remove_test_data()
    if remove_plots:
        mFp.remove_test_plots()
    if exit_after:
        exit()

# ------------------------------------------------------------------------------------ tMean: MAE, RMSE, ks-statistic --------------------------------------------------------------------------------------------------- #
    run_it = False
    if run_it:
        path = '/scratch/w40/cb4968/test/pr_tMean_all.nc' # file with metric for all datasets
        try:
            ds = xr.open_dataset(path)
            print('loaded tMean from temp file')
        except:
            print(f'Creating xarray dataset with tMean from all datasets')
            ds = xr.Dataset()
            for dataset, experiment in mF.run_dataset(var = var_name): 
                # print(f'getting {dataset} from experiment: {experiment}')
                da, region = vD.get_variable_data(switch, var_name, dataset, experiment)
                if dataset in mV.observations:
                    if mV.obs_years:
                        print(f'obs year range: {mV.obs_years[0]}')
                ds[dataset] = da.mean(dim = 'time')
            ds.to_netcdf(path, mode = 'w')
        # print(ds)
        # exit()

        plot = False                                                          # mean precipitation distribution (models and obs)
        if plot:
            print('plotting: model and obs')
            vmin, vmax = 0, 10
            fig, axes = mP.plot_dsScenes(ds, label = f'[{units}]', title = f'{var_name} tMean (obs: {mV.obs_years[0]})', vmin = vmin, vmax = vmax, cmap = 'Blues')
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'{var_name}_tMean.png')
        plot = False                                                                    # difference (model - obs)
        if plot:
            print('plotting: (model-obs)')
            vmin, vmax = -10, 10
            ds_new = xr.Dataset()
            for dataset in ds.data_vars:
                if dataset != mV.observations[0]:
                    ds_new[dataset] = ds[dataset] - ds[mV.observations[0]]
            fig, axes = mP.plot_dsScenes(ds_new, label = f'[{units}]', title = f'(model - obs) {var_name} tMean (obs: {mV.obs_years[0]})', vmin = vmin, vmax = vmax, cmap = 'RdBu_r', variable_list = ds_new.data_vars)
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'd_{var_name}_tMean.png')
        plot = False                                                                  # squared difference (model - obs)
        if plot:
            print('plotting: (model-obs)^2')
            vmin, vmax = 0, 50
            ds_new = xr.Dataset()
            for dataset in ds.data_vars:
                if dataset != mV.observations[0]:
                    ds_new[dataset] = get_d_squared(da_model = ds[dataset], da_obs = ds[mV.observations[0]])
            fig, axes = mP.plot_dsScenes(ds_new, label = f'[{units}]', title = f'(model - obs)^2 {var_name} tMean (obs: {mV.obs_years[0]})', vmin = vmin, vmax = vmax, cmap = 'Reds', variable_list = ds_new.data_vars)
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'd2_{var_name}_tMean.png')
        plot = False                                                                        # MAE (model - obs)
        if plot:
            print('plotting: MAE')
            ds_new = xr.Dataset()
            for dataset in ds.data_vars:
                if dataset != mV.observations[0]:
                    ds_new[dataset] = get_mae(da_model = ds[dataset], da_obs = ds[mV.observations[0]])
            fig, ax = bP.plot_dsBar(ds_new, title = f'MAE_{var_name}', ylabel = f'[{units}]')
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'MAE_{var_name}_barplot.png')
        plot = False                                                                         # RMSE (model - obs)
        if plot:
            print('plotting: RMSE')
            ds_new = xr.Dataset()
            for dataset in ds.data_vars:
                if dataset != mV.observations[0]:
                    ds_new[dataset] = get_rmse(da_model = ds[dataset], da_obs = ds[mV.observations[0]])
            fig, ax = bP.plot_dsBar(ds_new, title = f'RMSE_{var_name}', ylabel = f'[{units}]')
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'RMSE_{var_name}_barplot.png')
        plot = False                                                              # ks-statistic k-s two sample test (obs, model)
        if plot:
            print('plotting: ks-statistic from ks - 2 sample test')
            ds_new = xr.Dataset()
            for dataset in ds.data_vars:
                if dataset != mV.observations[0]:
                    ks_statistic, p_value = get_ks_2sample(da_model = ds[dataset], da_obs = ds[mV.observations[0]])
                    ds_new[dataset] = ks_statistic
            fig, ax = bP.plot_dsBar(ds_new, title = f'ks-statistic_{var_name}', ylabel = f'[]')
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'ks-statistic_{var_name}_barplot.png')
        plot = False                                                                # p-value k-s two sample test (obs, model)
        if plot:
            print('plotting: p-value from ks - 2 sample test')
            ds_new = xr.Dataset()
            for dataset in ds.data_vars:
                if dataset != mV.observations[0]:
                    ks_statistic, p_value = get_ks_2sample(da_model = ds[dataset], da_obs = ds[mV.observations[0]])
                    ds_new[dataset] = p_value
            print(ds_new['TaiESM1'])
            fig, ax = bP.plot_dsBar(ds_new, title = f'p-value_{var_name}', ylabel = f'[]')
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'p-value_{var_name}_barplot.png')


# ------------------------------------------------------------------------------- sMean: mean and variance --------------------------------------------------------------------------------------------------- #
    run_it = False
    if run_it:
        path = '/scratch/w40/cb4968/test/pr_sMean_all.nc' # file with metric for all datasets



















