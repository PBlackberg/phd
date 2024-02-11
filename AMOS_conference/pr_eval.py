'''
# ---------------------
#      pr eval
# ---------------------
Could double check that 1998-2022 was used (not just the later period)
'''

models_closeObs = ['CESM2-WACCM', 'EC-Earth3', 'MIROC-ES2L', 'MIROC6', 'NorESM2-MM', 'TaiESM1'] #(basd on pr and pwad RMSE)


# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import numpy as np
import xarray as xr


# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars               as mV   
import myFuncs              as mF     
import myFuncs_plots        as mFp

sys.path.insert(0, f'{os.getcwd()}/util-plot')
import map_plot             as mP
import bar_plot             as bP

sys.path.insert(0, f'{os.getcwd()}/util-data')
import get_data.variable_data as vD
import get_data.metric_data     as mD



# ------------------------
#   Calculate metric
# ------------------------
# ------------------------------------------------------------------------------------- pr_tMean ----------------------------------------------------------------------------------------------------- # 
def calc_pr_tMean(experiment = ''):
    ds = xr.Dataset()
    for dataset in mV.datasets:
        if experiment == 'historical' and dataset in mV.observations:
            experiment_alt = 'obs'
        elif experiment == 'ssp585' and dataset in mV.observations:
            continue
        else:
            experiment_alt = experiment
        da, _ = vD.get_variable_data(switch, 'pr', dataset, experiment_alt)
        ds[dataset] = da.mean(dim = 'time').compute()
    return ds

def get_pr_calc_ds(experiment = ''):
    folder = f'{mV.folder_scratch}/temp_calc'
    filename = f'pr_tMean_{experiment}.nc'
    path = f'{folder}/{filename}'
    try:
        ds = xr.open_dataset(path)
        print(f'loaded or_tMean from temp_calc folder')
    except Exception as e:  # It's a good practice to catch specific exceptions
        print(f'Failed to load pr_tMean from temp_calc folder: {e}')
        print(f'Creating pr_tMean for {experiment}')
        ds = calc_pr_tMean(experiment)
        ds.to_netcdf(path, mode = 'w')
        print(f'saved pr_tMean')
    return ds


 # -------------------------------------------------------------------------------------- RMSE ----------------------------------------------------------------------------------------------------- # 
def get_d_squared(da_model, da_obs):
    return (da_model - da_obs) ** 2

def get_rmse(da_model, da_obs):
    ''' Root Mean Squared Error '''
    return np.round(np.sqrt(get_d_squared(da_model, da_obs).mean()), 2)    



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':              
    print(f'precipitation evaluation started')
    print(f'obs year range: {mV.obs_years[0]}')
    remove_temp_calc = False
    mF.remove_temp_calc() if remove_temp_calc else None
    remove_plots = True
    mFp.remove_test_plots() if remove_plots else None



 # ------------------------------------------------------------------------------------- get data ----------------------------------------------------------------------------------------------------- # 
    var_name = 'pr'
    units = 'mm/day'
    switch = {'test_sample': False, 'ocean_mask': False}

    ds_historical = get_pr_calc_ds(experiment = mV.experiments[0])
    ds_warm = get_pr_calc_ds(experiment = mV.experiments[1])
    # print(ds_historical)
    # exit()


# ----------------------------------------------------------------------------------------- plot ----------------------------------------------------------------------------------------------------- # 
    plot = False                                                                   # plot mean (individual) 
    if plot:
        vmin, vmax = 0, 10
        ds = ds_historical
        for dataset in mV.datasets:
            da_mean = ds[dataset]
            fig, ax = mFp.plot_scene(da_mean, cmap = 'Blues', ax_title = dataset, fig_title = f'pr_tMean', vmin = 0, vmax = 10, label = f'[{units}]')
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'{dataset}_pr_tMean')
            print('plotted tMean individual')

    plot = False                                                                   # plot mean difference (all together) 
    if plot:
        ds = ds_historical
        vmin, vmax = -10, 10
        ds_diff = xr.Dataset()
        for dataset in ds.data_vars:
            if dataset != mV.observations[0]:
                ds_diff[dataset] = ds[dataset] - ds[mV.observations[0]]
        fig, axes = mP.plot_dsScenes(ds_diff, label = f'[{units}]', title = f'(model - obs) {var_name} tMean (obs: {mV.obs_years[0]})', vmin = vmin, vmax = vmax, cmap = 'RdBu_r', variable_list = ds_diff.data_vars)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'd_{var_name}_tMean.png')
        print('plotted obs model difference')

    plot = False                                                                        # RMSE
    if plot:
        ds = ds_historical
        ds_rmse = xr.Dataset()
        for dataset in ds.data_vars:
            if dataset != mV.observations[0]:
                ds_rmse[dataset] = get_rmse(da_model = ds[dataset], da_obs = ds[mV.observations[0]])
        title = '' # f'RMSE_pwad'
        fig, ax = bP.plot_dsBar(ds_rmse, title = title, ylabel = units)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'RMSE_pr_barplot.png')
        # mFp.show_plot(fig, show_type = 'save_cwd', filename = f'RMSE_pr_barplot.pdf')
        print('plotted RMSE')
        saveit = False
        if saveit:
            folder = f'{mV.folder_scratch}/temp_calc'
            filename = f'pr_rmse_rank_historical.nc'
            path = f'{folder}/{filename}'
            datasets, alist = [], []
            for dataset in ds_rmse.data_vars:
                datasets.append(dataset)
                alist.append(ds_rmse[dataset])
            sorted_indices = np.argsort(alist)
            sorted_datasets = [datasets[i] for i in sorted_indices]
            ds_rank = xr.Dataset({'model_rank': xr.DataArray(sorted_datasets)})
            ds_rank.to_netcdf(path, mode = 'w')
            print(ds_rank)
            print(ds_rank['model_rank'])
            print('saved pr RMSE list')

        color_pwad = False
        if color_pwad:
            folder = f'{mV.folder_scratch}/temp_calc'
            filename = f'pwad_rmse_rank_historical.nc'
            path = f'{folder}/{filename}'
            models = xr.open_dataset(path)['model_rank'].data[0:14]
            title = '' # f'RMSE_pwad'
            fig, ax = bP.plot_dsBar(ds_rmse, title = title, ylabel = units, highlight_given = True, models_highlight = models)
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'RMSE_pr_barplot_pwad_color.png')
            # mFp.show_plot(fig, show_type = 'save_cwd', filename = f'RMSE_pr_barplot_pwad_color.pdf')
            print('plotted RMSE pwad color')
        
        color_pr_pwad = False
        if color_pwad:
            folder = f'{mV.folder_scratch}/temp_calc'
            filename = f'pwad_rmse_rank_historical.nc'
            path = f'{folder}/{filename}'
            models_pwad = xr.open_dataset(path)['model_rank'].data[0:14]

            folder = f'{mV.folder_scratch}/temp_calc'
            filename = f'pr_rmse_rank_historical.nc'
            path = f'{folder}/{filename}'
            models_pr = xr.open_dataset(path)['model_rank'].data[0:14]
            models = np.intersect1d(models_pwad, models_pr)
            print(models)

            title = '' # f'RMSE_pwad'
            fig, ax = bP.plot_dsBar(ds_rmse, title = title, ylabel = units, highlight_given = True, models_highlight = models)
            # mFp.show_plot(fig, show_type = 'save_cwd', filename = f'RMSE_pr_barplot_pwad_pr_color.png')
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'RMSE_pr_barplot_pwad_pr_color.pdf')
            print('plotted RMSE pwad pr color')


    plot = False                                                                   # plot mean (model mean) 
    if plot:
        ds = ds_historical
        data_arrays = [ds[var] for var in ds.data_vars if var != mV.observations[0]]
        combined = xr.concat(data_arrays, dim='model')
        da_mean = combined.mean(dim='model')
        fig, ax = mFp.plot_scene(da_mean, cmap = 'Blues', ax_title = 'model-mean', fig_title = f'pr_tMean', vmin = 0, vmax = 10, label = f'[{units}]')
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'pr_tMean_modelMean.png')
        print('plotted tMean model_mean')

    plot = True                                                                   # plot mean difference (model mean) (all models)
    if plot:
        ds = ds_historical
        print(ds)
        data_arrays = [ds[var] for var in ds.data_vars if var != mV.observations[0]]
        combined = xr.concat(data_arrays, dim='model')
        da_mean = combined.mean(dim='model')
        da_mean_difference = da_mean - ds[mV.observations[0]]
        fig, ax = mFp.plot_scene(da_mean_difference, cmap = 'RdBu_r', ax_title = 'CMIP6 (model-mean)', fig_title = '', vmin = -5, vmax = 5, label = f'[{units}]')
        # mFp.show_plot(fig, show_type = 'save_cwd', filename = f'pr_tMean_difference_modelMean.png')
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'pr_tMean_difference_modelMean.pdf')
        print('plotted tMean difference model_mean')

    plot = False                                                                   # plot mean difference (model mean) (models close to obs)
    if plot:
        ds = ds_historical
        ds = ds[models_closeObs]
        data_arrays = [ds[var] for var in ds.data_vars if var != mV.observations[0]]
        combined = xr.concat(data_arrays, dim='model')
        da_mean = combined.mean(dim='model')
        da_mean_difference = da_mean - ds_historical[mV.observations[0]]
        fig, ax = mFp.plot_scene(da_mean_difference, cmap = 'RdBu_r', ax_title = 'CMIP6 (model-mean) (closer to obs)', fig_title = '', vmin = -5, vmax = 5, label = f'[{units}]')
        # mFp.show_plot(fig, show_type = 'save_cwd', filename = f'pr_tMean_difference_modelMean_closeObs.png')
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'pr_tMean_difference_modelMean_closeObs.pdf')
        print('plotted tMean difference model_mean with models closer to obs')
























































