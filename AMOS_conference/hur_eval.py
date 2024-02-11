'''
# ---------------------
#      hur eval
# ---------------------
'''
# models_closeObs = ['CESM2-WACCM', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3', 'MRI-ESM2-0', 'NorESM2-MM', 'TaiESM1'] #(basd on hur and pwad RMSE)
models_closeObs = ['ACCESS-ESM1-5', 'CESM2-WACCM', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3', 'MRI-ESM2-0', 'NorESM2-MM', 'TaiESM1'] #, 'ACCESS-CM2']


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
import get_data.metric_data     as mD



# ------------------------
#   Calculate metric
# ------------------------
# ------------------------------------------------------------------------------------ hur_tMean ----------------------------------------------------------------------------------------------------- # 
def calc_hur_tMean(experiment = ''):
    ds = xr.Dataset()
    for dataset in mV.datasets:
        if experiment == 'historical' and dataset in mV.observations:
            experiment_alt = 'obs'
        elif experiment == 'ssp585' and dataset in mV.observations:
            continue
        else:
            experiment_alt = experiment
        ds[dataset] = mD.load_metric_file('hur', 'hur_700hpa_tMean', dataset, experiment_alt).compute()
    return ds

def get_hur_tMean(experiment = ''):
    folder = f'{mV.folder_scratch}/temp_calc'
    filename = f'hur_tMean_{experiment}.nc'
    path = f'{folder}/{filename}'
    try:
        ds = xr.open_dataset(path)
        print(f'loaded hur_tMean from temp_calc folder')
    except Exception as e:
        print(f'Failed to load pr_tMean from temp_calc folder: {e}')
        print(f'Creating hur_tMean for {experiment}')
        ds = calc_hur_tMean(experiment)
        ds.to_netcdf(path, mode = 'w')
        print(f'saved hur_tMean')
    return ds


 # --------------------------------------------------------------------------------------- RMSE ----------------------------------------------------------------------------------------------------- # 
def get_d_squared(da_model, da_obs):
    return (da_model - da_obs) ** 2

def get_rmse(da_model, da_obs):
    ''' Root Mean Squared Error '''
    return np.round(np.sqrt(get_d_squared(da_model, da_obs).mean()), 2)    



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':              
    print(f'Relative humidity evaluation started')
    print(f'obs year range: {mV.obs_years[0]}')
    remove_temp_calc = False
    mF.remove_temp_calc() if remove_temp_calc else None
    remove_plots = True
    mFp.remove_test_plots() if remove_plots else None

    ds_historical = get_hur_tMean(experiment = mV.experiments[0])

    plot = False                                                                   # plot mean (individual) 
    if plot:
        label = 'Relative humidity [%]'
        vmin, vmax = 0, 100
        ds = ds_historical
        for dataset in mV.datasets:
            da_mean = ds[dataset]
            fig, ax = mFp.plot_scene(da_mean, cmap = 'Blues', ax_title = dataset, fig_title = f'hur_tMean', label = label, vmin = vmin, vmax = vmax)
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'{dataset}_hur_tMean')
            print('plotted hur_tMean individual')

    plot = False                                                                   # plot mean difference (all together) 
    if plot:
        label = 'Relative humidity [%]'
        ds = ds_historical
        vmin, vmax = -20, 20 #-10, 10
        ds_diff = xr.Dataset()
        for dataset in ds.data_vars:
            if dataset != mV.observations[0]:
                ds_diff[dataset] = ds[dataset] - ds[mV.observations[0]]
        fig, axes = mP.plot_dsScenes(ds_diff, label = label, title = f'(model - obs) hur tMean (obs: {mV.obs_years[0]})', vmin = vmin, vmax = vmax, cmap = 'RdBu_r', variable_list = ds_diff.data_vars)
        # mFp.show_plot(fig, show_type = 'save_cwd', filename = f'd_hur_tMean.png')
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'd_hur_tMean.pdf')
        print('plotted obs model difference')

    plot = True                                                                        # RMSE
    if plot:
        label = 'Relative humidity [%]'
        ds = ds_historical
        ds_rmse = xr.Dataset()
        for dataset in ds.data_vars:
            if dataset != mV.observations[0]:
                ds_rmse[dataset] = get_rmse(da_model = ds[dataset], da_obs = ds[mV.observations[0]])
        title = '' # f'RMSE_pwad'
        fig, ax = bP.plot_dsBar(ds_rmse, title = title, ylabel = label)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'RMSE_hur_barplot.png')
        # mFp.show_plot(fig, show_type = 'save_cwd', filename = f'RMSE_pr_barplot.pdf')
        print('plotted RMSE')
        saveit = True
        if saveit:
            folder = f'{mV.folder_scratch}/temp_calc'
            filename = f'hur_rmse_rank_historical.nc'
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
            print('saved hur RMSE list')

        color_pwad = True
        if color_pwad:
            folder = f'{mV.folder_scratch}/temp_calc'
            filename = f'pwad_rmse_rank_historical.nc'
            path = f'{folder}/{filename}'
            models = xr.open_dataset(path)['model_rank'].data[0:15]
            title = '' # f'RMSE_pwad'
            fig, ax = bP.plot_dsBar(ds_rmse, title = title, ylabel = label, highlight_given = True, models_highlight = models)
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'RMSE_hur_barplot_pwad_color.png')
            # mFp.show_plot(fig, show_type = 'save_cwd', filename = f'RMSE_pr_barplot_pwad_color.pdf')
            print('plotted RMSE pwad color')
        
        color_hur_pwad = True
        if color_hur_pwad:
            folder = f'{mV.folder_scratch}/temp_calc'
            filename = f'pwad_rmse_rank_historical.nc'
            path = f'{folder}/{filename}'
            models_pwad = xr.open_dataset(path)['model_rank'].data[0:14]

            folder = f'{mV.folder_scratch}/temp_calc'
            filename = f'hur_rmse_rank_historical.nc'
            path = f'{folder}/{filename}'
            models_hur = xr.open_dataset(path)['model_rank'].data[0:15]
            models = np.intersect1d(models_pwad, models_hur)
            print(f'models intersecting pwad and hur eval {models}')

            title = '' # f'RMSE_pwad'
            fig, ax = bP.plot_dsBar(ds_rmse, title = title, ylabel = label, highlight_given = True, models_highlight = models)
            # mFp.show_plot(fig, show_type = 'save_cwd', filename = f'RMSE_hur_barplot_pwad_hur_color.png')
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'RMSE_hur_barplot_pwad_hur_color.pdf')
            print('plotted RMSE pwad pr color')

    plot = False                                                                   # plot mean difference (model mean) (all models)
    if plot:
        label = 'Relative humidity [%]'
        vmin, vmax = -35, 35 #-10, 10
        ds = ds_historical
        data_arrays = [ds[var] for var in ds.data_vars if var != mV.observations[0]]
        combined = xr.concat(data_arrays, dim='model')
        da_mean = combined.mean(dim='model')
        da_mean_difference = da_mean - ds[mV.observations[0]]
        fig, ax = mFp.plot_scene(da_mean_difference, cmap = 'RdBu_r', ax_title = 'CMIP6 (model-mean)', fig_title = '', vmin = vmin, vmax = vmax, label = label)
        # mFp.show_plot(fig, show_type = 'save_cwd', filename = f'hur_tMean_difference_modelMean.png')
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'hur_tMean_difference_modelMean.pdf')
        print('plotted tMean difference model_mean')
        all_models = da_mean_difference 

    plot = True                                                                   # plot mean difference (model mean) (models close to obs)
    if plot:
        label = 'Relative humidity [%]'
        vmin, vmax = -35, 35 #-10, 10
        ds = ds_historical
        ds = ds[models_closeObs]
        data_arrays = [ds[var] for var in ds.data_vars if var != mV.observations[0]]
        combined = xr.concat(data_arrays, dim='model')
        da_mean = combined.mean(dim='model')
        da_mean_difference = da_mean - ds_historical[mV.observations[0]]
        fig, ax = mFp.plot_scene(da_mean_difference, cmap = 'RdBu_r', ax_title = 'CMIP6 (model-mean) (closer to obs)', fig_title = '', vmin = vmin, vmax = vmax, label = label)
        # mFp.show_plot(fig, show_type = 'save_cwd', filename = f'hur_tMean_difference_modelMean_closeObs.png')
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'hur_tMean_difference_modelMean_closeObs.pdf')
        print('plotted tMean difference model_mean with models closer to obs')
        all_models_closeObs = da_mean_difference 


    plot = False                                                                   # difference all and close to obs
    if plot:
        vmin, vmax = -10, 10 #-10, 10
        difference = all_models - all_models_closeObs
        fig, ax = mFp.plot_scene(da_mean_difference, cmap = 'RdBu_r', ax_title = 'CMIP6 (model-mean) (closer to obs)', fig_title = '', vmin = vmin, vmax = vmax, label = label)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'hur_tMean_d_model_closetoobs.png')































